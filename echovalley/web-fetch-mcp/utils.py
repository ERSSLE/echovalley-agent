from pyquery import PyQuery
from urllib.parse import urlparse
import re
from bs4 import BeautifulSoup
from logger import logger
from playwright.async_api._generated import Page
from enum import Enum
from config import Config
from pydantic import BaseModel,Field
from typing import Dict,List,Any
from markdownify import markdownify
from readability import Document
from abc import ABC,abstractmethod

class ToolState(str,Enum):
    IDLE = 'IDLE'
    USING = 'USING'
class UsingError(Exception):
    pass

config = Config()

class BaseExtractor(BaseModel,ABC):
    page_item_base_selectors: List[str] = Field(default_factory=lambda: [],description='页面要提取项的选择器列表')
    next_page_selector: str | None = None
    extract_method: str = 'process_page' # process_page,process_extract

    @abstractmethod
    def _item_filter(self,item:PyQuery):
        """过滤器，过滤掉一些不需要的项，比如广告，True剔除，False保留"""
    
    @abstractmethod
    def _extract(self,item:PyQuery) -> Dict[str,str]:
        """返回值应该是{title:str,link:str,domain:str,description:str}"""

    async def process_page(self,page: Page):
        content = await page.content()
        doc = PyQuery(content)
        results = []
        for selector in self.page_item_base_selectors:
            for item in doc(selector).items():
                # print(item)
                if self._item_filter(item):
                    continue
                else:
                    result = self._extract(item)
                    if result:
                        results.append(result)
        return results
    
    async def process_extract(self,page:Page,ctx:Dict[str,Any],
                              clickable_child:str,
                              try_num:int,
                              mode:str='text',
                              parse_method:str='general',
                              max_items=10):
        """
        通过点击页面元素获取各元素指向的页面内容
        Args:
            page: 父页面
            ctx: 上下文
            clickable_child: 每个项下实际可点击的子元素选择器
            try_num: 出错后的重试次数
            mode: 返回结果的格式 text,md
            parse_method: 解析方式
            max_items: 最大获取项数量
        """
        content = await page.content()
        doc = PyQuery(content)
        results = []
        item_count = 0
        for base_selector in self.page_item_base_selectors:
            for n,item in enumerate(doc(base_selector).items()):
                if self._item_filter(item):
                    continue
                else:
                    logger.info(f'正在访问{base_selector}的第{n+1}/{doc(base_selector).length}项...')
                    success = False
                    selector = f'{base_selector}:nth-child({n+1}) {clickable_child}'
                    result = self._extract(doc(f'{base_selector}:nth-child({n+1})'))
                    for attempt in range(try_num):
                        new_page = None
                        await page.bring_to_front()
                        try:
                            async with ctx['context'].expect_page() as new_page_info:
                                await page.locator(selector).click()
                            new_page = await new_page_info.value
                            await new_page.wait_for_load_state()
                            await new_page.bring_to_front()
                            result['content'] = await get_page_content(new_page,mode,source=parse_method)
                            success = True
                        except Exception as e:
                            logger.warning(f'访问第{n}个项失败,(尝试{attempt}): {str(e)}')
                        finally:
                            if new_page is not None:
                                await new_page.close()
                                if success:
                                    break
                    if not success:
                        result['content'] = ''
                    results.append(result)
                    item_count += 1
                    if item_count >= max_items:
                        break
            if item_count > max_items:
                break
        return results
    async def navigate_to_next_page(self, page: Page, page_num: int | None = None):
        """导航到搜索结果下一页（带重试）"""
        if self.next_page_selector is None:
            logger.warning("该提取器不支持导航到下一页")
            return False
        for attempt in range(config.access_retry):
            try:
                element = await page.wait_for_selector(self.next_page_selector,timeout=3000)
                await element.click()
                await page.wait_for_load_state(config.page_load_state)
                return True
            except Exception as e:
                if page_num is None:
                    logger.warning(f"导航到下一页失败 (尝试 {attempt}): {str(e)}")
                else:
                    logger.warning(f"导航到第 {page_num} 页失败 (尝试 {attempt}): {str(e)}")
                if attempt == config.access_retry:
                    return False
                await page.reload(wait_until=config.page_load_state)
        return False
    
class BingExtractor(BaseExtractor):
    def _item_filter(self, item:PyQuery):
        if ((item.find('.videosvc').length > 0) or \
            (item.parents('.b_ad').length > 0) or \
            (item.parents('.b_adTop').length > 0)
            ):
            return True
        else:
            return False
    def _extract(self, item:PyQuery):
        try:
            title_elem = item('h2 a')
            if not title_elem:
                return None
            title = title_elem.text().strip()
            link = title_elem.attr('href')
            if not link or 'bing.com/aclick?' in link:
                return None
            description = item('.b_caption p').text() or item('.b_lineclamp2').text()
            if not description:
                description = item('.b_paractl').text()  # 备用选择器
            description = re.sub(r'\s+', ' ', description).strip() if description else ""
            domain = ""
            try:
                domain = urlparse(link).netloc
                if domain.startswith("www."):
                    domain = domain[4:]
            except:
                pass
            return {
                'title': title,
                'link': link,
                'domain': domain,
                'description': description
            }
        except Exception as e:
            logger.warning(f"提取结果项失败: {str(e)}")
            return None

class GoogleExtractor(BaseExtractor):
    def _item_filter(self, item:PyQuery):
        if ((item('div[data-snf]').length > 0)
            ):
            return False
        else:
            return True
    def _extract(self, item:PyQuery):
        try:
            title_elem = item('div[data-snhf="0"] a.zReHs[href]')
            if not title_elem:
                return None
            title = title_elem('h3').text().strip()
            link = title_elem.attr('href')
            if not link:
                return None
            description = item('div[data-sncf="1"]').text()
            description = re.sub(r'\s+', ' ', description).strip() if description else ""
            domain = ""
            try:
                domain = urlparse(link).netloc
                if domain.startswith("www."):
                    domain = domain[4:]
            except:
                pass
            return {
                'title': title,
                'link': link,
                'domain': domain,
                'description': description
            }
        except Exception as e:
            logger.warning(f"提取结果项失败: {str(e)}")
            return None

class PageContentExtractor(BaseModel):
    enable_summary: bool = True
    enable_other_content: bool = True

    def _css_selector_region(self,html:str,content_region:str):
        if content_region:
            content = PyQuery(html)(content_region)
            html = content.html()
            return html
        return html
    def _readablility_summary(self,html:str):
        doc = Document(html)
        clean_html = doc.summary()
        return clean_html
    
    def _other_content(self,soup:BeautifulSoup):
        page_hrefs = []
        for a in soup('a'):
            href = a.attrs.get('href',None)
            if href and ('https://zhida.zhihu.com' not in href): #排除知乎zhida
                atext = a.get_text().strip()
                page_hrefs.append(f'标签文本：{atext}，链接：{href}')
        page_hrefs = '可能有价值的其它链接：\n' + '\n'.join(page_hrefs)
        return page_hrefs
    
    async def extract_content(self,page:Page,mode:str='text',
                              content_region:str|None = None,
                              decompose_elements: List[str]|None = None):
        """提取页面内容
        Args:
            mode: 输出格式md, text
            content_region: 要提取的页面内容区域选择器
            decompose_elements: 要剔除的一些元素
        """
        html_content = await page.content()
        if self.enable_summary:
            html_content = self._readablility_summary(html_content)
        if content_region is not None:
            html_content = self._css_selector_region(html_content,content_region)
        soup: BeautifulSoup = BeautifulSoup(html_content,'html.parser')
        if decompose_elements:
            for label in decompose_elements:
                label.decompose()
        if mode == 'text':
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines() if line.strip())
            text = '\n'.join(lines)
            if self.enable_other_content:
                other_content = self._other_content(soup)
                text = f'{text}\n{other_content}'
            return text
        elif mode == 'md':
            return markdownify(soup.prettify())
    
all_extractors = dict(
    bing = BingExtractor(
        page_item_base_selectors=['#b_context #topw .b_algo','#b_results .b_algo'],
        next_page_selector='a.sb_pagN'
    ),
    google = GoogleExtractor(
        page_item_base_selectors=['#search div[data-rpos]'],
        next_page_selector='#pnnext'
    ),
)

page_content_extractor = PageContentExtractor()

all_content_extractors = dict(
    bing = page_content_extractor.extract_content, # 这里bing与google是一样的
    google = page_content_extractor.extract_content,
    general = page_content_extractor.extract_content,
)

async def navigate_to_next_page(page: Page, page_num: int, source='bing'):
    """导航到搜索结果下一页（带重试）"""
    return await all_extractors[source].navigate_to_next_page(page,page_num)
    
async def process_results_page(page: Page,source='bing',
                               ctx:Dict[str,Any] | None=None,try_num:int=None,mode='text',max_items:int=10
                               ):
    if all_extractors[source].extract_method == 'process_page':
        return await all_extractors[source].process_page(page)
    elif all_extractors[source].extract_method == 'process_extract':
        # 根据需要可以扩展该功能
        return
    
async def get_page_content(page: Page,mode='md',source:str|None=None):
    return await all_content_extractors[source](page=page,mode=mode)
