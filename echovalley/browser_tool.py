import asyncio
from playwright.async_api import async_playwright
from playwright.async_api._generated import Page
from typing import Any,Dict,List
from utils import extract_html_main_content,prepare_clickable,WebAPI
from pydantic import BaseModel
from dom.dom import DomService,click_element_node,input_text_element_node, remove_highlights,DOMState
from logging import getLogger
from base_tool import ToolResult,ToolError,BaseTool,ToolState


logger = getLogger(__name__)

class BrowserToolConfig(BaseModel):
    """"""
    highlight_elements: bool = True,
    focus_element: int = -1,
    viewport_expansion: int = 500
    max_async_context_num: int = 10
    # browser_proxy_server: str | None = 'http://localhost:7890' # 代理设置
    browser_proxy_server: str | None = None
    headless: bool = False
    save_downloads_path: str | None = None
    timeout: int = 10000 # 毫秒
    scroll_amount: int = 800 # 像素
    load_state: str = 'load' # ["commit", "domcontentloaded", "load", "networkidle"]

browser_tool_config = BrowserToolConfig()

class BrowserTool(BaseTool):
    """
    base_contexts: {
        'contexts': [{'state': ToolState, 
                      'id': int, 
                      'context': BrowserContext,
                      'current_page_id':int
                      'pages': [{'page_id':int,'page':Page,'domstate': DOMState} ...]} ...]
        'playwright': PlayWright,
        'browser': Browser
    }
    """
    name: str = 'browser'
    # _current_page: Dict[str,str|DOMState|Page] | None = None
    config: BrowserToolConfig = None
    base_context: Dict[str, Any] | None = None

    @classmethod
    async def create(cls,tool_config = browser_tool_config):
        """
        """
        playwright = await async_playwright().start()
        proxy = {'server':tool_config.browser_proxy_server} if tool_config.browser_proxy_server else None
        browser = await playwright.chromium.launch(headless=tool_config.headless,
                                                   proxy=proxy,
                                                   args=[
                                                        '--disable-web-security',
                                                        '--disable-features=TranslateUI',
                                                        '--disable-extensions',
                                                        '--no-sandbox',
                                                        '--disable-dev-shm-usage',
                                                        '--window-size=1920,1080'
                                                    ]
                                                   )
        contexts = [{'state':ToolState.IDLE,'id':i, 'context': None,'current_page_id':None , 'pages': []} \
                    for i in range(tool_config.max_async_context_num)]
        return cls(base_context={'contexts':contexts,'playwright':playwright,'browser':browser}, config=tool_config)

    async def invoke(self,
                     method: str,
                     url: str = 'none',
                     text: str = 'none',
                     node_id: int = -1,
                     context_id: int = -1,
                     page_id: int = -1,
                     amount: int = -1,
                     keys: str = 'none',
                     seconds_to_wait: int = -1,
                     goal: str = 'none',
                     ):
        """通过浏览器资源，用合适的method和method对应的其它参数来执行操作。
        该方法包含了使用浏览器对网页进行的一些基本操作，包括导航，交互，页面滚动，内容提取，标签管理等
        根据不同的情况，不是所有的参数都是必要的，你应该根据上下文和逻辑推理确定该传入哪些必要的参数以执行操作，对不必要的参数使用该参数描述指定的缺省值替代。
        在同一个任务中，每次调用之间是上下文一致的，始终保持了浏览器操作的连续性。只有明确该任务完结时才会被释放之前操作的状态。
        当你需要浏览网页，填写表格，提取内容，点击按钮，进行网络搜索时，请使用此功能。
        关键的能力包括：
        导航：包含进行网页访问，刷新网页，返回操作等。
        交互：点击元素、输入文本、从下拉菜单中选择、发送键盘命令等。
        滚动：按像素数量向上/向下滚动或滚动到特定文本
        内容提取：从网页中提取内容,你应该在需要的时候使用此功能从当前页面提取内容。否则你只能看到一些可交互元素，而无法获取网页主要内容。该方法是extract_content。
        标签管理：在标签之间切换、打开新标签或关闭标签
        注意：使用元素索引时，请参考当前浏览器状态下显示的编号元素。

        Args:
            method: 浏览器的操作方法，也就是method可能的取值,包含 \
                new_page (创建新标签页并导航到指定url,该方法只需要额外参数[url]), \
                input_text (向目标文本框输入文本并确定，该方法只需要额外参数[text,node_id]),\
                switch_page (从当前页面切换到另外一个页面，该方法只需要额外参数[page_id]),\
                go_back (将当前页面回退到上一个访问的网址，该方法不需要额外参数),\
                go_fowward (将当前页面前进到下一个访问的网址，该方法不需要额外参数),\
                refresh (刷新当前页面，该方法不需要额外的参数),\
                scroll_up (向上滚动当前页面，该方法需要额外的参数[amount]),\
                scroll_down (向下滚动当前页面，该方法需要额外的参数[amount]),\
                scroll_to_text (将当前页面滚动到特定文本位置，该方法需要额外的参数[text]),\
                click_element (点击当前页面的元素，该方法需要额外参数[node_id]),\
                goto_url (将当前页面导航到特定url，该方法需要额外参数[url]),\
                send_keys (从键盘键入按键值，该方法需要额外参数[keys]),\
                get_dropdown_options (从当前页面点击下拉候选框，该方法需要额外参数[node_id]),\
                select_dropdown_option (从下拉候选框中选择一项，该方法需要额外参数[node_id,text]),\
                wait (在当前页面等待数秒，该方法需要额外参数[seconds_to_wait]),\
                extract_content (从浏览器中当前页面提取内容,该方法不需要额外的参数,\
                close_page (关闭标签页面，该方法需要额外参数[page_id]),\
                close_current_page (关闭当前标签页面，该方法不需要额外参数)

            url: 要访问的目标网址, 缺省值none
            text: 要输入的文字或者要查询的文字，缺省值none
            node_id: 页面中的节点id, 用于确定element, 缺省值-1
            context_id: 浏览器上下文id,这个参数应用会根据调用历史来确定，你只需要稳定地给出-1即可。缺省值-1
            page_id: 页面的id,用于区分不同的页面，缺省值-1
            amount: 页面向上或向下滚动的距离(像素值),缺省值-1,如果用缺省值则使用工具配置的默认值
            keys: 从键盘向当前页面输入的键盘键值，缺省值none
            seconds_to_wait: 当前页面等待的时间(秒)，缺省值-1
            goal: 从页面当中提取的内容，缺省值-1

        """
        if context_id == -1:
            ctx = await self._get_idle_context()
        elif context_id >= len(self.base_context['contexts']):
            return ToolResult(message={'result':'context_id超出资源限制，不能获取相应的上下文'},state='异常')
        else:
            ctx = await self._get_context_by_id(context_id)
        try:
            if method == 'new_page':
                if url == 'none' or '':
                    return ToolResult(message={'result':'方法new_page需要额外的参数url'},state='异常')
                return await self._new_page(ctx,url)
            elif method == 'input_text':
                if (node_id == -1) or (text == 'none'):
                    return ToolResult(message={'result':f'方法input_text需要额外的参数node_id与text'},state='异常')
                return await self._input_text(ctx,node_id,text)
            elif method == 'switch_page':
                if (page_id == -1):
                    return ToolResult(message={'result':f'方法switch_page需要额外的参数page_id'},state='异常')
                return await self._switch_page(ctx,page_id)
            elif method == 'go_back':
                return await self._go_back(ctx)
            elif method == 'go_forward':
                return await self._go_forward(ctx)
            elif method == 'refresh':
                return await self._refresh(ctx)
            elif method == 'scroll_up':
                return await self._scroll_up(ctx,amount)
            elif method == 'scroll_down':
                return await self._scroll_down(ctx,amount)
            elif method == 'scroll_to_text':
                if text == 'none':
                    return ToolResult(message={'result':f'方法scroll_to_text需要参数text'},state='异常')
                return await self._scroll_to_text(ctx,text)
            elif method == 'click_element':
                if node_id == -1:
                    return ToolResult(message={'result':f'方法click_element需要额外的参数node_id'},state='异常')
                return await self._click_element(ctx,node_id)
            elif method == 'goto_url': # 在当前标签页
                if url == 'none':
                    return ToolResult(message={'result':f'方法goto_url需要额外参数url'},state='异常')
                return await self._goto_url(ctx,url)
            elif method == 'send_keys':
                if keys == 'none':
                    return ToolResult(message={'result':f'方法send_keys需要额外参数keys'},state='异常')
                return await self._send_keys(ctx,keys)
            elif method == 'get_dropdown_options':
                if node_id == -1:
                    return ToolResult(message={'result':f'方法get_dropdown_options需要额外参数node_id'},state='异常')
                return await self._get_dropdown_options(ctx,node_id)
            elif method == 'select_dropdown_option':
                if (node_id == -1) or (text == 'none'):
                    return ToolResult(message={'result': f'方法select_dropdown_option需要参数node_id与text'}, state='异常')
                return await self._select_dropdown_option(ctx,node_id,text)
            elif method == 'wait':
                if (seconds_to_wait == -1):
                    return ToolResult(message={'result':f'方法wait需要参数seconds_to_wait'},state='异常')
                else:
                    await asyncio.sleep(seconds_to_wait)
                    return ToolResult(message={'result':f'等待了{seconds_to_wait}秒'},state='成功')
            elif method == 'extract_content':
                return await self._extract_content(ctx)
            elif method == 'close_page':
                if page_id == -1:
                    return ToolResult(message={'result': f'方法close_page需要额外的参数page_id'}, state='异常')
                return await self._close_page(ctx,page_id)
            elif method == 'close_current_page':
                return await self._close_current_page(ctx)
        except:
            return ToolResult(message={'result':f'无法正确创建新标签页并导航'},state='异常')
    
    #===============================================
    # 对应浏览器操作的原子方法
    async def _get_current_browser_states(self,ctx: Dict[str,Any]):
        page_nums = len(ctx['pages'])
        page_descs = []
        for page in ctx['pages']:
            page_url = page['page'].url
            page_title = await page['page'].title()
            page_id = page['page_id']

            scroll_y = await page['page'].evaluate('window.scrollY')
            viewport_height = await page['page'].evaluate('window.innerHeight')
            total_height = await page['page'].evaluate('document.documentElement.scrollHeight')
            pixels_above = scroll_y
            pixels_below = total_height - (scroll_y + viewport_height)
            page_desc = (f"page_id={page_id}: page_url={page_url},page_title={page_title},"
                        f"pixels_above={pixels_above},pixels_below={pixels_below}")
            page_descs.append(page_desc)
        page_descs = '\n'.join(page_descs)
        page_ids = [page['page_id'] for page in ctx['pages']]
        page = self.current_page(ctx)
        current_page_id = page['page_id'] if page else 'none'
        current_page_clickable_elements = page['domstate'].element_tree.clickable_elements_to_string() if page else 'none'
        current_page_clickable_elements = prepare_clickable(current_page_clickable_elements)
        states_str = f"""当前浏览器状态：
总共有{page_nums}个页面，页面page_id分别为{page_ids}
当前各个页面的状态：
{page_descs}

当前页面的page_id={current_page_id}
当前页面的可点击元素列在了下面，每个元素前面标明了node_id
<clickable_elements_start>\n{current_page_clickable_elements}\n<clickable_elements_end>
        """
        return states_str


    async def _new_page(self,ctx: Dict[str,Any],url:str) -> ToolResult:
        try:
            page = await ctx['context'].new_page()
            await page.goto(url)
            await page.wait_for_load_state(self.config.load_state)
            page = await self._save_page(ctx,page)
            await self.set_current_page(ctx,page)
            return ToolResult(message={'result':f'创建了新标签页，并成功导航到了{url}, 标签页id={page['page_id']}',
                                    'page_id':page['page_id']},state='成功')
        except Exception as e:
            return ToolResult(message={'result':f'创建新标签页并导航到{url}时出现错误{str(e)}'},state='异常')
        
    async def _input_text(self,ctx: Dict[str,Any], node_id:int,text:str) -> ToolResult:
        try:
            page = self.current_page(ctx)
            element = page['domstate'].selector_map.get(node_id,None)
            if element is None:
                return ToolResult(message={'result':f'本页面没有找到索引为{node_id}的元素'},state='异常')
            await input_text_element_node(page['page'],element,text)
            return ToolResult(message={'result':f'已将文本`{text}`输入到了索引为{node_id}的元素'},state='成功')
        except Exception as e:
            return ToolResult(message={'result':f'由于引发了异常,未能在索引为{node_id}的元素输入文本`{text}`，异常原因{str(e)}'},
                              state='异常')
    
    async def _click_element(self,ctx: Dict[str,Any], node_id: int) -> ToolResult:
        try:
            page = self.current_page(ctx)
            element = page['domstate'].selector_map.get(node_id,None)
            if element is None:
                return ToolResult(message={'result':f'本页面没有找到索引为{node_id}的元素'},state='异常')
            download_path = await click_element_node(page['page'],element,self.config.save_downloads_path)
            if len(ctx['context'].pages) > len(ctx['pages']):
                _ = await self._update_pages(ctx)
                await self.set_current_page(ctx,ctx['pages'][-1])
            else:
                await self._update_page(page)
            if download_path:
                return ToolResult(message={'result':f'点击了索引为{node_id}的元素，下载文件在{download_path}'},state='成功')
            else:
                return ToolResult(message={'result':f'点击了索引为{node_id}的元素'},state='成功')
        except Exception as e:
            return ToolResult(message={'result':f'没能成功点击索引为{node_id}的元素，出现异常{str(e)}'},state='异常')
        
    async def _switch_page(self,ctx: Dict[str,Any], page_id: int) -> ToolResult:
        try:
            page = self._get_page_by_id(ctx,page_id)
            await self.set_current_page(ctx,page)
            return ToolResult(message={'result':f'将page_id={page_id}的页面设置成了当前页面'},state='成功')
        except Exception as e:
            return ToolResult(message={'result':f'设置page_id={page_id}的页面为当前页面时出现异常{str(e)}'},state='异常')
        
    async def _go_back(self,ctx:Dict[str,Any]) -> ToolResult:
        try:
            page = self.current_page(ctx)
            await page['page'].go_back(timeout=self.config.timeout)
            await page['page'].wait_for_load_state(self.config.load_state)
            await self._update_page(page)
            return ToolResult(message={'result':f'页面回退到了上一个访问页'},state='成功')
        except Exception as e:
            return ToolResult(message={'result':f'页面在回退的时候出现异常{str(e)}'},state='异常')
        
    async def _go_forward(self,ctx:Dict[str,Any]) -> ToolResult:
        try:
            page = self.current_page(ctx)
            await page['page'].go_forward(timeout=self.config.timeout)
            await page['page'].wait_for_load_state(self.config.load_state)
            await self._update_page(page)
            return ToolResult(message={'result':f'页面前进到了下一个访问页'},state='成功')
        except Exception as e:
            return ToolResult(message={'result':f'页面在前进的时候出现异常{str(e)}'},state='异常')
        
    async def _refresh(self,ctx:Dict[str,Any]) -> ToolResult:
        try:
            page = self.current_page(ctx)
            await page['page'].reload(timeout=self.config.timeout)
            await page['page'].wait_for_load_state(self.config.load_state)
            await self._update_page(page)
            return ToolResult(message={'result':f'成功刷新了当前页面'},state='成功')
        except Exception as e:
            return ToolResult(message={'result':f'刷新当前页面时出现了异常{str(e)}'},state='异常')
        
    async def _scroll_up(self,ctx:Dict[str,Any],amount: int|None = None) -> ToolResult:
        try:
            amount = amount if (bool(amount) and (amount != -1)) else self.config.scroll_amount
            page = self.current_page(ctx)
            await page['page'].evaluate(f"window.scrollBy(0, {-1 * amount});")
            await page['page'].wait_for_load_state(self.config.load_state)
            await self._update_page(page)
            return ToolResult(message={'result':f'页面向上滚动了{amount}像素'},state='成功')
        except Exception as e:
            return ToolResult(message={'result':f'页面向上滚动时引发异常{str(e)}'},state='异常')
        
    async def _scroll_down(self,ctx:Dict[str,Any],amount: int|None = None) -> ToolResult:
        try:
            amount = amount if (bool(amount) and (amount != -1)) else self.config.scroll_amount
            page = self.current_page(ctx)
            await page['page'].evaluate(f"window.scrollBy(0, {1 * amount});")
            await page['page'].wait_for_load_state(self.config.load_state)
            await self._update_page(page)
            return ToolResult(message={'result':f'页面向下滚动了{amount}像素'},state='成功')
        except Exception as e:
            return ToolResult(message={'result':f'页面向下滚动时引发异常{str(e)}'},state='异常')
        
    async def _scroll_to_text(self,ctx:Dict[str,Any],text: str) -> ToolResult:
        try:
            page = self.current_page(ctx)
            locator = page['page'].get_by_text(text,exact=False)
            await locator.scroll_into_view_if_needed()
            await page['page'].wait_for_load_state(self.config.load_state)
            await self._update_page(page)
            return ToolResult(message={'result':f'滚动到了{text}'},state='成功')
        except Exception as e:
            return ToolResult(message={'result':f'未能滚动到{text}，异常{str(e)}'},state='异常')
        
    async def _goto_url(self,ctx: Dict[str,Any], url: str) -> ToolResult:
        try:
            page = self.current_page(ctx)
            if page is None:
                page = await ctx['context'].new_page()
                await page.goto(url)
                await page.wait_for_load_state(self.config.load_state)
                page = await self._save_page(ctx,page)
            else:
                await page['page'].goto(url)
                await page['page'].wait_for_load_state(self.config.load_state)
                await self._update_page(page)
            return ToolResult(message={'result':f'成功访问了{url}'},state='成功')
        except Exception as e:
            return ToolResult(message={'result':f'访问{url}时出现了异常{str(e)}'},state='异常')
        
    async def _send_keys(self,ctx: Dict[str,Any], keys:str) -> ToolResult:
        try:
            page = self.current_page(ctx)
            await page['page'].keyboard.press(keys)
            if len(ctx['context'].pages) > len(ctx['pages']):
                _ = await self._update_pages(ctx)
                await self.set_current_page(ctx,ctx['pages'][-1])
            else:
                await self._update_page(page)
            return ToolResult(message={'result':f'从键盘输入了键`{keys}`'},state='成功')
        except Exception as e:
            return ToolResult(message={'result':f'从键盘输入键`{keys}`时出现异常{str(e)}'},state='异常')
        
    async def _get_dropdown_options(self,ctx:Dict[str,Any],node_id:int) -> ToolResult:
        try:
            page = self.current_page(ctx)
            element = page['domstate'].selector_map.get(node_id,None)
            if element is None:
                return ToolResult(message={'result':f'本页面没有找到索引为{node_id}的元素'},state='异常')
            options = await page['page'].evaluate("""
                (xpath) => {
                    const select = document.evaluate(xpath, document, null,
                        XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue;
                    if (!select) return null;
                    return Array.from(select.options).map(opt => ({
                        text: opt.text,
                        value: opt.value,
                        index: opt.index
                    }));
                }
            """,
                element.xpath,
            )
            return ToolResult(message={'result':f"下拉选项：{options}"},state='成功')
        except Exception as e:
            return ToolResult(message={'result':f'索引为{node_id}的元素获取下拉选项时出现异常{str(e)}'},state='异常')
    
    async def _select_dropdown_option(self,ctx:Dict[str,Any],node_id:int, text:str):
        try:
            page = self.current_page(ctx)
            element = page['domstate'].selector_map.get(node_id,None)
            if element is None:
                return ToolResult(message={'result':f'本页面没有找到索引为{node_id}的元素'},state='异常')
            await page['page'].select_option(element.xpath,label=text)
            return ToolResult(message={'result': f'从索引为{node_id}的下拉框中选择了选项{text}'},state='成功')
        except Exception as e:
            return ToolResult(message={'result': f'从索引为{node_id}的元素以下拉框进行选择时引发异常{str(e)}'},state='异常')
        
    async def _close_page(self,ctx: Dict[str,Any], page_id: int) -> ToolResult:
        try:
            current_page = self.current_page(ctx)
            current_page_id = current_page['page_id']
            closed = False
            for i,page in enumerate(ctx['pages']):
                if page['page_id'] == page_id:
                    await page['page'].close()
                    closed = True
                    break
            if closed:
                del ctx['pages'][i]
            else:
                return ToolResult(message={'result':f'没有相关页面，无法关闭'},state='异常')
            if (page_id == current_page_id) and (len(ctx['pages'])>0):
                await self.set_current_page(ctx,ctx['pages'][-1])
            return ToolResult(message={'result':f'成功关闭了索引为`{page_id}`的页面'},state='成功')
        except Exception as e:
            return ToolResult(message={'result':f'关闭页面`{page_id}`时出现异常{str(e)}'},state='异常')
        
    async def _extract_content(self,ctx:Dict[str,Any]):
        try:
            page = self.current_page(ctx)
            html_content = await page['page'].content()
            page_content = extract_html_main_content(html_content)
            return ToolResult(message={'result':f'从页面提取的内容：{page_content}'},state='成功')
        except Exception as e:
            return ToolResult(message={'result':f'获取页面内容时出现异常{str(e)}'},state='异常')
        
    async def _close_current_page(self,ctx: Dict[str,Any]) -> ToolResult:
        page = self.current_page(ctx)
        return await self._close_page(ctx,page['page_id'])    

    # ==============================================
    # 辅助用的方法
    def current_page(self,ctx: Dict[str,Any]):
        page_id = ctx['current_page_id']
        for page in ctx['pages']:
            if page['page_id'] == page_id:
                return page
        return None
    
    def new_page_id(self,ctx: Dict[str,Any]):
        if len(ctx['pages']) == 0:
            return 0
        page_id_max = max(page['page_id'] for page in ctx['pages'])
        return page_id_max + 1
    
    async def set_current_page(self,ctx:Dict[str,Any],page: Dict[str,str|DOMState|Page] | None):
        ctx['current_page_id'] = page['page_id']
        if isinstance(page['page'],Page):
            await page['page'].bring_to_front()
            await page['page'].wait_for_load_state()

    async def get_interactive_elements(self,page: Page | None = None) -> DOMState:
        if page is None:
            page = self.current_page
        await remove_highlights(page)
        dom_service = DomService(page)
        domstate = await dom_service.get_clickable_elements(
            highlight_elements=self.config.highlight_elements,
            focus_element=self.config.focus_element,
            viewport_expansion=self.config.viewport_expansion)
        return domstate

    async def _save_page(self,ctx: Dict[str,Any], page:Page) -> Dict[str,str|DOMState|Page]:
        page_id = self.new_page_id(ctx)
        domstate = await self.get_interactive_elements(page)
        page = {'page_id':page_id, 'page':page, 'domstate': domstate}
        ctx['pages'].append(page)
        return page

    def _get_page_by_id(self,ctx: Dict[str,Any],id:int):
        if 'pages' not in ctx:
            raise ToolError(message='该上下文ctx中目前不存在标签页')
        for page in ctx['pages']:
            if page['page_id'] == id:
                return page
        raise ToolError(message='该上下文ctx中目前不存在要查找的标签页')
    
    async def _update_page(self, page: Dict[str,Page|str|DOMState]):
        page['domstate'] = await self.get_interactive_elements(page['page'])
    
    async def _update_pages(self,ctx: Dict[str,Any]) -> List[str]:
        i = len(ctx['pages']) # 从最后一个判断，减少循环次数
        page_ids = []
        while (len(ctx['context'].pages) > len(ctx['pages'])) and (i < len(ctx['context'].pages)):
            page_id = self.new_page_id(ctx)
            page = ctx['context'].pages[i]
            domstate = await self.get_interactive_elements(page)
            ctx['pages'].append({'page_id':page_id,
                                 'page':page,
                                 'domstate': domstate
                                 })
            i += 1
            page_ids.append(page_id)
        return page_ids
    
    async def _create_context(self,browser):
        return await browser.new_context(
            user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            viewport=None,
            locale='cn-ZH',
            timezone_id='Asia/Shanghai',
            storage_state='state.json',
        )
    
    async def _get_idle_context(self):
        """获取当前闲置的浏览器资源"""
        async with self.lock:
            if self.base_context is None:
                raise ToolError('该工具需要初始化context，但尚未初始化，可以使用类方法create进行初始化')
            for ctx in self.base_context['contexts']:
                if ctx.get('state',None) == ToolState.IDLE:
                    # ctx['context'] = await self.base_context['browser'].new_context()
                    ctx['context'] = await self._create_context(self.base_context['browser'])
                    ctx['state'] = ToolState.USING
                    return ctx
            raise ToolError(message='目前没有闲置的上下文空间')
    
    async def _get_context_by_id(self,id: int):
        """根据id获取浏览器资源"""
        async with self.lock:
            if self.base_context is None:
                raise ToolError('该工具需要初始化context，但尚未初始化，可以使用类方法create进行初始化')
            for ctx in self.base_context['contexts']:
                if ctx.get('id') == id:
                    if ctx['state'] == ToolState.IDLE:
                        # ctx['context'] = await self.base_context['browser'].new_context()
                        ctx['context'] = await self._create_context(self.base_context['browser'])
                        ctx['state'] = ToolState.USING
                    return ctx
            raise ToolError(message='无法获取该id指定的上下文')

    async def _release_context_by_id(self,id:int):
        """根据id来释放上下文"""
        async with self.lock:
            for ctx in self.base_context['contexts']:
                if ctx['id'] == id:
                    if 'context' in ctx:
                        await ctx['context'].close()
                        del ctx['context']
                        ctx['context'] = None
                    if 'pages' in ctx:
                        del ctx['pages']
                        ctx['pages'] = []
                    ctx['state'] = ToolState.IDLE
                    break
    async def cleanup(self):
        try:
            await self.base_context['browser'].close()
            await self.base_context['playwright'].stop()
        except Exception as e:
            print(e)

class BingSearch(BaseTool):
    name: str = 'bing'
    api: WebAPI|None = None 
    async def invoke(self,text:str|None = None):
        """使用bing接口查询相关网络信息。
        Args:
            text: 要查询的内容

        """
        try:
            text = '+'.join([item for item in text.strip().split(' ') if item != ''])
            links = await self.api.async_get_request(endpoint='search',query=text,max_pages=1,source='bing',has_proxy=False)
            urls = [link['link'] for link in links[:2]] # 选择最靠前的前2个检索结果
            extracted_contents = await self.get_urls(urls)
            extracted_contents = [f'第{i+1}个查询到的内容：\n{c.strip()}' for i,c in enumerate(extracted_contents)]
            extracted_contents = '\n\n'.join(extracted_contents)
            return ToolResult(message={'result':f'{extracted_contents}'},state='成功')
        except Exception as e:
            return ToolResult(message={'result':f'访问bing搜索时出现错误{str(e)}'},state='异常')

    async def get_urls(self,urls:List[str]):
            page_contents = await self.api.async_get_request(endpoint='web_content',urls=urls,
                          mode='text',source='general',has_proxy=False)
            return page_contents
        
    @classmethod
    async def create(cls,host:str,port:int):
        api = WebAPI(host=host,port=port)
        return cls(api=api)

if __name__ == '__main__':
    pass
