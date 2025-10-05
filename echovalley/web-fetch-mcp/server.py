from playwright.async_api import async_playwright
from playwright.async_api._generated import Page
import asyncio,os,re
from fastapi import FastAPI,Query
import httpx
from typing import Dict,List,Any,TypeVar,ParamSpec,Callable,Coroutine
from functools import wraps
import random
from contextlib import asynccontextmanager
from logger import logger
from utils import (config, process_results_page,
                       Config,ToolState,UsingError,
                       navigate_to_next_page,
                       get_page_content
                       )

class BrowserPool():
    async def init(self,config: Config):
        self.config = config
        self.lock = asyncio.Lock()
        await self._init_browsers()
        return self
    
    async def _init_browsers(self):
        self.playwright = await async_playwright().start()
        proxy = {'server':config.proxy_server} if self.config.proxy_server else None
        self.browser = await self.playwright.chromium.launch(headless=config.headless,
                                                #    proxy=proxy,
                                                   args=self.start_params)
        self.contexts = [{'state':ToolState.IDLE,'id':i, 'context': await self.new_context()} \
                    for i in range(self.config.browser_pool_size)]
        self.proxy_contexts = [{'state':ToolState.IDLE,'id':i, 'context': await self.new_context(proxy)} \
                    for i in range(self.config.browser_pool_size)]
        
    async def new_context(self,proxy=None):
        context = await self.browser.new_context(
            user_agent=random.choice(self.config.user_agents),
            viewport=None,
            # locale='cn-ZH',
            timezone_id='Asia/Shanghai',
            storage_state='state.json',
            proxy=proxy,
        )
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => undefined
            });
        """)
        return context

    @property
    def start_params(self):
        args = [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-infobars",
        "--disable-extensions",
        "--disable-popup-blocking",
        "--disable-notifications",
        "--ignore-certificate-errors",
        "--disable-blink-features=AutomationControlled",
        "--disable-software-rasterizer",
        "--disable-background-networking",
        "--disable-background-timer-throttling",
        "--disable-breakpad",
        "--disable-client-side-phishing-detection",
        "--disable-default-apps",
        "--disable-hang-monitor",
        "--disable-sync",
        "--metrics-recording-only",
        "--no-first-run",
        "--safebrowsing-disable-auto-update",
        ]
        if self.config.headless:
            args.extend(["--headless=new", "--disable-gpu"])
        return args
    
    async def get_idle_context(self,has_proxy: bool=False):
        """获取当前闲置的浏览器资源"""
        if has_proxy:
            contexts = self.proxy_contexts
        else:
            contexts = self.contexts
        async with self.lock:
            for ctx in contexts:
                if ctx.get('state',None) == ToolState.IDLE:
                    ctx['state'] = ToolState.USING
                    return ctx
            raise UsingError('没有闲置资源')
    
    async def release_context_by_id(self,id:int,has_proxy: bool=False):
        """根据id来释放上下文"""
        if has_proxy:
            contexts = self.proxy_contexts
        else:
            contexts = self.contexts
        async with self.lock:
            for ctx in contexts:
                if ctx['id'] == id:
                    ctx['state'] = ToolState.IDLE
                    for page in ctx['context'].pages:
                        await page.close()
                    break

    async def clean(self):
        try:
            await self.browser.close()
            await self.playwright.stop()
        except Exception as e:
            print(e)

browserpool: BrowserPool = None

@asynccontextmanager
async def browser_context(app: FastAPI):
    global browserpool
    try:
        browserpool = await BrowserPool().init(config)
        logger.info('浏览器已启动')
        yield
    except Exception as e:
        logger.error(f'浏览器错误:{str(e)}')
    finally:
        try:
            await browserpool.clean()
        except Exception as e:
            logger.info(f'浏览器关闭错误：{str(e)}')
        else:
            logger.info('浏览器已关闭')
    
app = FastAPI(
    title="搜索API",
    description="浏览器并发搜索",
    version="1.0.0",
    lifespan=browser_context
)


async def get_idle_context(browser_pool: BrowserPool, has_proxy:bool):
    ctx = None
    for i in range(browser_pool.config.wait_loop_num):
        try:
            ctx = await browser_pool.get_idle_context(has_proxy=has_proxy)
        except UsingError:
            await asyncio.sleep(browser_pool.config.wait_sleep_time)
        else:
            break
    return ctx

async def random_scroll(page: Page,min_steps=1):
    scroll_height = random.randint(200,800)
    scroll_steps = random.randint(min_steps,min_steps+3)
    for steps in range(scroll_steps):
        await page.evaluate(f"window.scrollBy(0, {scroll_height});")
        await asyncio.sleep(random.uniform(0.3,1.0))

P = ParamSpec('P')
R = TypeVar('R')

def query_endpoint_async(browser_pool:BrowserPool, has_proxy:bool
    ) -> Callable[[Callable[P,Coroutine[Any,Any,R]]], Callable[P,Coroutine[Any,Any,R]]]:
    def decorator(func: Callable[P,Coroutine[Any,Any,R]]) -> Callable[P,Coroutine[Any,Any,R]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            ctx = await get_idle_context(browser_pool,has_proxy)
            if ctx:
                kwargs['ctx'] = ctx
                try:
                    results = await func(*args,**kwargs)
                except Exception as e:
                    logger.error(f'网络出现错误:{str(e)}')
                    results = '网络出现错误'
                finally:
                    await browser_pool.release_context_by_id(ctx['id'],has_proxy)
                    return results
            else:
                return '没有足够浏览器资源'
        return wrapper
    return decorator

async def search_crawler(query:str, max_pages: int = 2, source='bing',ctx: Dict[str,Any] = None):
    """
    借助搜索引擎检索相关内容
    """
    page: Page = await ctx['context'].new_page()
    if source == 'bing':
        #await page.goto(f'{browserpool.config.bing}/search?q={query}')
        await page.goto(f'{browserpool.config.bing}')
        await page.wait_for_load_state(browserpool.config.page_load_state)
        await page.locator('body div.hpl.hp_cont div#sb_form_c #sb_form_q').fill(query) 
        await page.press('body div.hpl.hp_cont div#sb_form_c #sb_form_q','Enter')
    elif source == 'google':
        await page.goto(f'{browserpool.config.google}/search?q={query}')
    await page.wait_for_load_state(browserpool.config.page_load_state)
    processed_pages = 0
    results = []
    for page_num in range(1,max_pages+1):
        if page_num > 1:
            # 导航到下一页
            if not await navigate_to_next_page(page,page_num,source):
                logger.info(f"无法导航到第 {page_num} 页，停止处理")
                break
        logger.info(f'处理第{page_num}页结果')
        page_results = await process_results_page(page,source)
        results.extend(page_results)
        processed_pages += 1
    logger.info(f"处理了 {processed_pages} 页，共 {len(results)} 条结果")
    return results

async def search_crawler_urls(urls: List[str], mode='md',source:str='general',ctx: Dict[str,Any] = None):
    """
    获取所有urls所指向的网页内容。
	
    """
    page_nums = min(browserpool.config.visit_urls_page_nums,len(urls))
    pages = []
    urls_group = [[] for i in range(page_nums)]
    for page_num in range(page_nums):
        page: Page = await ctx['context'].new_page()
        pages.append(page)
        urls_group[page_num].extend(urls[page_num::page_nums])
    async def get_urls_results(page:Page,urls:List[str]):
        results = []
        for url in urls:
            success = False
            for attempt in range(browserpool.config.access_retry):
                try:
                    await page.goto(url)
                    await page.wait_for_load_state(browserpool.config.page_load_state)
                    await random_scroll(page)
                    result = await get_page_content(page,mode,source)
                    results.append(result)
                    success = True
                    break
                except Exception as e:
                    logger.warning(f'访问页面{url}失败,(尝试{attempt}): {str(e)}')
            if not success:
                results.append('')
        logger.info(f'成功获取{sum(1 for t in results if t != '')}/{len(urls)}个页面')
        return results
    results = await asyncio.gather(*[get_urls_results(pages[i],urls_group[i]) for i in range(page_nums)])
    finally_results = []
    for i in range(len(urls)):
        group_idx = i % page_nums
        loc_idx = i // page_nums
        if len(urls_group[group_idx]) > loc_idx:
            finally_results.append(results[group_idx][loc_idx])
    return finally_results

@app.get('/search')
async def search_endpoint(query:str,max_pages:int = 2, source: str = 'bing', has_proxy:bool=False):
    search_crawler_local = query_endpoint_async(browserpool,has_proxy=has_proxy)(search_crawler)
    return await search_crawler_local(query=query,max_pages=max_pages,source=source)

@app.get('/web_content')
async def search_web(urls: List[str] = Query(...),mode:str='md',source='general', has_proxy:bool=False):
    search_crawler_urls_local = query_endpoint_async(browserpool,has_proxy=has_proxy)(search_crawler_urls)
    return await search_crawler_urls_local(urls=urls,mode=mode,source=source)

@app.get('/test')
async def test(url:str, has_proxy:bool=True):
    ctx = await get_idle_context(browserpool,has_proxy)
    if ctx:
        try:
            page:Page = await ctx['context'].new_page()
            await page.goto(url)
            await page.wait_for_load_state(browserpool.config.page_load_state)
            results = 'success'
        except Exception as e:
            logger.error(f'出现错误:{str(e)}')
            results = 'error'
        else:
            input()
            await ctx['context'].storage_state(path="state.json")
        finally:
            await browserpool.release_context_by_id(ctx['id'])
            return results
    else:
        return '没有足够浏览器资源'

@app.get('/state')
async def get_state():
    return {'no_proxy':f'{sum(1 for ctx in browserpool.contexts if ctx['state']==ToolState.IDLE)}/{len(browserpool.contexts)}',
            'proxy':f'{sum(1 for ctx in browserpool.proxy_contexts if ctx['state']==ToolState.IDLE)}/{len(browserpool.proxy_contexts)}',
           }

@app.get('/')
async def get_help():
    return {
        'state':{},
        'test':{'url':'str','has_proxy':'bool'},
        'search':{'query':'str','max_pages':'int','source':'str','has_proxy':'bool'},
        'web_content':{'urls':'List[str]','mode':'str','source':'str','has_proxy':'bool'},
    }

#=============================================================================================
# MCP SERVER
from mcp.server.fastmcp import FastMCP
mcp = FastMCP('web request tools',mount_path='/',sse_path='/sse',message_path='/messages')

@mcp.tool()
async def get_url(url:str) -> str:
    """查询url所指定的网页的实际内容。

    Args:
        url: 要检索的网页地址
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(f'http://{config.server_host}:{config.server_port}/web_content',
                    params={'urls':[url],'mode':'md','source':'general','has_proxy':False},
                    timeout=300,
                    )
            response.raise_for_status()
            data = response.json()
            return data[0]
        except Exception as e:
            return f'访问{url}时出现网络异常：{str(e)}'
    
@mcp.tool()
async def search_web_address(query:str) -> List[Dict[str,str]]:
    """通过搜索引擎检索指定查询能查询到的关键网页元信息，
    该工具只能获得基本的相关网页元数据，无法获取网页具体内容，如果要获取该页面详细信息，还需要访问元数据提供的具体网址。

    Args:
        query: 查询字符串，要检索的目标。
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(f'http://{config.server_host}:{config.server_port}/search',
                    params = {'query':query,'max_pages':2,'source':'bing','has_proxy':False},
                    timeout=300
                    )
        response.raise_for_status()
        return response.json()
    
sse_mcp= mcp.sse_app()
app.mount('/mcp',sse_mcp)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        app,
        host = config.server_host,
        port = config.server_port,
    )

