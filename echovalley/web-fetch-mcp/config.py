from pydantic import BaseModel
from typing import List

class Config(BaseModel):
    bing: str = 'https://www.bing.com'
    google: str = 'https://www.google.com'
    # proxy_server: str | None = 'http://localhost:7890'
    proxy_server: str | None = None
    user_agents: List[str] =  [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36 Edg/115.0.1901.188"
    ]
    # headless: bool = False
    headless: bool = True
    browser_pool_size: int = 3 # 浏览器上下文数量
    wait_loop_num: int = 10 # 获取闲置上下文资源尝试次数
    wait_sleep_time: int = 5 # 获取闲置上下文资源等待间隔
    page_load_state: str = 'load' # ["commit", "domcontentloaded", "load", "networkidle"]
    access_retry: int = 3 # 引发异常后的尝试次数
    # scroll_amount: int = 500
    visit_urls_page_nums: int = 5 # 访问网页时同时发送的最大请求数量
    server_host:str = '127.0.0.1' # 开启服务的地址
    server_port:int = 8000 # 开启服务的端口

