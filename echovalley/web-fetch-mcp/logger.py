import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bing_crawler.log',encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)