import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

ERROR_LOG_FILE = os.path.join(LOG_DIR, 'errors.log')


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(ERROR_LOG_FILE, maxBytes=1024*1024, backupCount=5, encoding='utf-8')
    handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
