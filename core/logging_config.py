import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

ERROR_LOG_FILE = os.path.join(LOG_DIR, 'errors.log')
INFO_LOG_FILE = os.path.join(LOG_DIR, 'app.log')


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    error_handler = RotatingFileHandler(
        ERROR_LOG_FILE, maxBytes=1024 * 1024, backupCount=5, encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)

    info_handler = RotatingFileHandler(
        INFO_LOG_FILE, maxBytes=1024 * 1024, backupCount=5, encoding="utf-8"
    )
    info_handler.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    error_handler.setFormatter(formatter)
    info_handler.setFormatter(formatter)

    logger.addHandler(error_handler)
    logger.addHandler(info_handler)
