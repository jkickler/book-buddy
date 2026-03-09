import os

from loguru import logger

LOG_DIR = "logs"
LOG_FILE = "app.log"


def setup_logging():
    """Configure loguru with file output only."""
    logger.remove()

    os.makedirs(LOG_DIR, exist_ok=True)
    logger.add(
        os.path.join(LOG_DIR, LOG_FILE),
        rotation="daily",
        retention="7 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        level="INFO",
    )

    return logger
