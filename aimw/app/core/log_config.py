import datetime
import logging
import pathlib
import sys
from functools import lru_cache
from types import FrameType
from typing import Optional, cast

from loguru import logger
from pydantic_settings import BaseSettings

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent


class LoggingSettings(BaseSettings):
    """
    Logging Settings

    * TRACE	    5	logger.trace()
    * DEBUG	    10	logger.debug()
    * INFO	    20	logger.info()
    * SUCCESS	25	logger.success()
    * WARNING	30	logger.warning()
    * ERROR	    40	logger.error()
    * CRITICAL	50	logger.critical()
    """
    # Meta
    LOGGING_LEVEL: int = logging.INFO
    LOG_FILE_PATH: Optional[str] = None
    
    LOG_ROTATOR_SIZE_LIMIT: float = 2e+8

    class Config:
        case_sensitive = True
        env_file_encoding = 'utf-8'
        env_file = str(BASE_DIR / "conf/logger_conf.env")

# see https://loguru.readthedocs.io/en/stable/resources/recipes.html#rotating
# -log-file-based-on-both-size-and-time


class Rotator:

    def __init__(self, *, size, at):
        now = datetime.datetime.now()

        self._size_limit = size
        self._time_limit = now.replace(
            hour=at.hour, minute=at.minute, second=at.second)

        if now >= self._time_limit:
            # The current time is already past the target time so it would
            # rotate already.
            # Add one day to prevent an immediate rotation.
            self._time_limit += datetime.timedelta(days=1)

    def should_rotate(self, message, file):
        file.seek(0, 2)
        if file.tell() + len(message) > self._size_limit:
            return True
        if message.record["time"].timestamp() > self._time_limit.timestamp():
            self._time_limit += datetime.timedelta(days=1)
            return True
        return False

# See: https://loguru.readthedocs.io/en/stable/overview.html#entirely-
# compatible-with-standard-logging


class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:  # noqa: WPS609
            frame = cast(FrameType, frame.f_back)
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


def setup_app_logging(config: LoggingSettings) -> None:
    """Prepare custom logging for the api."""

    LOGGERS = ("uvicorn.asgi", "uvicorn.access")
    logging.getLogger().handlers = [InterceptHandler()]
    for logger_name in LOGGERS:
        logging_logger = logging.getLogger(logger_name)
        logging_logger.handlers = [InterceptHandler(
            level=config.LOGGING_LEVEL)]

    logger.configure(
        handlers=[{"sink": sys.stderr, "level": config.LOGGING_LEVEL}]
    )

    # Rotate file if over 200 MB (2e+8) or at midnight every day
    rotator = Rotator(
        size=config.LOG_ROTATOR_SIZE_LIMIT,
        at=datetime.time(0, 0, 0))

    file_log_path = (
        str(BASE_DIR / "logs/server.log"),
        config.LOG_FILE_PATH)[config.LOG_FILE_PATH is not None]

    logger.add(file_log_path, rotation=rotator.should_rotate)

    logger.add(f"file_{datetime.datetime.now().strftime("%Y%m%d")}.log",
               rotation="11:59:59", compression="zip", enqueue=True)


@lru_cache()
def get_log_settings() -> LoggingSettings:
    return LoggingSettings()
