import random
import string
import time
from ast import literal_eval
from contextlib import asynccontextmanager
from typing import Any

# from app.api.api_v1 import api
# from app.api.api_v1.api import api_router
# from app.core.config import get_basic_settings, get_settings
from app.core.log_config import LoggingSettings, setup_app_logging
from app.services import ai_model_loader
# from app.services.monitoring.monitoring_instrumentator import instrumentator
from app.services.security import api_key_auth
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from loguru import logger

# setup logging as early as possible
setup_app_logging(config=LoggingSettings())