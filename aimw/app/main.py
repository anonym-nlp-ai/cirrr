import random
import string
import time
from contextlib import asynccontextmanager
from typing import Any

from app.api.api_v1 import api
from app.api.api_v1.api import api_router
from app.core.aimw_config import get_basic_settings, get_settings
from app.core.log_config import LoggingSettings, setup_app_logging
from app.services.security import api_key_auth
from fastapi import APIRouter, Depends, FastAPI, Request, Response
from loguru import logger

# setup logging as early as possible
setup_app_logging(config=LoggingSettings())

root_router = APIRouter()
app = FastAPI(
    title=get_basic_settings().APP_NAME,
    description="APIs to expose CIR3 as a service.",
    version=get_basic_settings().APP_VERSION)


@root_router.get(path="/", status_code=200)
async def root():
    """Root GET Endpoint"""
    return {"message": "CIR3 - AIMW root API"}


@app.middleware("http")
async def add_process_time_header_middleware(request: Request, call_next) -> Any:
    """ Adds a pre-processing and post-processing to any HTTP operation.
    This middleware intercepts the entrypoint request and then forward
    the `request` to the corresponding path operation. This middleware
    calculates the time taken by the entrypoint and return the response
    time to the caller, by adding a response header `X-Process-Time` which
    will contain the response time.
    Args:
        request (Request): request from origin path operation
        call_next (_type_): origin path operation function/decorator
    Returns:
        Any: response
    """
    # Use unique ID for each request, so the same request logs can be traced
    idem = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
    logger.info(f"rid={idem} start request path={request.url.path}")

    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    formatted_process_time = \
        get_settings().PROCESS_TIME_FORMAT.format(process_time)
    # response.headers["X-Process-Time"] = str(f'{formatted_process_time} sec')
    response.headers["X-Process-Time"] = formatted_process_time

    logger.info(
        f"rid={idem} completed_in={formatted_process_time}ms status_code=\
            {response.status_code}")

    return response

app.include_router(
    api_router,
    prefix=get_settings().API_V1_STR,
    dependencies=[Depends(api_key_auth)]
    )
app.include_router(root_router, dependencies=[Depends(api_key_auth)])

# call from terminal using `python main.py`
if __name__ == "__main__":
    # Use this for debugging purposes only
    logger.warning(
        "Running in development mode. Do not run in production.")
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True, log_level="debug")
