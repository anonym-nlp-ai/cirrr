from aimw.app.api.api_v1.endpoints import cir3_api
from app.api.api_v1.endpoints import app_info
from app.core.aimw_config import get_settings
from fastapi import APIRouter

api_router = APIRouter()
api_router.include_router(app_info.router, prefix="/info", tags=["API Info"])
api_router.include_router(
    cir3_api.router, prefix="/id", tags=["ID Extraction"])


def get_modified_handler(url_path_for: str) -> str:
    """Construct modified handler.

    Args:
        url_path_for (str): name

    Returns:
        str: modified handler
    """
    
    modified_handler = get_settings().API_V1_STR + \
        api_router.url_path_for(url_path_for)
        
    return modified_handler