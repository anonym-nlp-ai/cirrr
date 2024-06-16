from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from aimw.app.core.aimw_config import get_settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")  # use token authentication


def api_key_auth(api_key: str = Depends(oauth2_scheme)):
    if api_key not in get_settings().API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Forbidden"
        )
