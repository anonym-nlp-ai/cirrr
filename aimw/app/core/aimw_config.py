import pathlib
from enum import Enum
from functools import lru_cache
from typing import Any, Union

from pydantic import root_validator, validator
from pydantic_settings import BaseSettings

from aimw.app.exceptions.exceptions import ValueConfigException

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent


class Environment(str, Enum):
    PRODUCTION = "production"
    STAGING = "staging"
    QA = "QA"
    DEVELOPMENT = "development"


class Settings(BaseSettings):
    """
    Main Settings
    """

    # Meta
    # logging: LoggingSettings = LoggingSettings()

    API_V1_STR: str = "/api/v1"
    API_KEYS: str

    SERVERS: list[dict[str, Union[str, Any]]]

    PROCESS_TIME_FORMAT: str = "0:.4f"
    DATE_FORMAT = "%d.%m.%Y"

    IGNORE_DIACRITIC_CHARS: bool = False
    DIACRITIC_ASCII_UTF8_MAP: dict[Union[bytes, str], Union[bytes, str]] = {
        b"\xc4\x8d": b"c",
        b"\xc5\xa1": b"s",
        b"\xc5\xbe": b"z",
    }

    @root_validator()
    @classmethod
    def construct_diacritic_charset(cls, field_values):
        """
        validate and encode the string using the codec registered for encoding.
        """
        if (
            field_values["IGNORE_DIACRITIC_CHARS"]
            and len(field_values["DIACRITIC_ASCII_UTF8_MAP"].keys()) > 0
        ):
            encoded_map: dict = {}
            # Construct the acritic chars encoded map
            for key in field_values["DIACRITIC_ASCII_UTF8_MAP"].keys():
                if isinstance(key, bytes):
                    encoded_map[key] = field_values["DIACRITIC_ASCII_UTF8_MAP"][key]
                elif isinstance(key, str):
                    encoded_map[key.encode("utf-8")] = field_values[
                        "DIACRITIC_ASCII_UTF8_MAP"
                    ][key].encode("utf-8")
                else:
                    raise ValueError(
                        "'DIACRITIC_ASCII_UTF8_MAP' expects str or bytes objects."
                    )
            field_values["DIACRITIC_ASCII_UTF8_MAP"] = encoded_map
        return field_values

    @validator("API_KEYS")
    def api_keys_must_exist(cls, v):
        if v is not None and len(v) > 0 and len("".join(v)) > 0:
            return v
        else:
            raise ValueConfigException("API_KEYS")

    class Config:
        case_sensitive = True
        env_file_encoding = "utf-8"
        # env_file = ".env"
        env_file = str(BASE_DIR / "conf/aimw_conf.env")


class BasicSettings(BaseSettings):
    APP_NAME: str = "CIR3"
    # import pkg_resources
    # my_version = pkg_resources.get_distribution('my-package-name').version
    APP_VERSION = "0.3.8"
    ENVIRONMENT: str = Environment.DEVELOPMENT
    ADMIN_EMAIL: str = "sami.saadaoui@citystgeorges.ac.uk"

    class Config:
        case_sensitive = True
        env_file_encoding = "utf-8"
        env_file = str(BASE_DIR / "conf/aimw_conf.env")


@lru_cache()
def get_settings() -> Settings:
    return Settings()


@lru_cache()
def get_basic_settings() -> BasicSettings:
    return BasicSettings()
