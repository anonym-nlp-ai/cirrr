import pathlib
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from aimw.app.exceptions.exceptions import ValueConfigException

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent


class CIR3Settings(BaseSettings):
    """AI Settings as a class holder for AI config."""

    max_qa_pairs: int = Field(default=7)
    max_subtopics: int = Field(default=10)
    max_inner_refinements: int = Field(default=10)
    max_outer_refinements: int = Field(default=10)
    temperature: float = Field(default=0.1)
    nucleus_sampling: float = Field(default=0.5)
    llm_models_info: dict = Field(default=...)

    groq_api_key: str
    langchain_api_key: str
    openai_api_key: str

    verbose: bool = Field(default=True)
    streaming: bool = Field(default=True)
    max_tokens: int = Field(default=5000)

    ai_model_kwargs: dict = Field(default={"seed": 178})

    M: int = 5
    N: int = 10
    L: int = 12
    K: int = 6

    qgen_model_path: str = "BeIR/query-gen-msmarco-t5-large-v1"

    @field_validator("groq_api_key")
    @classmethod
    def api_keys_must_exist(cls, v):
        if v is not None and len(v) > 0 and len("".join(v)) > 0:
            return v
        else:
            raise ValueConfigException("API_KEYS")

    @field_validator("llm_models_info")
    @classmethod
    def ai_api_keys_must_exist(cls, v):
        if v is not None and len(v) > 0 and len("".join(v)) > 0:
            return v
        else:
            raise ValueConfigException("llm_models_info")

    class Config:
        case_sensitive = True
        env_file_encoding = "utf-8"
        env_file = str(BASE_DIR / "conf/ai-core_conf.env")


@lru_cache()
def get_ai_settings() -> CIR3Settings:
    return CIR3Settings()
