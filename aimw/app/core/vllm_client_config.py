from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from typing import Optional, Literal
import pathlib
from functools import lru_cache


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent

class LLMConfig(BaseSettings):
    """Configuration for LLM client"""
    api_base: str = Field(default="https://localhost:8000", alias="LLM_API_BASE")
    api_key: str = Field(default="", alias="LLM_API_KEY")
    model: str = Field(default="google/gemma-3-27b-it", alias="LLM_MODEL")
    max_tokens: int = Field(default=5000, alias="LLM_MAX_TOKENS")
    temperature: float = Field(default=0.20, alias="LLM_TEMPERATURE")
    top_p: float = Field(default=0.70, alias="LLM_TOP_P")
    stream: bool = Field(default=False, alias="LLM_STREAM")
    client_type: Literal["openai", "requests"] = Field(default="openai", alias="LLM_CLIENT_TYPE")

    class Config:
        env_file = str(BASE_DIR / "conf" / "vllm-client-conf.env")
        case_sensitive = True
        populate_by_name = True

class MessageContent(BaseModel):
    """Base model for message content"""
    type: Literal["text", "image_url"]
    text: Optional[str] = None
    image_url: Optional[dict] = None

class Message(BaseModel):
    """Message structure for LLM requests"""
    role: Literal["system", "user", "assistant"]
    content: list[MessageContent]

class LLMRequest(BaseModel):
    """Complete LLM request structure"""
    model: str
    messages: list[Message]
    max_tokens: int
    temperature: float
    top_p: float
    stream: bool 


@lru_cache()
def get_vllm_client_settings() -> LLMConfig:
    return LLMConfig()
