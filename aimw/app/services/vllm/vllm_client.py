from abc import ABC, abstractmethod
import openai
import requests
from typing import Optional, Union, List
import base64
from pathlib import Path
from loguru import logger

from aimw.app.core.vllm_client_config import LLMConfig, Message, MessageContent, LLMRequest


class BaseLLMClient(ABC):
    """Base class for LLM clients"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        logger.debug(f"Initializing client with config: {config.model_dump()}")
    
    @abstractmethod
    def chat_completion(self, messages: List[Message], stream: Optional[bool] = None) -> Union[str, dict]:
        """Get chat completion from the LLM"""
        pass

class OpenAIClient(BaseLLMClient):
    """OpenAI-based LLM client"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        logger.debug(f"Creating OpenAI client with base_url={config.api_base}")
        self.client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.api_base
        )
    
    def chat_completion(self, messages: List[Message], stream: Optional[bool] = None) -> Union[str, dict]:
        stream = stream if stream is not None else self.config.stream
        logger.debug(f"Making chat completion request with stream={stream}")
        
        try:
            completion = self.client.chat.completions.create(
                model=self.config.model,
                messages=[msg.model_dump() for msg in messages],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stream=stream
            )
            
            if stream:
                return completion
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in chat completion: {str(e)}")
            raise

class RequestsClient(BaseLLMClient):
    """Requests-based LLM client"""
    
    def chat_completion(self, messages: List[Message], stream: Optional[bool] = None) -> Union[str, dict]:
        stream = stream if stream is not None else self.config.stream
        
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Accept": "text/event-stream" if stream else "application/json"
        }
        
        payload = LLMRequest(
            model=self.config.model,
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stream=stream
        ).model_dump()
        
        response = requests.post(
            f"{self.config.api_base}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if stream:
            return response
        return response.json()

def create_client(config: Optional[LLMConfig] = None) -> BaseLLMClient:
    """Factory function to create the appropriate client"""
    if config is None:
        config = LLMConfig()
    
    if config.client_type == "openai":
        return OpenAIClient(config)
    return RequestsClient(config)

def encode_image(image_path: Union[str, Path]) -> str:
    """Helper function to encode image to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8') 