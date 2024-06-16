import uuid
from typing import Any, List, Optional
from uuid import UUID

from langchain.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSequence
from pydantic import BaseModel

from aimw.app.schemas.enum.ai_enums import Role


class Agent(BaseModel):
    """Agent Class"""

    uuid: UUID = uuid.uuid4()
    role: Role
    ai_model_name: Optional[str] = "llama3-8b-8192"
    version: Optional[str]
    model: Any
    params: Optional[dict] = {}


class Chain(BaseModel):
    """Chain Class"""

    uuid: UUID = uuid.uuid4()
    prompt_templates: Optional[List[PromptTemplate]]
    agent: Optional[Agent]
    runnable_sequences: Optional[List[RunnableSequence]]
