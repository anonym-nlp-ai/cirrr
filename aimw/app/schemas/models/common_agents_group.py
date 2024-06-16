import uuid
from abc import ABC, abstractmethod
from typing import List

from aimw.app.schemas.enum.ai_enums import Role
from aimw.app.schemas.models.agent import Agent
from aimw.app.services.factory.model_factory_service import Factory


class Group(ABC):
    def __init__(self, role: Role, count: int = 1, agents_group: List[Agent] = []):
        self.role = role
        self.count = count
        self.agents_group = agents_group
        self.factory = Factory()

    @abstractmethod
    def describe(self):
        pass
