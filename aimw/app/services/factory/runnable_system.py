from abc import ABC
from contextlib import asynccontextmanager

from aimw.app.schemas.enum.ai_enums import Role
from aimw.app.schemas.models.decentralized_agents_group import Writers
from aimw.app.services.factory.model_factory_service import Factory


class RunnableCIR3(ABC):
    def __init__(self):
        self.factory = Factory()
        self.classifier_runnable = self.factory.build_runnable_sequence(
            agent_role=Role.CLASSIFIER
        )
        self.moderator_runnable = self.factory.build_runnable_sequence(Role.MODERATOR)
        self.curmudgeon_runnable = self.factory.build_runnable_sequence(Role.CURMUDGEON)

    def setup_writers_group(self, moderator_response: list):
        self.writers = Writers(moderator_response)
        self.writers.describe()


@asynccontextmanager
async def lifespan():
    # Instantiate a singleton CIR3

    runnable_system = RunnableCIR3()
    yield
    # Clean up the AI models and release the resources
    runnable_system.clear()


runnable_cir3 = RunnableCIR3()
