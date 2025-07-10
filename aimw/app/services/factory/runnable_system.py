from abc import ABC
from contextlib import asynccontextmanager

from aimw.app.core.ai_config import get_ai_settings
from aimw.app.schemas.enum.ai_enums import Role
from aimw.app.schemas.models.decentralized_heterogeneous_agents_group import (
    HeterogeneousWritersGroup,
)
from aimw.app.schemas.models.decentralized_homogeneous_agents_group import (
    HomogeneousWritersGroup,
)
from aimw.app.services.factory.model_factory_service import Factory


class RunnableCIR3(ABC):
    def __init__(self):
        self.factory = Factory()
        self.classifier_runnable = self.factory.build_runnable_sequence(
            agent_role=Role.CLASSIFIER,
            model_params=get_ai_settings().llm_models_info[Role.CLASSIFIER]
        )
        self.moderator_runnable = self.factory.build_runnable_sequence(
            Role.MODERATOR,
            model_params=get_ai_settings().llm_models_info[Role.MODERATOR]
        )
        self.curmudgeon_runnable = self.factory.build_runnable_sequence(
            Role.CURMUDGEON,
            model_params=get_ai_settings().llm_models_info[Role.CURMUDGEON]
        )

    def setup_writers_group(self, moderator_response: list):
        if isinstance(get_ai_settings().llm_models_info["writer"], dict): # deprecated, future version will remove this. USE only list of models for both homogeneous and heterogeneous writers setup.
            self.writers = HomogeneousWritersGroup(moderator_response)
        elif isinstance(get_ai_settings().llm_models_info["writer"], list): # list of models
            self.writers = HeterogeneousWritersGroup(
                moderator_response,
                models_info=get_ai_settings().llm_models_info["writer"],
            )
        else:
            raise ValueError(f"Invalid writers group model type: {get_ai_settings().llm_models_info['writer']}")
        self.writers.describe()


@asynccontextmanager
async def lifespan():
    # Instantiate a singleton CIR3

    runnable_system = RunnableCIR3()
    yield
    # Clean up the AI models and release the resources
    runnable_system.clear()


runnable_cir3 = RunnableCIR3()
