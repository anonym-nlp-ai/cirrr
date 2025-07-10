import uuid
from abc import ABC, abstractmethod
from typing import Any, List, Optional
from uuid import UUID

from langchain.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSequence
from loguru import logger
from pydantic import BaseModel

from aimw.app.core.ai_config import get_ai_settings
from aimw.app.exceptions.exceptions import ValueTypeException
from aimw.app.resources.icl_templates import icl_cir3_templates_factory
from aimw.app.schemas.enum.ai_enums import Role
from aimw.app.schemas.models.agent import Agent
from aimw.app.schemas.models.common_agents_group import Group
from aimw.app.services.factory.model_factory_service import Factory
import random


class HeterogeneousWritersGroup(Group):
    def __init__(self, moderator_response, models_info: list):
        self.role = Role.WRITER
        self.moderator_instructions = []
        self.writers_qadraft_runnables = []
        self.writers_inner_refine_runnables = []
        self.agents_group = []
        self.models_info = models_info

        logger.debug(f"models_info: {models_info}")

        if isinstance(moderator_response, list) or isinstance(moderator_response, dict):
            if isinstance(moderator_response, dict):
                moderator_response = moderator_response["writers_instructions"]
            self.count = len(moderator_response)
            super().__init__(self.role, self.count, self.agents_group)

            available_models = []
            for i, moderation in enumerate(moderator_response):

                # NOTE: Specific requirements per model (speciality, personality, finetuned, etc.) 
                # for heterogeneous writers can be implemented here + by amending .env file.

                chosen = {}
                # Select a model for each writer: random fair selection or random unfair selection
                if not available_models:
                    available_models = self.models_info.copy()
                    if not get_ai_settings().allow_repeats_writers_model:
                        random.shuffle(available_models)
                if get_ai_settings().allow_repeats_writers_model:
                    chosen = random.choice(self.models_info)
                else:
                    chosen = available_models.pop()

                self.agents_group.append(
                    self.factory.create_agent(
                        agent_role=Role.WRITER,
                        model_params=chosen,
                        params={"perspective": moderation["perspective"]},
                    )
                )

                if all(
                    elem in moderation.keys() for elem in ["task", "instructions"]
                ):  # "task" in moderation.keys():
                    logger.debug(f"Concatenating moderator's instructions ...")
                    self.moderator_instructions.append(
                        f"Task: {moderation['task']}.\n"
                        + f"Instructions: {' '.join(moderation['instructions'])}"
                    )
                    logger.debug(
                        f"modertaor instructions {i}: {self.moderator_instructions[i]}"
                    )
                else:
                    logger.debug(f"Stringifying moderator's instructions ...")
                    self.moderator_instructions.append(str(moderation))

                self.writers_qadraft_runnables.append(
                    self.factory.build_custom_runnable_sequence(
                        agent=self.agents_group[i],
                        model_params=chosen,
                        template=icl_cir3_templates_factory.PromptFactory.get_prompt(
                            Role.WRITER_INITIAL,
                            chosen["provider"]
                            ) # icl_cir3_templates.writer_initial_prompt,
                    )
                )

                self.writers_inner_refine_runnables.append(
                    self.factory.build_custom_runnable_sequence(
                        agent=self.agents_group[i],
                        template=None,
                        model_params=chosen
                    )
                )
        else:
            raise ValueTypeException(
                param_name="moderator_response",
                expected_type=list,
                actual_type=moderator_response.__class__,
            )

    def describe(self):
        logger.info(f"A decentralized group of {self.count} {self.role.value} agents.")
