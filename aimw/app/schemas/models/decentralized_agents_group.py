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
from aimw.app.resources.icl_templates import icl_cir3_templates
from aimw.app.schemas.enum.ai_enums import Role
from aimw.app.schemas.models.agent import Agent
from aimw.app.schemas.models.common_agents_group import Group
from aimw.app.services.factory.model_factory_service import Factory


class Writers(Group):
    def __init__(self, moderator_response):
        self.role = Role.WRITER
        self.count = len(moderator_response)
        self.moderator_instructions = []
        self.writers_qadraft_runnables = []
        self.writers_inner_refine_runnables = []
        self.agents_group = []
        super().__init__(self.role, self.count, self.agents_group)

        if isinstance(moderator_response, list) or isinstance(moderator_response, dict):
            if isinstance(moderator_response, dict):
                moderator_response = moderator_response["writers_instructions"]


            for i, moderation in enumerate(moderator_response):
                self.agents_group.append(
                    self.factory.create_agent(
                        agent_role=Role.WRITER,
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
                        template=icl_cir3_templates.writer_initial_prompt,
                    )
                )

                self.writers_inner_refine_runnables.append(
                    self.factory.build_custom_runnable_sequence(
                        agent=self.agents_group[i],
                        template=None,
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
