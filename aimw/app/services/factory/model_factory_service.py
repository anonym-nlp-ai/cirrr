from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.base import RunnableSequence
from langchain_groq import ChatGroq

from aimw.app.core.ai_config import get_ai_settings
from aimw.app.resources.icl_templates import icl_cir3_templates_factory
from aimw.app.schemas.enum.ai_enums import Role
from aimw.app.schemas.models.agent import Agent, Chain


class Factory:
    def create_agent(self, agent_role: Role, params: dict = {}) -> Agent:
        model_params = get_ai_settings().llm_models_info[agent_role]
        model = Agent(
            role=agent_role,
            model_name=model_params["ai_model_name"],
            version=model_params["version"],
            params=params,
            model=ChatGroq(
                model=model_params["ai_model_name"],
                temperature=get_ai_settings().temperature,
                model_kwargs={
                    "top_p": get_ai_settings().nucleus_sampling,
                    "seed": get_ai_settings().ai_model_kwargs["seed"],
                },
                streaming=get_ai_settings().streaming,
                api_key=get_ai_settings().groq_api_key,
            ),
        )
        return model

    def build_agents(self) -> List[Agent]:
        agents = []
        for role in get_ai_settings().llm_models_info.keys():
            agents.append(self.create_agent(role))

        return agents

    def build_group_agents(self, count: int, role: Role) -> List[Agent]:
        return [self.create_agent(role) for _ in range(count)]

    def build_chain(self, agent_role: Role) -> Chain:
        agent = self.create_agent(agent_role)
        template = icl_cir3_templates_factory.prompt_templates[agent_role]
        runnable_sequence = template | agent.model | JsonOutputParser()

        prompt_templates = [template]
        runnable_sequences = [runnable_sequence]

        chain = Chain(
            agent=agent,
            prompt_templates=prompt_templates,
            runnable_sequences=runnable_sequences,
        )

        return chain

    def build_runnable_sequence(self, agent_role: Role) -> RunnableSequence:
        agent = self.create_agent(agent_role)
        template = icl_cir3_templates_factory.prompt_templates[agent_role]
        runnable_sequence = template | agent.model | JsonOutputParser()

        return runnable_sequence

    def build_custom_runnable_sequence(
        self, agent_role: Role, template: PromptTemplate = None
    ) -> RunnableSequence:
        agent = self.create_agent(agent_role)
        runnable_sequence = (
            (
                (icl_cir3_templates_factory.prompt_templates[agent_role], template)[
                    template is not None
                ]
            )
            | agent.model
            | JsonOutputParser()
        )

        return runnable_sequence

    def build_custom_runnable_sequence(
        self, agent: Agent, template: PromptTemplate = None
    ) -> RunnableSequence:
        runnable_sequence = (
            (
                (icl_cir3_templates_factory.prompt_templates[agent.role], template)[
                    template is not None
                ]
            )
            | agent.model
            | JsonOutputParser()
        )

        return runnable_sequence
