from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables.base import RunnableSequence
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from aimw.app.core.ai_config import get_ai_settings
from aimw.app.core.vllm_client_config import get_vllm_client_settings
from aimw.app.services.vllm import vllm_client
from aimw.app.resources.icl_templates import icl_cir3_templates_factory
from aimw.app.schemas.enum.ai_enums import Role
from aimw.app.schemas.models.agent import Agent, Chain, LLM
from loguru import logger


class Factory:

    def create_llm(self, model_name: str, params: dict = {}) -> LLM:
        model = LLM(
            model_name=model_name,
            params=params,
            model=ChatGroq(
                model=model_name,
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

    def create_agent(
        self, agent_role: Role, model_params: dict = None, params: dict = {}
    ) -> Agent:

        # if model_params is None:
        #     model_params = get_ai_settings().llm_models_info[agent_role]

        logger.info(
            f"Creating agent for {agent_role} with model {model_params['ai_model_name']}"
        )

        if model_params["provider"] == "groq":

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
        elif model_params["provider"] == "openai":
            model = Agent(
                role=agent_role,
                model_name=model_params["ai_model_name"],
                version=model_params["version"],
                params=params,
                model=ChatOpenAI(
                    model=model_params["ai_model_name"],
                    temperature=get_ai_settings().temperature,
                    model_kwargs={
                        "top_p": get_ai_settings().nucleus_sampling,
                        "seed": get_ai_settings().ai_model_kwargs["seed"],
                    },
                    streaming=get_ai_settings().streaming,
                    api_key=get_ai_settings().openai_api_key,
                ),
            )
        elif model_params["provider"] == "anthropic":
            model = Agent(
                role=agent_role,
                model_name=model_params["ai_model_name"],
                version=model_params["version"],
                params=params,
                model=ChatAnthropic(
                    model=model_params["ai_model_name"],
                    temperature=get_ai_settings().temperature,
                    top_p=get_ai_settings().nucleus_sampling,
                    model_kwargs={},
                    streaming=get_ai_settings().streaming,
                    api_key=get_ai_settings().anthropic_api_key,
                ),
            )
        elif model_params["provider"] == "self-hosted-vllm":
            # incorporate self-hosted model (e.g. Gemma 27B IT) using vllm
            # or use model=vllm_client.create_client(get_vllm_client_settings())
            model = Agent(
                role=agent_role,
                model_name=model_params["ai_model_name"],
                version=model_params["version"],
                params=params,
                model=ChatOpenAI(
                    base_url=get_vllm_client_settings().api_base,
                    api_key=get_vllm_client_settings().api_key,
                    model=model_params["ai_model_name"],
                    temperature=get_ai_settings().temperature,
                    model_kwargs={
                        "top_p": get_ai_settings().nucleus_sampling,
                        "seed": get_ai_settings().ai_model_kwargs["seed"],
                    },
                    streaming=get_ai_settings().streaming,
                ),
            )
        else:
            raise ValueError(f"Invalid provider: {model_params['provider']}")

        return model

    def build_agents(self) -> List[Agent]:
        agents = []
        for role in get_ai_settings().llm_models_info.keys():
            agents.append(
                self.create_agent(
                    role, model_params=get_ai_settings().llm_models_info[role]
                )
            )

        return agents

    def build_group_agents(self, count: int, role: Role) -> List[Agent]:
        return [
            self.create_agent(
                role, model_params=get_ai_settings().llm_models_info[role]
            )
            for _ in range(count)
        ]

    def build_chain(self, agent_role: Role, model_params: dict = None) -> Chain:
        agent = self.create_agent(
            agent_role, model_params=model_params
        )
        # template = icl_cir3_templates_factory.prompt_templates[agent_role]
        template = icl_cir3_templates_factory.PromptFactory.get_prompt(
            agent_role, model_params["provider"]
        )
        runnable_sequence = template | agent.model | JsonOutputParser()

        prompt_templates = [template]
        runnable_sequences = [runnable_sequence]

        chain = Chain(
            agent=agent,
            prompt_templates=prompt_templates,
            runnable_sequences=runnable_sequences,
        )

        return chain

    def build_runnable_sequence(self, agent_role: Role, model_params: dict = None) -> RunnableSequence:
        agent = self.create_agent(
            agent_role, model_params=model_params
        )
        # template = icl_cir3_templates_factory.prompt_templates[agent_role]
        template = icl_cir3_templates_factory.PromptFactory.get_prompt(
            agent_role, model_params["provider"]
        )
        runnable_sequence = template | agent.model | JsonOutputParser()

        return runnable_sequence

    def build_custom_runnable_sequence(
        self, agent_role: Role, template: PromptTemplate = None, model_params: dict = None
    ) -> RunnableSequence:
        agent = self.create_agent(
            agent_role, model_params=model_params
        )
        runnable_sequence = (
            (
                (
                    icl_cir3_templates_factory.PromptFactory.get_prompt(
                        agent_role,
                        model_params["provider"],
                    ),
                    template,
                )[template is not None]
            )
            | agent.model
            | JsonOutputParser()
        )

        return runnable_sequence

    def build_custom_runnable_sequence(
        self, agent: Agent, template: PromptTemplate = None, model_params: dict = None
    ) -> RunnableSequence:
        runnable_sequence = (
            (
                (
                    icl_cir3_templates_factory.PromptFactory.get_prompt(
                        agent.role,
                        model_params["provider"],
                    ),
                    template,
                )[template is not None]
            )
            | agent.model
            | JsonOutputParser()
        )

        return runnable_sequence
