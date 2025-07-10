from aimw.app.services.factory import runnable_system
from aimw.app.core.ai_config import get_ai_settings
from aimw.app.schemas.enum.ai_enums import Role
from loguru import logger
import time
try:
    from openai.error import RateLimitError as OpenAIRateLimitError
except ImportError:
    OpenAIRateLimitError = None
try:
    from langchain_core.exceptions import RateLimitError as LangchainRateLimitError
except ImportError:
    LangchainRateLimitError = None


class CrossModelClassifier:
    def __init__(self, include_cir3_classifier: bool = True):
        self.include_cir3_classifier = include_cir3_classifier
        self.cross_classifier_agents_params = (
            get_ai_settings().cross_classifier_agents_params
        )
        self.cir3_classifier_params = get_ai_settings().llm_models_info["classifier"]
        self.cross_classifier_agents = self._create_runnable_cross_classifier_agents(
            self.cross_classifier_agents_params
        )
        self.cir3_classifier_runnable = (
            runnable_system.runnable_cir3.factory.build_runnable_sequence(
                agent_role=Role.CLASSIFIER,
                model_params=self.cir3_classifier_params,
            )
        )

    def _create_runnable_cross_classifier_agents(
        self, cross_classifier_agents_params: list
    ) -> list:
        cross_classifier_agents = []
        for model_params in cross_classifier_agents_params:
            logger.debug(f"Model: {model_params['ai_model_name']}")
            classifier_runnable = (
                runnable_system.runnable_cir3.factory.build_runnable_sequence(
                    agent_role=Role.CLASSIFIER,
                    model_params=model_params,
                )
            )
            cross_classifier_agents.append(
                {
                    "model_name": model_params["ai_model_name"],
                    "runnable": classifier_runnable,
                }
            )
        return cross_classifier_agents

    def _invoke_with_rate_limit_retry(self, runnable, payload):
        while True:
            try:
                return runnable.invoke(payload)
            except Exception as e:
                # Try to catch known RateLimitError types
                is_rate_limit = False
                wait_time = None
                error_message = str(e)
                # Check OpenAI RateLimitError
                if OpenAIRateLimitError and isinstance(e, OpenAIRateLimitError):
                    is_rate_limit = True
                # Check Langchain RateLimitError
                elif LangchainRateLimitError and isinstance(e, LangchainRateLimitError):
                    is_rate_limit = True
                # Fallback: check for error code in dict (legacy/generic)
                elif hasattr(e, 'args') and e.args and isinstance(e.args[0], dict):
                    error = e.args[0].get('error', {})
                    if error.get('code') == 'rate_limit_exceeded':
                        is_rate_limit = True
                        error_message = error.get('message', error_message)
                # If rate limit, parse wait time and sleep
                if is_rate_limit:
                    import re
                    match = re.search(r"Please try again in ([0-9]+)m([0-9]+\.[0-9]+)s", error_message)
                    if match:
                        minutes = int(match.group(1))
                        seconds = float(match.group(2))
                        wait_time = minutes * 60 + seconds
                    else:
                        # Default to 60 seconds if not found
                        wait_time = 60
                    logger.warning(f"Rate limit hit. Waiting {wait_time:.2f} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
                raise

    def evaluate_cross_classifier_agents(self, document_list: list, m: int) -> list:
        for document in document_list:
            document["perpectives"] = {}
            classifier_subtopics_list = []
            
            for classifier_agent in self.cross_classifier_agents:
                classifier_response = self._invoke_with_rate_limit_retry(
                    classifier_agent["runnable"], {"M": m, "document": document}
                )
                logger.debug(classifier_response)
                classifier_subtopics = classifier_response["subtopics"]
                classifier_subtopics_list.append(
                    {
                        "model_name": classifier_agent["model_name"],
                        "subtopics": classifier_subtopics,
                    }
                )
            document["perpectives"][
                "cross_model_classification"
            ] = classifier_subtopics_list

            if self.include_cir3_classifier:
                document = self.cir3_classify(document, m)

        return document_list

    def cir3_classify(self, document: dict, m: int) -> dict:
        cir3_classifier_response = self._invoke_with_rate_limit_retry(
            self.cir3_classifier_runnable, {"M": m, "document": document}
        )
        logger.debug(f"CIR3 classifier response: {cir3_classifier_response}")
        cir3_classifier_subtopics = cir3_classifier_response["subtopics"]
        if "perpectives" not in document:
            document["perpectives"] = {}
        document["perpectives"]["cir3_classifier_subtopics"] = cir3_classifier_subtopics
        return document
