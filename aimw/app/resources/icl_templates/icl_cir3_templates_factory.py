from aimw.app.resources.icl_templates.icl_cir3_templates import iCLCir3Templates
from aimw.app.schemas.enum.ai_enums import Role
from aimw.app.core.ai_config import get_ai_settings
from loguru import logger

class PromptFactory:
    @staticmethod
    def get_prompt(role: Role, provider: str):
        """
        Returns the appropriate prompt template based on role and provider.
        
        Args:
            role (Role): The role of the agent (CLASSIFIER, MODERATOR, WRITER, WRITER_INITIAL, CURMUDGEON)
            provider (str): The LLM provider (llama, gemma3, openai, claude)
            
        Returns:
            PromptTemplate: The appropriate prompt template
        """
        # generic template
        template = """{system_message}
        {user_prompt}"""

        match provider:
            case "groq":
                template = get_ai_settings().template_llama
            case "self-hosted-vllm":
                template = get_ai_settings().template_gemma3
            case "openai":
                template = get_ai_settings().template_openai
            case "anthropic":
                template = get_ai_settings().template_claude

        prompt = iCLCir3Templates.format_prompt(template=template, role=role)

        if role not in Role:
            raise ValueError(f"Invalid role: {role}")
            
        if provider not in ["groq", "self-hosted-vllm", "openai", "anthropic"]:
            raise ValueError(f"Invalid provider: {provider} for role: {role}")

        logger.debug(f"----> Using prompt for role: {role} and provider: {provider}: {prompt}")

        return prompt