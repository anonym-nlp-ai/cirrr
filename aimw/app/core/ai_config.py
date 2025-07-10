import pathlib
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

from aimw.app.exceptions.exceptions import ValueConfigException

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent.parent


class CIR3Settings(BaseSettings):
    """AI Settings as a class holder for AI config."""

    max_qa_pairs: int = Field(default=7)
    max_subtopics: int = Field(default=10)
    max_inner_refinements: int = Field(default=10)
    max_outer_refinements: int = Field(default=10)
    temperature: float = Field(default=0.1)
    nucleus_sampling: float = Field(default=0.5)
    llm_models_info: dict = Field(default=...)
    allow_repeats_writers_model: bool = Field(default=False)

    langchain_api_key: str
    groq_api_key: str
    openai_api_key: str
    anthropic_api_key: str

    verbose: bool = Field(default=True)
    streaming: bool = Field(default=True)
    max_tokens: int = Field(default=5000)
    compute_device: str = Field(default="cuda")
    ai_model_kwargs: dict = Field(default={"seed": 201})

    M: int = 5
    N: int = 10
    L: int = 12
    K: int = 6

    qgen_model_path: str = "BeIR/query-gen-msmarco-t5-large-v1"

   # Curmudgeon strategies: "vendi_only", "curmudgeon_only", "curmudgeon_vendi", "random_rejection"
    curmudgeon_strategy: str = Field(default="curmudgeon_vendi")
    rondon_disagreement_probability: float = Field(default=0.8)
    # Vendi base model: "ngram_score", "bert_score", "bge" or "simcse_score"
    vendi_base_model: str = Field(default="simcse_score")
    # or path can be used instead (implemntation supports model tag or model path)
    vendi_base_model_path: str = Field(default="princeton-nlp/unsup-simcse-bert-base-uncased")

    # Empirically, SimCSE produces similarity scores in the range of 1 to 2, 
    # with 1 indicating perfect similarity, typically observed between a given 
    # context and its corresponding concatenated answers:
    # relative weighting of diversity (qa)
    alpha_qa: float = Field(default=0.5)
    # relative weighting of alignment (ca)
    alpha_ca: float = Field(default=0.5)

    balanced_g_score_threshold: float = Field(default=1.2)

    template_llama: str = Field(default="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
    {system_message}
    <|eot_id|><|start_header_id|>user<|end_header_id|>
    {user_prompt}
    <|eot_id|><|start_header_id|>assistant<|end_header_id|>""")

    template_gemma3: str = Field(default="""<start_of_turn>user
    {system_message}
    {user_prompt}
    <end_of_turn>
    <start_of_turn>model""")

    template_openai: str = Field(default="""{system_message}
    {user_prompt}""")

    template_claude: str = Field(default="""<system>
    {system_message}
    </system>
    <user>
    {user_prompt}
    </user>
    <assistant>""")

    cross_classifier_agents_params: list = Field(default=...)

    @field_validator("groq_api_key")
    @classmethod
    def api_keys_must_exist(cls, v):
        if v is not None and len(v) > 0 and len("".join(v)) > 0:
            return v
        else:
            raise ValueConfigException("API_KEYS")

    @field_validator("llm_models_info")
    @classmethod
    def ai_api_keys_must_exist(cls, v):
        if v is not None and len(v) > 0 and len("".join(v)) > 0:
            return v
        else:
            raise ValueConfigException("llm_models_info")

    class Config:
        case_sensitive = True
        env_file_encoding = "utf-8"
        env_file = str(BASE_DIR / "conf/ai-core_conf.env")


@lru_cache()
def get_ai_settings() -> CIR3Settings:
    return CIR3Settings()
