# **CIR3**: Coordinated LLM Multi-Agent Systems for Collaborative Question-Answer Generation

Highlights:
- Effective QA Generation through Group Coordination and Efficient Communication in LLM-based Multiagent Systems

## Overview
**C**ollective **I**ntentional **R**eading through **R**eflection and **R**efinement (**CIR3**), a question-answer generator that leverages a collaborative and systematic process emphasizing in-depth engagement with the input text. CIR3 is a novel system designed to generate comprehensive and truthful set of QA pairs from information-dense documents. The core idea of CIR3 lies in its efficient flow of information, which employs LLM-based agents to conduct an in-depth analysis of the input context. This is achieved through a combination of transactive reasoning, multi-perspective assessment, and a balanced collective convergence process.

## Getting started

1. Clone the repository:

```sh
git clone https://github.com/anonym-nlp-ai/cirrr.git
cd cirrr/aimw
```

2. Setup API keys: `Groq`, `langchain` and `OpenAI` in `./cirrr/conf/ai-core_conf.env`
```sh
groq_api_key="groq_key_goes_here"
langchain_api_key="langchain_key_goes_here"
openai_api_key="openai_key_goes_here"
```

3. Install the required packages:

You can use either `poetry` or `conda`:

**Poetry**

```sh
poetry install
```

**Conda**
```sh

conda create -n cir3 python=3.13
conda activate cir3
pip install uv
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 # win
pip3 install torch torchvision torchaudio # linux cud 12.6 
# for Linux
sudo apt-get update && sudo apt-get install -y build-essential
poetry export -f requirements.txt --output requirements.txt
uv pip install -r requirements.txt
```

**Conda with minimum requirements**
```sh
conda create -n cir3 python=3.13
conda activate cir3
pip install uv
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126 # win
pip3 install torch torchvision torchaudio # linux cud 12.6 

# for Linux
sudo apt-get update && sudo apt-get install -y build-essential

# Optional: to visualize the agents graph on Jupyter Notebook
conda install conda-forge::pygraphviz

# install minimum requirements
uv pip install -r ./aimw/minimum_req.txt
```

4. Running CIR3 

You can use one of the following methods:

* Jupyter Notebooks: 
    * `./re/gen/cir3_demo.ipynb`
    * `./re/gen/cir3_step-by-step_demo.ipynb`
* Python script:
    ```sh
    cd ./quick-run
    python ./run_cir3.py --output-dir your_out_directory
    # Or simply
    python ./run_cir3.py 
    ```

* [Optional] Containerization (under development):

We also provide CIR3 as a containerized API service.
To build the necessary docker images and run containers, use the following:

```sh
docker-compose up --build
```

APIs can be accessed through `OpenAPI Swagger` at `https://127.0.0.1:8443/docs`.

# Note
If you wish to run the framework using commercial providers (Anthropic, OpenAI, Groq), please check their respective pricing pages:

- [OpenAI Pricing](https://platform.openai.com/docs/pricing)
- [Anthropic Pricing](https://www.anthropic.com/pricing#api)
- [Groq Pricing](https://groq.com/pricing/)


# TODO

- add vllm via docker-compose + instrumnentation + docs
- amend .envs
- writers heterougenous: aimw/app/schemas/models/decentralized_agents_group.py
- load vendi models only once
- compress the inner state memory: keep last qas only in a list
- Review the doc.


combined_vendi_score: {'score_q': 2.7691658, 'score_a': 2.721003, 'score_ca': 1.1520605, 'balanced_g_score': 1.2965118885040283, 'comp_diversity_score': 1.372542142868042, 'faith_alignment_score': -0.07603025436401367}

combined_vendi_score: {'score_q': 3.1531253, 'score_a': 2.9122965, 'score_ca': 1.1676774, 'balanced_g_score': 1.4325168132781982, 'comp_diversity_score': 1.5163555145263672, 'faith_alignment_score': -0.08383870124816895}





# INFO: homogeneous writers require a dict, and heterogeneous writers setup requires a list, however, you can can also trandform a heterogeneous setup to homogeneous setup by provider a list of the same model type, where the list can contain one or more models of the same type. Future version will remove homogeneous setup as a dict. 

# NOTE: Specific requirements per model (speciality, personality, finetuned, etc.) for heterogeneous writers can be implemented here + by amending .env file.














# LLM Usage in CIR3

## Overview
CIR3 supports multiple LLM providers and configurations for different agent roles. The system is designed to be flexible, allowing both homogeneous (same model type) and heterogeneous (different model types) setups for the writer agents.

## Supported LLM Providers

1. **Groq**
   - Models: llama3-8b-8192, llama3-70b-8192
   - Configuration: Requires `groq_api_key` in `conf/ai-core_conf.env`

2. **OpenAI**
   - Models: gpt-4o-mini
   - Configuration: Requires `openai_api_key` in `conf/ai-core_conf.env`

3. **Anthropic**
   - Models: claude-3-7-sonnet-20250219
   - Configuration: Requires `anthropic_api_key` in `conf/ai-core_conf.env`

4. **Self-hosted VLLM**
   - Models: google/gemma-3-27b-it
   - Configuration: Requires VLLM server setup

## Configuration

The LLM configuration is managed through the `llm_models_info` setting in `conf/ai-core_conf.env`. This setting supports both homogeneous and heterogeneous setups.

### Homogeneous Setup Example
```json
{
    "moderator": {"provider": "groq", "ai_model_name": "llama3-8b-8192", "version": "0.0.1"},
    "classifier": {"provider": "groq", "ai_model_name": "llama3-8b-8192", "version": "0.0.1"},
    "writer": {"provider": "groq", "ai_model_name": "llama3-8b-8192", "version": "0.0.1"},
    "curmudgeon": {"provider": "groq", "ai_model_name": "llama3-70b-8192", "version": "0.0.1"}
}
```

### Heterogeneous Setup Example
```json
{
    "moderator": {"provider": "openai", "ai_model_name": "gpt-4o-mini", "version": "0.0.1"},
    "classifier": {"provider": "openai", "ai_model_name": "gpt-4o-mini", "version": "0.0.1"},
    "writer": [
        {"provider": "groq", "ai_model_name": "llama3-8b-8192", "version": "0.0.1"},
        {"provider": "openai", "ai_model_name": "gpt-4o-mini", "version": "0.0.1"},
        {"provider": "anthropic", "ai_model_name": "claude-3-7-sonnet-20250219", "version": "0.0.1"},
        {"provider": "self-hosted-vllm", "ai_model_name": "google/gemma-3-27b-it", "version": "0.0.1"}
    ],
    "curmudgeon": {"provider": "openai", "ai_model_name": "gpt-4o-mini", "version": "0.0.1"}
}
```

## Model Templates

Each provider has its own prompt template format:

- **Llama3**: `<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system_message}<|eot_id|><|start_header_id|>user<|end_header_id|>{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>`
- **Gemma3**: `<start_of_turn>user{system_message}{user_prompt}<end_of_turn><start_of_turn>model`
- **OpenAI**: `{system_message}{user_prompt}`
- **Claude**: `<system>{system_message}</system><user>{user_prompt}</user><assistant>`

## Important Notes

1. **Writer Agents**: 
   - Can be configured as either homogeneous (single model type) or heterogeneous (multiple model types)
   - For homogeneous setup, use a dictionary configuration
   - For heterogeneous setup, use a list of model configurations
   - You can create a homogeneous setup using multiple instances of the same model type in a list

2. **API Keys**:
   - Ensure all required API keys are properly configured in `conf/ai-core_conf.env`
   - Check provider pricing pages for current rates:
     - [OpenAI Pricing](https://platform.openai.com/docs/pricing)
     - [Anthropic Pricing](https://www.anthropic.com/pricing#api)
     - [Groq Pricing](https://groq.com/pricing/)

3. **VLLM Setup**:
   - For self-hosted models, ensure the VLLM server is properly configured and running
   - Follow the containerization setup instructions for VLLM deployment

## Best Practices

1. **Model Selection**:
   - Choose models based on your specific requirements for each agent role
   - Consider cost, performance, and availability when selecting models
   - Test different model combinations to find the optimal setup for your use case

2. **Configuration Management**:
   - Keep API keys secure and never commit them to version control
   - Use environment variables or secure configuration management for sensitive data
   - Document any changes to the LLM configuration

3. **Performance Monitoring**:
   - Monitor API usage and costs
   - Track model performance and response times
   - Adjust configurations based on performance metrics 