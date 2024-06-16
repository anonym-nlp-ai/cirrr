# **CIR3**: Comprehensive and Faithful Question-Answer Generation

## Overview
**C**ollective **I**ntentional **R**eading through **R**eflection and **R**efinement (**CIR3**), a question-answer generator that leverages a collaborative and systematic process emphasising in-depth engagement with the input text. CIR3 is a novel system designed to generate comprehensive and truthful set of QA pairs from information-dense documents. The core idea of CIR3 lies in its efficient flow of information, which employs LLM-based agents to conduct an in-depth analysis of the input context. This is achieved through a combination of transactive reasoning, multi-perspective assessment, and a balanced collective convergence process

## Getting started

1. Clone the git repository:

```sh
git clone https://github.com/anonym-nlp-ai/cirrr.git
cd cirrr/aimw
```

2. Install the required packages.

```sh
poetry install
```

3. Setup API keys: `Groq`, `langchain` and `OpenAI` in `./cirrr/conf/ai-core_conf.env`
```sh
groq_api_key="groq_key_goes_here"
langchain_api_key="lanchain_key_goes_here"
openai_api_key="openai_key_goes_here"
```

4. Running CIR3 locally
You can use either of the following notebooks `./re/gen/cir3_demo.ipynb` or `./re/gen/cir3_step-by-step_demo.ipynb`

We also provide CIR3 as a containerized service using `docker` and `docker-compose`. To do so, you can build and fire-up the necessary containers as follow:

```sh
docker-compose up --build
```

You can access the `OpenAPI Swagger` at `https://127.0.0.1:8443/docs`.

**Note**: do not forget to setup the external parameters in `.env` and `docker-compose.yml` file.  This service is still under progress.
