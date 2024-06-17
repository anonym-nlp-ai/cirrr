# **CIR3**: Comprehensive and Faithful Question-Answer Generation

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
conda create -n cir3 python=3.11
conda activate cir3
poetry export -f requirements.txt --output requirements.txt
pip install -r requirements.txt
```

4. Running CIR3 

You can use one of the following methods:

* Jupyter Notebooks: 
    * `./re/gen/cir3_demo.ipynb`
    * `./re/gen/cir3_step-by-step_demo.ipynb`
* Python script:
    ```sh
    python .\run_cir3.py --output-dir your_out_directory
    # Or simply
    python .\run_cir3.py 
    ```

* [Optional] Containerization (under development):

We also provide CIR3 as a containerized API service.
To build the necessary docker images and run containers, use the following:

```sh
docker-compose up --build
```

APIs can be accessed through `OpenAPI Swagger` at `https://127.0.0.1:8443/docs`.
