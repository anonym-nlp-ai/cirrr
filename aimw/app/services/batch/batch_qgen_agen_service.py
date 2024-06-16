import time

import torch
from langchain.cache import InMemoryCache
from langchain.globals import set_debug, set_llm_cache, set_verbose
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from loguru import logger
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from aimw.app.schemas.models.agent import LLM


class QAGenerator:
    def __init__(
        self,
        template: str,
        number_of_questions: str,
        verbose: bool,
        model: LLM,
        qgen_model_path="BeIR/query-gen-msmarco-t5-large-v1",
    ):
        # Answer Generator
        self.template = template
        self.number_of_questions = number_of_questions
        self.verbose = verbose
        self.debug = True
        self.format_instructions = None
        self.a_gen_model = model

        # Question Generator
        self.model_path = qgen_model_path
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
        self.q_gen_model = T5ForConditionalGeneration.from_pretrained(self.model_path)

    def set_debug_mode(self, debug):
        """
        Sets the debug mode for LangChain.

        Args:
            debug (bool): The debug mode to be set. True for enabling LangChain debug mode, False for disabling it.

        Returns:
            None
        """
        self.debug = debug

    def build_format_instructions(self, name: str, description: str, type: str) -> str:
        """
        A function that builds format instructions based on the provided name, description, and type.

        :param name: A string representing the name.
        :param description: A string representing the description.
        :param type: A string representing the type.
        :return: A string containing the format instructions.
        :rtype: str
        """
        if self.format_instructions is None:
            query_schema = ResponseSchema(name=name, description=description, type=type)
            query_response_schemas = [query_schema]
            query_output_parser = StructuredOutputParser.from_response_schemas(
                query_response_schemas
            )
            format_instructions = query_output_parser.get_format_instructions()
            self.query_output_parser = query_output_parser
            self.format_instructions = format_instructions
        return self.format_instructions

    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def generate_questions(self, doc: str) -> str:
        """BEIR generate questions for a given document

        Args:
            doc (str): document

        Returns:
            str: questions
        """

        self.set_seed(42)
        input_ids = self.tokenizer.encode(doc, return_tensors="pt")
        outputs = self.q_gen_model.generate(
            input_ids=input_ids,
            max_length=64,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=7,
        )

        logger.debug(f"passdage: {doc}")
        queries = ""
        logger.debug("\nGenerated Queries:")
        for i in range(len(outputs)):
            query = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            queries = queries + f"\n{i + 1}: {query}"

        return queries

    def generate_query_aspects(
        self,
        docs: list,
        key_name: str = "text",
        doc_id_name: str = "id",
        sleep_time: int = 0,
    ) -> tuple[list[dict], list[str]]:
        """
        Generates query aspects for a list of documents.

        Args:
            docs (list): A list of documents.
            key_name (str, optional): The key name for the document text. Defaults to "text".
            doc_id_name (str, optional): The key name for the document ID. Defaults to "id".
            sleep_time (int, optional): The sleep time between each document processing. Defaults to 0.

        Returns:
            tuple[list[dict], list[str]]: A tuple containing the processed documents and the list of failed document IDs.

        Raises:
            NotImplementedError: If the input `docs` is not a list.

        Side Effects:
            - Appends a dictionary to `failed_doc_ids` for each document that fails to process.
            - Adds a key "cct_saar" to each document in `docs` with the parsed output.

        """
        if not isinstance(docs, list):
            raise NotImplementedError("Only list of dicts is supported")

        format_instructions = self.build_format_instructions(
            name="queries_aspects",
            description="queries and aspects about the context",
            type="string",
        )

        bare_prompt_template = "{content}"
        bare_template = ChatPromptTemplate.from_template(template=bare_prompt_template)
        prompt_template = ChatPromptTemplate.from_template(template=self.template)

        query_aspects_generation_chain = bare_template | self.a_gen_model.model

        failed_doc_ids = []

        i = 0
        for doc in tqdm(docs):

            # BEIR Query Generato
            generated_questions = self.generate_questions(doc[key_name])
            logger.info(
                f"doc_id: {i} - \n context: {doc[key_name]} \nquestions: {generated_questions}"
            )
            messages = prompt_template.format_messages(
                context=doc[key_name],
                questions=generated_questions,
                format_instructions=format_instructions,
            )
            time.sleep(sleep_time)
            try:
                response = query_aspects_generation_chain.invoke({"content": messages})
            except Exception as e:
                logger.error(e)
                logger.info(f"Failed doc id: {i}. Continue processing...")
                failed_doc_ids.append(
                    {doc_id_name: i, "issue": "Network"}
                )  # doc[doc_id_name])
                continue
            try:
                output_dict = self.query_output_parser.parse(response.content)
            except Exception as e:
                logger.error(e)
                logger.info(f"Failed doc id: {i}. Continue processing...")
                # failed_doc_ids.append(doc[doc_id_name])
                failed_doc_ids.append(
                    {doc_id_name: i, "issue": "Parsing"}
                )  # doc[doc_id_name])
                continue

            doc["cct_saar"] = output_dict
            doc[doc_id_name] = i
            i = i + 1

        return docs, failed_doc_ids
