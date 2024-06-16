import logging
import time
from typing import Any, Dict, List

from langchain.cache import InMemoryCache
from langchain.globals import set_debug, set_llm_cache, set_verbose
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from tqdm import tqdm

from aimw.src.core.logging import LoggingHandler


from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)


class QAGenerator:
    def __init__(
        self,
        template,
        model_name,
        groq_api_key,
        temperature,
        verbose,
        streaming,
        max_tokens,
        model_kwargs,
    ):
        self.template = template
        self.model_name = model_name
        self.groq_api_key = groq_api_key
        self.temperature = temperature
        self.verbose = verbose
        self.streaming = streaming
        self.max_tokens = max_tokens
        self.model_kwargs = model_kwargs
        self.debug = True
        self.format_instructions = None

        # self.model_path = "BeIR/query-gen-msmarco-t5-base-v1"

        self.model_path = "BeIR/query-gen-msmarco-t5-large-v1"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_path)

        # if torch.cuda.is_available():
        #     print("CUDA is available. Moving the model to the CUDA device.")
        #     self.model = self.model.to('cuda')
  
    def set_debug_mode(self, debug):
        """
        Sets the debug mode for LangChain.

        Args:
            debug (bool): The debug mode to be set. True for enabling LangChain debug mode, False for disabling it.

        Returns:
            None
        """
        self.debug = debug

    def get_llm_generator(self) -> ChatGroq:
        """
        Returns a ChatGroq instance with the same parameters as the current instance.

        :return: A ChatGroq instance with the same parameters as the current instance.
        :rtype: ChatGroq
        """
        cct_saar_generator = ChatGroq(
            temperature=self.temperature,
            verbose=self.verbose,
            groq_api_key=self.groq_api_key,
            model_name=self.model_name,
            streaming=self.streaming,
            max_tokens=self.max_tokens,
            model_kwargs=self.model_kwargs,
        )
        return cct_saar_generator

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

    # TODO: implement batching
    # Manage network exceptions


####################################################################

    def set_seed(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def generate_questions(self, doc: str) -> str:



  
        self.set_seed(42)

        # para = "Mutual funds and exchange-traded funds represent a popular way for investors to diversify but they have some key differences. While ETFs can be traded intra-day like stocks, mutual funds can only be purchased at the end of each trading day based on a calculated price known as the net asset value. Mutual funds in their present form have been around for almost a century, with the first mutual fund launched in 1924. Exchange-traded funds are relatively new entrants in the investment arena, with the first ETF launched in January 1993: the SPDR S&P 500 ETF Trust (SPY)."

        input_ids = self.tokenizer.encode(doc, return_tensors='pt')
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=64,
            do_sample=True,
            top_p=0.95,
            num_return_sequences=7)

        print("Paragraph:")
        print(doc)
        queries = ""
        print("\nGenerated Queries:")
        for i in range(len(outputs)):
            query = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            queries = queries + f'\n{i + 1}: {query}'
            # print(f'{i + 1}: {query}')


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

        query_aspects_generation_chain = bare_template | self.get_llm_generator()

        failed_doc_ids = []

        ##################### BEIR Query Generator ###########################
        
        i = 0
        ##################################################
        # i = 0
        # while i < 101:
        for doc in tqdm(docs):
            # doc = docs[i]
            generated_questions = self.generate_questions(doc[key_name])
            logging.info(f"doc_id: {i} - \n context: {doc[key_name]} \nquestions: {generated_questions}")
            messages = prompt_template.format_messages(
                context=doc[key_name],
                questions=generated_questions,
                format_instructions=format_instructions,
            )
            # i = i + 1
            time.sleep(sleep_time)
            try:
                response = query_aspects_generation_chain.invoke({"content": messages})
            except Exception as e:
                logging.error(e)
                logging.info(
                    f"Failed doc id: {i}. Continue processing..."
                )
                failed_doc_ids.append(
                    {doc_id_name: i, "issue": "Network"}
                )  # doc[doc_id_name])
                continue
            try:
                output_dict = self.query_output_parser.parse(response.content)
            except Exception as e:
                logging.error(e)
                logging.info(
                    f"Failed doc id: {i}. Continue processing..."
                )
                # failed_doc_ids.append(doc[doc_id_name])
                failed_doc_ids.append(
                    {doc_id_name: i, "issue": "Parsing"}
                )  # doc[doc_id_name])
                continue

            doc["cct_saar"] = output_dict
            doc[doc_id_name] = i
            i = i + 1
            

        return docs, failed_doc_ids

    