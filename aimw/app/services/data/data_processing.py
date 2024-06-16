import pandas as pd
from loguru import logger
from transformers import AutoTokenizer


def id_to_text(collection, queries):
    """Returns two dictionaries mapping id to text.

    Returns:
        qid_to_text: dictionary
            key - qid
            value - question text
        docid_to_text: dictionary
            key - docid
            value - answer text
    ----------
    Arguments:
        collection: dataframe
            Dataframe containing docids and answers
        queries: dataframe
            Dataframe containing qids and questions
    """
    qid_to_text = {}
    docid_to_text = {}

    for _, row in queries.iterrows():
        qid_to_text[row["qid"]] = row["question"]

    for _, row in collection.iterrows():
        docid_to_text[row["docid"]] = row["doc"]

    return qid_to_text, docid_to_text


def get_top_qids_with_most_rel_docs(top_n: int, qid_docid: pd.DataFrame) -> list[int]:

    qidx_with_max_rel = qid_docid["qid"].value_counts().index[:top_n]
    logger.debug(
        f"Top {top_n} queries with most relevant docs: idx_max_rel={qidx_with_max_rel}\n"
    )
    logger.debug(
        f'Max number of relevant docs: {qid_docid["qid"].value_counts().max()}'
    )
    for i in qidx_with_max_rel:
        logger.debug(
            f'Query id={i} has {qid_docid[qid_docid["qid"]==i]["qid"].count()} docs'
        )

    logger.debug(
        f'\n\nTop {top_n} most relevant docs: {qid_docid[qid_docid["qid"] == qidx_with_max_rel[0]]}\n'
    )

    return qidx_with_max_rel.to_list()


def get_top_docids_with_most_associated_queries(
    top_n: int, qid_docid: pd.DataFrame
) -> list[int]:
    """
    Retrieves the top `top_n` document IDs with the most associated queries from the given DataFrame `qid_docid`.

    Parameters:
        top_n (int): The number of top document IDs to retrieve.
        qid_docid (pd.DataFrame): The DataFrame containing the query-document ID pairs.

    Returns:
        list[int]: A list of the top `top_n` document IDs with the most associated queries.
    """
    docid_with_max_qrel = qid_docid["docid"].value_counts().index[:top_n]
    logger.info(
        f"Top {top_n} documents with most associated queries: docid_with_max_qrel={docid_with_max_qrel}\n"
    )
    logger.info(
        f'Max number of docs with most associated queries: {qid_docid["docid"].value_counts().max()}'
    )
    for i in docid_with_max_qrel:
        logger.info(
            f'Doc id={i} has {qid_docid[qid_docid["docid"]==i]["docid"].count()} queries'
        )

    logger.info(
        f'\n\nTop {top_n} document with most associated queries: {qid_docid[qid_docid["docid"] == docid_with_max_qrel[0]]}\n'
    )
    return docid_with_max_qrel.to_list()


def add_tokenized_size(
    data: pd.DataFrame, column_name: str, max_size=512
) -> pd.DataFrame:
    """Returns a dataframe with the size of tokenized input.

    Returns:
        data: Dataframe
    ----------
    Arguments:
        data: Dataframe with additional column `tokenized_size` and 'max_seq_len_exceeded'
    """

    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-large-en-v1.5")

    for i, row in data[:].iterrows():
        logger.debug(f"Encoding row: {i}")
        encoded_input = tokenizer(
            row[column_name], padding=True, truncation=False, return_tensors="pt"
        )

        data.at[i, "tokenized_size"] = encoded_input.input_ids.size()[1]
        data.at[i, "max_seq_len_exceeded"] = (
            encoded_input.input_ids.size()[1] > max_size
        )

    return data
