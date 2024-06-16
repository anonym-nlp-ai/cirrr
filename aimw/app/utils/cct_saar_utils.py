import json
from loguru import logger
import os
import re

from aimw.app.services.batch.batch_qa_gen_service import CCTSAARGenerator


def apply_cct_saar(
    cct_saar_generator: CCTSAARGenerator,
    r_dir: str,
    w_dir: str,
    start_index: int = 0,
    end_index: int = -1,
    sleep_time: int = 0,
) -> tuple[list[dict], list[str]]:
    """
    Applies CCT-SAAR to a corpus of documents.

    Args:
        cct_saar_generator (CCTSAARGenerator): An instance of the CCTSAARGenerator class.
        r_dir (str): The directory where the input corpus is located.
        w_dir (str): The directory where the output files will be saved.
        start_index (int, optional): The index of the first document to process. Defaults to 0.
        end_index (int, optional): The index of the last document to process. Defaults to -1.
            If end_index is less than 0, all documents after start_index will be processed.
        sleep_time (int, optional): The time to sleep between processing documents. Defaults to 0.

    Returns:
        tuple[list[dict], list[str]]: A tuple containing two lists. The first list contains the generated query aspects,
        and the second list contains the failed query aspects.

    Raises:
        ValueError: If start_index is less than 0.

    Processes each document in the input corpus by generating query aspects using the CCTSAARGenerator instance.
    Saves the generated and failed query aspects to output files.
    """

    if start_index < 0:
        raise ValueError("start_index must be >= 0")

    file_names = os.listdir(r_dir)

    start_index = (len(os.listdir(r_dir)), start_index)[start_index >= 0]
    end_index = (len(file_names), end_index)[end_index > 0]

    for i in range(start_index, end_index):
        logger.info(f"Processing file number {str(i)}: corpus_cln_split_{str(i)}.json")
        with open(f"{r_dir}corpus_cln_split_{str(i)}.json", encoding="utf-8") as f:
            data = json.load(f)

        cct_saar_generated, cct_saar_failed = cct_saar_generator.generate_query_aspects(
            docs=data, key_name="doc", doc_id_name="docid", sleep_time=sleep_time
        )

        with open(
            f"{w_dir}cct_saar_corpus_cln_split_{str(i)}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(cct_saar_generated, f)

        if len(cct_saar_failed) > 0:  # save cct_saar_failed
            with open(
                f"{w_dir}cct_saar_failed_corpus_cln_split_{str(i)}.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(cct_saar_failed, f)
    return cct_saar_generated, cct_saar_failed


def reprocess_cct_saar(
    cct_saar_generator: CCTSAARGenerator, dir: str, sleep_time: int = 0
):
    """
    Reprocesses the CCT-SAAR data based on the provided CCTSAARGenerator, directory, and optional sleep time.

    Args:
        cct_saar_generator (CCTSAARGenerator): The CCTSAARGenerator object used for reprocessing.
        dir (str): The directory path where the CCT-SAAR data is stored.
        sleep_time (int, optional): The amount of time to sleep between processing each file. Defaults to 0.

    Returns:
        None
    """
    pattern = r"cct_saar_failed_corpus_cln_split_(?P<number>\d+)\.json"

    file_numbers = sorted(
        [
            int(match.group(1))
            for filename in os.listdir(dir)
            if (match := re.search(pattern, filename))
        ]
    )

    logger.info(f"Found indices: {file_numbers}")

    for i in file_numbers:
        logger.info(
            f"Processing file number {str(i)}: cct_saar_corpus_cln_split_{str(i)}.json"
        )

        # Load the JSON data from json1 and json2
        with open(
            f"{dir}cct_saar_corpus_cln_split_{str(i)}.json", encoding="utf-8"
        ) as f1, open(
            f"{dir}cct_saar_failed_corpus_cln_split_{str(i)}.json", encoding="utf-8"
        ) as f2:
            cct_saar_processed = json.load(f1)
            cct_saar_failed = json.load(f2)

        # Define the key column in json2
        key_column = "docid"

        if len(cct_saar_failed) > 0:
            # Filter objects from cct_saar_processed based on 'docid' in cct_saar_failed
            for failed_obj in cct_saar_failed:
                for obj in cct_saar_processed:
                    if (
                        failed_obj[key_column] == obj[key_column]
                        and "cct_saar" not in obj
                    ):
                        # Apply CCT-SAAR on a list that contains only one object
                        success, fail = cct_saar_generator.generate_query_aspects(
                            docs=[obj],
                            key_name="doc",
                            doc_id_name="docid",
                            sleep_time=sleep_time,
                        )
                        if len(success) > 0:
                            #  obj["cct_saar"] = success[0]  # One object
                            failed_obj["reprocess_success"] = "yes"
                        else:
                            failed_obj["reprocess_success"] = "no"
        else:
            logger.info(f"No failures in file number {i}")

        logger.info(f"Failed Json Content for doc {i}: {cct_saar_failed}")

        with open(
            f"{dir}cct_saar_corpus_cln_split_new_{str(i)}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(cct_saar_processed, f)

        with open(
            f"{dir}cct_saar_failed_corpus_cln_new_split_{str(i)}.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(cct_saar_failed, f)
