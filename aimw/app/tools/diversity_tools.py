from ast import Dict, List

from tqdm import TqdmWarning
from vendi_score import text_utils

import warnings

# Ignore Vendi Score Package / sklearn `UserWarning`
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


def get_vendi_scores(sents: List, base_metric: str = "all") -> Dict:
    """Compute Vendi Score

    Args:
        sents (List): sequences
        base_metric (str): `ngram_score`, `bert_score`, `bge` or `simcse_score` or `all` scores.

    Returns:
        Dict: scores
    """

    scores = {}
    if base_metric == "ngram_score":
        ngram_score = text_utils.ngram_vendi_score(sents=sents, ns=[1, 2])
        scores["ngram_score"] = ngram_score
    elif base_metric == "simcse_score":
        simcse_score = text_utils.embedding_vendi_score(
            sents=sents, model_path="princeton-nlp/unsup-simcse-bert-base-uncased"
        )
        scores["simcse_score"] = simcse_score
    elif base_metric == "bert_score":
        bert_score = text_utils.embedding_vendi_score(
            sents=sents, model_path="bert-base-uncased"
        )
        scores["bert_score"] = bert_score
    elif base_metric == "bge":
        bge_score = text_utils.embedding_vendi_score(
            sents=sents, model_path="BAAI/bge-large-en-v1.5"
        )
        scores["bge_score"] = bge_score
    else:
        scores = {
            "ngram_score": text_utils.ngram_vendi_score(sents=sents, ns=[1, 2]),
            "simcse_score": text_utils.embedding_vendi_score(
                sents=sents, model_path="princeton-nlp/unsup-simcse-bert-base-uncased"
            ),
            "bert_score": text_utils.embedding_vendi_score(
                sents=sents, model_path="bert-base-uncased"
            ),
            "bge_score": text_utils.embedding_vendi_score(
                sents=sents, model_path="BAAI/bge-large-en-v1.5"
            ),
        }

    return scores
