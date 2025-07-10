from ast import Dict, List

from tqdm import TqdmWarning
from vendi_score import text_utils
from aimw.app.core.ai_config import get_ai_settings
import warnings

# Ignore Vendi Score Package / sklearn `UserWarning`
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", FutureWarning)


# TODO: use model instead of model_path, and load model only once.

def get_vendi_scores(sents: List, vendi_base_model_path: str = "princeton-nlp/unsup-simcse-bert-base-uncased") -> Dict:
    """Compute Vendi Score

    Args:
        sents (List): sequences
        vendi_base_model_path (str): model path

    Returns:
        Dict: scores
    """
    scores = {}
    if vendi_base_model_path == "ngram_score":
        scores["vendi_scores"] = text_utils.ngram_vendi_score(sents=sents, ns=[1, 2])
    else:
        scores["vendi_scores"] = text_utils.embedding_vendi_score(
            sents=sents, model_path=vendi_base_model_path,
            device=get_ai_settings().compute_device
        )
    return scores

def get_vendi_scores_tag(sents: List, base_metric: str = "simcse_score") -> Dict:
    """Compute Vendi Score

    Args:
        sents (List): sequences
        base_metric (str): `ngram_score`, `bert_score`, `bge` or `simcse_score`.

    Returns:
        Dict: scores
    """

    scores = {}
    if base_metric == "ngram_score":
        ngram_score = text_utils.ngram_vendi_score(sents=sents, ns=[1, 2])
        scores["ngram_score"] = ngram_score
    elif base_metric == "bge":
        bge_score = text_utils.embedding_vendi_score(
            sents=sents, model_path="BAAI/bge-large-en-v1.5",
            device=get_ai_settings().compute_device
        )
        scores["bge_score"] = bge_score
    elif base_metric == "bert_score":
        bert_score = text_utils.embedding_vendi_score(
            sents=sents, model_path="bert-base-uncased",
            device=get_ai_settings().compute_device
        )
        scores["bert_score"] = bert_score
    else: # "simcse_score":
        simcse_score = text_utils.embedding_vendi_score(
            sents=sents, model_path="princeton-nlp/unsup-simcse-bert-base-uncased",
            device=get_ai_settings().compute_device
        )
        scores["simcse_score"] = simcse_score

    return scores

def get_all_vendi_scores(sents: List) -> Dict:
    """Compute all Vendi Scores

    Args:
        sents (List): sequences

    Returns:
        Dict: scores
    """
    scores = {
        "ngram_score": get_vendi_scores(sents, "ngram_score")["ngram_score"],
        "simcse_score": get_vendi_scores(sents, "simcse_score")["simcse_score"],
        "bert_score": get_vendi_scores(sents, "bert_score")["bert_score"],
        "bge_score": get_vendi_scores(sents, "bge_score")["bge_score"],
    }
    return scores
