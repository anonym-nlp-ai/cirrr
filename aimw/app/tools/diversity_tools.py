from ast import Dict, List

from vendi_score import text_utils


def get_vendi_scores(sents: List) -> Dict:
    ngram_score = text_utils.ngram_vendi_score(sents=sents, ns=[1, 2])
    bert_score = text_utils.embedding_vendi_score(
        sents=sents, model_path="bert-base-uncased"
    )
    simcse_score = text_utils.embedding_vendi_score(
        sents=sents, model_path="princeton-nlp/unsup-simcse-bert-base-uncased"
    )

    return {
        "ngram_score": ngram_score,
        "bert_score": bert_score,
        "simcse_score": simcse_score,
    }
