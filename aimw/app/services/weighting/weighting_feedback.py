from typing import Dict, Any
import random
import numpy as np

def construct_curmudgeon_feedback(
    vendi_scores: Dict[str, Any], balanced_g_score_threshold: float = 1.2
) -> str:

    if vendi_scores["balanced_g_score"] >= balanced_g_score_threshold:
        return "Question and answer pair meets diversity and alignment criteria."
    else:
        return "Question and answer pair does not meet diversity and / or alignment criteria."

def bernoulli_disagreement_control(disagreement_probability: float = 0.8) -> str:
    """Return 'continue' with probability p using Bernoulli distribution."""
    result = np.random.binomial(1, disagreement_probability)
    return "continue" if result == 1 else "agreement"


def random_disagreement_control(disagreement_probability: float = 0.8) -> str:
    """Return 'continue' with probability = threshold, otherwise 'agreement'."""
    return "continue" if random.random() < disagreement_probability else "agreement"

def construct_random_curmudgeon_feedback(disagreement_probability: float = 0.8) -> str:
    if random_disagreement_control(disagreement_probability) == "agreement":
        return "Question and answer pair meets diversity and alignment criteria."
    else:
        return "Question and answer pair does not meet diversity and / or alignment criteria."