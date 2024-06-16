from typing import List, Tuple
from scipy.stats import ttest_rel

from loguru import logger


def conduct_ttest(
    a: List[float], b: List[float], alpha: float = 0.05
) -> Tuple[float, float]:
    """Calculate the t-test on TWO RELATED samples of scores, a and b.
    This is a test for the null hypothesis that two related or repeated samples have identical average (expected) values

    Args:
        a (List[float]): a
        b (List[float]): b
        alpha (float, optional): p_value alpha. Defaults to 0.05.

    Raises:
        Exception: _description_

    Returns:
        Tuple[float, float]: _description_
    """
    if len(a) != len(b):
        raise Exception("Paired lists must have the same size.")

    t_statistic, p_value = ttest_rel(a, b)

    if p_value < alpha:
        logger.info("Statistically significant difference (p =", p_value)
    else:
        logger.info("No significant difference, (p =", p_value)

    return (t_statistic, p_value)
