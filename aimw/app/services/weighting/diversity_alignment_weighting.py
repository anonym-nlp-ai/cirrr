from typing import Dict, List, Any
from loguru import logger
from aimw.app.tools import diversity_tools
from aimw.app.core.ai_config import get_ai_settings

class DiversityAlignmentWeighting:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DiversityAlignmentWeighting, cls).__new__(cls)
        return cls._instance

    def calculate_vendi_scores(self, questions: List[str], answers: List[str], document: str, granularity: bool = False) -> Dict[str, Any]:
        """
        Calculate vendi scores for questions, answers, and their alignment with the document.
        
        Args:
            questions: List of question strings
            answers: List of answer strings
            document: The original document text
            
        Returns:
            Dictionary containing individual vendi scores and combined score
        """

        alpha_qa = get_ai_settings().alpha_qa
        alpha_ca = get_ai_settings().alpha_ca
        vendi_base_model_path = get_ai_settings().vendi_base_model_path

        # Calculate vendi scores for questions and answers
        vendi_q = diversity_tools.get_vendi_scores(
            sents=questions,
            vendi_base_model_path=vendi_base_model_path
        )["vendi_scores"]
        
        vendi_a = diversity_tools.get_vendi_scores(
            sents=answers,
            vendi_base_model_path=vendi_base_model_path
        )["vendi_scores"]
        
        # Calculate vendi score for answer-document alignment
        vendi_ca = diversity_tools.get_vendi_scores(
            sents=[" ".join(answers), document],
            vendi_base_model_path=vendi_base_model_path
        )["vendi_scores"]

        # Calculate balanced_g_score
        # Note: SimCSE produces similarity scores in range 1-2, with 1 indicating perfect similarity
        # balanced_g_score = ((alpha_qa / 2) * (vendi_q + vendi_a)) + alpha_ca * (1 - vendi_ca)
        comp_diversity_score = (alpha_qa / 2) * (vendi_q + vendi_a)
        faith_alignment_score = alpha_ca * (1 - vendi_ca)
        balanced_g_score = comp_diversity_score + faith_alignment_score

        weighted_scores = {
            "score_q": vendi_q,
            "score_a": vendi_a,
            "score_ca": vendi_ca,
            "balanced_g_score": balanced_g_score,
        }

        if granularity:
            weighted_scores["comp_diversity_score"] = comp_diversity_score
            weighted_scores["faith_alignment_score"] = faith_alignment_score

        logger.debug(f"combined_vendi_score: {weighted_scores}")

        return weighted_scores
