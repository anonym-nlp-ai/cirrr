# enhanced_geval.py
import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from tabulate import tabulate

def load_openai_api_key():
    """Load OpenAI API key from ai-core_conf.env file."""
    env_file_path = "./../../conf/ai-core_conf.env"
    try:
        with open(env_file_path, "r") as f:
            for line in f:
                if line.strip().startswith("openai_api_key="):
                    api_key = line.split("=", 1)[1].strip().strip('"')
                    return api_key
    except FileNotFoundError:
        print(f"Warning: Could not find {env_file_path}")
        return None


# Load and set API key before importing deepeval
openai_api_key = load_openai_api_key()
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key
    print("OpenAI API key loaded and set in environment")
else:
    print("Warning: OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")

# Now import deepeval after setting the API key
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from deepeval import evaluate


class EnhancedGEval(GEval):
    """
    Enhanced G-EVAL metric that extends the base GEval class with structured
    aspect-based evaluation for QA generation tasks.
    """
    
    def __init__(
        self,
        name: str,
        evaluation_aspects: List[str],
        evaluation_steps: List[str],
        evaluation_params: Optional[List[LLMTestCaseParams]] = None,
        criteria: Optional[str] = None,
        rubric: Optional[List] = None,
        threshold: float = 0.5,
        model: str = "gpt-4o",
        top_logprobs: int = 20,
        async_mode: bool = True,
        strict_mode: bool = False,
        verbose_mode: bool = False,
        _include_g_eval_suffix: bool = True,
    ):
        """
        Initialize Enhanced G-EVAL metric with aspect-based evaluation.
        
        Args:
            name: Name of the metric (e.g., "Comprehensiveness", "Faithfulness")
            evaluation_aspects: List of specific aspects being evaluated
            evaluation_steps: List of evaluation steps for systematic assessment
            evaluation_params: Parameters for evaluation (default: INPUT and ACTUAL_OUTPUT)
            threshold: Threshold for pass/fail determination
            model: LLM model to use for evaluation
            async_mode: Whether to run evaluation asynchronously
            strict_mode: Whether to use strict evaluation mode
            verbose_mode: Whether to enable verbose output
        """
        # Set default evaluation parameters if not provided
        if evaluation_params is None:
            evaluation_params = [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT]
        
        # Initialize parent GEval class
        super().__init__(
            name=name,
            evaluation_steps=evaluation_steps,
            evaluation_params=evaluation_params,
            criteria=criteria,
            rubric=rubric,
            threshold=threshold,
            model=model,
            top_logprobs=top_logprobs,
            async_mode=async_mode,
            strict_mode=strict_mode,
            verbose_mode=verbose_mode,
            _include_g_eval_suffix=_include_g_eval_suffix,
        )
        
        # Store additional attributes
        self.evaluation_aspects = evaluation_aspects
        self.metric_name = name
    
    def get_aspects(self) -> List[str]:
        """Return the evaluation aspects for this metric."""
        return self.evaluation_aspects
    
    def get_metric_info(self) -> dict:
        """Return comprehensive information about this metric."""
        return {
            "name": self.metric_name,
            "aspects": self.evaluation_aspects,
            "num_steps": len(self.evaluation_steps),
            "threshold": self.threshold,
            "model": self.model,
        }


def create_comprehensiveness_metric() -> EnhancedGEval:
    """
    Create an Enhanced G-EVAL metric for comprehensiveness evaluation
    based on Coverage, Depth, Accuracy, and Coherence aspects.
    """
    return EnhancedGEval(
        name="Comprehensiveness",
        evaluation_aspects=["Coverage", "Depth", "Accuracy", "Coherence"],
        evaluation_steps=[
            # Coverage
            "Examine the source document to identify all key topics, concepts, and important information covered",
            "Review the set of question-answer pairs to determine what aspects of the document they address",
            "Check if the questions cover all major themes and subtopics from the document",
            
            # Depth
            "Consider depth of coverage - are complex topics explored adequately or only superficially?",
            "Assess whether important details, relationships, and nuances are captured in the QA pairs",
            
            # Accuracy
            "Verify that the questions accurately reflect the document's content and answers are factually correct",
            
            # Coherence
            "Evaluate the logical flow and connection between questions and their relationship to document structure",
            
            # Final Assessment
            "Score HIGH if the QA set demonstrates comprehensive coverage, adequate depth, accuracy, and coherence",
            "Score LOW if major topics are missing, coverage is superficial, inaccurate, or lacks coherence",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )


def create_faithfulness_metric() -> EnhancedGEval:
    """
    Create an Enhanced G-EVAL metric for faithfulness evaluation
    based on Accuracy, Exaggeration, Consistency, Justification, Plausibility, and Misrepresentation aspects.
    """
    return EnhancedGEval(
        name="Faithfulness",
        evaluation_aspects=[
            "Accuracy", "Exaggeration", "Consistency", 
            "Justification", "Plausibility", "Misrepresentation"
        ],
        evaluation_steps=[
            # Accuracy
            "Carefully read the source document to understand the factual information presented",
            "Examine each answer to verify factual accuracy against the source document",
            
            # Exaggeration
            "Check for any statements that overstate or embellish information from the document",
            
            # Consistency
            "Look for contradictions or deviations from facts presented in the source material",
            
            # Justification
            "Verify that all claims in answers are well-supported by evidence from the document",
            
            # Plausibility
            "Assess whether answers represent reasonable inferences based on the document content",
            
            # Misrepresentation
            "Check for any distortion or misleading presentation of facts from the source",
            
            # Final Assessment
            "Score HIGH if answers demonstrate accuracy, avoid exaggeration, maintain consistency, are well-justified, plausible, and avoid misrepresentation",
            "Score LOW if answers contain inaccuracies, exaggerations, inconsistencies, poor justification, implausible claims, or misrepresentations",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )


def create_diversity_metric() -> EnhancedGEval:
    """
    Create an Enhanced G-EVAL metric for diversity evaluation.
    """
    return EnhancedGEval(
        name="Diversity",
        evaluation_aspects=[
            "Question Types", "Perspective Variety", "Complexity Range", 
            "Linguistic Diversity", "Coverage Breadth"
        ],
        evaluation_steps=[
            "Analyze the variety of question types in the set (factual, analytical, comparative, etc.)",
            "Check for diversity in the aspects of the topic being explored by different questions",
            "Look for questions that approach the subject from different angles or perspectives",
            "Assess whether questions vary in complexity and cognitive demand (recall vs. analysis vs. synthesis)",
            "Identify any repetitive patterns or overly similar questions that reduce diversity",
            "Consider linguistic diversity - variety in question structures and phrasing",
            "Evaluate whether the QA set explores different levels of detail (high-level vs. specific)",
            "Score HIGH if questions show strong variety in type, perspective, and complexity",
            "Score LOW if questions are repetitive, similar in structure, or lack variety",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    )




def format_qa_pairs_for_evaluation(qa_set: List[Dict[str, str]]) -> str:
    """Format question-answer pairs for G-eval evaluation."""
    formatted_pairs = []
    for i, qa_pair in enumerate(qa_set, 1):
        formatted_pairs.append(f"Q{i}: {qa_pair['question']}")
        formatted_pairs.append(f"A{i}: {qa_pair['answer']}")
    return "\n".join(formatted_pairs)


def evaluate_qa_set_with_enhanced_metrics(
    qa_set: List[Dict[str, str]], 
    document: str,
    iteration_idx: int
) -> Dict[str, Any]:
    """Evaluate a QA set using Enhanced G-eval metrics."""
    print(f"Enhanced G-eval evaluation for iteration {iteration_idx}")
    
    # Create enhanced metrics
    comprehensiveness_metric = create_comprehensiveness_metric()
    faithfulness_metric = create_faithfulness_metric()
    diversity_metric = create_diversity_metric()
    
    qa_formatted = format_qa_pairs_for_evaluation(qa_set)
    
    # Create test case
    test_case = LLMTestCase(
        input=f"Document: {document}",
        actual_output=qa_formatted,
        context=[
            "Evaluate this set of question-answer pairs against the source document. "
            "Consider comprehensiveness, faithfulness, and diversity aspects."
        ],
    )
    
    results = {}
    
    # Evaluate comprehensiveness
    print(f"  Evaluating comprehensiveness (aspects: {', '.join(comprehensiveness_metric.get_aspects())})...")
    comp_result = evaluate([test_case], [comprehensiveness_metric])
    results['comprehensiveness'] = {
        'score': comp_result.test_results[0].metrics_data[0].score,
        'aspects': comprehensiveness_metric.get_aspects(),
        'qa_count': len(qa_set)
    }
    
    # Evaluate faithfulness
    print(f"  Evaluating faithfulness (aspects: {', '.join(faithfulness_metric.get_aspects())})...")
    faith_result = evaluate([test_case], [faithfulness_metric])
    results['faithfulness'] = {
        'score': faith_result.test_results[0].metrics_data[0].score,
        'aspects': faithfulness_metric.get_aspects(),
        'qa_count': len(qa_set)
    }
    
    # Evaluate diversity
    print(f"  Evaluating diversity (aspects: {', '.join(diversity_metric.get_aspects())})...")
    div_result = evaluate([test_case], [diversity_metric])
    results['diversity'] = {
        'score': div_result.test_results[0].metrics_data[0].score,
        'aspects': diversity_metric.get_aspects(),
        'qa_count': len(qa_set)
    }
    
    # Calculate average
    avg_score = (results['comprehensiveness']['score'] + 
                results['faithfulness']['score'] + 
                results['diversity']['score']) / 3
    
    results['average_score'] = avg_score
    results['iteration'] = iteration_idx
    
    return results






