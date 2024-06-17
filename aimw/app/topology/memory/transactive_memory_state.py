from typing import List, Dict, Any, Optional

from typing_extensions import TypedDict


### Transactive Memory
class GraphState(TypedDict):
    """
    Represents the state of CIR3 graph.

    Attributes:
        document: document
        M: maximum number of subtopics / writers
        N: maximum number of question-answer pairs to be generated
        L: maximum inner-refinement cycles
        K: maximum outer-refinement cycles
        subtopics: list of subtopics/perspectives
        inner_transactive_memory: Inner transactive memory
        outer_transactive_memory: Outer transactive memory
        num_steps: all steps
        final_qas: final set of question-answer pairs
        moderator_instructions: moderator instructions to writers
        initial_writers_res: initial writers output
        question_answer_list: list et of question-answer pairs
        status: agreement or continue status, generated by writers
        feedback_list: list of writers feedback
        curmudgeon_feedback_list: curmudgeon feedback list
        curmudgeon_status: agreement or continue status, generated by curmudgeon
        num_inner_steps: number of inner-refinement steps
        num_outer_steps: number of outer-refinement steps
    """

    document: str
    M: int
    N: int
    L: int
    K: int
    subtopics: Dict
    inner_transactive_memory: List[Any]
    outer_transactive_memory: List[Any]
    num_steps: int
    final_qas: List[dict]
    # trans_mem_status_inner_iter: List[str]
    # Optional - The following can be removed
    moderator_instructions: Optional[Dict]
    initial_writers_res: Optional[Dict]
    question_answer_list: Optional[Dict]
    status: Optional[str]
    feedback_list: Optional[Dict]
    curmudgeon_feedback_list: Optional[Dict]
    curmudgeon_status: Optional[str]
    num_inner_steps: Optional[int]
    num_outer_steps: Optional[int]
