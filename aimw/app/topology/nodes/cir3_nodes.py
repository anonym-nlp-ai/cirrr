from loguru import logger
import asyncio

from aimw.app.resources.icl_templates.icl_cir3_templates import iCLCir3Templates
from aimw.app.services.factory import runnable_system
from aimw.app.services.weighting.diversity_alignment_weighting import (
    DiversityAlignmentWeighting,
)
from aimw.app.tools import diversity_tools
from aimw.app.core.ai_config import get_ai_settings
import aimw.app.services.weighting.weighting_feedback as weighting_feedback
from aimw.app.services.weighting.vendi_score_logger import vendi_logger

# setup logging as early as possible
# setup_app_logging(config=LoggingSettings())


def identify_perspectives(state):
    """take the document and categorize it into one main topic and a list of subtopics"""

    logger.info("--- Entering 'identify_perspectives' node ---")

    document = state["document"]
    M = state["M"]
    num_steps = int(state["num_steps"])
    num_steps += 1

    subtopics = runnable_system.runnable_cir3.classifier_runnable.invoke(
        {"M": M, "document": document}
    )
    logger.info(f"Step: {num_steps} - subtopics: {subtopics}")

    logger.info("--- Leaving 'identify_perspectives' node ---")

    return {"subtopics": subtopics, "num_steps": num_steps}


def moderate_writers(state):

    logger.info("--- Entering `moderate_writers` node ---")
    document = state["document"]
    subtopics = state["subtopics"]
    M = state["M"]
    N = state["N"]
    num_steps = state["num_steps"]
    num_steps += 1

    moderator_response = runnable_system.runnable_cir3.moderator_runnable.invoke(
        {
            "M": M,
            "N": N,
            "document": document,
            "subtopics": subtopics,
            "moderator_examplar": iCLCir3Templates.moderator_examplar,
        }
    )
    logger.debug(f"type(moderator_response): {type(moderator_response)}")
    logger.debug(moderator_response)

    # Setup Writers Network
    runnable_system.runnable_cir3.setup_writers_group(moderator_response)
    runnable_system.runnable_cir3.writers.describe()

    logger.info("--- Leaving `moderate_writers` node ---")

    return {"num_steps": num_steps}


def individual_draft_qas_writer(state):

    logger.info("--- Entering `individual_draft_qas_writer` node ---")

    # Get the state
    document = state["document"]
    num_steps = state["num_steps"]
    num_steps += 1

    inner_transactive_memory = []  # Init only once per document
    outer_transactive_memory = []  # Init only once per document

    inner_iter_response = []  # Init at every inner iteration
    trans_mem_qas_inner_iter = []  # Init at every inner iteration
    trans_mem_feed_inner_iter = []  # Init at every inner iteration
    trans_mem_status_inner_iter = []  # Init at every inner iteration

    for i, runnable in enumerate(
        runnable_system.runnable_cir3.writers.writers_qadraft_runnables
    ):
        individual_writer_response = runnable.invoke(
            {
                "document": document,
                "moderator_prompt": runnable_system.runnable_cir3.writers.moderator_instructions[
                    i
                ],
            }
        )
        inner_iter_response.append(individual_writer_response)
        trans_mem_qas_inner_iter.append(
            individual_writer_response["question_answer_list"]
        )
        trans_mem_feed_inner_iter.append(individual_writer_response["writer_feedback"])
        trans_mem_status_inner_iter.append(individual_writer_response["writer_status"])

    inner_transactive_memory.append(
        (trans_mem_qas_inner_iter, trans_mem_feed_inner_iter)
    )

    logger.info(f"Step: {num_steps}")
    logger.debug(f"Inner_transactive_memory: {inner_transactive_memory}")
    logger.debug(f"QAs from last inner iteration: {inner_transactive_memory[-1][0]}")
    logger.debug(f"Feedback from last inner Iteration {inner_transactive_memory[-1][1]}")
    logger.info(f"Status from last inner Iteration {trans_mem_status_inner_iter}")

    logger.info("--- Leaving `individual_draft_qas_writer` node ---")

    return {
        "inner_transactive_memory": inner_transactive_memory,
        "num_steps": num_steps,
    }


def group_inner_refine(state):
    logger.info("--- Entering `group_inner_refine` node ---")

    # Get the state
    document = state["document"]
    num_steps = state["num_steps"]
    num_steps += 1

    L = state["L"]
    inner_transactive_memory = state["inner_transactive_memory"]
    outer_transactive_memory = state["outer_transactive_memory"]

    # trans_mem_status_inner_iter = state["trans_mem_status_inner_iter"]

    status = True
    l = 0
    while (l < L) and status:

        # status = all([s["status"] == "agreement" for s in trans_mem_status_inner_iter])

        old_feedback = {}
        inner_iter_response = []  # Init at every inner iteration
        trans_mem_qas_inner_iter = []  # Init at every inner iteration
        trans_mem_feed_inner_iter = []  # Init at every inner iteration
        trans_mem_status_inner_iter = []  # Init at every inner iteration

        curmudgeon_feedback = (
            outer_transactive_memory[-1]
            if outer_transactive_memory is not None
            and len(outer_transactive_memory) > 0
            else outer_transactive_memory
        )
        for i, runnable in enumerate(
            runnable_system.runnable_cir3.writers.writers_inner_refine_runnables
        ):
            individual_writer_response = runnable.invoke(
                {
                    "document": document,
                    "old_question_answer_list": inner_transactive_memory[-1][0],
                    "old_writer_feedback": old_feedback,
                    "curmudgeon_feedback": curmudgeon_feedback,
                    "perspective": runnable_system.runnable_cir3.writers.agents_group[
                        i
                    ].params["perspective"],
                }
            )

            inner_iter_response.append(individual_writer_response)
            trans_mem_qas_inner_iter.append(
                individual_writer_response["new_question_answer_list"]
            )
            trans_mem_feed_inner_iter.append(
                individual_writer_response["new_writer_feedback"]
            )
            trans_mem_status_inner_iter.append(
                individual_writer_response["writer_status"]
            )

        inner_transactive_memory.append(
            (trans_mem_qas_inner_iter, trans_mem_feed_inner_iter)
        )

        logger.info(f"Step: {num_steps}")
        logger.debug(f"Inner_transactive_memory: {inner_transactive_memory}")
        logger.debug(f"QAs from last inner iteration: {inner_transactive_memory[-1][0]}")
        logger.debug(
            f"Feedback from last inner Iteration {inner_transactive_memory[-1][1]}"
        )
        logger.info(f"Status from last inner Iteration {trans_mem_status_inner_iter}")

        status = all([s == "continue" for s in trans_mem_status_inner_iter])
        old_feedback = inner_transactive_memory[-1][1]
        l += 1

    logger.info("--- Leaving `group_inner_refine` node ---")
    return {
        "inner_transactive_memory": inner_transactive_memory,
        "num_steps": num_steps,
    }


def outer_refine(state):
    logger.info("---Entering 'outer_refine' Node ---")
    # Get the state
    document = state["document"]
    inner_transactive_memory = state["inner_transactive_memory"]
    outer_transactive_memory = (state["outer_transactive_memory"], [])[
        state["outer_transactive_memory"] is None
    ]

    K = state["K"]
    N = state["N"]

    num_steps = state["num_steps"]
    num_steps += 1

    if K >= 0:
        questions = [
            d["question"]
            for sublist in inner_transactive_memory[-1][0]  # tuple > list > list
            for d in sublist
        ]
        answers = [
            d["answer"] for sublist in inner_transactive_memory[-1][0] for d in sublist
        ]

        logger.debug(f"===========> K: {K} - questions: {questions}")
        logger.debug(f"===========> K: {K} - answers: {answers}")

        old_curmudgeon_feedback = (
            outer_transactive_memory[-1]
            if outer_transactive_memory is not None
            and len(outer_transactive_memory) > 0
            else outer_transactive_memory
        )


        #######################################################################################
        # Curmudgeon Strategy
        #######################################################################################
        if get_ai_settings().curmudgeon_strategy == "vendi_only":

            weighting = DiversityAlignmentWeighting()
            vendi_scores = weighting.calculate_vendi_scores(
                questions, answers, document, granularity=True
            )

            # Add scores to logger
            vendi_logger.add_score(vendi_scores)

            vendi_scores.pop("comp_diversity_score")
            vendi_scores.pop("faith_alignment_score")

            curmudgeon_input["weighted_scores"] = vendi_scores

            # Construct curmudgeon response using questions and answers
            curmudgeon_feedback = []
            for q, a in zip(questions, answers):
                curmudgeon_feedback.append(
                    {
                        "question": q,
                        "answer": a,
                        "curmudgeon_feedback": weighting_feedback.construct_curmudgeon_feedback(
                            vendi_scores=vendi_scores,
                            balanced_g_score_threshold=get_ai_settings().balanced_g_score_threshold,
                        ),
                    }
                )

            status = (
                "continue"
                if vendi_scores["balanced_g_score"]
                < get_ai_settings().balanced_g_score_threshold
                else "agreement"
            )

            curmudgeon_response = {
                "new_curmudgeon_feedback": curmudgeon_feedback,
                "curmudgeon_status": status,
            }

        elif get_ai_settings().curmudgeon_strategy == "random_rejection":

            curmudgeon_feedback = []
            for q, a in zip(questions, answers):
                curmudgeon_feedback.append(
                    {
                        "question": q,
                        "answer": a,
                        "curmudgeon_feedback": weighting_feedback.construct_random_curmudgeon_feedback(
                            disagreement_probability=get_ai_settings().rondon_disagreement_probability
                        ),
                    }
                )

            status = weighting_feedback.random_disagreement_control(
                disagreement_probability=get_ai_settings().rondon_disagreement_probability
            )
            curmudgeon_response = {
                "new_curmudgeon_feedback": curmudgeon_feedback,
                "curmudgeon_status": status,
            }

        elif get_ai_settings().curmudgeon_strategy == "curmudgeon_only":

            curmudgeon_input = {
                "document": document,
                "N": N,
                "question_answer_list": inner_transactive_memory[-1][0],
                "old_curmudgeon_feedback": old_curmudgeon_feedback,
                "last_iteration": K == 0,
            }
            curmudgeon_response = (
                runnable_system.runnable_cir3.curmudgeon_runnable.invoke(
                    curmudgeon_input
                )
            )

        else:  # curmudgeon_vendi: curmudgeon with vendi as tool
            curmudgeon_input = {
                "document": document,
                "N": N,
                "question_answer_list": inner_transactive_memory[-1][0],
                "old_curmudgeon_feedback": old_curmudgeon_feedback,
                "last_iteration": K == 0,
            }
            weighting = DiversityAlignmentWeighting()
            vendi_scores = weighting.calculate_vendi_scores(
                questions, answers, document, granularity=True
            )

            # Add scores to logger
            vendi_logger.add_score(vendi_scores)

            vendi_scores.pop("comp_diversity_score")
            vendi_scores.pop("faith_alignment_score")

            curmudgeon_input["weighted_scores"] = vendi_scores

            curmudgeon_response = (
                runnable_system.runnable_cir3.curmudgeon_runnable.invoke(
                    curmudgeon_input
                )
            )

        outer_transactive_memory.append(curmudgeon_response)

        logger.debug(f"type(curmudgeon_response): {type(curmudgeon_response)}")

        #######################################################################################
        # End of Curmudgeon Strategy
        #######################################################################################

    K -= 1
    logger.info(f"Step: {num_steps} - curmudgeon iter {K}: \n {curmudgeon_response}")
    logger.info("---Leaving 'outer_refine' Node ---")

    return {
        "K": K,
        "num_steps": num_steps,
        "outer_transactive_memory": outer_transactive_memory,
    }


def halt(state):
    logger.info("--- Terminate Process---")
    ## Get the state

    outer_transactive_memory = state["outer_transactive_memory"]
    K = state["K"]

    num_steps = state["num_steps"]
    num_steps += 1

    # Flush any remaining vendi scores
    vendi_logger.flush_scores()

    # write_markdown_file(str(question_answer_list), "final_QAs")
    logger.info(
        f"Step: {num_steps} - finale QAs: {outer_transactive_memory[-1]['new_curmudgeon_feedback']}"
    )

    final_qas = [
        {"question": item["question"], "answer": item["answer"]}
        for item in outer_transactive_memory[-1]["new_curmudgeon_feedback"]
    ]
    return {
        "final_qas": final_qas,
        "num_steps": num_steps,
    }


def log_transactive_memory(state):
    """print the state"""
    logger.info("--- Transactive Memory State---")

    inner_transactive_memory = state["inner_transactive_memory"]
    outer_transactive_memory = state["outer_transactive_memory"]

    logger.info(f"inner_transactive_memory: {inner_transactive_memory}")
    logger.info(f"outer_transactive_memory: {outer_transactive_memory}")
