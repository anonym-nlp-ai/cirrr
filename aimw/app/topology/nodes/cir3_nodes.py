from loguru import logger

from aimw.app.resources.icl_templates import icl_cir3_templates
from aimw.app.services.factory import runnable_system
from aimw.app.tools import diversity_tools

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
            "moderator_examplar": icl_cir3_templates.moderator_examplar,
        }
    )
    logger.info(f"type(moderator_response): {type(moderator_response)}")
    logger.info(moderator_response)

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
    logger.info(f"Inner_transactive_memory: {inner_transactive_memory}")
    logger.info(f"QAs from last inner iteration: {inner_transactive_memory[-1][0]}")
    logger.info(f"Feedback from last inner Iteration {inner_transactive_memory[-1][1]}")
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
        logger.info(f"Inner_transactive_memory: {inner_transactive_memory}")
        logger.info(f"QAs from last inner iteration: {inner_transactive_memory[-1][0]}")
        logger.info(
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
            for sublist in inner_transactive_memory[-1][0]
            for d in sublist
        ]
        answers = [
            d["answer"] for sublist in inner_transactive_memory[-1][0] for d in sublist
        ]

        logger.debug(f"questions: {questions}")
        logger.debug(f"answers: {answers}")

        vendi_q = diversity_tools.get_vendi_scores(questions, "bge")["bge_score"]
        vendi_a = diversity_tools.get_vendi_scores(answers, "bge")["bge_score"]
        vendi_ca = diversity_tools.get_vendi_scores(
            [" ".join(answers), document], "bge"
        )["bge_score"]

        logger.debug(f"vendi_q: {vendi_q}")
        logger.debug(f"vendi_a: {vendi_a}")
        logger.debug(f"vendi_ca: {vendi_ca}")

        vendi_scores = {"score_1": vendi_q, "score_2": vendi_a, "score_3": vendi_ca}

        old_curmudgeon_feedback = (
            outer_transactive_memory[-1]
            if outer_transactive_memory is not None
            and len(outer_transactive_memory) > 0
            else outer_transactive_memory
        )
        curmudgeon_response = runnable_system.runnable_cir3.curmudgeon_runnable.invoke(
            {
                "document": document,
                "N": N,
                "question_answer_list": inner_transactive_memory[-1][0],
                "old_curmudgeon_feedback": old_curmudgeon_feedback,
                "diversity_scores": vendi_scores,
                "last_iteration": K == 0,
            }
        )

        outer_transactive_memory.append(curmudgeon_response)

        logger.debug(f"type(curmudgeon_response): {type(curmudgeon_response)}")

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
