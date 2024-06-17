from loguru import logger


def route_outer_refinement(state):
    """
    Route to writers or moderator
    Args:
        state (dict): The current graph state
    Returns:
        str: Next node to call
    """

    logger.info("---'route_outer_refinement'---")

    outer_transactive_memory = state["outer_transactive_memory"]
    K = state["K"]

    num_steps = state["num_steps"]
    num_steps += 1

    route = ("terminate", "debate")[
        (outer_transactive_memory[-1]["curmudgeon_status"] == "continue") and (K >= 0)
    ]

    logger.info(f"Route: {route}")

    return route
