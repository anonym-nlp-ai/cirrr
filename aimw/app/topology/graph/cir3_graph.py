from langgraph.graph import END, StateGraph
from loguru import logger

from aimw.app.topology.edges import cir3_edges
from aimw.app.topology.memory.transactive_memory_state import GraphState
from aimw.app.topology.nodes import cir3_nodes


class Cir3Graph:
    def __init__(self) -> None:
        # Setup Topology Nodes
        self.cir3_workflow = StateGraph(GraphState)

        # Define the nodes
        self.cir3_workflow.add_node(
            "identify_perspectives", cir3_nodes.identify_perspectives
        )
        self.cir3_workflow.add_node("moderate_writers", cir3_nodes.moderate_writers)
        self.cir3_workflow.add_node(
            "individual_draft_qas_writer", cir3_nodes.individual_draft_qas_writer
        )
        self.cir3_workflow.add_node("group_inner_refine", cir3_nodes.group_inner_refine)
        self.cir3_workflow.add_node("outer_refine", cir3_nodes.outer_refine)
        self.cir3_workflow.add_node("halt", cir3_nodes.halt)

        # Setup Topology Edges
        self.cir3_workflow.set_entry_point("identify_perspectives")
        self.cir3_workflow.add_edge("identify_perspectives", "moderate_writers")
        self.cir3_workflow.add_edge("moderate_writers", "individual_draft_qas_writer")
        self.cir3_workflow.add_edge("individual_draft_qas_writer", "group_inner_refine")
        self.cir3_workflow.add_edge("group_inner_refine", "outer_refine")
        self.cir3_workflow.add_conditional_edges(
            "outer_refine",
            cir3_edges.route_outer_refinenemnt,
            {
                "debate": "group_inner_refine",
                "terminate": "halt",
            },
        )
        self.cir3_workflow.add_edge("halt", END)

        # Compile
        # self.app = self.cir3_workflow.compile()
