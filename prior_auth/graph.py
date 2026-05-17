from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from prior_auth.state import PriorAuthState
from prior_auth.nodes import (
    supervisor_node,
    policy_lookup_node,
    clinical_criteria_node,
    decision_node,
    route_from_supervisor,
)


def build_graph():
    graph = StateGraph(PriorAuthState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("policy_lookup", policy_lookup_node)
    graph.add_node("clinical_criteria", clinical_criteria_node)
    graph.add_node("decision", decision_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "policy_lookup": "policy_lookup",
            "clinical_criteria": "clinical_criteria",
            "decision": "decision",
            "FINISH": END,
        },
    )
    graph.add_edge("policy_lookup", "supervisor")
    graph.add_edge("clinical_criteria", "supervisor")
    graph.add_edge("decision", "supervisor")

    return graph.compile(checkpointer=MemorySaver())
