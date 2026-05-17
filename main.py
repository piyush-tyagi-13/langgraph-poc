from prior_auth import build_graph
from prior_auth.nodes import clean_output

INITIAL_STATE = {
    "messages": ["Starting prior authorization workflow"],
    "next_agent": "",
    "procedure_code": "MRI-70553",
    "patient_info": "45-year-old patient with chronic lower back pain, persistent for 6 months, failed physical therapy",
    "policy_result": "",
    "clinical_result": "",
    "final_decision": "",
}

CONFIG = {"configurable": {"thread_id": "prior-auth-case-001"}}


def run() -> None:
    graph = build_graph()

    print("Starting Prior Authorization Workflow...\n")
    print("=" * 60)

    for chunk in graph.stream(INITIAL_STATE, config=CONFIG):
        for node_name, node_output in chunk.items():
            print(f"\n[{node_name.upper()}]")
            for msg in node_output.get("messages", []):
                print(f"  {msg}")
            if node_output.get("final_decision"):
                print(f"\n{'=' * 60}")
                print("FINAL DECISION:")
                print(clean_output(node_output["final_decision"]))
                print("=" * 60)


if __name__ == "__main__":
    run()
