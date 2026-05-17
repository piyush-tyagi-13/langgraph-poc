from typing import TypedDict, Annotated, Literal
import operator
import os
import re
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

class PriorAuthState(TypedDict):
    messages: Annotated[list, operator.add]
    next_agent: str
    procedure_code: str
    patient_info: str
    policy_result: str
    clinical_result: str
    final_decision: str

@tool
def lookup_coverage_policy(procedure_code: str) -> str:
    """Look up whether a medical procedure is covered under the patient's insurance plan.
    Use this when you need to check coverage policy for a specific procedure code."""
    policies = {
        "MRI-70553": "Covered. Prior authorization required. Medical necessity must be documented.",
        "CT-71250": "Covered. No prior authorization needed for first occurrence.",
        "SURG-27447": "Covered. Prior authorization mandatory. Two specialist opinions required.",
        "PT-97110": "Covered. Limited to 20 sessions per year. No prior auth for first 6 sessions."
    }
    return policies.get(procedure_code, f"Policy not found for {procedure_code}. Manual review required.")


@tool
def check_clinical_criteria(procedure_code: str, patient_info: str) -> str:
    """Check whether a patient meets the clinical criteria for a requested procedure.
    Use this when you have the procedure code and patient information and need to
    evaluate medical necessity."""
    criteria_met = {
        "MRI-70553": "chronic" in patient_info.lower() or "persistent" in patient_info.lower(),
        "CT-71250": True,
        "SURG-27447": "failed conservative" in patient_info.lower(),
        "PT-97110": True
    }
    met = criteria_met.get(procedure_code, False)
    if met:
        return f"Clinical criteria MET for {procedure_code}. Patient presentation supports medical necessity."
    else:
        return f"Clinical criteria NOT MET for {procedure_code}. Insufficient documentation of medical necessity."


class SupervisorDecision(BaseModel):
    next_agent: Literal["policy_lookup", "clinical_criteria", "decision", "FINISH"]
    reasoning: str


supervisor_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a prior authorization supervisor managing specialist agents.

                Your available agents are:
                - policy_lookup: Checks insurance coverage policy. Use when policy_result is empty.
                - clinical_criteria: Evaluates clinical necessity. Use when clinical_result is empty.
                - decision: Makes final authorization decision. Use when BOTH policy_result AND clinical_result are populated AND final_decision is empty.
                - FINISH: Use when final_decision is already populated. DO NOT route to decision again if final_decision exists.
                
                CRITICAL RULE: If final_decision is anything other than empty string or 'Not made yet', you MUST respond with FINISH."""),
                    ("human", """
                Procedure Code: {procedure_code}
                Patient Info: {patient_info}
                Policy Result: {policy_result}
                Clinical Result: {clinical_result}
                Final Decision: {final_decision}
                
                What should happen next?
    """)
])

structured_llm = llm.with_structured_output(SupervisorDecision)
supervisor_chain = supervisor_prompt | structured_llm


def supervisor_node(state: PriorAuthState):
    result = supervisor_chain.invoke({
        "procedure_code": state["procedure_code"],
        "patient_info": state["patient_info"],
        "policy_result": state.get("policy_result", "Not checked yet"),
        "clinical_result": state.get("clinical_result", "Not checked yet"),
        "final_decision": state.get("final_decision", "Not made yet")
    })
    return {
        "next_agent": result.next_agent,
        "messages": [f"Supervisor: routing to {result.next_agent}. Reasoning: {result.reasoning}"]
    }


tools = [lookup_coverage_policy, check_clinical_criteria]
llm_with_tools = llm.bind_tools(tools)


def policy_lookup_node(state: PriorAuthState):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a policy specialist. Use the lookup_coverage_policy tool to check coverage."),
        ("human", "Check coverage policy for procedure code: {procedure_code}")
    ])
    chain = prompt | llm_with_tools
    result = chain.invoke({"procedure_code": state["procedure_code"]})

    # Handle tool calls
    if result.tool_calls:
        tool_result = lookup_coverage_policy.invoke(
            result.tool_calls[0]["args"]
        )
        return {
            "policy_result": tool_result,
            "messages": [f"Policy Lookup: {tool_result}"]
        }
    return {
        "policy_result": result.content,
        "messages": [f"Policy Lookup: {result.content}"]
    }


def clinical_criteria_node(state: PriorAuthState):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a clinical specialist. Use the check_clinical_criteria tool to evaluate medical necessity."),
        ("human", "Evaluate clinical criteria for procedure {procedure_code} for patient: {patient_info}")
    ])
    chain = prompt | llm_with_tools
    result = chain.invoke({
        "procedure_code": state["procedure_code"],
        "patient_info": state["patient_info"]
    })

    if result.tool_calls:
        tool_result = check_clinical_criteria.invoke(
            result.tool_calls[0]["args"]
        )
        return {
            "clinical_result": tool_result,
            "messages": [f"Clinical Criteria: {tool_result}"]
        }
    return {
        "clinical_result": result.content,
        "messages": [f"Clinical Criteria: {result.content}"]
    }


decision_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a prior authorization decision specialist.
                    Based on the policy result and clinical criteria evaluation, make a final decision.
                    Be specific and cite both the policy and clinical findings in your decision."""),
    ("human", """
                    Procedure: {procedure_code}
                    Patient: {patient_info}
                    Policy Finding: {policy_result}
                    Clinical Finding: {clinical_result}
                    
                    Make the final prior authorization decision.
    """)
])

def clean_output(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

def decision_node(state: PriorAuthState):
    chain = decision_prompt | llm | StrOutputParser()
    result = chain.invoke({
        "procedure_code": state["procedure_code"],
        "patient_info": state["patient_info"],
        "policy_result": state["policy_result"],
        "clinical_result": state["clinical_result"]
    })
    return {
        "final_decision": result,
        "messages": [f"Decision: {clean_output(result)}"]
    }


def route_from_supervisor(state: PriorAuthState) -> str:
    # Safety net: if decision exists, always finish regardless of what supervisor said
    if state.get("final_decision") and state["final_decision"] not in ["", "Not made yet"]:
        return "FINISH"
    return state["next_agent"]


def build_graph():
    from langgraph.checkpoint.memory import MemorySaver
    checkpointer = MemorySaver()

    graph = StateGraph(PriorAuthState)

    # Add nodes
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("policy_lookup", policy_lookup_node)
    graph.add_node("clinical_criteria", clinical_criteria_node)
    graph.add_node("decision", decision_node)

    # Entry point
    graph.add_edge(START, "supervisor")

    # Supervisor routes conditionally
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {
            "policy_lookup": "policy_lookup",
            "clinical_criteria": "clinical_criteria",
            "decision": "decision",
            "FINISH": END
        }
    )

    # Specialists always return to supervisor
    graph.add_edge("policy_lookup", "supervisor")
    graph.add_edge("clinical_criteria", "supervisor")
    graph.add_edge("decision", "supervisor")

    return graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":
    graph = build_graph()

    print(graph.get_graph().draw_mermaid())

    initial_state = {
        "messages": ["Starting prior authorization workflow"],
        "next_agent": "",
        "procedure_code": "MRI-70553",
        "patient_info": "45-year-old patient with chronic lower back pain, persistent for 6 months, failed physical therapy",
        "policy_result": "",
        "clinical_result": "",
        "final_decision": ""
    }

    config = {"configurable": {"thread_id": "prior-auth-case-001"}}

    print("Starting Prior Authorization Workflow...\n")
    print("=" * 60)

    # for chunk in graph.stream(initial_state, config=config):
    #     for node_name, node_output in chunk.items():
    #         print(f"\n[{node_name.upper()}]")
    #         if "messages" in node_output:
    #             for msg in node_output["messages"]:
    #                 print(f"  {msg}")
    #         if "final_decision" in node_output and node_output["final_decision"]:
    #             print(f"\n{'=' * 60}")
    #             print("FINAL DECISION:")
    #             print(clean_output(node_output["final_decision"]))  # add clean_output here
    #             print("=" * 60)