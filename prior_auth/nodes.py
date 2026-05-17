import re
from langchain_core.output_parsers import StrOutputParser

from prior_auth.config import create_llm
from prior_auth.models import SupervisorDecision
from prior_auth.prompts import (
    supervisor_prompt,
    policy_lookup_prompt,
    clinical_criteria_prompt,
    decision_prompt,
)
from prior_auth.state import PriorAuthState
from prior_auth.tools import ALL_TOOLS, lookup_coverage_policy, check_clinical_criteria

_llm = create_llm()
_llm_with_tools = _llm.bind_tools(ALL_TOOLS)
_supervisor_chain = supervisor_prompt | _llm.with_structured_output(SupervisorDecision)


def clean_output(text: str) -> str:
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def supervisor_node(state: PriorAuthState) -> dict:
    result = _supervisor_chain.invoke({
        "procedure_code": state["procedure_code"],
        "patient_info": state["patient_info"],
        "policy_result": state.get("policy_result", "Not checked yet"),
        "clinical_result": state.get("clinical_result", "Not checked yet"),
        "final_decision": state.get("final_decision", "Not made yet"),
    })
    return {
        "next_agent": result.next_agent,
        "messages": [f"Supervisor: routing to {result.next_agent}. Reasoning: {result.reasoning}"],
    }


def policy_lookup_node(state: PriorAuthState) -> dict:
    chain = policy_lookup_prompt | _llm_with_tools
    result = chain.invoke({"procedure_code": state["procedure_code"]})

    if result.tool_calls:
        tool_result = lookup_coverage_policy.invoke(result.tool_calls[0]["args"])
        return {"policy_result": tool_result, "messages": [f"Policy Lookup: {tool_result}"]}
    return {"policy_result": result.content, "messages": [f"Policy Lookup: {result.content}"]}


def clinical_criteria_node(state: PriorAuthState) -> dict:
    chain = clinical_criteria_prompt | _llm_with_tools
    result = chain.invoke({
        "procedure_code": state["procedure_code"],
        "patient_info": state["patient_info"],
    })

    if result.tool_calls:
        tool_result = check_clinical_criteria.invoke(result.tool_calls[0]["args"])
        return {"clinical_result": tool_result, "messages": [f"Clinical Criteria: {tool_result}"]}
    return {"clinical_result": result.content, "messages": [f"Clinical Criteria: {result.content}"]}


def decision_node(state: PriorAuthState) -> dict:
    chain = decision_prompt | _llm | StrOutputParser()
    result = chain.invoke({
        "procedure_code": state["procedure_code"],
        "patient_info": state["patient_info"],
        "policy_result": state["policy_result"],
        "clinical_result": state["clinical_result"],
    })
    return {"final_decision": result, "messages": [f"Decision: {clean_output(result)}"]}


def route_from_supervisor(state: PriorAuthState) -> str:
    if state.get("final_decision") and state["final_decision"] not in ["", "Not made yet"]:
        return "FINISH"
    return state["next_agent"]
