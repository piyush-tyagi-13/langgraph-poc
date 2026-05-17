from typing import TypedDict, Annotated
import operator


class PriorAuthState(TypedDict):
    messages: Annotated[list, operator.add]
    next_agent: str
    procedure_code: str
    patient_info: str
    policy_result: str
    clinical_result: str
    final_decision: str
