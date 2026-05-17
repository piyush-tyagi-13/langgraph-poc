from typing import Literal
from pydantic import BaseModel


class SupervisorDecision(BaseModel):
    next_agent: Literal["policy_lookup", "clinical_criteria", "decision", "FINISH"]
    reasoning: str
