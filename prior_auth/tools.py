from langchain_core.tools import tool

_COVERAGE_POLICIES: dict[str, str] = {
    "MRI-70553": "Covered. Prior authorization required. Medical necessity must be documented.",
    "CT-71250": "Covered. No prior authorization needed for first occurrence.",
    "SURG-27447": "Covered. Prior authorization mandatory. Two specialist opinions required.",
    "PT-97110": "Covered. Limited to 20 sessions per year. No prior auth for first 6 sessions.",
}


@tool
def lookup_coverage_policy(procedure_code: str) -> str:
    """Look up whether a medical procedure is covered under the patient's insurance plan.
    Use this when you need to check coverage policy for a specific procedure code."""
    return _COVERAGE_POLICIES.get(
        procedure_code,
        f"Policy not found for {procedure_code}. Manual review required.",
    )


@tool
def check_clinical_criteria(procedure_code: str, patient_info: str) -> str:
    """Check whether a patient meets the clinical criteria for a requested procedure.
    Use this when you have the procedure code and patient information and need to
    evaluate medical necessity."""
    criteria_met = {
        "MRI-70553": "chronic" in patient_info.lower() or "persistent" in patient_info.lower(),
        "CT-71250": True,
        "SURG-27447": "failed conservative" in patient_info.lower(),
        "PT-97110": True,
    }
    met = criteria_met.get(procedure_code, False)
    if met:
        return f"Clinical criteria MET for {procedure_code}. Patient presentation supports medical necessity."
    return f"Clinical criteria NOT MET for {procedure_code}. Insufficient documentation of medical necessity."


ALL_TOOLS = [lookup_coverage_policy, check_clinical_criteria]
