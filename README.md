# Prior Authorization Agent - LangGraph POC

A proof-of-concept multi-agent system for automating healthcare prior authorization workflows using LangGraph and LLMs.

## Overview

This project demonstrates a **supervisor-worker agent pattern** where an LLM-powered supervisor coordinates specialized agents to evaluate prior authorization requests. The system:

- **Evaluates policy compliance** - Checks if a procedure is covered under insurance policies
- **Assesses clinical necessity** - Validates if a patient meets clinical criteria for a procedure
- **Makes authorization decisions** - Synthesizes policy and clinical findings into a final approval/denial decision

The workflow is driven by the supervisor agent, which orchestrates the flow between specialist agents based on the current state of information.

## Architecture

### Multi-Agent Pattern

The system uses a **supervisor routing pattern** where:

1. **Supervisor Agent** - Routes between specialists, determines workflow state
2. **Policy Lookup Agent** - Queries coverage policies for procedure codes
3. **Clinical Criteria Agent** - Evaluates medical necessity based on patient information
4. **Decision Agent** - Makes final prior authorization determination

### Agentic Workflow

```
START → Supervisor
   ↓
   ├→ Policy Lookup → Supervisor
   ├→ Clinical Criteria → Supervisor
   └→ Decision → Supervisor
   ↓
END (when final_decision is made)
```

The supervisor uses structured LLM output (`SupervisorDecision`) to deterministically route between agents, ensuring the workflow completes when all required information has been gathered.

## Project Structure

### Modular Organization

The project is split by responsibility into focused modules:

```
prior_auth/
├── __init__.py           # Package exports
├── state.py              # State schema (PriorAuthState TypedDict)
├── models.py             # Pydantic models (SupervisorDecision)
├── tools.py              # Tool definitions for agents
├── prompts.py            # LLM prompts for all agents
├── config.py             # LLM configuration (Groq)
├── nodes.py              # Node implementations for graph
└── graph.py              # Graph construction & compilation
```

#### Module Responsibilities

| Module | Purpose |
|--------|---------|
| **state.py** | Defines the `PriorAuthState` TypedDict with all state fields managed across the workflow |
| **models.py** | Pydantic model for structured LLM output (`SupervisorDecision`) |
| **tools.py** | Tool definitions (`lookup_coverage_policy`, `check_clinical_criteria`) for LLM invocation |
| **prompts.py** | System and human prompts for all agents (supervisor, policy, clinical, decision) |
| **config.py** | LLM client initialization (Groq Llama 3.3 70B) |
| **nodes.py** | Node function implementations that execute agent logic |
| **graph.py** | Constructs and compiles the LangGraph StateGraph |

### Entry Points

- **main.py** - Run the prior authorization workflow with a sample case
- **prior_auth_agent.py** - Monolithic reference implementation (shows all logic in one file)

## Getting Started

### Prerequisites

- Python 3.10+
- Groq API key (for LLM access)

### Installation

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd langgraph-poc
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your Groq API key
   GROQ_API_KEY=your_api_key_here
   ```

### Running the Workflow

Execute the prior authorization workflow:

```bash
python main.py
```

### Using Phoenix for Observability

Phoenix provides real-time tracing and debugging for LLM applications. To use it with the monolithic script:

1. Start the Phoenix UI in a separate terminal:
   ```bash
   phoenix serve --host 0.0.0.0 --port 6006
   ```
   The UI will be available at `http://localhost:6006`

2. Add Phoenix instrumentation to your script (e.g., `prior_auth_agent.py`) at the very top, before any LangChain imports:
   ```python
   from prior_auth.phoenix_setup import setup_phoenix
   setup_phoenix()
   
   # ... rest of your imports and code
   ```

   Or manually instrument:
   ```python
   import phoenix as px
   from openinference.instrumentation.langchain import LangChainInstrumentor
   from phoenix.otel import register

   px.launch_app()
   tracer_provider = register(endpoint="http://localhost:6006/v1/traces")
   LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
   ```

3. Run your workflow and observe traces in the Phoenix UI in real-time:
   ```bash
   python prior_auth_agent.py
   ```

**Phoenix Dashboard Features:**
- View LLM traces and chain execution flow
- Inspect prompt/completion tokens and latency
- Debug tool calls and agent routing decisions
- Monitor LLM costs and performance metrics

**Sample Output:**
```
Starting Prior Authorization Workflow...

============================================================

[SUPERVISOR]
  Supervisor: routing to policy_lookup. Reasoning: Policy not yet checked...

[POLICY_LOOKUP]
  Policy Lookup: Covered. Prior authorization required...

[SUPERVISOR]
  Supervisor: routing to clinical_criteria. Reasoning: Policy confirmed...

[CLINICAL_CRITERIA]
  Clinical Criteria: Clinical criteria MET for MRI-70553...

[SUPERVISOR]
  Supervisor: routing to decision. Reasoning: All criteria met...

[DECISION]

============================================================
FINAL DECISION:
Prior authorization APPROVED...
============================================================
```

## State Management

### PriorAuthState

The workflow maintains the following state fields:

| Field | Type | Purpose |
|-------|------|---------|
| `messages` | `list[str]` | Accumulates agent outputs and reasoning |
| `next_agent` | `str` | Supervisor's next routing decision |
| `procedure_code` | `str` | Medical procedure code (e.g., "MRI-70553") |
| `patient_info` | `str` | Patient clinical context and history |
| `policy_result` | `str` | Insurance policy lookup result |
| `clinical_result` | `str` | Clinical necessity evaluation result |
| `final_decision` | `str` | Final prior authorization decision |

## Customization

### Adding New Procedures

Update the policy and criteria mappings in `prior_auth/tools.py`:

```python
_COVERAGE_POLICIES = {
    "YOUR-CODE": "Coverage details...",
}

criteria_met = {
    "YOUR-CODE": lambda patient_info: your_logic_here,
}
```

### Modifying Agent Prompts

Edit `prior_auth/prompts.py` to customize:
- Supervisor routing logic
- Policy evaluation strategy
- Clinical criteria assessment
- Decision reasoning

### Changing the LLM

Update `prior_auth/config.py`:

```python
def create_llm() -> ChatGroq:
    return ChatGroq(
        model="your-model-name",
        api_key=os.getenv("YOUR_API_KEY"),
    )
```

## Key Design Decisions

### Separation of Concerns

- **State** defined separately from logic (state.py vs nodes.py)
- **Prompts** centralized in prompts.py for easy modification
- **Tools** isolated in tools.py for reusability and testing
- **LLM configuration** abstracted in config.py

### Structured Output

Uses Pydantic models for deterministic routing:
```python
class SupervisorDecision(BaseModel):
    next_agent: Literal["policy_lookup", "clinical_criteria", "decision", "FINISH"]
    reasoning: str
```

This ensures the supervisor reliably routes to the correct next node.

### State Accumulation

Uses `Annotated[list, operator.add]` for messages to automatically append new outputs without manual list merging.

## Comparing Implementations

### Monolithic (prior_auth_agent.py)
- ✅ Single file, easy to understand initially
- ❌ Difficult to maintain as logic grows
- ❌ Hard to reuse components
- ❌ State, tools, prompts, nodes all mixed together

### Modular (prior_auth/ package)
- ✅ Clear separation of concerns
- ✅ Easy to test individual components
- ✅ Reusable modules and tools
- ✅ Easier to scale and extend
- ✅ Prompts and configs centralized for quick iteration

## Future Enhancements

- [ ] Add database integration for historical decisions
- [ ] Implement user feedback loop for model refinement
- [ ] Add API endpoint for workflow submission
- [ ] Implement appeal/escalation workflows
- [ ] Add multi-step clinical validation
- [ ] Support for multiple insurance plan policies
- [ ] Performance metrics and analytics

## Dependencies

See [requirements.txt](requirements.txt) for exact ve
- **Arize Phoenix** - LLM observability and tracing
- **OpenInference LangChain Instrumentation** - Automatic LangChain tracingrsions.

- **LangChain Core** - LLM abstractions and tools
- **LangGraph** - Agentic workflow framework
- **LangChain Groq** - Groq LLM integration
- **Pydantic** - Data validation and structured output
- **python-dotenv** - Environment variable management