"""
Phoenix instrumentation setup for LangChain/LangGraph tracing.
"""
import phoenix as px
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register


def setup_phoenix(endpoint: str = "http://localhost:6006/v1/traces") -> None:
    """
    Initialize Phoenix for LLM observability.
    
    Args:
        endpoint: Phoenix server endpoint URL. Default is local development server.
        
    Example:
        >>> from prior_auth.phoenix_setup import setup_phoenix
        >>> setup_phoenix()
        >>> # Now run your LangChain/LangGraph code and traces will appear in Phoenix UI
    """
    # Launch Phoenix UI (only needed in development)
    px.launch_app()
    
    # Register tracer provider with Phoenix
    tracer_provider = register(endpoint=endpoint)
    
    # Instrument LangChain for automatic tracing
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    
    print(f"✓ Phoenix instrumentation enabled (traces: {endpoint})")
