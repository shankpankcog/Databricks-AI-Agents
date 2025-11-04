# server.py
from fastapi import FastAPI
from langserve import add_routes
from pyspark.sql import SparkSession
from langgraph.graph import StateGraph, END
from functools import partial
import uuid
import json

# Import your existing multi-agent system components
from multi_agent_system import (
    AgentState,
    get_table_schemas,
    cache_agent,
    router_agent,
    intent_agent,
    rag_agent,
    text_to_sql_agent,
    mixed_intent_agent,
    voice_summarization_agent,
    context_engineer_agent,
    response_agent,
    llm_error_agent,
    save_cache_agent,
    llm,
    vsc
)
import guardrails
from models import ErrorResponse
from logger_config import get_logger

logger = get_logger(__name__)

# Initialize the FastAPI app
app = FastAPI(
    title="Multi-Agent System Server",
    version="1.0",
    description="A server for the multi-agent system.",
)

# --- Assemble the LangGraph Application ---
workflow = StateGraph(AgentState)

rag_agent_with_vsc = partial(rag_agent, vsc=vsc)
mixed_intent_agent_with_vsc = partial(mixed_intent_agent, vsc=vsc)

workflow.add_node("cache_agent", cache_agent)
workflow.add_node("router", router_agent)
workflow.add_node("intent_agent", intent_agent)
workflow.add_node("rag_agent", rag_agent_with_vsc)
workflow.add_node("text_to_sql_agent", text_to_sql_agent)
workflow.add_node("mixed_intent_agent", mixed_intent_agent_with_vsc)
workflow.add_node("voice_summarization_agent", voice_summarization_agent)
workflow.add_node("context_engineer_agent", context_engineer_agent)
workflow.add_node("response_agent", response_agent)
workflow.add_node("llm_error_agent", llm_error_agent)
workflow.add_node("save_cache", save_cache_agent)

workflow.set_entry_point("cache_agent")

workflow.add_conditional_edges("cache_agent", lambda state: "continue" if not state.get("from_cache") else "end", {"continue": "router", "end": END})
workflow.add_edge("router", "intent_agent")
workflow.add_conditional_edges(
    "intent_agent",
    lambda state: state["intent"],
    {
        "structured": "text_to_sql_agent",
        "unstructured": "rag_agent",
        "mixed": "mixed_intent_agent",
        "voice": "voice_summarization_agent"
    }
)

workflow.add_conditional_edges("rag_agent", lambda state: "error" if state.get("error") else "continue", {"continue": "context_engineer_agent", "error": "llm_error_agent"})
workflow.add_conditional_edges("text_to_sql_agent", lambda state: "error" if state.get("error") else "continue", {"continue": "context_engineer_agent", "error": "llm_error_agent"})
workflow.add_conditional_edges("mixed_intent_agent", lambda state: "error" if state.get("error") else "continue", {"continue": "context_engineer_agent", "error": "llm_error_agent"})
workflow.add_conditional_edges("voice_summarization_agent", lambda state: "error" if state.get("error") else "continue", {"continue": "context_engineer_agent", "error": "llm_error_agent"})

workflow.add_edge("context_engineer_agent", "response_agent")
workflow.add_edge("response_agent", "save_cache")
workflow.add_edge("llm_error_agent", "save_cache")
workflow.add_edge("save_cache", END)

langgraph_app = workflow.compile()

# --- Guarded Application Logic ---
def guarded_app(inputs: dict) -> dict:
    """
    Wraps the main LangGraph application with input and output guardrails.

    Args:
        inputs (dict): The input dictionary for the LangGraph app.

    Returns:
        dict: The final output from the LangGraph app, after being processed by the guardrails.
    """
    if not guardrails.is_content_safe(llm, inputs['user_query']):
        error_model = ErrorResponse(
            session_id=inputs.get("session_id"),
            user_query=inputs.get("user_query"),
            error_message="Input query failed guardrail check."
        )
        return {"final_response": error_model.model_dump_json()}

    final_output = None
    for output in langgraph_app.stream(inputs, {"recursion_limit": 10}):
        if "final_response" in output:
            final_output = output

    if final_output and final_output["final_response"]:
        response_str = final_output["final_response"] if isinstance(final_output["final_response"], str) else json.dumps(final_output["final_response"])
        filtered_response = guardrails.filter_output(llm, response_str)
        final_output["final_response"] = filtered_response
        return final_output

    return {"error": "Failed to get a final response from the agent workflow."}


# --- Add LangServe Routes ---
add_routes(
    app,
    guarded_app,
    path="/agent",
)

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
