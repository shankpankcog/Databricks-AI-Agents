# multi_agent_system.py
import os
import datetime
from typing import TypedDict, Annotated, Sequence, Union
import operator
import hashlib
import uuid
from functools import partial
import json

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores.databricks import DatabricksVectorSearch
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from pyspark.sql import SparkSession

# Import project assets
import config
import prompts
import cache_manager
from logger_config import get_logger
from models import RAGResponse, SQLResponse, MixedResponse, ErrorResponse, BaseResponse

logger = get_logger(__name__)

# --- Agent State Definition ---
class AgentState(TypedDict):
    """
    Represents the state of the agent workflow. It includes fields for session management,
    caching, and the results from different agentic steps.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    session_id: str
    intent: str
    sql_query: str
    sql_result: str
    rag_result: str
    error: str
    final_response: Union[BaseResponse, str]
    from_cache: bool

# --- LLM Initialization ---
llm = ChatOpenAI(
    model=config.ENDPOINT_NAME,
    api_key=config.DATABRICKS_TOKEN,
    base_url=f"{config.DATABRICKS_HOST}/serving-endpoints"
)

# --- Agent Definitions ---

def get_table_schemas(spark: SparkSession, catalog: str, schema: str) -> str:
    """Dynamically fetches schema information for all tables."""
    logger.info(f"Fetching schemas from {catalog}.{schema}")
    try:
        tables = spark.sql(f"SHOW TABLES IN {catalog}.{schema}")
        schema_details = [f"Table '{row['tableName']}': " + ", ".join([f"{c['col_name']} ({c['data_type']})" for c in spark.sql(f"DESCRIBE TABLE {catalog}.{schema}.{row['tableName']}").collect()]) for row in tables.collect()]
        return "\n".join(schema_details)
    except Exception as e:
        logger.error(f"Could not retrieve table schemas: {e}")
        return "Could not retrieve table schemas."


def cache_agent(state: AgentState) -> dict:
    """
    Checks if a response for the user's query exists in the cache Delta table.
    """
    logger.info("---CACHE AGENT---")
    spark = SparkSession.builder.appName("CacheAgent").getOrCreate()
    cached_response = cache_manager.check_cache(spark, state['user_query'])
    if cached_response:
        return {"final_response": cached_response, "from_cache": True}
    return {"from_cache": False}

def router_agent(state: AgentState) -> dict:
    """Initializes the conversation history for a new query."""
    logger.info("---ROUTER AGENT---")
    state['messages'] = [HumanMessage(content=state['user_query'])]
    return {"intent": "clarify_intent"}

def intent_agent(state: AgentState) -> dict:
    """Determines the user's intent."""
    logger.info("---INTENT AGENT---")
    prompt = ChatPromptTemplate.from_template(prompts.INTENT_AGENT_PROMPT)
    chain = prompt | llm | StrOutputParser()
    intent = chain.invoke({"user_query": state['user_query']}).strip().lower()
    logger.info(f"Detected intent: {intent}")

    if "structured" in intent: return {"intent": "structured"}
    if "unstructured" in intent: return {"intent": "unstructured"}
    if "mixed" in intent: return {"intent": "mixed"}
    return {"intent": "unstructured"}

def rag_agent(state: AgentState, vsc: VectorSearchClient) -> dict:
    """
    Performs a hybrid RAG search.
    """
    logger.info("---RAG AGENT (HYBRID)---")
    try:
        index = DatabricksVectorSearch(vsc.get_index(endpoint_name=config.VECTOR_SEARCH_ENDPOINT_NAME, index_name=config.VECTOR_SEARCH_INDEX_NAME))
        retriever = index.as_retriever()
        document_context = retriever.invoke(state['user_query'])

        spark = SparkSession.builder.appName("RAGAgent").getOrCreate()
        schema_context = get_table_schemas(spark, config.UNITY_CATALOG_NAME, config.UNITY_CATALOG_SCHEMA_NAME)

        prompt = ChatPromptTemplate.from_template(prompts.RAG_AGENT_PROMPT)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"question": state['user_query'], "document_context": document_context, "schema_context": schema_context})
        return {"rag_result": result}
    except Exception as e:
        logger.error(f"Error in RAG agent: {e}")
        return {"error": f"Error in RAG agent: {e}"}

def text_to_sql_agent(state: AgentState) -> dict:
    """Generates and executes a SQL query with a self-correction loop."""
    logger.info("---TEXT TO SQL AGENT---")
    spark = SparkSession.builder.appName("MultiAgentSystem").getOrCreate()
    schema_context = get_table_schemas(spark, config.UNITY_CATALOG_NAME, config.UNITY_CATALOG_SCHEMA_NAME)

    prompt = ChatPromptTemplate.from_template(prompts.TEXT_TO_SQL_AGENT_PROMPT)
    chain = prompt | llm | StrOutputParser()
    sql_query = chain.invoke({
        "catalog": config.UNITY_CATALOG_NAME,
        "schema": schema_context,
        "user_query": state['user_query']
    }).strip()

    for attempt in range(2):
        try:
            logger.info(f"Executing SQL Query (Attempt {attempt + 1}): {sql_query}")
            result_df = spark.sql(sql_query)
            result = result_df.limit(100).toPandas().to_string()
            return {"sql_query": sql_query, "sql_result": result}
        except Exception as e:
            error_message = str(e)
            logger.warning(f"SQL Query failed: {error_message}")

            correction_prompt = ChatPromptTemplate.from_template(prompts.SQL_CORRECTION_PROMPT)
            correction_chain = correction_prompt | llm | StrOutputParser()
            sql_query = correction_chain.invoke({
                "user_query": state['user_query'],
                "faulty_sql": sql_query,
                "error_message": error_message
            }).strip()

    return {"error": f"Failed to execute SQL query after correction: {error_message}"}


def mixed_intent_agent(state: AgentState, vsc: VectorSearchClient) -> dict:
    """Handles mixed-intent queries with a self-correction loop for the SQL part."""
    logger.info("---MIXED INTENT AGENT---")
    rag_output = rag_agent(state, vsc)
    if "error" in rag_output:
        return {"error": rag_output['error']}
    rag_context = rag_output.get("rag_result", "")

    spark = SparkSession.builder.appName("MultiAgentSystem").getOrCreate()
    schema_context = get_table_schemas(spark, config.UNITY_CATALOG_NAME, config.UNITY_CATALOG_SCHEMA_NAME)
    prompt = ChatPromptTemplate.from_template(prompts.MIXED_INTENT_SQL_PROMPT)
    chain = prompt | llm | StrOutputParser()
    sql_query = chain.invoke({
        "rag_context": rag_context,
        "catalog": config.UNITY_CATALOG_NAME,
        "schema": schema_context,
        "user_query": state['user_query']
    }).strip()

    for attempt in range(2):
        try:
            logger.info(f"Executing Mixed-Intent SQL Query (Attempt {attempt + 1}): {sql_query}")
            result_df = spark.sql(sql_query)
            sql_result = result_df.limit(100).toPandas().to_string()
            return {"rag_result": rag_context, "sql_result": sql_result, "sql_query": sql_query}
        except Exception as e:
            error_message = str(e)
            logger.warning(f"Mixed-Intent SQL Query failed: {error_message}")

            correction_prompt = ChatPromptTemplate.from_template(prompts.SQL_CORRECTION_PROMPT)
            correction_chain = correction_prompt | llm | StrOutputParser()
            sql_query = correction_chain.invoke({
                "user_query": state['user_query'],
                "faulty_sql": sql_query,
                "error_message": error_message
            }).strip()

    return {"error": f"Failed to execute mixed-intent SQL query after correction: {error_message}"}


def response_agent(state: AgentState) -> dict:
    """
    Formats the final response into a structured Pydantic model.
    """
    logger.info("---RESPONSE AGENT---")
    rag_result = state.get("rag_result")
    sql_result = state.get("sql_result")
    session_id = state.get("session_id")
    user_query = state.get("user_query")

    if rag_result and sql_result:
        prompt = ChatPromptTemplate.from_template(prompts.RESPONSE_AGENT_MIXED_PROMPT)
        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({"user_query": user_query, "rag_result": rag_result, "sql_result": sql_result})
        response_model = MixedResponse(
            session_id=session_id,
            user_query=user_query,
            synthesized_answer=answer,
            rag_sources=[],
            sql_query=state.get("sql_query", "")
        )
        return {"final_response": response_model.model_dump_json()}

    elif sql_result:
        prompt = ChatPromptTemplate.from_template(prompts.RESPONSE_AGENT_SQL_PROMPT)
        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({"user_query": user_query, "sql_result": sql_result})
        response_model = SQLResponse(
            session_id=session_id,
            user_query=user_query,
            summary=summary,
            sql_query=state.get("sql_query", ""),
            tabular_result=sql_result
        )
        return {"final_response": response_model.model_dump_json()}

    elif rag_result:
        response_model = RAGResponse(
            session_id=session_id,
            user_query=user_query,
            rag_content=rag_result,
            sources=[]
        )
        return {"final_response": response_model.model_dump_json()}

    error_model = ErrorResponse(
        session_id=session_id,
        user_query=user_query,
        error_message=state.get("error", "An unknown error occurred in the response agent.")
    )
    return {"final_response": error_model.model_dump_json()}

def llm_error_agent(state: AgentState) -> dict:
    """Handles errors and formats the response as a Pydantic model."""
    logger.error("---LLM ERROR AGENT---")
    error_message = state.get("error", "An unknown error occurred.")

    spark = SparkSession.builder.appName("ErrorLogging").getOrCreate()
    create_optimized_error_log_table(spark)

    try:
        from pyspark.sql.functions import current_timestamp
        error_data = [(state.get("session_id"), state.get("user_query"), error_message)]
        error_df = spark.createDataFrame(error_data, ["session_id", "user_query", "error_message"])
        error_df = error_df.withColumn("timestamp", current_timestamp())

        error_df.write.format("delta").mode("append").saveAsTable(config.ERROR_LOG_TABLE)
        logger.info(f"Error successfully logged to {config.ERROR_LOG_TABLE}")
    except Exception as e:
        logger.critical(f"Failed to log error to Delta table: {e}")

    prompt = ChatPromptTemplate.from_template(prompts.ERROR_AGENT_PROMPT)
    chain = prompt | llm | StrOutputParser()
    suggestion = chain.invoke({"error": error_message})

    error_model = ErrorResponse(
        session_id=state.get("session_id"),
        user_query=state.get("user_query"),
        error_message=error_message,
        suggested_fix=suggestion
    )
    return {"final_response": error_model.model_dump_json()}

def create_optimized_error_log_table(spark: SparkSession):
    """Creates the error log table with an explicit schema and optimized properties."""
    table_name = config.ERROR_LOG_TABLE
    if not spark.catalog.tableExists(table_name):
        logger.info(f"Error log table '{table_name}' not found. Creating it.")
        spark.sql(f"""
            CREATE TABLE {table_name} (
                timestamp TIMESTAMP,
                session_id STRING,
                user_query STRING,
                error_message STRING
            ) USING DELTA
            TBLPROPERTIES (
                delta.autoOptimize.optimizeWrite = true,
                delta.autoOptimize.autoCompact = true
            )
        """)

def save_cache_agent(state: AgentState) -> dict:
    """Saves the final response (as JSON) to the cache."""
    logger.info("---SAVE CACHE AGENT---")
    spark = SparkSession.builder.appName("SaveCacheAgent").getOrCreate()
    response_to_cache = state['final_response'] if isinstance(state['final_response'], str) else state['final_response']
    cache_manager.save_to_cache(spark, state['user_query'], response_to_cache, state['session_id'])
    return {}

# --- Main Workflow ---
def main():
    logger.info("Starting the multi-agent system...")
    vsc = VectorSearchClient()

    workflow = StateGraph(AgentState)

    rag_agent_with_vsc = partial(rag_agent, vsc=vsc)
    mixed_intent_agent_with_vsc = partial(mixed_intent_agent, vsc=vsc)

    workflow.add_node("cache_agent", cache_agent)
    workflow.add_node("router", router_agent)
    workflow.add_node("intent_agent", intent_agent)
    workflow.add_node("rag_agent", rag_agent_with_vsc)
    workflow.add_node("text_to_sql_agent", text_to_sql_agent)
    workflow.add_node("mixed_intent_agent", mixed_intent_agent_with_vsc)
    workflow.add_node("response_agent", response_agent)
    workflow.add_node("llm_error_agent", llm_error_agent)
    workflow.add_node("save_cache", save_cache_agent)

    workflow.set_entry_point("cache_agent")

    workflow.add_conditional_edges("cache_agent", lambda state: "continue" if not state.get("from_cache") else "end", {"continue": "router", "end": END})
    workflow.add_edge("router", "intent_agent")
    workflow.add_conditional_edges("intent_agent", lambda state: state["intent"], {"structured": "text_to_sql_agent", "unstructured": "rag_agent", "mixed": "mixed_intent_agent"})

    workflow.add_conditional_edges("rag_agent", lambda state: "error" if state.get("error") else "continue", {"continue": "response_agent", "error": "llm_error_agent"})
    workflow.add_conditional_edges("text_to_sql_agent", lambda state: "error" if state.get("error") else "continue", {"continue": "response_agent", "error": "llm_error_agent"})
    workflow.add_conditional_edges("mixed_intent_agent", lambda state: "error" if state.get("error") else "continue", {"continue": "response_agent", "error": "llm_error_agent"})

    workflow.add_edge("response_agent", "save_cache")
    workflow.add_edge("llm_error_agent", "save_cache")
    workflow.add_edge("save_cache", END)

    app = workflow.compile()

    logger.info("Multi-agent system is ready.")

    session_id = str(uuid.uuid4())
    query = "What are our top selling products?"
    inputs = {"user_query": query, "session_id": session_id, "messages": []}

    for output in app.stream(inputs, {"recursion_limit": 10}):
        for key, value in output.items():
            logger.info(f"--- Node: '{key}' ---")
            if value and "final_response" in value and value["final_response"]:
                try:
                    response_json = json.loads(value['final_response'])
                    logger.info(f"Final Response:\n{json.dumps(response_json, indent=2)}")
                except (json.JSONDecodeError, TypeError):
                    logger.info(f"Final Response:\n{value['final_response']}")
            else:
                logger.info(f"State Update: {value}")
        logger.info("\n" + "="*40 + "\n")

if __name__ == "__main__":
    main()
