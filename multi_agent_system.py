# multi_agent_system.py
import os
import datetime
from typing import TypedDict, Annotated, Sequence, Union, Dict, Any
import operator
import hashlib
import uuid
from functools import partial
import json
import re
import whisper

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
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
from models import RAGResponse, SQLResponse, MixedResponse, ErrorResponse, VoiceResponse, BaseResponse

logger = get_logger(__name__)

# --- Agent State Definition ---
class AgentState(TypedDict):
    """
    Represents the state of the agent workflow.

    Attributes:
        messages: The history of messages in the conversation.
        user_query: The initial query from the user for the current turn.
        session_id: A unique identifier for the entire conversation session.
        intent: The determined intent of the user's query.
        intent_details: The structured JSON output from the intent agent.
        sql_query: The generated SQL query, if applicable.
        sql_result: The stringified result of the executed SQL query.
        rag_result: The result from the RAG agent.
        error: Any error message that occurs during the workflow.
        engineered_context: A consolidated block of text with all context.
        final_response: The final, user-facing response.
        from_cache: A flag to indicate if the response was from the cache.
    """
    messages: Annotated[Sequence[BaseMessage], operator.add]
    user_query: str
    session_id: str
    intent: str
    intent_details: Dict[str, Any]
    sql_query: str
    sql_result: str
    rag_result: str
    error: str
    engineered_context: str
    final_response: Union[BaseResponse, str]
    from_cache: bool

# --- LLM, VSC, and Whisper Initialization ---
llm = ChatOpenAI(model=config.ENDPOINT_NAME, api_key=config.DATABRICKS_TOKEN, base_url=f"{config.DATABRICKS_HOST}/serving-endpoints")
vsc = VectorSearchClient()
logger.info("Pre-loading Whisper model...")
whisper_model = whisper.load_model("base")
logger.info("Whisper model loaded.")

# --- Agent Definitions ---

def get_table_schemas(spark: SparkSession, catalog: str, schema: str) -> str:
    """
    Dynamically fetches schema information for all tables in a given Unity Catalog schema.

    Args:
        spark: The active Spark session.
        catalog: The name of the Unity Catalog.
        schema: The name of the schema within the catalog.

    Returns:
        A formatted string describing the schemas of all tables.
    """
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
    Checks if a response for the user's query exists in the cache.

    Args:
        state: The current state of the workflow.

    Returns:
        A dictionary with the cached response or a cache miss flag.
    """
    logger.info("---CACHE AGENT---")
    spark = SparkSession.builder.appName("CacheAgent").getOrCreate()
    cached_response = cache_manager.check_cache(spark, state['user_query'])
    if cached_response:
        return {"final_response": cached_response, "from_cache": True}
    return {"from_cache": False}

def router_agent(state: AgentState) -> dict:
    """
    Initializes the conversation history for a new query.

    Args:
        state: The current state of the workflow.

    Returns:
        A dictionary indicating the next step.
    """
    logger.info("---ROUTER AGENT---")
    state['messages'] = [HumanMessage(content=state['user_query'])]
    return {"intent": "clarify_intent"}

def intent_agent(state: AgentState) -> dict:
    """
    Determines the user's intent and extracts structured details.

    Args:
        state: The current state of the workflow.

    Returns:
        A dictionary containing the structured intent details.
    """
    logger.info("---INTENT AGENT---")
    prompt = ChatPromptTemplate.from_template(prompts.INTENT_AGENT_PROMPT)
    chain = prompt | llm | StrOutputParser()

    try:
        response_json = chain.invoke({"user_query": state['user_query']})
        intent_details = json.loads(response_json)
        logger.info(f"Detected Intent Details: {intent_details}")
        intent = intent_details.get("intent", "Descriptive").lower()
        return {"intent": intent, "intent_details": intent_details}
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Failed to parse intent JSON: {e}. Defaulting to Descriptive.")
        return {"intent": "descriptive", "intent_details": {"intent": "Descriptive"}}

def get_all_column_names(schema_text: str) -> set:
    """
    Extracts all column names from the formatted schema block.

    Args:
        schema_text: The formatted string of table schemas.

    Returns:
        A set of all column names found in the schema.
    """
    return set(re.findall(r"(\w+)\s*\(\w+\)", schema_text))

def text_to_sql_agent(state: AgentState) -> dict:
    """
    Generates and executes a SQL query with anti-hallucination and self-correction.

    Args:
        state: The current state of the workflow.

    Returns:
        A dictionary with the SQL query and its result, or an error.
    """
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

    available_columns = get_all_column_names(schema_context)
    queried_columns = set(re.findall(r"\b(\w+)\b", sql_query))

    for col in queried_columns:
        if col.upper() not in ["SELECT", "FROM", "WHERE", "GROUP", "BY", "ORDER", "LIMIT", "AS", "ON", "JOIN", "INNER", "LEFT", "RIGHT", "OUTER", "SUM", "AVG", "COUNT", "MIN", "MAX", "DATE_TRUNC", "CAST", "YEAR"] and col not in available_columns:
            logger.error(f"Anti-hallucination check failed. Column '{col}' not in schema.")
            return {"error": f"The column '{col}' is not present in the table schema and cannot be used in the query."}

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

def rag_agent(state: AgentState, vsc: VectorSearchClient) -> dict:
    """
    Performs a hybrid RAG search.

    Args:
        state: The current state of the workflow.
        vsc: The initialized Databricks Vector Search client.

    Returns:
        A dictionary with the RAG result or an error message.
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

def mixed_intent_agent(state: AgentState, vsc: VectorSearchClient) -> dict:
    """
    Handles mixed-intent queries with a self-correction loop.

    Args:
        state: The current state of the workflow.
        vsc: The initialized Databricks Vector Search client.

    Returns:
        A dictionary containing both RAG and SQL results, or an error.
    """
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

def voice_summarization_agent(state: AgentState) -> dict:
    """
    Transcribes an audio file and generates a summary.

    Args:
        state: The current state of the workflow.

    Returns:
        A dictionary containing the final response as a Pydantic model.
    """
    logger.info("---VOICE SUMMARIZATION AGENT---")
    user_query = state['user_query']

    match = re.search(r"[\s\"']?([^\s\"'/]+\.(mp3|wav|m4a|flac))[\s\"']?", user_query)
    if not match:
        return {"error": "Could not find a valid audio file name in the query."}

    audio_filename = match.group(1)
    audio_file_path = os.path.join(config.UNSTRUCTURED_DATA_PATH, audio_filename)

    if not os.path.exists(audio_file_path):
        return {"error": f"Audio file '{audio_filename}' not found in the configured data path."}

    try:
        logger.info(f"Transcribing audio file: {audio_file_path}")
        result = whisper_model.transcribe(audio_file_path)
        transcription = result["text"]

        state['messages'].append(AIMessage(content=f"Full transcription:\n{transcription}"))

        logger.info("Transcription complete. Generating summary...")
        prompt = ChatPromptTemplate.from_template(prompts.VOICE_SUMMARY_PROMPT)
        chain = prompt | llm | StrOutputParser()
        summary = chain.invoke({"transcription": transcription})

        response_model = VoiceResponse(
            session_id=state['session_id'],
            user_query=user_query,
            summary=summary,
            transcription=transcription
        )
        return {"final_response": response_model}

    except Exception as e:
        logger.error(f"Error during voice processing: {e}")
        return {"error": f"Error during voice processing: {e}"}

def context_engineer_agent(state: AgentState) -> dict:
    """
    Assembles all available context into a single, structured block of text.

    Args:
        state: The current state of the workflow.

    Returns:
        A dictionary updating the `engineered_context` field in the state.
    """
    logger.info("---CONTEXT ENGINEER AGENT---")
    context_parts = []
    if state.get("messages"):
        history = "\n".join([f"{msg.type}: {msg.content}" for msg in state["messages"]])
        context_parts.append(f"Conversation History:\n{history}")
    if state.get("rag_result"):
        context_parts.append(f"Information from Documents:\n{state['rag_result']}")
    if state.get("sql_result"):
        context_parts.append(f"Database Query Results:\n{state['sql_result']}")
    if not context_parts:
        return {"error": "No context was generated by the previous steps."}
    engineered_context = "\n\n---\n\n".join(context_parts)
    logger.info(f"Engineered Context:\n{engineered_context}")
    return {"engineered_context": engineered_context}

def response_agent(state: AgentState) -> dict:
    """
    Generates the final response using the engineered context.

    Args:
        state: The current state of the workflow.

    Returns:
        A dictionary containing the final response as a JSON string.
    """
    logger.info("---RESPONSE AGENT (REFACTORED)---")
    if isinstance(state.get("final_response"), BaseResponse):
        return {"final_response": state["final_response"].model_dump_json()}

    engineered_context = state.get("engineered_context")
    if not engineered_context:
        return llm_error_agent({"error": "Context engineering failed."})
    prompt = ChatPromptTemplate.from_template(prompts.RESPONSE_AGENT_SIMPLE_PROMPT)
    chain = prompt | llm | StrOutputParser()
    final_answer = chain.invoke({"user_query": state['user_query'], "engineered_context": engineered_context})
    response_model = RAGResponse(
        session_id=state['session_id'],
        user_query=state['user_query'],
        rag_content=final_answer,
        sources=["Synthesized from multiple sources"]
    )
    return {"final_response": response_model.model_dump_json()}

def create_optimized_error_log_table(spark: SparkSession):
    """
    Creates the error log Delta table.

    Args:
        spark: The active Spark session.
    """
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

def llm_error_agent(state: AgentState) -> dict:
    """
    Handles errors by logging them and suggesting a fix.

    Args:
        state: The current state containing the error message.

    Returns:
        A dictionary containing the error response as a JSON string.
    """
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

def save_cache_agent(state: AgentState) -> dict:
    """
    Saves the final response to the cache.

    Args:
        state: The current state containing the final response.

    Returns:
        An empty dictionary.
    """
    logger.info("---SAVE CACHE AGENT---")
    spark = SparkSession.builder.appName("SaveCacheAgent").getOrCreate()
    response_to_cache = state['final_response'] if isinstance(state['final_response'], str) else state['final_response']
    cache_manager.save_to_cache(spark, state['user_query'], response_to_cache, state['session_id'])
    return {}
