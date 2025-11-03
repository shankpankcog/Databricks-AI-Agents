# multi_agent_system.py
import os
import datetime
from typing import TypedDict, Annotated, Sequence, Union
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
    Represents the state of the agent workflow. This TypedDict is used by LangGraph
    to manage the flow of data between different nodes in the graph.

    Attributes:
        messages (Sequence[BaseMessage]): The history of messages in the conversation.
        user_query (str): The initial query from the user for the current turn.
        session_id (str): A unique identifier for the entire conversation session.
        intent (str): The determined intent of the user's query (e.g., 'structured').
        sql_query (str): The generated SQL query, if applicable.
        sql_result (str): The stringified result of the executed SQL query.
        rag_result (str): The result from the RAG agent.
        error (str): Any error message that occurs during the workflow.
        final_response (Union[BaseResponse, str]): The final, user-facing response, which
            can be a Pydantic model (as a JSON string) or a plain string from the cache.
        from_cache (bool): A flag to indicate if the response was retrieved from the cache.
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

# --- LLM and VSC Initialization ---
llm = ChatOpenAI(
    model=config.ENDPOINT_NAME,
    api_key=config.DATABRICKS_TOKEN,
    base_url=f"{config.DATABRICKS_HOST}/serving-endpoints"
)
vsc = VectorSearchClient()

# --- Pre-load Models ---
logger.info("Pre-loading Whisper model...")
whisper_model = whisper.load_model("base")
logger.info("Whisper model loaded.")

# --- Agent Definitions ---

def get_table_schemas(spark: SparkSession, catalog: str, schema: str) -> str:
    """
    Dynamically fetches schema information for all tables in a given Unity Catalog schema.

    Args:
        spark (SparkSession): The active Spark session.
        catalog (str): The name of the Unity Catalog.
        schema (str): The name of the schema within the catalog.

    Returns:
        str: A formatted string describing the schemas of all tables, or an error message.
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
    Checks if a response for the user's query exists in the cache Delta table.
    This is the entry point of the graph.

    Args:
        state (AgentState): The current state of the workflow.

    Returns:
        dict: A dictionary with the cached response if found, or a flag indicating a cache miss.
    """
    logger.info("---CACHE AGENT---")
    spark = SparkSession.builder.appName("CacheAgent").getOrCreate()
    cached_response = cache_manager.check_cache(spark, state['user_query'])
    if cached_response:
        return {"final_response": cached_response, "from_cache": True}
    return {"from_cache": False}

def router_agent(state: AgentState) -> dict:
    """
    Initializes the conversation history for a new query that was not found in the cache.

    Args:
        state (AgentState): The current state of the workflow.

    Returns:
        dict: A dictionary indicating the next step is to clarify intent.
    """
    logger.info("---ROUTER AGENT---")
    state['messages'] = [HumanMessage(content=state['user_query'])]
    return {"intent": "clarify_intent"}

def intent_agent(state: AgentState) -> dict:
    """
    Determines the user's intent (structured, unstructured, mixed, or voice) using the LLM.

    Args:
        state (AgentState): The current state of the workflow.

    Returns:
        dict: A dictionary containing the determined intent.
    """
    logger.info("---INTENT AGENT---")
    prompt = ChatPromptTemplate.from_template(prompts.INTENT_AGENT_PROMPT)
    chain = prompt | llm | StrOutputParser()
    intent = chain.invoke({"user_query": state['user_query']}).strip().lower()
    logger.info(f"Detected intent: {intent}")

    if "structured" in intent: return {"intent": "structured"}
    if "unstructured" in intent: return {"intent": "unstructured"}
    if "mixed" in intent: return {"intent": "mixed"}
    if "voice" in intent: return {"intent": "voice"}
    return {"intent": "unstructured"} # Default fallback

def rag_agent(state: AgentState, vsc: VectorSearchClient) -> dict:
    """
    Performs a hybrid RAG search by combining context from both unstructured documents
    (via Vector Search) and structured table schemas from Unity Catalog.

    Args:
        state (AgentState): The current state of the workflow.
        vsc (VectorSearchClient): The initialized Databricks Vector Search client.

    Returns:
        dict: A dictionary with the synthesized RAG result, or an error message.
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
    """
    Generates and executes a SQL query. Includes a self-correction loop to attempt
    to fix and retry failed queries.

    Args:
        state (AgentState): The current state of the workflow.

    Returns:
        dict: A dictionary with the SQL query and its result, or an error if it fails after retries.
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
    """
    Handles mixed-intent queries by first performing a RAG search to gather context,
    then using that context to generate a more informed SQL query. Includes a
    self-correction loop for the SQL execution part.

    Args:
        state (AgentState): The current state of the workflow.
        vsc (VectorSearchClient): The initialized Databricks Vector Search client.

    Returns:
        dict: A dictionary containing both RAG and SQL results, or an error.
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
    Transcribes an audio file using Whisper and generates a summary with an LLM.

    Args:
        state (AgentState): The current state of the workflow.

    Returns:
        dict: A dictionary containing the final response as a Pydantic model.
    """
    logger.info("---VOICE SUMMARIZATION AGENT---")
    user_query = state['user_query']

    # Use regex to find just the filename, not the full path
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
    Assembles all available context into a clear, structured format.
    This node runs before the final response generation to ensure the LLM
    has all the necessary information. This is a placeholder for more complex
    context manipulation in the future.

    Args:
        state (AgentState): The current state of the workflow.

    Returns:
        dict: An empty dictionary as no state change is needed for this version.
    """
    logger.info("---CONTEXT ENGINEER AGENT---")
    return {}

def response_agent(state: AgentState) -> dict:
    """
    Formats the final response into a structured Pydantic model. It synthesizes
    results for mixed-intent queries or formats single-source results.

    Args:
        state (AgentState): The current state containing results from previous agents.

    Returns:
        dict: A dictionary containing the final response as a JSON string
              serialized from a Pydantic model.
    """
    logger.info("---RESPONSE AGENT---")
    if isinstance(state.get("final_response"), BaseResponse):
        return {"final_response": state["final_response"].model_dump_json()}

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

def create_optimized_error_log_table(spark: SparkSession):
    """
    Creates the error log Delta table with optimized properties if it does not exist.

    Args:
        spark (SparkSession): The active Spark session.
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
    Handles errors by logging them to a Delta table and using the LLM to suggest a fix.

    Args:
        state (AgentState): The current state containing the error message.

    Returns:
        dict: A dictionary containing the error response as a JSON string from a Pydantic model.
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
        state (AgentState): The current state containing the final response.

    Returns:
        dict: An empty dictionary.
    """
    logger.info("---SAVE CACHE AGENT---")
    spark = SparkSession.builder.appName("SaveCacheAgent").getOrCreate()
    response_to_cache = state['final_response'] if isinstance(state['final_response'], str) else state['final_response']
    cache_manager.save_to_cache(spark, state['user_query'], response_to_cache, state['session_id'])
    return {}
