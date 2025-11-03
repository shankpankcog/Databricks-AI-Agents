# prompts.py

# This file contains the system prompts for each agent in the multi-agent system.

INTENT_AGENT_PROMPT = """
Given the user query, determine the intent. The possible intents are:
'structured': The user is asking a question that can be answered with a SQL query.
'unstructured': The user is asking a question that can be answered by searching documents.
'mixed': The user is asking a question that requires both SQL and document search.
'voice': The user is asking to transcribe or summarize an audio file.

User query: {user_query}
"""

VOICE_SUMMARY_PROMPT = """
You are a helpful assistant. The user has provided the following transcription of an audio file.
Please provide a concise summary of the text.

Transcription:
{transcription}

Summary:
"""

RAG_AGENT_PROMPT = """
Answer the following question based on the provided context from documents and table schemas.

Context from Documents:
{document_context}

Context from Table Schemas:
{schema_context}

Question:
{question}
"""

TEXT_TO_SQL_AGENT_PROMPT = """
Given the user query, generate a SQL query for Databricks Unity Catalog.

Unity Catalog: {catalog}
Schema: {schema}
User query: {user_query}

SQL Query:
"""

MIXED_INTENT_SQL_PROMPT = """
Given the user query and additional context from documents, generate a SQL query for Databricks Unity Catalog.

Additional Context from Documents: {rag_context}
Unity Catalog: {catalog}
Schema: {schema}
User query: {user_query}

SQL Query:
"""

RESPONSE_AGENT_SQL_PROMPT = """
You are a helpful assistant. The user asked: "{user_query}"
A SQL query returned the following: {sql_result}
Provide a clear, business-friendly summary of the result.
"""

RESPONSE_AGENT_MIXED_PROMPT = """
You have results from a document search and a database query.
Synthesize them to answer the user's question.

User Question: {user_query}
Document Search Result: {rag_result}
Database Query Result: {sql_result}

Provide a comprehensive, synthesized answer:
"""

ERROR_AGENT_PROMPT = """
An error occurred: {error}. Please suggest a fix.
"""

SQL_CORRECTION_PROMPT = """
The following SQL query failed to execute. Your task is to fix it.

User's Original Question: "{user_query}"

The Faulty SQL Query:
{faulty_sql}

The Error Message Received:
{error_message}

Please provide only the corrected SQL query, without any additional explanation.
Corrected SQL Query:
"""
