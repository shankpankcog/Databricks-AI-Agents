# models.py
from pydantic import BaseModel
from typing import Optional, List

class BaseResponse(BaseModel):
    """A base model for all agent responses."""
    response_type: str
    session_id: str
    user_query: str

class RAGResponse(BaseResponse):
    """Pydantic model for a RAG agent response."""
    response_type: str = "rag"
    rag_content: str
    sources: List[str] # To hold metadata about the source documents

class SQLResponse(BaseResponse):
    """Pydantic model for a Text-to-SQL agent response."""
    response_type: str = "sql"
    summary: str
    sql_query: str
    tabular_result: str # The result as a string-formatted table

class MixedResponse(BaseResponse):
    """Pydantic model for a mixed-intent response."""
    response_type: str = "mixed"
    synthesized_answer: str
    rag_sources: List[str]
    sql_query: str

class ErrorResponse(BaseResponse):
    """Pydantic model for an error response."""
    response_type: str = "error"
    error_message: str
    suggested_fix: Optional[str] = None
