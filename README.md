# Production-Grade Multi-Agent System for Databricks

This project implements a sophisticated, production-ready multi-agent system using LangGraph, designed for a Databricks environment. The system is exposed via a FastAPI server with a built-in chat UI, and it can handle a variety of user queries by routing them to specialized agents.

## Key Features

- **Web UI & API**: Built with FastAPI and LangServe to provide a chat interface and API endpoints.
- **Router & Intent Agents**: Intelligently route user queries to the appropriate specialized agent.
- **Voice Summarization Agent**: Transcribes audio files using OpenAI's Whisper and provides a summary.
- **Hybrid RAG Agent**: Synthesizes information from both unstructured documents (via Vector Search) and structured table schemas from Unity Catalog.
- **Self-Correcting Text-to-SQL Agent**: Converts natural language to SQL and automatically attempts to fix failed queries.
- **Structured Responses**: Uses Pydantic models to return predictable, structured JSON responses.
- **Query Caching & Session Management**: Caches responses and manages conversational history for follow-up questions.
- **Structured Logging & Guardrails**: Provides clear, monitorable logs and content safety checks for all interactions.
- **Optimized Delta Tables**: Creates all Delta tables with production-ready properties for performance and efficiency.

## Project Structure

- `server.py`: The main entry point for the application. Runs a FastAPI server with LangServe.
- `multi_agent_system.py`: A library containing all the agent definitions and the LangGraph workflow.
- `ingest_data.py`: A script to process and load unstructured documents into a Delta table for indexing.
- `setup_vector_search.py`: A script to create and configure the Databricks Vector Search endpoint and index.
- `cache_manager.py`: A module to handle the logic for the query cache.
- `logger_config.py`: A module to provide standardized, structured logging.
- `models.py`: Defines the Pydantic models for all structured agent responses.
- `guardrails.py`: A module for content validation and filtering.
- `config.py`: The central configuration file for all Databricks settings.
- `prompts.py`: A file to store and manage all system prompts for the agents.
- `requirements.txt`: A list of the required Python libraries.
- `README.md`: This file.

## Setup & Execution

1.  **Clone the Repository and Install Dependencies:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    pip install -r requirements.txt
    ```

2.  **Configure the System:**
    Open `config.py` and update the placeholders with your specific Databricks information.

3.  **Ingest Unstructured Data:**
    -   Place your PDF, DOCX, and audio files (e.g., `.mp3`, `.wav`) into the folder specified by `UNSTRUCTURED_DATA_PATH` in `config.py`.
    -   Run the ingestion script to process your documents and load them into the source Delta table.
    ```bash
    python ingest_data.py
    ```

4.  **Create the Vector Search Index:**
    Run the setup script to create the Vector Search endpoint and a continuous Delta Sync index.
    ```bash
    python setup_vector_search.py
    ```

5.  **Run the Web Server:**
    Execute the `server.py` script to start the FastAPI server.
    ```bash
    uvicorn server:app --host 0.0.0.0 --port 8000 --reload
    ```

6.  **Access the UI:**
    Open your web browser and navigate to `http://localhost:8000/agent/playground/`. You can now interact with the multi-agent system through the chat interface.

## How to Use the Voice Agent

To use the voice summarization feature, ask a question that includes the name of an audio file located in your `UNSTRUCTURED_DATA_PATH`. For example:

`"Can you transcribe and summarize the meeting_notes.mp3 file?"`

The system will automatically find the file, process it, and return a structured response with the summary and the full transcription.
