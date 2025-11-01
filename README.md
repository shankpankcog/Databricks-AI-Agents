# Production-Grade Multi-Agent System for Databricks

This project implements a sophisticated, production-ready multi-agent system using LangGraph, designed for a Databricks environment. The system can handle a variety of user queries by routing them to specialized agents, with support for hybrid search, caching, session management, structured logging, and Pydantic-based responses.

## Key Features

- **Router & Intent Agents**: Intelligently route user queries to the appropriate specialized agent.
- **Hybrid RAG Agent**: Synthesizes information from both unstructured documents (via Vector Search) and structured table schemas from Unity Catalog.
- **Self-Correcting Text-to-SQL Agent**: Converts natural language to SQL and automatically attempts to fix failed queries.
- **Structured Responses**: Uses Pydantic models to return predictable, structured JSON responses.
- **Query Caching**: Caches responses in an optimized Delta table for faster retrieval of repeated questions.
- **Session Management**: Manages conversational history for follow-up questions.
- **Structured Logging**: Provides clear, monitorable logs for each step of the agentic workflow.
- **Optimized Delta Tables**: Creates all Delta tables with production-ready properties for performance and efficiency.

## Project Structure

- `multi_agent_system.py`: The main application file with the LangGraph workflow.
- `ingest_data.py`: A script to process and load unstructured documents into a Delta table for indexing.
- `setup_vector_search.py`: A script to create and configure the Databricks Vector Search endpoint and index.
- `cache_manager.py`: A module to handle the logic for the query cache.
- `logger_config.py`: A module to provide standardized, structured logging across the application.
- `models.py`: Defines the Pydantic models for all structured agent responses.
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
    -   Place your PDF and DOCX files into the folder specified by `UNSTRUCTURED_DATA_PATH` in `config.py`.
    -   Run the ingestion script to process your documents and load them into the source Delta table. This will create an optimized Delta table if it doesn't exist.
    ```bash
    python ingest_data.py
    ```

4.  **Create the Vector Search Index:**
    Run the setup script to create the Vector Search endpoint and a continuous Delta Sync index.
    ```bash
    python setup_vector_search.py
    ```

5.  **Run the Multi-Agent System:**
    Execute the main script in your Databricks environment. The output will be structured JSON logs.
    ```bash
    python multi_agent_system.py
    ```

The system will automatically create the necessary cache and error log tables with optimized properties on its first run.
