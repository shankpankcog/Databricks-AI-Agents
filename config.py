# config.py

# Databricks Configuration
DATABRICKS_HOST = "https://your-databricks-instance.cloud.databricks.com"
DATABRICKS_TOKEN = "YOUR_DATABRICKS_API_TOKEN"
VECTOR_SEARCH_ENDPOINT_NAME = "YOUR_VECTOR_SEARCH_ENDPOINT"

# Language Model Configuration
LLM_PROVIDER = "anthropic"  # or "openai", "azure_openai", etc.
# Example for Anthropic Claude Sonnet on Databricks
ENDPOINT_NAME = "claude-3-sonnet-20240229"

# Vector Search Configuration
VECTOR_SEARCH_INDEX_NAME = "your_vector_search_index"
DELTA_SYNC_TABLE = "your_delta_sync_table"

# Logging Configuration
ERROR_LOG_TABLE = "error_log_delta_table"
QUERY_CACHE_TABLE = "query_cache_delta_table"

# Data Sources
UNITY_CATALOG_NAME = "your_unity_catalog"
UNITY_CATALOG_SCHEMA_NAME = "your_schema"

# File Paths for local testing (optional)
CSV_FILE_PATH = "path/to/your/data.csv"
EXCEL_FILE_PATH = "path/to/your/data.xlsx"
UNSTRUCTURED_DATA_PATH = "path/to/your/unstructured/data" # PDF/DOCX folder
