# setup_vector_search.py
import os
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
import time

# Import project assets
import config
from logger_config import get_logger

logger = get_logger(__name__)

def wait_for_endpoint_to_be_ready(vsc: VectorSearchClient, endpoint_name: str):
    """
    Waits for a Databricks Vector Search endpoint to become ONLINE.

    This function polls the endpoint's status and waits until it is fully provisioned.
    It includes a timeout to prevent indefinite waiting.

    Args:
        vsc (VectorSearchClient): The initialized Vector Search client.
        endpoint_name (str): The name of the endpoint to check.

    Raises:
        Exception: If the endpoint enters a non-recoverable state or times out.
    """
    logger.info(f"Waiting for endpoint '{endpoint_name}' to become ready...")
    for i in range(180): # Wait for up to 30 minutes (180 * 10 seconds)
        endpoint = vsc.get_endpoint(name=endpoint_name)
        status = endpoint.get("endpoint_status", {}).get("state", "UNKNOWN")
        if status == "ONLINE":
            logger.info(f"Endpoint '{endpoint_name}' is online.")
            return
        elif status == "PROVISIONING":
            logger.info(f"Endpoint is still provisioning... (Status: {status})")
            time.sleep(10)
        else:
            raise Exception(f"Endpoint entered a non-recoverable state: {status}")
    raise Exception(f"Endpoint '{endpoint_name}' did not become ready in time.")

def setup_vector_search_index():
    """
    Sets up a Databricks Vector Search endpoint and a continuous Delta Sync index.

    This function is idempotent. It will:
    1. Create the Vector Search endpoint if it does not already exist.
    2. Wait for the endpoint to be online.
    3. Create the continuous Delta Sync index if it does not already exist.
       If the index does exist, it will trigger a sync to process any updates.
    """
    vsc = VectorSearchClient()

    # --- Step 1: Create a Vector Search Endpoint ---
    try:
        vsc.get_endpoint(name=config.VECTOR_SEARCH_ENDPOINT_NAME)
        logger.info(f"Endpoint '{config.VECTOR_SEARCH_ENDPOINT_NAME}' already exists.")
    except Exception as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e):
            logger.info(f"Endpoint '{config.VECTOR_SEARCH_ENDPOINT_NAME}' not found. Creating...")
            vsc.create_endpoint(name=config.VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")
            logger.info("Endpoint created. Waiting for it to be ready...")
        else:
            raise e

    wait_for_endpoint_to_be_ready(vsc, config.VECTOR_SEARCH_ENDPOINT_NAME)

    # --- Step 2: Create or Sync the Vector Search Index ---
    try:
        vsc.create_delta_sync_index(
            endpoint_name=config.VECTOR_SEARCH_ENDPOINT_NAME,
            index_name=config.VECTOR_SEARCH_INDEX_NAME,
            source_table_name=config.DELTA_SYNC_TABLE,
            pipeline_type="CONTINUOUS",
            primary_key="chunk_id", # This must match the primary key in your source Delta table
            embedding_source_column="text_content", # The column containing the text to embed
            embedding_model_endpoint_name="databricks-bge-large-en" # A recommended embedding model
        )
        logger.info(f"Successfully created index '{config.VECTOR_SEARCH_INDEX_NAME}'.")
    except Exception as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            logger.info(f"Index '{config.VECTOR_SEARCH_INDEX_NAME}' already exists. Attempting to sync.")
            vsc.get_index(endpoint_name=config.VECTOR_SEARCH_ENDPOINT_NAME, index_name=config.VECTOR_SEARCH_INDEX_NAME).sync()
        else:
            logger.error(f"An error occurred while creating/updating the index: {e}")

if __name__ == "__main__":
    logger.info("Setting up Databricks Vector Search Index...")
    # This script assumes that the source Delta table specified in config.py
    # has already been created and populated by running the ingest_data.py script.
    setup_vector_search_index()
    logger.info("Setup complete.")
