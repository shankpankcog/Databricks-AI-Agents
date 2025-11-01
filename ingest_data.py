# ingest_data.py
import os
import hashlib
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

# Import project assets
import config
from logger_config import get_logger

logger = get_logger(__name__)

def get_document_id(file_path: str) -> str:
    """
    Generates a unique and deterministic ID for a document based on its file path.

    Args:
        file_path (str): The absolute or relative path to the document file.

    Returns:
        str: A unique MD5 hash of the file path.
    """
    return hashlib.md5(file_path.encode()).hexdigest()

def create_optimized_delta_table(spark: SparkSession, table_name: str, schema: StructType):
    """
    Creates an optimized Delta table with a specific schema if it doesn't already exist.
    The table is configured with properties that are beneficial for Vector Search indexing.

    Args:
        spark (SparkSession): The active Spark session.
        table_name (str): The full, three-level name of the Delta table to create.
        schema (StructType): The PySpark schema to use for the table.
    """
    if not spark.catalog.tableExists(table_name):
        logger.info(f"Table '{table_name}' does not exist. Creating it with optimized properties.")
        # Construct the CREATE TABLE SQL statement from the schema
        schema_sql = ", ".join([f"{field.name} {field.dataType.simpleString()}" for field in schema.fields])
        spark.sql(f"""
            CREATE TABLE {table_name} ({schema_sql})
            USING DELTA
            TBLPROPERTIES (
                delta.enableChangeDataFeed = true,
                delta.autoOptimize.optimizeWrite = true,
                delta.autoOptimize.autoCompact = true
            )
        """)
    else:
        logger.info(f"Table '{table_name}' already exists. Skipping creation.")

def process_documents(spark: SparkSession, folder_path: str, target_table: str):
    """
    Loads PDF and DOCX documents from a folder, splits them into chunks, and upserts
    them into a target Delta table using a robust schema and SQL MERGE command.

    Args:
        spark (SparkSession): The active Spark session.
        folder_path (str): The path to the folder containing the documents.
        target_table (str): The name of the Delta table to upsert the chunks into.
    """
    if not os.path.exists(folder_path):
        logger.error(f"The folder path '{folder_path}' does not exist.")
        return

    logger.info(f"Processing documents from: {folder_path}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    all_chunks = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.lower().endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            continue

        logger.info(f"Loading and chunking document: {filename}")
        documents = loader.load()
        chunks = text_splitter.split_documents(documents)

        doc_hash = get_document_id(file_path)

        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_hash}_{i}"
            all_chunks.append((chunk_id, file_path, doc_hash, i, chunk.page_content))

    if not all_chunks:
        logger.info("No new documents to process.")
        return

    # Define the explicit schema for the chunks DataFrame
    schema = StructType([
        StructField("chunk_id", StringType(), False),
        StructField("source", StringType(), True),
        StructField("doc_hash", StringType(), True),
        StructField("chunk_index", IntegerType(), True),
        StructField("text_content", StringType(), True)
    ])

    # Ensure the target Delta table exists with the correct schema and properties
    create_optimized_delta_table(spark, target_table, schema)

    chunks_df = spark.createDataFrame(all_chunks, schema=schema)
    logger.info(f"Generated {chunks_df.count()} chunks from the documents.")

    # Use a SQL MERGE for robust upserting
    chunks_df.createOrReplaceTempView("_new_chunks")

    merge_sql = f"""
        MERGE INTO {target_table} t
        USING _new_chunks s
        ON t.chunk_id = s.chunk_id
        WHEN MATCHED THEN UPDATE SET
            t.text_content = s.text_content,
            t.doc_hash = s.doc_hash
        WHEN NOT MATCHED THEN INSERT *
    """

    logger.info("Upserting chunks into Delta table with SQL MERGE...")
    spark.sql(merge_sql)
    logger.info("Upsert complete.")

if __name__ == "__main__":
    spark = SparkSession.builder.appName("DataIngestion").getOrCreate()

    if not os.path.exists(config.UNSTRUCTURED_DATA_PATH):
        logger.warning(f"Creating a sample directory at '{config.UNSTRUCTURED_DATA_PATH}'")
        os.makedirs(config.UNSTRUCTURED_DATA_PATH)
        logger.info("Please add your PDF and DOCX files to this directory and run again.")
    else:
        process_documents(spark, config.UNSTRUCTURED_DATA_PATH, config.DELTA_SYNC_TABLE)

    logger.info("Data ingestion script finished.")
