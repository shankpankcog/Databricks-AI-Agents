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
    """Generates a unique ID for a document based on its file path."""
    return hashlib.md5(file_path.encode()).hexdigest()

def create_optimized_delta_table(spark: SparkSession, table_name: str, schema: StructType):
    """Creates an optimized Delta table if it doesn't already exist."""
    if not spark.catalog.tableExists(table_name):
        logger.info(f"Table '{table_name}' does not exist. Creating it with optimized properties.")
        spark.sql(f"""
            CREATE TABLE {table_name} (
                {', '.join([f'{field.name} {field.dataType.simpleString()}' for field in schema.fields])}
            ) USING DELTA
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
    Loads documents from a folder, splits them into chunks, and upserts them into a Delta table
    using a robust schema and SQL MERGE command.
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

    # Define the explicit schema for our chunks DataFrame
    schema = StructType([
        StructField("chunk_id", StringType(), False),
        StructField("source", StringType(), True),
        StructField("doc_hash", StringType(), True),
        StructField("chunk_index", IntegerType(), True),
        StructField("text_content", StringType(), True)
    ])

    # Create the Delta table with this schema if it doesn't exist
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
