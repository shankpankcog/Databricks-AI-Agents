# cache_manager.py
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
import hashlib

# Import project assets
import config
from logger_config import get_logger

logger = get_logger(__name__)

def create_optimized_cache_table(spark: SparkSession):
    """Creates the cache table with an explicit schema and optimized properties."""
    schema = StructType([
        StructField("query_hash", StringType(), False),
        StructField("user_query", StringType(), True),
        StructField("final_response", StringType(), True),
        StructField("session_id", StringType(), True),
        StructField("timestamp", TimestampType(), True)
    ])

    table_name = config.QUERY_CACHE_TABLE
    if not spark.catalog.tableExists(table_name):
        logger.info(f"Cache table '{table_name}' not found. Creating it.")
        spark.sql(f"""
            CREATE TABLE {table_name} (
                query_hash STRING,
                user_query STRING,
                final_response STRING,
                session_id STRING,
                timestamp TIMESTAMP
            ) USING DELTA
            TBLPROPERTIES (
                delta.autoOptimize.optimizeWrite = true,
                delta.autoOptimize.autoCompact = true
            )
        """)
    else:
        logger.info(f"Cache table '{table_name}' already exists.")

def get_query_hash(query: str) -> str:
    """Creates a unique hash for a query string."""
    return hashlib.md5(query.encode()).hexdigest()

def check_cache(spark: SparkSession, query: str):
    """Checks the cache for a response for a given query."""
    create_optimized_cache_table(spark) # Ensure table exists before checking

    query_hash = get_query_hash(query)
    try:
        cached_result = spark.read.table(config.QUERY_CACHE_TABLE) \
            .filter(col("query_hash") == query_hash) \
            .select("final_response") \
            .first()

        if cached_result:
            logger.info(f"CACHE HIT for query: '{query}'")
            return cached_result['final_response']
    except Exception as e:
        logger.error(f"Error checking cache: {e}")

    logger.info(f"CACHE MISS for query: '{query}'")
    return None

def save_to_cache(spark: SparkSession, query: str, response: str, session_id: str):
    """Saves a new query and its response to the cache."""
    create_optimized_cache_table(spark) # Ensure table exists before saving

    query_hash = get_query_hash(query)

    from pyspark.sql.functions import current_timestamp
    cache_df = spark.createDataFrame([
        (query_hash, query, response, session_id)
    ], ["query_hash", "user_query", "final_response", "session_id"])
    cache_df = cache_df.withColumn("timestamp", current_timestamp())

    cache_df.createOrReplaceTempView("_new_cache_entry")

    merge_sql = f"""
        MERGE INTO {config.QUERY_CACHE_TABLE} t
        USING _new_cache_entry s
        ON t.query_hash = s.query_hash
        WHEN MATCHED THEN UPDATE SET
            t.final_response = s.final_response,
            t.session_id = s.session_id,
            t.timestamp = s.timestamp
        WHEN NOT MATCHED THEN INSERT *
    """

    logger.info("Saving response to cache...")
    spark.sql(merge_sql)
    logger.info("Cache save complete.")
