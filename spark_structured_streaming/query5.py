from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, window, desc, count_distinct
from pyspark.sql.types import StructType, StringType, TimestampType

"""
In this query you must detect if a user has suspicious or bot-like behaviour. We will consider
users who click on more than 20 articles in a minute as suspicious
"""

# JSON data format
schema = StructType() \
    .add("user_id", StringType()) \
    .add("article_id", StringType()) \
    .add("timestamp", TimestampType()) \
    .add("category", StringType()) \
    .add("location", StringType()) \
    .add("device_type", StringType()) \
    .add("session_id", StringType())


KAFKA_BROKER = "172.31.29.187:9094"

# Create spark session
spark = SparkSession.builder \
    .appName("SpikeArticles") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BROKER) \
    .option("subscribe", "news_events") \
    .option("startingOffsets", "latest") \
    .load()

# Extract features
parsed = df.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")


result = parsed \
    .groupBy(window(col("timestamp"), "1 minute"), col("user_id").alias("suspicius_users")) \
    .count()

filtered = result.filter(col("count") > 20)

query = filtered.writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination()
