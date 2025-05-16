from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, window, desc
from pyspark.sql.types import StructType, StringType, TimestampType


def kafkasetup():
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
        .appName("TopCountriesFinance") \
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
    
    return parsed


parsed = kafkasetup()

# Filtrar por categoría finance y aplicar ventana
finance_events = parsed.filter(col("category") == "finance")

# Agrupar por ventana de 15 minutos y país
finance_windowed = finance_events \
    .withWatermark("timestamp", "10 minutes") \
    .groupBy(window(col("timestamp"), "15 minutes"), col("location")) \
    .count()

# Ordenar por conteo descendente (Spark mostrará todo, puedes cortar visualmente el top 10)
ordered = finance_windowed.orderBy(desc("count"))


# Show the results in the console
query = ordered.writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("truncate", False) \
    .start()

query.awaitTermination()