from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def create_spark_session(app_name="NYC Taxi Analytics"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

def load_data(spark, path, file_format="parquet"):
    return spark.read.parquet(path)

# Revenue per hour by pickup zone
def revenue_per_hour_by_zone(df, output_path):
    result = df.groupBy("hour", "PU_Zone") \
               .agg(F.sum("total_amount").alias("total_revenue")) \
               .orderBy(F.desc("total_revenue"))

    result.write.mode("overwrite").csv(output_path, header=True)

# Most frequent pickup/dropoff pairs
def most_frequent_pickup_dropoff_pairs(df, output_path):
    result = df.groupBy("PU_Zone", "DO_Zone") \
               .count() \
               .orderBy(F.desc("count"))

    result.write.mode("overwrite").csv(output_path, header=True)

# Average speed per trip
def average_speed(df, output_path):
    result = df.withColumn("trip_duration_hours",
                           (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 3600) \
               .withColumn("avg_speed_mph",
                           F.col("trip_distance") / F.col("trip_duration_hours")) \
               .select("PU_Zone", "DO_Zone", "avg_speed_mph") \
               .orderBy(F.desc("avg_speed_mph"))

    result.write.mode("overwrite").csv(output_path, header=True)

# Average tip by hour
def average_tip_by_hour(df, output_path):
    result = df.groupBy("hour") \
               .agg(F.avg("tip_amount").alias("average_tip")) \
               .orderBy("hour")

    result.write.mode("overwrite").csv(output_path, header=True)

# Number of trips by day of week and hour
def trips_by_day_hour(df, output_path):
    result = df.groupBy("day_of_week", "hour") \
               .count() \
               .orderBy("day_of_week", "hour")

    result.write.mode("overwrite").csv(output_path, header=True)



if __name__ == "__main__":
    spark = create_spark_session()

    # Load preprocessed data
    data_path = "data/full_tripdata.parquet"
    df = load_data(spark, data_path)

    # Run analytical queries
    revenue_per_hour_by_zone(df, "output/revenue_per_hour_by_zone")
    most_frequent_pickup_dropoff_pairs(df, "./output/most_frequent_pickup_dropoff_pairs")
    average_speed(df, "output/average_speed")
    average_tip_by_hour(df, "output/average_tip_by_hour")
    trips_by_day_hour(df, "output/trips_by_day_hour")

    spark.stop()
