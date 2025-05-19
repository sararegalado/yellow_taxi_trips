from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType


def create_spark_session(app_name="NYC Taxi Analytics"):
    spark = SparkSession.builder \
        .appName(app_name) \
        .getOrCreate()
    return spark

def load_data(spark, path):
    print(f"Loading data from: {path}")
    return spark.read.parquet(path)


# Revenue per hour by pickup zone (top20)
def revenue_per_hour_by_zone(df, output_path):
    result = df.groupBy("hour", "PU_Zone") \
               .agg(F.round(F.sum("total_amount"), 2).alias("total_revenue")) \
               .orderBy(F.desc("total_revenue")) \
               .limit(20)

    result.write.mode("overwrite").parquet(output_path)

# Most frequent pickup/dropoff pairs (top20)
def most_frequent_pickup_dropoff_pairs(df, output_path):
    result = df.groupBy("PU_Zone", "DO_Zone") \
               .count() \
               .orderBy(F.desc("count")) \
               .limit(20)

    result.write.mode("overwrite").parquet(output_path)

# Average speed per trip (top20)
def average_speed(df, output_path):
    df = df.withColumn("trip_duration_hours", 
                       (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")) / 3600.0)

    # Trips of at least 3 minutes to filter outliers
    df_valid = df.filter((F.col("trip_duration_hours") > 0.05) & (F.col("trip_distance") > 0))

    df_valid = df_valid.withColumn("avg_speed_mph",
                                   F.col("trip_distance") / F.col("trip_duration_hours"))

    result = df_valid.select("PU_Zone", "DO_Zone", "avg_speed_mph") \
                .filter(F.col("avg_speed_mph") < 120).orderBy(F.desc("avg_speed_mph")) \
                .limit(20)

    result.write.mode("overwrite").parquet(output_path)

# Average tip by hour (top20)
def average_tip_by_hour(df, output_path):
    result = df.groupBy("hour") \
               .agg(F.avg("tip_amount").alias("average_tip")) \
               .orderBy("average_tip", ascending=False) \
               .limit(20)

    result.write.mode("overwrite").parquet(output_path)

# Number of trips by day of week and hour (top20)
def trips_by_day_hour(df, output_path):
    df = df.withColumn("day_name", F.when(df["day_of_week"] == 1, "Monday")
                                     .when(df["day_of_week"] == 2, "Tuesday")
                                     .when(df["day_of_week"] == 3, "Wednesday")
                                     .when(df["day_of_week"] == 4, "Thursday")
                                     .when(df["day_of_week"] == 5, "Friday")
                                     .when(df["day_of_week"] == 6, "Saturday")
                                     .when(df["day_of_week"] == 7, "Sunday")
                                     .otherwise("Unknown"))

    result = df.groupBy("day_name", "hour") \
               .count() \
               .orderBy("count", ascending=False) \
               .limit(20)

    result.write.mode("overwrite").parquet(output_path)



if __name__ == "__main__":
    spark = create_spark_session()

    # Input and output paths in HDFS
    input_path = "/NY_taxi_trips/data/processed.parquet"
    output_base = "/NY_taxi_trips/analytical_queries/output/"

    df = load_data(spark, input_path)


    # Run each analytical function
    revenue_per_hour_by_zone(df, output_base + "revenue_per_hour_by_zone")
    most_frequent_pickup_dropoff_pairs(df, output_base + "most_frequent_pickup_dropoff_pairs")
    average_speed(df, output_base + "average_speed")
    average_tip_by_hour(df, output_base + "average_tip_by_hour")
    trips_by_day_hour(df, output_base + "trips_by_day_hour")

    print("Analytics completed.")
    spark.stop()
