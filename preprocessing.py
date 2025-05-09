from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

def create_spark_session(app_name="NYC Taxi Analysis"):
    spark = SparkSession.builder.appName(app_name).getOrCreate()
    return spark

def load_data(spark, path, file_format):
    if file_format == "csv":
        df = spark.read.csv(path, header=True, inferSchema=True)
    else:
        df = spark.read.parquet(path)
    return df

#Preprocess taxi trip data: extract time-based features, trip duration and speed.
def preprocess_data(taxi_trips_df):

    taxi_trips_df = taxi_trips_df.filter(F.col("tpep_dropoff_datetime") > F.col("tpep_pickup_datetime")) \
        .withColumn("hour", F.hour("tpep_pickup_datetime")) \
        .withColumn("day_of_week", F.dayofweek("tpep_pickup_datetime")) \
        .withColumn("trip_duration_seconds",
               (F.unix_timestamp("tpep_dropoff_datetime") - F.unix_timestamp("tpep_pickup_datetime")).cast(DoubleType())) \
        .withColumn("speed_mph",
               F.when(F.col("trip_duration_seconds") > 0, F.col("trip_distance") / (F.col("trip_duration_seconds") / 3600)).otherwise(0.0)) \
        .withColumn("extras", F.expr("extra + mta_tax + tolls_amount + improvement_surcharge + congestion_surcharge + airport_fee"))
    

    taxi_trips_df = taxi_trips_df.drop("RatecodeID", "store_and_fwd_flag", "extra", "mta_tax", "tolls_amount", "improvement_surcharge", "congestion_surcharge", "airport_fee")
    
    return taxi_trips_df



# Join taxi trip data with locations data
def join_with_zones(df, zones_df):
    df = df.join(
        zones_df
            .select("LocationID", "Borough", "Zone")  # Solo las columnas necesarias
            .withColumnRenamed("LocationID", "PULocationID")
            .withColumnRenamed("Borough", "PU_Borough")
            .withColumnRenamed("Zone", "PU_Zone"),
        on="PULocationID",
        how="left"
    )

    df = df.join(
        zones_df
            .select("LocationID", "Borough", "Zone")
            .withColumnRenamed("LocationID", "DOLocationID")
            .withColumnRenamed("Borough", "DO_Borough")
            .withColumnRenamed("Zone", "DO_Zone"),
        on="DOLocationID",
        how="left"
    )

    return df



if __name__ == "__main__":
    spark = create_spark_session()

    # Example: Load CSV trip data
    trips_df = load_data(spark, "data/yellow_tripdata_2024-02.parquet", file_format="parquet")
    trips_df = preprocess_data(trips_df)

    # Load Parquet taxi zone lookup
    zones_df = load_data(spark, "data/taxi_zone_lookup.csv", file_format="csv")

    # Join with zones
    df = join_with_zones(trips_df, zones_df)
    df.write.mode("overwrite").csv("data/full_tripdata.parquet")



    spark.stop()



