from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

def create_spark_session(app_name="NYC Taxi Preprocessing"):
    spark = SparkSession.builder \
        .appName(app_name).getOrCreate()
    return spark

def load_data(spark, path, file_format):
    print(f"Loading data from {path}")
    if file_format == "csv":
        df = spark.read.csv(path, header=True, inferSchema=True)
    else:
        df = spark.read.parquet(path)
    return df

#Preprocess taxi trip data: extract time-based features, trip duration and speed.
def preprocess_data(taxi_trips_df):
    print("Preprocessing taxi trips data...")

    taxi_trips_df = taxi_trips_df.filter(F.col("tpep_dropoff_datetime") > F.col("tpep_pickup_datetime")) \
        .withColumn("hour", F.hour("tpep_pickup_datetime")) \
        .withColumn("day_of_week", F.dayofweek("tpep_pickup_datetime")) \
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

    # HDFS paths
    trips_path = "hdfs:///user/ec2-user/data/yellow_tripdata_2024-02.parquet"
    zones_path = "hdfs:///user/ec2-user/data/taxi_zone_lookup.csv"
    output_path = "hdfs:///user/ec2-user/data/processed.parquet"


    trips_df = load_data(spark, trips_path, file_format="parquet")
    zones_df = load_data(spark, zones_path, file_format="csv")

    # Preprocess and join
    trips_df = preprocess_data(trips_df)
    final_df = join_with_zones(trips_df, zones_df)

    # Write output
    print(f"Writing result to: {output_path}")
    final_df.write.mode("overwrite").parquet(output_path)


    print("Preprocessing complete.")
    spark.stop()



