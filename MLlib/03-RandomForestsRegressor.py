import os
import pyspark
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession
from yellow_taxi_trips.preprocessing import preprocess_data


spark = SparkSession.builder \
    .appName("NYC Taxi RF Regressor") \
    .getOrCreate()

# Load the data
df = spark.read.parquet("../data/yellow_tripdata_2024-02.parquet")

# Preprocess the data
df = preprocess_data(df)

# Split the data
(train_df, test_df) = df.randomSplit([0.8, 0.2], seed=42)

# Assemble features
feature_columns = ["trip_distance", "hour", "day_of_week", "extras", "fare_amount", "tip_amount"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="skip")
train_df = assembler.transform(train_df)
test_df = assembler.transform(test_df)

# Train the Random Forest Regressor
rf = RandomForestRegressor(numTrees=15, maxDepth=5, seed=42, featuresCol="features", labelCol="total_amount")
model = rf.fit(train_df)

# Evaluate the model
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="total_amount", metricName="rmse")
rmse = evaluator.evaluate(model.transform(test_df))

print(f"RMSE: {rmse.__round__(2)}")

# Save the model
model_path = "models/yellow_taxi_rf_model"
if not os.path.exists(model_path):
    model.save("models/yellow_taxi_rf_model", )
    print(f"Model saved to {model_path}.")
else:
    print(f"Model already exists at {model_path}.")



