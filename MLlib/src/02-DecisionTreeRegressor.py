from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np
from pyspark.sql.types import DoubleType, IntegerType, LongType

# Spark Session
spark = SparkSession.builder \
    .appName("Yellow Taxi Decision Tree Regressor") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Load the data
df = spark.read.parquet("data/processed.parquet")

# Relevant columns
selected_cols = ["trip_distance", "passenger_count", "PULocationID", "DOLocationID", 
                 "VendorID", "total_amount", "payment_type","hour", "day_of_week"]
relevant_df = df.select(*selected_cols)

# Train and test split
train_data, test_data = relevant_df.randomSplit([0.8, 0.2], seed=42)

# Cleaning data
train_data = train_data.na.drop(subset=["trip_distance", "passenger_count", "PULocationID", "DOLocationID", 
                                            "VendorID", "total_amount", "payment_type","hour", "day_of_week"
])

test_data = test_data.na.drop(subset=["trip_distance", "passenger_count", "PULocationID", "DOLocationID", 
                                        "VendorID", "total_amount", "payment_type","hour", "day_of_week"
])

amount_encoder = OneHotEncoder(
    inputCols=["PULocationID", "DOLocationID", "VendorID", "payment_type"],
    outputCols=["PULocationID_ohe", "DOLocationID_ohe", "VendorID_ohe", "payment_type_ohe"]
)

# Building the pipeline
feature_cols = [
    "trip_distance", "passenger_count", "hour", "day_of_week",
    "PULocationID_ohe", "DOLocationID_ohe", "VendorID_ohe", "payment_type_ohe"
]

# Assemble features
amount_assembler = VectorAssembler(
    inputCols= feature_cols,
    outputCol="features"
)

total_amount_dt = DecisionTreeRegressor(labelCol="total_amount",
                                        featuresCol="features",
                                        maxDepth=5)

pipeline_total = Pipeline(stages=[amount_encoder, amount_assembler, total_amount_dt])
model_total = pipeline_total.fit(train_data)


# Evaluate the model
predictions = model_total.transform(test_data)

rmse = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="rmse").evaluate(predictions)
mae = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="mae").evaluate(predictions)
r2 = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="r2").evaluate(predictions)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"R2: {r2}")

