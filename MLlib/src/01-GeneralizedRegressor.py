from pyspark.sql.functions import col, unix_timestamp
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.feature import OneHotEncoder
from pyspark.sql.types import DoubleType, IntegerType, LongType

# Spark Session
spark = SparkSession.builder \
    .appName("Yellow Taxi Generalized Regressor") \
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

encoder = OneHotEncoder(
    inputCols=["PULocationID", "DOLocationID", "VendorID", "payment_type"],
    outputCols=["PULocationID_ohe", "DOLocationID_ohe", "VendorID_ohe", "payment_type_ohe"]
)

# Building the pipeline
feature_cols = [
    "trip_distance", "passenger_count", "hour", "day_of_week",
    "PULocationID_ohe", "DOLocationID_ohe", "VendorID_ohe", "payment_type_ohe"
]
# Assemble features
assembler = VectorAssembler(
    inputCols= feature_cols,
    outputCol="features"
)

# Model creation and training
glr = GeneralizedLinearRegression(
    labelCol="total_amount", 
    featuresCol="features", 
    maxIter=10, 
    regParam=0.1,
    family="gaussian", 
    link="identity"
)

pipeline = Pipeline(stages= [encoder, assembler, glr])
model = pipeline.fit(train_data)

predictions = model.transform(test_data)

# Evaluators and metrics
evaluator_rmse = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="rmse")
evaluator_mae = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="mae")
evaluator_r2 = RegressionEvaluator(labelCol="total_amount", predictionCol="prediction", metricName="r2")

rmse = evaluator_rmse.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

print("====Model Evaluation Metrics====")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")
print("=================================")

# Ploting the results

# Convertir a pandas (muestra pequeña)
sample_pd = predictions.select("total_amount", "prediction").dropna().sample(False, 0.01, seed=42).toPandas()

# Gráfico
plt.figure(figsize=(8, 6))
sns.scatterplot(x="total_amount", y="prediction", data=sample_pd, alpha=0.4)
plt.plot([0, sample_pd["total_amount"].max()], [0, sample_pd["total_amount"].max()], color="red", linestyle="--")
plt.xlabel("Valor real (total_amount)")
plt.ylabel("Predicción del modelo")
plt.title("Predicción vs. valor real del total a pagar")
plt.grid(True)
plt.show()
