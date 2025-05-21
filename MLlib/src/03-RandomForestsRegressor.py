import os
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel
from pyspark.sql import SparkSession
from pyspark.ml.regression import RandomForestRegressionModel
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


spark = SparkSession.builder \
    .appName("NYC Taxi RF Regressor") \
    .config("spark.driver.memory", "8g") \
    .getOrCreate()
    

# Load the data
df = spark.read.parquet("../../data/processed.parquet")

# Split the data
(train_df, test_df) = df.randomSplit([0.8, 0.2], seed=42)

# Assemble features
feature_columns = ["trip_distance", "hour", "day_of_week", "extras", "fare_amount", "tip_amount","passenger_count", "payment_type"]
assembler = VectorAssembler(inputCols=feature_columns, outputCol="features", handleInvalid="skip")
train_df = assembler.transform(train_df)
test_df = assembler.transform(test_df)

train_df = train_df.select("features", "total_amount")
test_df = test_df.select("features", "total_amount")


# Train the Random Forest Regressor and perform hyperparameter tuning
rf = RandomForestRegressor(numTrees=15, maxDepth=5, seed=42, featuresCol="features", labelCol="total_amount")

param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [10, 20, 30]) \
    .addGrid(rf.maxDepth, [5, 10, 15]) \
    .build()

cross_validator = CrossValidator(estimator=rf, estimatorParamMaps=param_grid, evaluator=RegressionEvaluator(predictionCol="prediction", labelCol="total_amount", metricName="rmse"), numFolds=3)

model = cross_validator.fit(train_df)

# Evaluate the model
evaluator_rmse = RegressionEvaluator(predictionCol="prediction", labelCol="total_amount", metricName="rmse")
rmse = evaluator_rmse.evaluate(model.transform(test_df))

evaluator_mae = RegressionEvaluator(predictionCol="prediction", labelCol="total_amount", metricName="mae")
mae = evaluator_mae.evaluate(model.transform(test_df))

evaluator_r2 = RegressionEvaluator(predictionCol="prediction", labelCol="total_amount", metricName="r2")
r2 = evaluator_r2.evaluate(model.transform(test_df))

print("Model Evaluation Metrics")
print("-" * 35)
print(f"{'RMSE':<10}: {rmse:.2f}")
print(f"{'MAE':<10}: {mae:.2f}")
print(f"{'R²':<10}: {r2:.4f}")

# Save the model
model_path = "../models/RFRegressor"
if not os.path.exists(model_path):
    model.save(model_path)
    print(f"Model saved to {model_path}.")
else:
    print(f"Model already exists at {model_path}.")

# Load the model
loaded_model = CrossValidatorModel.load(model_path)
predictions = loaded_model.transform(test_df)
predictions = predictions.select("total_amount", "prediction")
predictions = predictions.toPandas()
predictions = predictions.sort_values(by="total_amount", ascending=False).head(100).reset_index(drop=True)
predictions['index'] = np.arange(len(predictions))


# Plot the predictions
plt.figure(figsize=(12, 6))
plt.scatter(predictions["index"], predictions["total_amount"], label="Real (total_amount)", color="blue", alpha=0.6)
plt.scatter(predictions["index"], predictions["prediction"], label="Predicted", color="orange", alpha=0.6)

plt.xlabel("Sample Index")
plt.ylabel("Amount ($)")
plt.title("Random Forest Regression: Real vs Predicted Values")
plt.legend()
plt.tight_layout()
plt.savefig("../plots/RFRegressor_scatter.png")
plt.show()





