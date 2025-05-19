import os
from pyspark.ml.classification import GBTClassifier
from pyspark.sql import SparkSession
from pyspark.sql.functions import when, col
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import VectorAssembler

spark = SparkSession.builder.appName("GBTClassifier").getOrCreate()

# Load data
df = spark.read.parquet("../../data/processed.parquet")

# Add the boolean column for tip given
df = df.withColumn("tip_given", when(col("tip_amount") > 0, 1).otherwise(0))

assembler = VectorAssembler(inputCols=["total_amount", "trip_distance", "fare_amount", "extras", "passenger_count"], outputCol="features", handleInvalid="skip")
df = assembler.transform(df)
df = df.select("features", "tip_given")

# Split the data into training and test sets
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Train the GBTClassifier
gbt = GBTClassifier(labelCol="tip_given", featuresCol="features", maxIter=10)

param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [5, 10, 15]) \
    .addGrid(gbt.maxBins, [10, 20, 30]) \
    .addGrid(gbt.maxIter, [10, 20, 30]) \
    .build()

cross_validator = CrossValidator(estimator=gbt, estimatorParamMaps=param_grid, evaluator=BinaryClassificationEvaluator(labelCol="tip_given", rawPredictionCol="rawPrediction", metricName="areaUnderROC"), numFolds=5)

model = cross_validator.fit(train_df)

# Evaluate the model
evaluator = BinaryClassificationEvaluator(labelCol="tip_given", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

auc = evaluator.evaluate(model.transform(test_df))

print(f"Area Under ROC: {auc}")

# Save the model
model_path = "../models/GBTClassifier"
if not os.path.exists(model_path):
    os.makedirs(model_path)
    model.save(model_path)
else:
    print(f"Model already exists at {model_path}.")