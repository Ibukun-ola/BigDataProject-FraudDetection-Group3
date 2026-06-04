from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("FraudDetection-Evaluation") \
    .config("spark.driver.memory", "1g") \
    .config("spark.executor.memory", "1g") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\n>>> Loading features...")
df = spark.read.parquet("hdfs://localhost:9000/fraud_project/features/")
df = df.sample(fraction=0.05, seed=42)

fraud_count = df.filter(F.col("isFraud") == 1).count()
legit_count = df.filter(F.col("isFraud") == 0).count()
ratio = legit_count / fraud_count

df = df.withColumn("classWeight",
    F.when(F.col("isFraud") == 1, ratio).otherwise(1.0))

type_indexer = StringIndexer(inputCol="type", outputCol="typeIndex")
feature_cols = ["amount", "typeIndex", "txn_count_15min",
    "amt_sum_15min", "balance_drop", "balance_wiped",
    "oldbalanceOrg", "newbalanceOrig"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
rf = RandomForestClassifier(
    labelCol="isFraud", featuresCol="features",
    weightCol="classWeight", numTrees=10, maxDepth=4, seed=42)

pipeline = Pipeline(stages=[type_indexer, assembler, rf])
train, test = df.randomSplit([0.8, 0.2], seed=42)

print("\n>>> Training model...")
model = pipeline.fit(train)

print("\n>>> Evaluating...")
predictions = model.transform(test)

recall = MulticlassClassificationEvaluator(
    labelCol="isFraud", predictionCol="prediction",
    metricName="weightedRecall").evaluate(predictions)

precision = MulticlassClassificationEvaluator(
    labelCol="isFraud", predictionCol="prediction",
    metricName="weightedPrecision").evaluate(predictions)

f1 = MulticlassClassificationEvaluator(
    labelCol="isFraud", predictionCol="prediction",
    metricName="f1").evaluate(predictions)

print("\n>>> MODEL EVALUATION RESULTS")
print(">>> Recall:    " + str(round(recall, 4)))
print(">>> Precision: " + str(round(precision, 4)))
print(">>> F1 Score:  " + str(round(f1, 4)))

print("\n>>> Confusion Matrix:")
predictions.groupBy("isFraud", "prediction") \
           .count() \
           .orderBy("isFraud", "prediction") \
           .show()

rf_model = model.stages[-1]
importances = rf_model.featureImportances
feature_importance = sorted(
    zip(feature_cols, importances),
    key=lambda x: -x[1]
)

print("\n>>> Top 3 Features:")
for rank, (feat, score) in enumerate(feature_importance[:3], 1):
    print(str(rank) + ". " + feat + ": " + str(round(score, 4)))

print("\n>>> Evaluation Complete!")
spark.stop()
