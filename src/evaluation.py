cat > src/evaluation.py << 'EOF'
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder \
    .appName("FraudDetection-Evaluation") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("\n>>> Loading engineered features from HDFS...")
df = spark.read.parquet(
    "hdfs://localhost:9000/fraud_project/features/"
)

_, test = df.randomSplit([0.8, 0.2], seed=42)

print("\n>>> Loading trained model from HDFS...")
model = PipelineModel.load(
    "hdfs://localhost:9000/fraud_project/model/"
)

print("\n>>> Generating predictions...")
predictions = model.transform(test)

auc_eval = BinaryClassificationEvaluator(
    labelCol="isFraud",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc = auc_eval.evaluate(predictions)

recall_eval = MulticlassClassificationEvaluator(
    labelCol="isFraud",
    predictionCol="prediction",
    metricName="weightedRecall"
)
recall = recall_eval.evaluate(predictions)

precision_eval = MulticlassClassificationEvaluator(
    labelCol="isFraud",
    predictionCol="prediction",
    metricName="weightedPrecision"
)
precision = precision_eval.evaluate(predictions)

f1_eval = MulticlassClassificationEvaluator(
    labelCol="isFraud",
    predictionCol="prediction",
    metricName="f1"
)
f1 = f1_eval.evaluate(predictions)

print(f"\n>>> MODEL EVALUATION RESULTS")
print(f">>> AUC-ROC:   {auc:.4f}")
print(f">>> Recall:    {recall:.4f}  <- Primary Metric")
print(f">>> Precision: {precision:.4f}")
print(f">>> F1 Score:  {f1:.4f}")

print("\n>>> Confusion Matrix:")
predictions.groupBy("isFraud", "prediction") \
           .count() \
           .orderBy("isFraud", "prediction") \
           .show()

feature_cols = [
    "amount", "typeIndex", "txn_count_15min",
    "amt_sum_15min", "balance_drop", "balance_wiped",
    "oldbalanceOrg", "newbalanceOrig"
]

rf_model = model.stages[-1]
importances = rf_model.featureImportances
feature_importance = sorted(
    zip(feature_cols, importances),
    key=lambda x: -x[1]
)

print(f"\n{'Rank':<6} {'Feature':<25} {'Importance Score'}")
print("-" * 45)
for rank, (feat, score) in enumerate(feature_importance[:3], 1):
    print(f"  {rank}    {feat:<25} {score:.4f}")

print("\n>>> Evaluation Complete!")
spark.stop()
EOF
