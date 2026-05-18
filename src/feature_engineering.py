from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Start Spark Session
spark = SparkSession.builder \
    .appName("FraudDetection-FeatureEngineering") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Load Data from HDFS
print ("\n>>> Loading PaySim dataset from HDFS...")
df = spark.read.csv(
    "hdfs://localhost:9000/fraud_project/paysim.csv",
    header=True,
    inferSchema=True
)

# Data Cleaning
print ("\n>>> Checking for null values...")
df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c)
           for c in df.columns]).show()

df = df.dropna()
print(f">>> Rows after cleaning: {df.count():,}")

# Exploratory Data Analysis
print ("\n>>> Transaction types distribution:")
df.groupBy("type").count().orderBy("count", ascending=False).show()

print ("\n>>> Fraud by transaction type:")
df.groupBy("type", "isFraud").count()\
  .orderBy("type", "isFraud").show()

# Velocity Features (Spark Windowing)
# Window: per sender (nameOrig), ordered by step (1 step = 1 hour)
# rangeBetween(-1, 0) approximates a 15-minute window
print ("\n>>> Engineering Velocity Features...")

windowSpec = Window.partitionBy("nameOrig") \
                   .orderBy("step") \
                   .rangeBetween(-1, 0)

# Number of transactions by sender in last ~15 mins
df = df.withColumn("txn_count_15min", F.count("step").over(windowSpec))

# Total amount sent by sender in last ~15 mins
df = df.withColumn("amt_sum_15min", F.sum("amount").over(windowSpec))

# Balance drop: difference between old and new balance for sender 
df = df.withColumn("balance_drop",
    F.col("oldbalanceOrg") - F.col("newbalanceOrig"))

# Flag: balance wiped out completely (common in fraud)
df = df.withColumn("balance_wiped",
    F.when((F.col("oldbalanceOrg") > 0) &
           (F.col("newbalanceOrig") == 0), 1).otherwise(0))

# Show Engineered Features 
print ("\n>>> Sample of Engineered Features:")
df.select("nameOrig", "step", "amount", "txn_count_15min", "amt_sum_15min", "balance_drop", "balance_wiped", "isFraud").show(10)

# Save to  HDFS
print ("\n>>> Saving engineered dataset to HDFS...")
df.write.mode("overwrite").parquet(
    "hdfs://localhost:9000/fraud_project/features/"
)

print ("\n>>> Feature Engineering Complete!")
spark.stop()
