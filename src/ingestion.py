from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Start Spark Session
spark = SparkSession.builder \
    .appName("FraudDetection-Ingestion") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

# Load Data from HDFS
print ("\n>>> Loading PaySim dataset from HDFS...")
df = spark.read.csv("hdfs://localhost:9000/fraud_project/paysim.csv", header=True, inferSchema=True)

# Validate Schema
print ("\n>>> Dataset Schema:")
df.printSchema()

# Basic EDA
total = df.count()
fraud = df.filter(F.col("isFraud") == 1).count()
legit = total - fraud

print (f"\n>>> Total Transactions: {total:,}")
print (f">>> Fraudulent:          {fraud:,}")
print (f">>> Legitimate:         {legit:,}")
print (f">>> Fraud Ratio:        {fraud/total*100:.4f}%")

# Show Sample
print ("\n>>> Sample Data:")
df.show(5)

print ("\n>>> Ingestion Complete!")
spark.stop()
