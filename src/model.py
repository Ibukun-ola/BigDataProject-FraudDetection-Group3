
from pyspark.sql import SparkSession
from pyspark.sql import function as F
from pyspark.ml import pipeline
from pyspark.ml.feature import VectorAssembler, StringIndex
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#------Start Spark Session--------------------------------------

spark=SparkSession.builder\
	.appName("FraudDetection-Modelling")\
	.getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

#------Loading Engineered features from HDFS...")

print("\n>>>Loading engineered features from HDFS...")
df=spark.read.parquet(
	"hdfs://localhost:9000/fraud_project/features/"
)


#-----Handle Class Imbalance (Class Weighting)------

print("\n>>> Handling class imbalance...")
fraud_count=df.filter(F.col("isFraud")==1).count()
legit_count=df.filter(F.col("isFraud")==0).count()
ratio=legit_count/fraud_count

print(f">>>Fraud count: {fraud_count:,}")
print(f">>>Legitimate count: {legit_count:,})
print(f">>>Weight ratio: {ratio:.2f}")

df=df.withColumn("classWeight",
	F.when(F.col("isFraud")==1, ratio).otherwise(1.0))


#---------Encode Transaction Type----------------------

type_indexer=StringIndexer(inputCol="type", outputCol="typeIndex")

#--------Assemble Feature Vector-----------------------

feature_cols=[
	"amount",
	"typeIndex",
	"txn_count_15min",
	"amt_sum_15min",
	"balance_drop",
	"balanced_wiped",
	"oldbalanceOrg",
	"newbalanceOrig"
]

assembler=VectorAssembler(
	inputCols=feature_cols,
	outputcol="features"
)


#--------Random Forest Classifier---------------

rf=RandomForestClassifier(
	labelCol="isFraud",
	featuresCol="features",
	weightCol="classWeight",
	numTrees=100,
	maxDepth=10,
	seed=42
	)

#-------Build Pipeline-----------------------------

pipeline=Pipeline(stages=[type_indexer, assembler,rf])


#-------Train/Test split---------------------------

print("\n>>> Splitting data into train and test sets (80/20)...")
train, test =df.randomSplit([0.8,0.2], seed =42
print(f">>>Training rows: {train.count}():,}")
print(f">>>Testing rows: (test.count():,}")


#--------Train the model------------------------------
print("\n>>> Training Random Forest model...")
model = pipeline.fit(train)
print (">>>Training Complete!")



#--------Save Model to HDFS--------------------------

print("\n>>> Saving model to HDFS...")
mode.write()overwrite()save(
	"hdfs://localhost:9000/fraud_project/model/"
)

print("\n>>>Modelling Complete!")
spark.stop()

