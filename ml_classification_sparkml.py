import findspark
findspark.init()

from pyspark.sql import SparkSession

from pyspark.ml.feature import (StringIndexer, VectorAssembler)
from pyspark.ml.classification import LogisticRegression

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# for downloading data
import os
import urllib.request

# create spark session
spark = SparkSession.builder.appName("Classification using SparkML").getOrCreate()

# load the data
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/drybeans.csv"
if not os.path.exists("drybeans.csv"):
    urllib.request.urlretrieve(url,"drybeans.csv")

beans_data = spark.read.csv("drybeans.csv", header=True,inferSchema=True)

# print schema of dataset
beans_data.printSchema()
# print top 5 rows of selected column
beans_data.select(["Area","Perimeter","Solidity","roundness","Compactness","Class"]).show(5)
# print the value counts for the column 'Class'
beans_data.groupBy('Class').count().orderBy('count').show()
# convert class column from string to numerical values
indexer = StringIndexer(inputCol="Class", outputCol="label")
beans_data = indexer.fit(beans_data).transform(beans_data)
# print the value counts for the column 'label'
beans_data.groupBy('label').count().orderBy('count').show()

# identify the label column and the input columns
assembler = VectorAssembler(inputCols=["Area","Perimeter","Solidity","roundness","Compactness"], outputCol="features")
beans_transformed_data = assembler.transform(beans_data)

beans_transformed_data.select("features","label").show()

# split the data
(training_data, testing_data) = beans_transformed_data.randomSplit([0.7,0.3], seed=42)

# build and train a logistic regression model
lr = LogisticRegression(featuresCol="features", labelCol="label")
model =lr.fit(training_data)

# evaluate the model
predictions = model.transform(testing_data)

# evaluate the model performance
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction',metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print("Accuracy =", accuracy)

