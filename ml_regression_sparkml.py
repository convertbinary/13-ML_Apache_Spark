import findspark
findspark.init()

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

import urllib.request
import os


# create spark session
spark = SparkSession.builder.appName("Regressing using SparkML").getOrCreate()

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/mpg.csv"

if not os.path.exists("mpg.csv"):
    urllib.request.urlretrieve(url,"mpg.csv")

# load mpg dataset
mpg_data = spark.read.csv("mpg.csv", header = True, inferSchema = True)

mpg_data.printSchema()

mpg_data.show(5)

# prepare feature vector
assembler = VectorAssembler(inputCols=["Cylinders", "Engine Disp", "Horsepower", "Weight", "Accelerate", "Year"], outputCol="features")
mpg_transformed_data = assembler.transform(mpg_data)

# display the assembled "features" and the label column "MPG"
mpg_transformed_data.select("features", "MPG").show()

# split the data
(training_data, testing_data) = mpg_transformed_data.randomSplit([0.7,0.3], seed=42)

# build and train
lr = LinearRegression(featuresCol="features", labelCol="MPG")
model = lr.fit(training_data)

# evaluate the model
predictions = model.transform(testing_data)

# r squred
evaluator = RegressionEvaluator(labelCol="MPG", predictionCol="prediction", metricName = "r2")
r2 = evaluator.evaluate(predictions)
print("R Squared = ", r2)

# Root Mean Squared Error
evaluator = RegressionEvaluator(labelCol="MPG", predictionCol="prediction", metricName = "rmse")
rmse = evaluator.evaluate(predictions)
print("RMSE = ", rmse)

# Mean Absolute Error
evaluator = RegressionEvaluator(labelCol="MPG", predictionCol="prediction", metricName = "mae")
mae = evaluator.evaluate(predictions)
print("MAE = ", mae)

spark.stop()