import findspark
findspark.init()

from pyspark.sql import SparkSession

import wget

# [task 1] create a spark session
spark = SparkSession.builder.appName("Getting started with spark").getOrCreate()

# [task 2] download the data file
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/mpg.csv"
wget.download(url)

# [task 3] load the data
mpg_data = spark.read.csv("mpg.csv", header=True, inferSchema=True)

# [task 4] explore the dataset
mpg_data.printSchema()

# [task 5] stop the spark session
spark.stop()