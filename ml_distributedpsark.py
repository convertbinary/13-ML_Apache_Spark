# import necessary spark lib
import findspark
findspark.init()

from pyspark.sql import SparkSession

# initializing SparkContext

from pyspark import SparkContext
from datetime import datetime
#wget not working
#import wget
import urllib.request
import os

sc = SparkContext(appName="RetailStoreSalesAnalysis")
url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/XXlNzqYcxqkTbllc-tL_0w/Retailsales.csv"
# error even specify file name
#wget(url,out="Retailsales.csv")
# use urllib from the standard lib instead
if not os.path.exists("Retailsales.csv"):
    urllib.request.urlretrieve(url, "Retailsales.csv")

# load data
raw_data = sc.textFile("Retailsales.csv")

# parsing and cleaning data
def parse_line(line):
    # split the line by comma to get fields
    fields = line.split(",")
    # Return a dictionary with parsed fields
    return {
        'product_id': fields[0],
        'store_id': fields[1],
        'date': fields[2],
        'sales': float(fields[3]),
        'revenue': float(fields[4]),
        'stock': float(fields[5]),
        'price': float(fields[6]),
        'promo_type_1': fields[7],
        'promo_type_2': fields[9]
    }

# remove the header line
header = raw_data.first()

raw_data_no_header = raw_data.filter(lambda line: line != header)

# parse the lines into a structured format
parsed_data = raw_data_no_header.map(parse_line)
parsed_data = parsed_data.filter(lambda x: x is not None)

# filter out records with missing or invalid data
cleaned_data = parsed_data.filter(lambda x: x['sales'] > 0 and x['price'] > 0)

# partitioning
# check the number of partitions
print(f"Number of partitions in cleaned_data: {cleaned_data.getNumPartitions()}")

# partition-wise count
def count_in_partition(index, iterator):
    count = sum(1 for _ in iterator)
    yield (index, count)

# get the count of records in each partition
partitions_info = cleaned_data.mapPartitionsWithIndex(count_in_partition).collect()
print("Number of records in each partition:")
for partition, count in partitions_info:
    print(f"Partition {partition}: {count} records")

# total sales and revenue per product
sales_revenue_per_product = cleaned_data.map(lambda x: (x['product_id'],(x['sales'],x['revenue']))).reduceByKey(lambda a, b: (a[0]+b[0], a[1]+b[1]))
print(f"Nuber of partitions in cleaned_data: {cleaned_data.getNumPartitions()}")

# total sales and revenue per store
sales_revenue_per_store = cleaned_data.map(lambda x: (x['store_id'], (x['sales'],x['revenue']))).reduceByKey(lambda a,b: (a[0]+b[0], a[1]+b[1]))

# average price per product
total_price_count_per_product = cleaned_data.map(lambda x: (x['product_id'], (x['price'], 1))). reduceByKey(lambda a,b: (a[0]+b[0],a[1]+b[1]))
average_price_per_product = total_price_count_per_product.mapValues(lambda x: x[0]/x[1])

# sales and revenue per promotion type
sales_revenue_per_promo_1 = cleaned_data.map(lambda x: (x['promo_type_1'], (x['sales'],x['revenue']))).reduceByKey(lambda a,b: (a[0]+a[0],a[1]+b[1]))
sales_revenue_per_promo_2 = cleaned_data.map(lambda x: (x['promo_type_2'], (x['sales'],x['revenue']))).reduceByKey(lambda a,b: (a[0]+a[0],a[1]+b[1]))

# stock analysis per product
stock_per_product = cleaned_data.map(lambda x: (x['product_id'], x['stock'])).reduceByKey(lambda a,b: a+b)

# saving result
sales_revenue_per_product.saveAsTextFile("./sales_revenue_per_product")
sales_revenue_per_store.saveAsTextFile("./sales_revenue_per_store")
average_price_per_product.saveAsTextFile("./average_price_per_product")
sales_revenue_per_promo_1.saveAsTextFile("./sales_revenue_per_promo_1")
sales_revenue_per_promo_2.saveAsTextFile("./sales_revenue_per_promo_2")
stock_per_product.saveAsTextFile("./stock_per_product")

# printing result
print("Total sales and revenue per product:")
print("=" * 35)
for product in sales_revenue_per_product.collect():
    format_string = f"{{:<5}} | {{:<9}} | {{:<9}}"
    print(format_string.format(str(product[0]), str(round(product[1][0],2)), str(round(product[1][1],2))))

print("\n\nTotal Sales and Revenue per Store:")
print("=" * 35)
for store in sales_revenue_per_store.collect():
    format_string = f"{{:<5}} | {{:<9}} | {{:<9}}"
    print(format_string.format(str(store[0]), str(round(store[1][0],2)), str(round(store[1][1],2))))

print("\n\nAverage Price per Product:")
print("=" * 30)

for product in average_price_per_product.collect():
    format_string = f"{{:<5}} | {{:<9}}"
    print(format_string.format(str(product[0]), str(round(product[1],2))))

print("\n\nSales and Revenue per Promotion Type 1:")
print("=" * 40)
for promo in sales_revenue_per_promo_1.collect():
    format_string = f"{{:<5}} | {{:<9}} | {{:<9}}"
    print(format_string.format(str(promo[0]), str(round(promo[1][0],2)), str(round(promo[1][1],2))))

print("\n\nSales and Revenue per Promotion Type 2:")
print("=" * 40)
for promo in sales_revenue_per_promo_2.collect():
    format_string = f"{{:<5}} | {{:<9}} | {{:<9}}"

    print(format_string.format(str(promo[0]), str(round(promo[1][0],2)), str(round(promo[1][1],2))))

print("\n\nStock per Product:")
print("=" * 20)
for product in stock_per_product.collect():
    format_string = f"{{:<5}} | {{:<9}}"
    print(format_string.format(str(product[0]), str(round(product[1],2))))

# stop the spark context
sc.stop()