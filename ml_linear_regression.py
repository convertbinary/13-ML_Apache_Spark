# def warn(*args, **kwargs):
#     pass
# import warnings
# warnings.warn = warn
# warnings.filterwarnings('ignore')

import pandas as pd 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt 

# [task 1] load the data in a csv file into a data frame

# the data set is available at the url below.
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/mpg.csv"

# using the read_csv function in the pandas library, we load the data into a dataframe.
df = pd.read_csv(URL)

print(df.sample(5))
# print(df.shape)

plt.scatter(df["Horsepower"], df["MPG"])
plt.show(block=False)
plt.pause(1)

# [task 2] identify the target column and the data columns

target = df["MPG"]
features = df[["Horsepower", "Weight"]]

# [task 3] build and train a linear regression model

lr = LinearRegression()

lr.fit(features, target)

# [task 4] evaluate the model and make predictions

print("R² score:", lr.score(features, target))

print("Prediction:", lr.predict([[100, 2000]]))