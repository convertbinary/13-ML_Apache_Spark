import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from math import sqrt
import matplotlib.pyplot as plt 

# [task 1] Load data

URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/mpg.csv"

df = pd.read_csv(URL)

df.sample(5)

df.plot.scatter(x="Weight",y="MPG")

#plt.show()

# [task 2] identify the target column and the data columns

y = df["MPG"]

X = df[["Horsepower", "Weight"]]

print(X.sample(5))

# [task 3] split datasets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# [task 4] build and train the model

lr = LinearRegression()

lr.fit(X_train, y_train)

# [task 5] evaluate the model

lr.score(X_test,y_test)

original_values = y_test
predicted_values = lr.predict(X_test)

print(r2_score(original_values,predicted_values))

print(mean_absolute_error(original_values,predicted_values))

print(mean_squared_error(original_values,predicted_values))

print(sqrt(mean_squared_error(original_values,predicted_values)))