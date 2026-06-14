import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# [task 1] load data in a csv file into a dataframe
# the data set is available at the url below.
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/iris.csv"

# load data into dataframe
df = pd.read_csv(URL)

# show 5 random rows from the dataset
print(df.sample(5))

print(df.shape)

df.Species.value_counts().plot.bar()
#plt.show()

# [task 2] identify the target column and the data columns
target = df["Species"]
features = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]

# [task 3] build and train a classifier
classifier = LogisticRegression()

classifier.fit(features,target)

# [task 4] evaluate the model
classifier.score(features,target)
classifier.predict([[5.4,2.6,4.1,1.3]])
