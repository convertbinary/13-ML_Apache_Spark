import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate sample data for clustering
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

# Apply k-means clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Print cluster censters
kmeans.cluster_centers_

# Plot the clusters and cluster centers
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='*',s=400,color='black')

# [task 1] load data
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-BD0231EN-SkillsNetwork/datasets/customers.csv"

df = pd.read_csv(URL)

print(df.sample(5))

df.shape

df.hist()


# [task 2] decide how many clusters to create

number_of_clusters = 3

# [task 3] create a clustering model

cluster = KMeans(n_clusters=number_of_clusters)

result = cluster.fit_transform(df)

cluster.cluster_centers_

# [task 4] make predictions

df['cluster_number'] = cluster.predict(df)

print(df.sample(5))

df.cluster_number.value_counts()



plt.show()