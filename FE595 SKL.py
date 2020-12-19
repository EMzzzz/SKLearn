from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt

# Question 1
# Load data
boston_data = datasets.load_boston()
price = boston_data.target
feature = boston_data.data

# fit linear model
model_boston = LinearRegression()
model_boston.fit(feature, price)
value = model_boston.coef_
names = boston_data.feature_names
display = pd.DataFrame({'name': names,
                        'value': value})
# reorder
display = display.sort_values(by='value', ascending=False)
print(display)

# Question 2
wine = datasets.load_wine()
iris = datasets.load_iris()
wine = pd.DataFrame(wine.data, columns=wine['feature_names'])
iris = pd.DataFrame(iris.data, columns=iris['feature_names'])

# graph to show 3 is the correct number of populations to use
sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(wine)
    wine["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.plot(list(sse.keys())[2], list(sse.values())[2], 'ro', color='red')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(iris)
    data["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.plot(list(sse.keys())[2], list(sse.values())[2], 'ro', color='red')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()
