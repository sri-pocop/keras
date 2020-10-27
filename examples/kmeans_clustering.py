# -*- coding: (utf-8 -*-
"""
Created on Wed Apr 29 15:43:21 2020

@author: srinivasan.c
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

dataset = pd.read_csv('data/Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values 
#y=  dataset.iloc[:,4].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11),wcss)
plt.show()
plt.clf()

kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

plt.scatter(x[y_kmeans == 0,0], x[y_kmeans == 0,1], c = 'red', label = 'Cluseter 1')
plt.scatter(x[y_kmeans == 1,0], x[y_kmeans == 1,1], c = 'blue',label = 'Cluseter 2')
plt.scatter(x[y_kmeans == 2,0], x[y_kmeans == 2,1], c = 'green',label = 'Cluseter 3')
plt.scatter(x[y_kmeans == 3,0], x[y_kmeans == 3,1], c = 'cyan',label = 'Cluseter 4')
plt.scatter(x[y_kmeans == 4,0], x[y_kmeans == 4,1], c = 'magenta',label = 'Cluseter 5')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], c = 'yellow', s = 150, label = 'centeroid')
plt.title('KMeans Clustering')
plt.xlabel('Salary(k)')
plt.ylabel('Spent')
plt.legend()
plt.show()