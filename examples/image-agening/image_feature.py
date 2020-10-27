# -*- coding: utf-8 -*-
"""
Created on Sat May  9 07:05:20 2020

@author: srinivasan.c
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from collections import Counter

dataset = pd.read_csv('data/original_data.csv')
#x = dataset.iloc[:,1:23].values 
x = dataset.iloc[:,[1,2,3,5,6,8,12,13]].values 
#y=  dataset.iloc[:,4].values

from sklearn.cluster import KMeans
wcss = []

for i in range(1,5):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 3000, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,5),wcss)
plt.show()

kmeans = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 3000, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)
y_k = Counter(y_kmeans)

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x, method = 'ward'))

from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=4)
y_hc = hc.fit_predict(x)
y_h = Counter(y_hc)