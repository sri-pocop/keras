# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:50:22 2020

@author: srinivasan.c
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

dataset = pd.read_csv('data/Mall_Customers.csv')
x = dataset.iloc[:,[3,4]].values 
#y=  dataset.iloc[:,4].values


import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5)
y_hc = hc.fit_predict(x)

plt.show()
plt.clf()

plt.scatter(x[y_hc == 0,0], x[y_hc == 0,1], c = 'red', label = 'Cluseter 1')
plt.scatter(x[y_hc == 1,0], x[y_hc == 1,1], c = 'blue',label = 'Cluseter 2')
plt.scatter(x[y_hc == 2,0], x[y_hc == 2,1], c = 'green',label = 'Cluseter 3')
plt.scatter(x[y_hc == 3,0], x[y_hc == 3,1], c = 'cyan',label = 'Cluseter 4')
plt.scatter(x[y_hc == 4,0], x[y_hc == 4,1], c = 'magenta',label = 'Cluseter 5')
plt.title('HC Clustering')
plt.xlabel('Salary(k)')
plt.ylabel('Spent')
plt.legend()
plt.show()

#print(hc.fit_predict([[15,56]]))