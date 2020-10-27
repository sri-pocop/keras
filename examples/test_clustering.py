# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 18:33:23 2020

@author: srinivasan.c
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 17:50:22 2020

@author: srinivasan.c
"""
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

dataset = pd.read_csv('data/Mall_Customers.csv')
x = dataset.iloc[:,[2,3,4]].values 
#y=  dataset.iloc[:,4].values

fig = plt.figure()
ax = plt.axes(projection='3d')

import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=5)
y_hc = hc.fit_predict(x)

plt.show()
plt.clf()

ax.scatter3D(x[y_hc == 0,0], x[y_hc == 0,1],x[y_hc == 0,2], cmap = 'red', label = 'Cluseter 1')
ax.scatter3D(x[y_hc == 1,0], x[y_hc == 1,1],x[y_hc == 1,2], cmap = 'blue',label = 'Cluseter 2')
ax.scatter3D(x[y_hc == 2,0], x[y_hc == 2,1],x[y_hc == 2,2], cmap = 'green',label = 'Cluseter 3')
ax.scatter3D(x[y_hc == 3,0], x[y_hc == 3,1],x[y_hc == 3,2], cmap = 'cyan',label = 'Cluseter 4')
ax.scatter3D(x[y_hc == 4,0], x[y_hc == 4,1],x[y_hc == 4,2], cmap = 'magenta',label = 'Cluseter 5')
"""plt.title('HC Clustering')
plt.xlabel('Salary(k)')
plt.ylabel('Spent')
plt.legend()
plt.show()"""

#print(hc.fit_predict([[15,56]]))