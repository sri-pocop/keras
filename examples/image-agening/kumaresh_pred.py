# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:24:27 2020

@author: srinivasan.c
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ss = StandardScaler()
from sklearn.cluster import KMeans
from sklearn import cluster

dataset = pd.read_csv('data/image_features2.csv')
data = dataset
#x = dataset.iloc[:,1:23].values 
x = dataset.iloc[:,1:23].values 

label_encoder = LabelEncoder()
dataset.iloc[:,0] = label_encoder.fit_transform(dataset.iloc[:,0]).astype('float64')
y = dataset.iloc[:,0].values

#from sklearn.decomposition import PCA
#pca = PCA(n_components=2)
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda
#pca = lda(n_components=2)
from sklearn.decomposition import KernelPCA
pca = KernelPCA(n_components=2)
principalComponents = pca.fit_transform(x,y)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, dataset[['SAMPLE']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1, 2, 3]
colors = ['r', 'g', 'b', 'y']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['SAMPLE'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()