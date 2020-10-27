# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:54:20 2020

@author: srinivasan.c
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd

dataset = pd.read_csv('data/Social_Network_Ads.csv')
x = dataset.iloc[:,[2,3]].values 
y=  dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/4, random_state =0)

from sklearn.preprocessing import StandardScaler
s_scaler = StandardScaler()
x_train_scaler = s_scaler.fit_transform(x_train)
x_test_scaler = s_scaler.fit_transform(x_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear')
classifier.fit(x_train_scaler,y_train)

y_pre =  classifier.predict(x_test_scaler)

from sklearn.metrics import confusion_matrix
x_cm = confusion_matrix(y_test,y_pre)

x_set, y_set = x_train_scaler,y_train

"""for i in range(x_cm.shape[0]):
    sum += x_cm[i][i]
    
accuracy = sum/x_test.shape[0]
print(accuracy)"""

x1,x2 = np.meshgrid(np.arange (start = x_set[:,0].min() - 1, stop = x_set[:,0].max() + 1, step = 0.01),
                    np.arange (start = x_set[:,1].min() - 1, stop = x_set[:,1].max() + 1, step = 0.01))
plt.contourf (x1, x2, classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
    alpha = 0.75, cmap = ListedColormap({'red','green'}))
plt.xlim( x1.min(), x1.max())
plt.xlim( x2.min(), x2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0],x_set[y_set == j,1], c= ListedColormap({'red','green'})(i),label = j)

plt.show()
x_set, y_set = x_test_scaler,y_test

x1,x2 = np.meshgrid(np.arange (start = x_set[:,0].min() - 1, stop = x_set[:,0].max() + 1, step = 0.01),
                    np.arange (start = x_set[:,1].min() - 1, stop = x_set[:,1].max() + 1, step = 0.01))
plt.contourf (x1, x2, classifier.predict(np.array([x1.ravel(),x2.ravel()]).T).reshape(x1.shape),
    alpha = 0.75, cmap = ListedColormap(('red','green')))
plt.xlim( x1.min(), x1.max())
plt.xlim( x2.min(), x2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j,0],x_set[y_set == j,1], c= ListedColormap(('red','green'))(i),label = j)
plt.show()
"""
import statsmodels.regression.linear_model as sm
x = np.append(arr = np.ones((50,1)).astype(int), values = x , axis = 1)

x_opt = x[:, [0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()"""