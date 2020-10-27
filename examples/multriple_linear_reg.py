# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 13:45:07 2020

@author: srinivasan.c
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('data/50_Startups.csv')
x = dataset.iloc[:,:-1].values 
y=  dataset.iloc[:,4].values
#print(y)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncode_x = LabelEncoder()
x[:,3] = labelEncode_x.fit_transform(x[:,3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=1/3)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

regressor.fit(x_train,y_train)
#print(x_test)
y_pre =  regressor.predict(x_test)

import statsmodels.regression.linear_model as sm
x = np.append(arr = np.ones((50,1)).astype(int), values = x , axis = 1)

x_opt = x[:, [0,1,2,3,4,5]]
regressor_ols = sm.OLS(endog = y, exog = x_opt).fit()
regressor_ols.summary()