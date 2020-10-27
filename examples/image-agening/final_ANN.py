# -*- coding: utf-8 -*-
"""
Created on Mon May 18 18:24:30 2020

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
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('data/im_data.csv')
 

label_encoder = LabelEncoder()
dataset.iloc[:,-1] = label_encoder.fit_transform(dataset.iloc[:,-1]).astype('float64')

data = dataset.iloc[:,1:]
"""
corr = data.corr()
columns = np.full((corr.shape[0],), True, dtype=bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns = data.columns[columns]
data = data[selected_columns]

#selected_columns = selected_columns[1:].values
import statsmodels.api as sm
def backwardElimination(x, Y, sl, columns):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
                    columns = np.delete(columns, j)
                    
    regressor_OLS.summary()
    return x, columns
SL = 0.05
data_modeled, selected_columns = backwardElimination(data.iloc[:,1:].values, data.iloc[:,0].values, SL, selected_columns)
data = data[selected_columns]
"""
x = data.iloc[:,:3].values 
y = data.iloc[:,-1].values 

x = ss.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/5, random_state = 0)

n_cols = x_train.shape[1]
model = Sequential()
model.add(Dense(3, activation='relu', input_shape=(n_cols,)))
model.add(Dense(3, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error',metrics = ['accuracy'])

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=5)#train model
#model.fit(x_train, y_train, validation_split=0.2, epochs=200, callbacks=[early_stopping_monitor])
r = model.fit(x, y, validation_split=0.2, epochs=200, callbacks=[early_stopping_monitor])

#y_pred = model.predict(x_test)

#y_pred_a = np.around(y_pred)
"""
cm = confusion_matrix(y, y_pred_a)

c = 0
w = 0
for i in range(0, len(y_test)):
    if y_test[i] == y_pred_a[i]:
        c = c + 1
    else:
        w = w + 1

print(c,w,c/len(y_test))

cm_for_label = confusion_matrix(r.labels_,y)

#for getting Exact Labels
original_labels = []
for i in range(0,4):
    max_val = 0
    max_val_index = 0
    for j in range(0,k):    
        #print(i,j,cm_for_label[i][j])
        if cm_for_label[i][j] > max_val:
            max_val = cm_for_label[i][j]
            max_val_index = j
    original_labels.append(max_val_index)
#For predictiong on new data
"""
data_pred = pd.read_csv('data/pred_data.csv')
data_for_pred = data_pred[["contr","energ","maxpr"]].values
data_pred["pred_class"] = ''
y_pred = model.predict(data_for_pred)
y_pred = np.around(y_pred)
for i in range(0, len(y_pred)):
    data_pred["pred_class"][i] = y_pred[i]#label_encoder.classes_[original_labels[y_pred[i]]]
data_pred.to_csv("data/pred_data.csv", index=False)

"""
classifier = Sequential()
classifier.add(Dense(units = 22, kernel_initializer = 'uniform' , activation = 'relu' , input_dim = 22))

classifier.add(Dense(units = 22, kernel_initializer = 'uniform' , activation = 'relu'))

classifier.add(Dense(1)#, kernel_initializer = 'uniform' , activation = 'sigmoid' ))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

classifier.fit(x_train, y_train , batch_size = 10 , nb_epoch = 50)


#selected_columns = selected_columns[1:].values
"""