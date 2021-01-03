# -*- coding: utf-8 -*-
"""
put train data under : data/im_data.csv
put test data under : data/pred_data.csv
    
pred_data.csv file must have "contr","energ","maxpr" colums

The predicted class will be saved in next column once the program runs, 
    with "pred_class" column name

"""

#Inporting Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn import cluster
from sklearn.metrics import accuracy_score

#importing Dataset and seperating Dependent and independent Variables
dataset = pd.read_csv('data/im_data.csv')
x = dataset.iloc[:,1:4].values 
label_encoder = LabelEncoder()
dataset.iloc[:,4] = label_encoder.fit_transform(dataset.iloc[:,4]).astype('float64')
y = dataset.iloc[:,4].values

#Splitting training data and test Data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/4, random_state = 0)

#Defining Clusters
k = 4

#Defining KMeans Model using Train Data
kmeans = cluster.KMeans(n_clusters = k)
clf_ = kmeans.fit(x_train,y_train)
cm_for_label = confusion_matrix(clf_.labels_,y_train)
print(cm_for_label)

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

#For calculating accuracy
def for_accuracy(y, y_pred):    
    correct = 0
    wrong = 0
    total = len(y)
    for i in range(0, total):
        if label_encoder.classes_[int(y[i])] == label_encoder.classes_[original_labels[y_pred[i]]]:
            correct = correct + 1
        else:
            wrong = wrong + 1       
        #print(int(y[i]), original_labels[clf_.labels_[i]])  
    return str(round((correct/total)*100, 2)) + ' %'
print('Train Accuracy : ', for_accuracy(y_train, clf_.labels_))          
#Predicting using Test Data
y_pred = kmeans.predict(x_test)
y_pred_classes = []
for i in range(0, len(y_pred)):
    y_pred_classes.append(label_encoder.classes_[original_labels[y_pred[i]]])
y_pred_classes = np.array(y_pred_classes)

#Visualizing Predicted data using confusion matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
print('Test Accuracy : ', for_accuracy(y_test, y_pred))  


#Graphical Visulization of Data
#Plot Code Starts

from sklearn.decomposition import KernelPCA
pca = KernelPCA(n_components=2)
principalComponents = pca.fit_transform(x,y)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, dataset[['CLASS']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Component 1', fontsize = 15)
ax.set_ylabel('Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = [0, 1, 2, 3]
colors = ['r', 'g', 'b', 'y']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['CLASS'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()
#Plot Code Ends

# For three 3d plots - Starts
from mpl_toolkits import mplot3d
from sklearn.preprocessing import StandardScaler
s_s = StandardScaler()
for_3d_plot = pd.DataFrame(data = (x), columns = ['p1', 'p2','p3'])
for_3d_plot = pd.concat([for_3d_plot, dataset[['CLASS']]], axis = 1)
for_3d_plot['color'] = ''
for i in range(0, len(for_3d_plot)):
    for_3d_plot["color"][i] = colors[int(for_3d_plot['CLASS'][i])]
threedee = plt.figure().gca(projection='3d')
threedee.scatter(for_3d_plot.index, for_3d_plot['p1'], for_3d_plot['p2']
                     , c = for_3d_plot['color'])
threedee.set_xlabel('index')
threedee.set_ylabel('p1')
threedee.set_zlabel('p2')

threedee = plt.figure().gca(projection='3d')
threedee.scatter(for_3d_plot.index, for_3d_plot['p2'], for_3d_plot['p3']
                     , c = for_3d_plot['color'])
threedee.set_xlabel('index')
threedee.set_ylabel('p2')
threedee.set_zlabel('p3')

threedee = plt.figure().gca(projection='3d')
threedee.scatter(for_3d_plot.index, for_3d_plot['p1'], for_3d_plot['p3']
                     , c = for_3d_plot['color'])
threedee.set_xlabel('index')
threedee.set_ylabel('p1')
threedee.set_zlabel('p3')
# For three 3d plots - Ends