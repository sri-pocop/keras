"""
SVM Algorithm
"""

#Inporting Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn import cluster
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#importing Dataset and seperating Dependent and independent Variables
dataset = pd.read_csv('data/im_data.csv')
x = dataset.iloc[:,1:4].values 
label_encoder = LabelEncoder()
dataset.iloc[:,4] = label_encoder.fit_transform(dataset.iloc[:,4]).astype('float64')
y = dataset.iloc[:,4].values

#Splitting training data and test Data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/10, random_state = 0)

#Defining SVM Model using Train Data
clf = SVC(kernel='linear')
clf_ = clf.fit(x_train,y_train)

#Predicting using Test Data
y_pred = clf.predict(x_test)

#Visualizing Predicted data using confusion matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

#Prediction on new data
data_pred = pd.read_csv('data/pred_data.csv')
data_for_pred = data_pred[["contr","energ","maxpr"]].values
data_pred["pred_SVM"] = ''
y_pred = clf.predict(data_for_pred)
y_pred = y_pred.astype(int)
for i in range(0, len(y_pred)):
    data_pred["pred_SVM"][i] = label_encoder.classes_[y_pred[i]]
data_pred.to_csv("data/pred_data.csv", index=False)

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


#decision boundary code starts
h = .02
# create a mesh to plot in
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
z_min, z_max = x[:, 2].min() - 1, x[:, 2].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
xx, zz = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(z_min, z_max, h))

for i, clf in enumerate(([0,1],[0,2],[1,2])):
    #plt.subplot(2,3, i + 1)   #include if 3 plot needed in single plot
    #plt.subplots_adjust(wspace=0.4, hspace=0.4)  #include if 3 plot needed in single plot

    Z = clf_.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=.8)

    # Plot also the training points
    plt.scatter(x[:, clf[0]], x[:, clf[1]], c=y)#, cmap=plt.cm.coolwarm)
    plt.xlabel('feature ' + str(clf[0]))
    plt.ylabel('feature ' + str(clf[1]))
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title('feature' + str(clf[0]) + '&' + str(clf[1]) )
    plt.show() #remove if 3 plot needed in single plot
plt.show()

#decision boundary code ends
