#Inporting Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn import cluster
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as mcm
import warnings
warnings.filterwarnings("ignore")

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
kmeans.fit(x_train)

#Predicting using Test Data
y_pred = kmeans.predict(x_test)

#Visualizing Predicted data using confusion matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# silhouette plot

"""cluster_labels = y_pred
clusterer = kmeans"""

X = x

range_n_clusters = [2,3,4,5,6]

for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    # lie within [-0.1, 1]
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = mcm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = mcm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')
    
    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

plt.show()

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
ax.set_xlabel('F 1', fontsize = 15)
ax.set_ylabel('F 2', fontsize = 15)
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

#Plot Code Ends# -*- coding: utf-8 -*-
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
