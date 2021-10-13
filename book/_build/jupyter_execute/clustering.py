#!/usr/bin/env python
# coding: utf-8

# # Clustering

# Clustering is an unsupervised learning method that divides the entire dataset into groups, or clusters, based on patterns in the data. Clustering is the task of dividing all of the data points in groups based on some similarity measure. The objective is to group each data point into a group of similar data points, while keeping that group dissimilar to the other groups. Clustering can be useful for discovering hidden patterns in a dataset. There are many different clustering algorithms, including DBSCAN, Aggolmerative Clustering, Affinity Propagation and the algorithm we are going use, K-Means.

# **K-Means Algorithm**

# K-means clustering is one of the simplest and most popular machine learning algorithms. K-means groups the data into a defined number (k) of clusters. The similarity of any two points is determined by the distance between them, Euclidean distance is one of the most commonly used measures of distance.  

# $$
#  d\left( p,q\right)   = \sqrt {\sum _{i=1}^{n}  \left( q_{i}-p_{i}\right)^2 } 
# $$(EuclideanDistance)

# The K-means algorithm is iterative, after the number of clusters, k, is specified, it works by executing the following steps:  
# 1. Select k random points from the data as the initial centroids.
# 2. Allocate all other points to the nearest cluster centroid.
# 3. Recompute the centroids of the new clusters.
# 4. Repeat steps 3 and 4, until either:
#     -  the centroids have stabilized (don't change between interations), or
#     - the maximum number of iterations have completed.

# ## Import Libraries & Read In Data

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import intercluster_distance
import plotly.graph_objs as go
import seaborn as sns


# In[2]:


processed_df = pd.read_csv('../data/processed.csv', index_col=0)
processed_df.head()                        


# In[3]:


scaled = processed_df.iloc[:,2:]


# ## Determine Number of Clusters

# For K-means number of clusters, k, must be pre-defined by the user. Therefore, it is up to us to determine the optimal number of cluser our the dataset. However, the optimal number of clusters in often subjective and depends on the similarity measure and the partitioning parameters.
# To choose an appropiate number of clusters, we will use two common methods, the Elbow Method and the Silhouette Coefficient. 

# ### The Elbow Method

# The Elbow method can be summarized as follows:
#   1. Compute the K-means algorithm for different values of k.
#   2. For each k, calculate and record the total within cluster sum of square (WSS):

# $$
#   WSS   = \sum _{i=1}^{n_{c}}{\sum _{x\in{ci}}  \left( x, \bar{x_{ci}}\right)^2 }, 
# $$ (wss)

#  where, ${c_{i} = Cluster}$$,  $${n_{c} = NumberOfClusters} $$,  $${\bar{x_{ci}} = ClusterCentroid}$ 

#  3. Plot WSS against k.
#  4. Identify the knee in the plot as the optimal k.

# In[4]:


# fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
for cluster in range(1,15):
    kmeans = KMeans(n_clusters = cluster, init='k-means++', random_state=42)
    kmeans.fit(processed_df.iloc[:,2:])
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,15), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o', lw=2.5)
plt.xlabel('Number of clusters', fontsize = 14)
plt.ylabel('Inertia', fontsize = 14)
plt.grid()
plt.show()


# Inertia decreases as the number of clusters increase. We're looking for the sweet spot (elbow point) where the curve starts to bend.

# **Confirm Elbow using kneed**

# In[5]:


kl = KneeLocator(range(1, 15), SSE, curve="convex", direction="decreasing")
print('Elbow at k = ' + str(kl.elbow))


# ### The Silhouettee coefficient

# Measure of how similar a data point is within-cluster (cohesion) compared to other clusters (seperation).

# The formula for the Silhouette score, S, is:
# $$
#   S = \frac{b-a}{max(a,b)}
# $$
# Where:
# - the mean distance between the observation and all other data points in the same cluster (i.e., intra-cluster distance) is denoted by **a**, and
# - The mean distance between the observation and all other data points of the next nearest cluster (mean nearest-cluster distance) is denoted by **b**.

# In[6]:


silhouette_coefs = []
for cluster in range(2,6):
    
    kmeans = KMeans(n_clusters = cluster, init='k-means++', random_state=42)

    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
    visualizer.fit(scaled)
    visualizer.show()
    
    kmeans.fit(scaled)
    silhouette_coef = silhouette_score(scaled, kmeans.predict(scaled))
    silhouette_coefs.append(silhouette_coef)


# In[7]:


# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(2,6), 'sil_co':silhouette_coefs})
fig = plt.figure(figsize=(12,6))
ax = fig.gca()
plt.plot(frame['Cluster'], frame['sil_co'], marker='o', lw=2.5)
plt.xlabel('Number of clusters', fontsize = 14)
plt.ylabel('Silhouette Coefficient', fontsize = 14)
ax = fig.gca()
ax.xaxis.set_ticks(np.arange(2, 6, 1))
plt.grid()
plt.show()


# The Silhouette coefficient is maximized at k = 2.

# **Final Choice**

# Both the Elbow method and the Silhouette coefficient maximized, suggest 2 different number of clusters. Since, the elbow point is at k = 4, and the silhouette coefficient is maximized at k = 2, we'll split it down the middle and set k = 3.

# ## KMeans on Current Dataset

# In[8]:


kmeans_scale = KMeans(n_clusters=4, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(scaled)
print('KMeans Silhouette Score: {}'.format(silhouette_score(scaled, kmeans_scale.labels_, metric='euclidean')))
labels_scale = kmeans_scale.labels_


# In[9]:


clusters_scale = pd.concat([scaled, pd.DataFrame({'cluster':labels_scale}, index=scaled.index)], axis=1)
clusters_scale


# ## KMeans on Principal Components

# Prinicipal component analysis (PCA) is a method to reduce the dimensionality of a dataset. PCA retains most of the variance (typically 90%) of information from the original high-dimensional dataset, despite reducing the dimensionality. The information from the original features are "squeezed" into principal components (PCs). 2-3 PCs are usually chosen, as humans cannot visualize anything above 3 features.

# ### Determine Optimal Number of PCs

# In[10]:


#n_components=5 because there are 5 features that can be clustered on in processed_df
pca = PCA(n_components=5)
pca.fit(scaled)
variance = pca.explained_variance_ratio_
var = np.cumsum(np.round(variance, 3)*100)
fig = plt.figure(figsize=(12,6))
ax = fig.gca()
plt.ylabel('% Variance Explained', fontsize = 14)
plt.xlabel('# of Features', fontsize = 14)
plt.title('PCA Analysis', fontsize = 14)
plt.xlim(0,4)
ax.xaxis.set_ticks(np.arange(0, 5, 1))
plt.ylim(0,100.5)
plt.plot(var, marker='o', lw=2.5)
plt.show()


# The above plot shows the amount of variance each PC retains. The first 2 PCs encompass approximately 82% of the variance, while the first 3 PCs empompass approximately 89%. We'll choose 3 PCs, as it encopasses the most variance, while still being able to be visualized.

# ### Feature reduction using PCA

# In[11]:


pca = PCA(n_components=3)
pca_scale = pca.fit_transform(scaled)
pca_df = pd.DataFrame(pca_scale, columns=['pc1','pc2','pc3'])
pca_df.to_csv('../data/pca_components.csv')
pca_df.head()


# ### Determine Number of Clusters

# We once again need to determine a suitable number of clusters, this time for the reduced PCA dataframe. We'll repeat both the Elbow Method and calculate the Silhouette Coefficient.

# **Elbow Method**

# In[12]:


# fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
for cluster in range(1,15):
    kmeans = KMeans(n_clusters = cluster, init='k-means++', random_state=42)
    kmeans.fit(pca_df)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,15), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o', lw=2.5)
plt.xlabel('Number of clusters', fontsize = 14)
plt.ylabel('Inertia', fontsize = 14)
plt.grid()
plt.show()


# In[13]:


kl = KneeLocator(range(1, 15), SSE, curve="convex", direction="decreasing")
print('Elbow at k = ' + str(kl.elbow))


# **Silhouette Coefficient**

# In[14]:


silhouette_coefs = []
for cluster in range(2,15):
    
    kmeans = KMeans(n_clusters = cluster, init='k-means++', random_state=42)
    kmeans.fit(pca_df)
    silhouette_coef = silhouette_score(pca_df, kmeans.predict(pca_df))
    silhouette_coefs.append(silhouette_coef)


# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(2,15), 'sil_co':silhouette_coefs})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['sil_co'], marker='o', lw=2.5)
plt.xlabel('Number of clusters', fontsize = 14)
plt.ylabel('Silhouette Coefficient', fontsize = 14)
plt.grid()
plt.show()


# Combining the results from the elbow and silhouette method, we will set the number of clusters to 4.

# ### Kmeans

# In[15]:


kmeans_pca = KMeans(n_clusters=4, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(pca_df)
labels_pca = kmeans_pca.labels_
clusters_pca = pd.concat([pca_df, pd.DataFrame({'pca_clusters':labels_pca})], axis=1)
centroids_whole_dataset = kmeans_pca.cluster_centers_
pd.DataFrame(centroids_whole_dataset).to_csv('../data/centroids_entire.csv')


# **Visualize the Clusters**

# In[16]:


Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'),zaxis = dict(title  = 'PC3'))
labels = labels_pca
trace = go.Scatter3d(x=clusters_pca.pc1, y=clusters_pca.pc2, z=clusters_pca.pc3, mode='markers',marker=dict(color = labels, colorscale='Viridis', size = 10, line = dict(color = 'gray',width = 5)))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene, height = 1000,width = 1000)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()


# In[17]:


fig = plt.figure(figsize = (15,7))
fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig.figsize = (10,10)
plt.subplot(1,3,1)
sns.scatterplot(clusters_pca.iloc[:,0],clusters_pca.iloc[:,1], hue=labels_pca+1, palette='Set1', s=100, alpha=0.2).set_title('KMeans Clusters (5) Derived from PCA', fontsize=15)
plt.legend()
plt.subplot(1,3,2)
sns.scatterplot(clusters_pca.iloc[:,0],clusters_pca.iloc[:,1], hue=processed_df.reset_index().Study, palette='Set1', s=100, alpha=0.2).set_title('PCs Colour Coded by Study', fontsize=15)
plt.legend()
# plt.show()
# plt.figure(figsize = (10,10))
plt.subplot(1,3,3)
sns.scatterplot(clusters_pca.iloc[:,0],clusters_pca.iloc[:,1], hue=processed_df.reset_index().PayScheme, palette='Set1', s=100, alpha=0.2).set_title('PCs Colour Coded by Payoff Scheme', fontsize=15)
plt.legend()
plt.show()


# There doesn't appear to be any correlation between the clusers and what study the subjects were apart of. Similarly, there's no apparent relationship between the clusters and the payoff scheme applicable to the subject. This suggests, for this group of studies, that a subject's performance in the IGT is not affected by what study they partook in. Study organizers appear to have sucessfully conducted unbiased trials. Therefore, I would not object to this combinations of trials being grouped together and used as a control group for future analysis.

# In[ ]:




