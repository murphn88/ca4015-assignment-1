#!/usr/bin/env python
# coding: utf-8

# # K-Means Privacy Preservation 

# We will now try to cluster the dataset, whilst maintaining the privacy of individual labs/ study. There will be no sharing of subjects' data between labs. Each lab will "report" clusters and those will be shared among labs. To further protect privacy, and to improve clustering results, we will cluster on the principal components.

# Method:
#   - For each study use the elbow method to deteremine a suitable number of cluseters, k.
#   - Combine all of the clusters from all of the studies.
#   - Cluster on the clusters.

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
from scipy.spatial import distance


# In[2]:


processed_df = pd.read_csv('../data/processed.csv', index_col=0)
pca_df = pd.read_csv('../data/pca_components.csv', index_col=0)
centroids_prev = pd.read_csv('../data/centroids_entire.csv', index_col=0)
pca_df.head()


# ## Seperate on Study and Perform K-Means Individually

# In[3]:


study_pca = pd.concat([processed_df.Study, pca_df], axis=1)
study_pca.head()


# In[4]:


all_centroids = []
for study,values in study_pca.groupby('Study'):
    
    pcs = values.iloc[:, 1:]
    
    SSE = []
    for cluster in range(1,8):
        kmeans = KMeans(n_clusters = cluster, init='k-means++', random_state=42)
        kmeans.fit(pcs)
        SSE.append(kmeans.inertia_)
        
    k = KneeLocator(range(1, 8), SSE, curve="convex", direction="decreasing").elbow
    
    kmeans_pca = KMeans(n_clusters=k, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(pcs)
    labels_pca = kmeans_pca.labels_

    centroids = kmeans.cluster_centers_[:k].tolist()
    for i in range(0,k):
        all_centroids.append(centroids[i])    


# In[5]:


centroids = pd.DataFrame(all_centroids)
centroids.head()


# In[6]:


kmeans_comb = KMeans(n_clusters=4, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(centroids)
labels_comb = kmeans_comb.labels_
clusters_comb = pd.concat([centroids, pd.DataFrame({'_clusters':labels_comb})], axis=1)


# In[7]:


Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'),zaxis = dict(title  = 'PC3'))
labels = labels_comb
trace = go.Scatter3d(x=clusters_comb[0], y=clusters_comb[1], z=clusters_comb[2], mode='markers',marker=dict(color = labels, colorscale='Viridis', size = 10, line = dict(color = 'gray',width = 5)))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene, height = 1000,width = 1000)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()


# **Compare to K-means on entire PCA dataset**

# In[8]:


centroids_prev


# In[9]:


centroids_curr = pd.DataFrame(kmeans_comb.cluster_centers_)
centroids_curr.head()


# **Comments**  
# This is a very simplistic approach to perservinig privacy through K-means. A more sophisticated approach would likely give a weighting to how many datapoints are in each cluster. It would also iterate through each of the studies multiple times, each time adjusting the clusters based on the centroids from the other studies.

# In[ ]:




