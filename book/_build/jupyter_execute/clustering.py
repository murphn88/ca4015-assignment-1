#!/usr/bin/env python
# coding: utf-8

# # Clustering

# Clustering is an unsupervised learning technique that divides the entire dataset into groups, or clusters, based on patterns in the data.

# **K-means algorithm**

# K-means clustering is one of the simplest and most popular machine learning algorithms. K-means groups the data into a defined number (k) of clusters.   
# k-means algorithm:

# 1. Specify the number of clusters k
# 2. Select k random points from the data as the initial centroids
# 3. Allocate all other points to the nearest cluster centroid
# 4. Recompute the centroids of the new clusters
# 5. Repeat steps 3 and 4, until either:
#     -  the centroids have stabilized (don't change between interations) or
#     - the maximum number of iterations have completed

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
import plotly.graph_objs as go
import seaborn as sns


# In[2]:


processed_df = pd.read_csv('../data/processed.csv', index_col=0)
processed_df.head()                        


# ## Determine Number of Clusters

# To choose an appropiate number of clusters, we will use two common methods, the Elbow Method and the Silhouette Coefficient. 

# ### The Elbow Method

# Involves running several k-means, incrementing k with each iteration, and note the Sum of the Squared Error (SSE) each time.

# In[3]:


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


# Inertia decreases as the number of clusters increase. We're looking for the sweet spot (elbow point) where the curve starts to bend. Here, the elbow looks to be located at k = 2.

# **Confirm Elbow using kneed**

# In[4]:


kl = KneeLocator(range(1, 15), SSE, curve="convex", direction="decreasing")
print('Elbow at k = ' + str(kl.elbow))


# ### The Silhouettee coefficient

# Measures how close each data point in one cluster is to data points in neighbouring clusters. 

# The formula for the Silhouette score, S, is:
# $$
#   S = \frac{b-a}{max(a,b)}
# $$
# Where:
# - the mean distance between the observation and all other data points in the same cluster (i.e., intra-cluster distance) is denoted by **a**, and
# - The mean distance between the observation and all other data points of the next nearest cluster (mean nearest-cluster distance) is denoted by **b**.

# In[5]:


for cluster in range(2,6):
    
    kmeans = KMeans(n_clusters = cluster, init='k-means++', random_state=42)

    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
    visualizer.fit(processed_df.iloc[:,2:])
    visualizer.show()


# In[6]:


# optimal from this seems to be 3, so some comment about how many clusters to pick


# ## K-means

# In[7]:


from yellowbrick.cluster import intercluster_distance

kmeans = KMeans(n_clusters = 4, init='k-means++', random_state=42)
intercluster_distance(kmeans, 
                      processed_df.iloc[:,2:], 
                      embedding='mds', 
                      random_state=12) # other option for embedding 'tsne'


# ## KMeans on Original Dataset

# In[8]:


kmeans_orig = KMeans(n_clusters=4, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(processed_df.iloc[:,2:])
print('KMeans Scaled Silhouette Score: {}'.format(silhouette_score(processed_df.iloc[:,2:], kmeans_orig.labels_, metric='euclidean')))
labels_orig = kmeans_orig.labels_
clusters_orig = pd.concat([processed_df.iloc[:,2:], pd.DataFrame({'cluster':labels_scale}, index=processed_df.index)], axis=1)


# In[121]:


clusters_orig


# In[125]:


pca2 = PCA(n_components=3).fit(processed_df.iloc[:,2:])
pca2d = pca2.transform(processed_df.iloc[:,2:])
plt.figure(figsize = (10,10))
sns.scatterplot(pca2d[:,0], pca2d[:,1], 
                hue=labels_orig, 
                palette='Set1',
                s=100, alpha=0.2).set_title('KMeans Clusters (4) Derived from Original Dataset', fontsize=15)
plt.legend()
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.show()


# In[35]:


Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'),zaxis = dict(title  = 'PC3'))
labels = labels_scale
trace = go.Scatter3d(x=pca2d[:,0], y=pca2d[:,1], z=pca2d[:,2], mode='markers',marker=dict(color = labels, colorscale='Viridis', size = 10, line = dict(color = 'gray',width = 5)))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene, height = 1000,width = 1000)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()


# ## KMeans on Principal Components

# Prinicipal component analysis (PCA) is a method to reduce the dimensionality of a dataset. PCA retains most of the variance (typically 90%) of information from the original high-dimensional dataset, despite reducing the dimensionality. The information from the original features are "squeezed" into principal components (PCs). 2-3 PCs are usually chosen, as humans cannot visualize anything above 3 features.

# ### Determine Optimal Number of PCs

# In[47]:


#n_components=5 because there are 5 features that can be clustered on in processed_df
pca = PCA(n_components=5)
pca.fit(processed_df.iloc[:,2:])
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
plt.plot(var, lw=2.5)
plt.show()


# The above plot shows the amount of variance each PC retains. The first 2 PCs encompass approximately 94% of the variance, while the first 3 PCs empompass approximately 98%. 94% is satisfactory, so we'll stick to 2 PCs.

# ### Feature reduction using PCA

# In[152]:


pca = PCA(n_components=3)
pca_scale = pca.fit_transform(processed_df.iloc[:,2:])
pca_df = pd.DataFrame(pca_scale, columns=['pc1','pc2','pc3'])
pca_df.head()


# ### Determine Number of Clusters

# We once again need to determine a suitable number of clusters, this time for the reduced PCA dataframe. We'll repeat both the Elbow Method and calculate the Silhouette Coefficient.

# **Elbow Method**

# In[153]:


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


# In[154]:


kl = KneeLocator(range(1, 15), SSE, curve="convex", direction="decreasing")
print('Elbow at k = ' + str(kl.elbow))


# **Silhouette Coefficient**

# In[155]:


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


# **Kmeans**

# In[173]:


kmeans_pca = KMeans(n_clusters=4, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(pca_df)
labels_pca = kmeans_pca.labels_
clusters_pca = pd.concat([pca_df, pd.DataFrame({'pca_clusters':labels_pca})], axis=1)


# In[174]:


plt.figure(figsize = (10,10))
sns.scatterplot(clusters_pca.iloc[:,0],clusters_pca.iloc[:,1], hue=labels_pca+1, palette='Set1', s=100, alpha=0.2).set_title('KMeans Clusters (4) Derived from PCA', fontsize=15)
plt.legend()
plt.show()


# In[130]:


plt.figure(figsize = (10,10))
sns.scatterplot(clusters_pca.iloc[:,0],clusters_pca.iloc[:,1], hue=processed_df.reset_index().Study, palette='Set1', s=100, alpha=0.2).set_title('KMeans Clusters (4) Derived from PCA', fontsize=15)
plt.legend()
plt.show()


# In[131]:


plt.figure(figsize = (10,10))
sns.scatterplot(clusters_pca.iloc[:,0],clusters_pca.iloc[:,1], hue=processed_df.reset_index().PayScheme, palette='Set1', s=100, alpha=0.2).set_title('KMeans Clusters (4) Derived from PCA', fontsize=15)
plt.legend()
plt.show()


# In[98]:


processed_df


# In[90]:


processed_df[['Study']].reset_index(drop=True)


# In[75]:


clusters_pca


# In[148]:


pd.DataFrame(labels_orig).value_counts()


# In[150]:


pd.DataFrame(labels_pca).value_counts()


# In[185]:


Scene = dict(xaxis = dict(title  = 'PC1'),yaxis = dict(title  = 'PC2'),zaxis = dict(title  = 'PC3'))
labels = labels_pca
trace = go.Scatter3d(x=clusters_pca.pc1, y=clusters_pca.pc2, z=clusters_pca.pc3, mode='markers',marker=dict(color = labels, colorscale='Viridis', size = 10, line = dict(color = 'gray',width = 5)))
layout = go.Layout(margin=dict(l=0,r=0),scene = Scene, height = 1000,width = 1000)
data = [trace]
fig = go.Figure(data = data, layout = layout)
fig.show()


# In[145]:


pd.DataFrame(labels_orig == labels_pca).value_counts()


# In[156]:


kmeans_pca = KMeans(n_clusters=4, n_init=100, max_iter=400, init='k-means++', random_state=42).fit(pca_df)
labels_pca = kmeans_pca.labels_
clusters_pca = pd.concat([pca_df, pd.DataFrame({'pca_clusters':labels_pca})], axis=1)


# In[167]:


clusters_pca

