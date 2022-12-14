# -*- coding: utf-8 -*-
"""
Machine Learning FCT NOVA - Assignment 2

@author: Luís Almas
@author: Duarte Belo

"""

from tp2_aux import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans


def fit_pca(x):
    x = x - x.mean()
    pca = PCA(n_components=6)
    #podemos dar só data = pca.fit_transform(x) ?
    pca.fit(x)
    data = pca.transform(x)
    #print(pca.explained_variance_ratio_.cumsum()) cumulative sum of variance
    return data

def fit_kernel_pca(x):
    x = x - x.mean()
    kernel_pca = KernelPCA(n_components=6, kernel='rbf') #Error in n_component
    #podemos dar só data = kernel_pca.fit_transform(x)
    kernel_pca.fit(x)
    data = kernel_pca.transform(x)
    return data
    
def isometric_mapping(x):
    isomap = Isomap(n_components=6)
    #podemos dar só data = isomap.fit_transform(x) ?
    isomap.fit(x)
    data = isomap.transform(x)    
    return data

# clusters
def hierarchical_cluster(data):
    labels = AgglomerativeClustering(n_clusters=3).fit_predict(data)
    
    plt.title("Hierarchical scatter")
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()
    
    return labels
    
def spectral_cluster(data):
    labels = SpectralClustering(n_clusters=3,assign_labels='cluster_qr').fit_predict(data)
    
    plt.title("Spectral scatter")
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()
    
    return labels
    
def kMeans_cluster(data):
    labels = KMeans(n_clusters=3).fit_predict(data)
    
    plt.title("Kmeans scatter")
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()
    
    return labels
    
#data
def getData():
    try:
        data = np.load('features.npz')
        total_features = data['x']
    except IOError:
        x = images_as_matrix()
        x_pca = fit_pca(x)
        x_kPCA = fit_kernel_pca(x)
        x_iso = isometric_mapping(x)
        total_features = np.append(x_pca, np.append(x_kPCA, x_iso, axis=1), axis=1)
        np.savez('features.npz', x = total_features)
    return total_features, x_pca, x_kPCA, x_iso

def select_features(features):
    #should be double?
    mat = np.loadtxt('labels.txt', delimiter=',', dtype='int');
    #matrix
    mx = mat[mat[:,1]>0,:][:,0]
    labels = mat[mat[:,1]>0,:,1]
    
    #fazer data frame ou
    #new_x = SelectKBest(f_classif, k=2).fit_transform(x, y)
    #plot_iris(new_x, y, "Best_K_classify")
    
    f, prob = f_classif(features[mx,],labels)
    
    bestFeatures = features[:,f>19]
    #necessary?
    bestLabelled = best_Features[mat[:,1]>0,:]
    
    return bestFeatures

def feature_mean_stdv(bestFeatures):
    bestFeatures = bestFeatures[:,0:-1:]
    meanFeats = np.mean(bestFeatures,0);
    stdevFeats = np.std(bestFeatures,0);
    return bestFeatures, meanFeats, stdevFeats

def getDistances(mat,k):
    kneighbors = KNeighborsClassifier(n_neighbors=k)
    kneighbors.fit(mat, np.zeros(mat.shape[0]))
    neigh_dist, neigh_ind = kneighbors.kneighbors(return_distance=True)
    dist_graph = np.zeros([mat.shape[0]])
    for ix in range(0,mat.shape[0]):
        dist_graph[ix]=neigh_dist[ix][k-1]
    np.ndarray.sort(dist_graph)
    return dist_graph[::-1]
    
features, pca_data, kernel_pca_data, isometric_data = getData()
bestFeatures = select_features(features)
bestFeatures, meanFeats, stdevFeats = feature_mean_stdv(bestFeatures)
stdBestFeatures = (bestFeatures-meanFeats)/stdevFeats;

# should be 5?
dbsGraph = getDistances(stdBestFeatures, 5)

#ids
ids = np.zeros(dbsGraph.shape[0])
for ix in range(0, len(ids)):
    ids[ix] = ix
    
#missing preformance and labels
    
#clusters
hierarchical_cluster(pca_data)
hierarchical_cluster(kernel_pca_data)
hierarchical_cluster(isometric_data)

spectral_cluster(pca_data)
spectral_cluster(kernel_pca_data)
spectral_cluster(isometric_data)

kMeans_cluster(pca_data)
kMeans_cluster(kernel_pca_data)
kMeans_cluster(isometric_data)

# report clusters, missing labels
report_clusters(ids, labels, "file_.html")
report_clusters(ids, labels, "file_.html")
report_clusters(ids, labels, "file_.html")