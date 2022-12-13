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
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
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

def hierarchical_cluster(data):
    labels = AgglomerativeClustering(n_clusters=3).fit_predict(data)
    plt.title("Hierarchical scatter")
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()
    
def spectral_cluster(data):
    labels = SpectralClustering(n_clusters=3,assign_labels='cluster_qr').fit_predict(data)
    plt.title("Spectral scatter")
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()
    
    
def kMeans_cluster(data):
    labels = KMeans(n_clusters=3).fit_predict(data)
    plt.title("Kmeans scatter")
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()
    
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
    return total_features

def select_features(features):
    #should be double?
    mat = np.loadtxt('labels.txt', delimiter=',', dtype='double');
    #matrix
    mx = mat[mat[:,1]>0,:][:,0]
    labels = mat[mat[:,1]>0,:,1]
    
    f, prob = f_classif(X[mx,],labels)
    
    bestFeatures = features[:,f>19]
    bestLabelled = best_features[mat[:,1]>0,:]
    
    return bestFeatures, labels, bestLabelled, f[f>19]
    
features = getData()
bestFeatures, labels, bestLabelled, f = select_features(features)
bestFeats = pd.DataFrame(bestFeatures)

#scatter matrix
#to-do

labels = get_labels_from_txt()
data = tp2_aux.images_as_matrix()
#print(labels)

pca_data = fit_pca(data)
kernel_pca_data =  fit_kernel_pca(data)
isometric_data = isometric_mapping(data)

hierarchical_cluster(pca_data)
hierarchical_cluster(kernel_pca_data)
hierarchical_cluster(isometric_data)

spectral_cluster(pca_data)
spectral_cluster(kernel_pca_data)
spectral_cluster(isometric_data)

kMeans_cluster(pca_data)
kMeans_cluster(kernel_pca_data)
kMeans_cluster(isometric_data)

# report clusters, missing matrix, labels
report_clusters(ids, labels, "file_.html")
report_clusters(ids, labels, "file_.html")
report_clusters(ids, labels, "file_.html")