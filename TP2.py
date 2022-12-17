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
from sklearn.preprocessing import StandardScaler
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
    
def spectral_cluster(data, n_cluster):
    labels = SpectralClustering(n_clusters=n_cluster,assign_labels='cluster_qr').fit_predict(data)
    
    plt.title("Spectral scatter")
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()
    
    return labels
    
def kMeans_cluster(data, n_cluster):
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

def select_features(total_features):
    #should be double?
    mat = np.loadtxt('labels.txt', delimiter=',', dtype='int');
    
    #matrix
    labels = mat[mat[:,1] != 0]
    labeled_data = np.array([total_features[int(i)] for i in labels[:,0]])
    
    #f-value to choose in the 18 features
    f, prob = f_classif(labeled_data,labels[:,1])
    print(f)
    print(prob)
    #how many features?
    bestFeatures = SelectKBest(f_classif,k=5)
    
    mat_idx = bestFeatures.get_support()
    total_features = total_features[:,mat_idx]
    
    return total_features, bestFeatures, labels

def feature_mean_stdv(bestFeatures):
    auxBestFeatures = bestFeatures[:,0:-1:]
    meanFeats = np.mean(auxBestFeatures,0);
    stdevFeats = np.std(auxBestFeatures,0);
    return meanFeats, stdevFeats

    
features, pca_data, kernel_pca_data, isometric_data = getData()
features, bestFeatures, labels = select_features(features)

#standardized best features
scaler = StandardScaler
bestFeatures = scaler.fit_transform(bestFeatures)

#ids
ids = labels[:,0]
    
#missing preformance and labels


#clusters
hierarchical_cluster(bestFeatures)

spectral_cluster(bestFeatures)

kMeans_cluster(bestFeatures)

# report clusters kMeans
n_cluster = int(input("Number of clusters for KMEAN?"))
labels_kmeans = kMeans_cluster(bestFeatures, n_clusters = n_cluster)
report_clusters(ids, labels_kmeans, "KMEANS-" + str(n_cluster) + ".html")

report_clusters(ids, labels, "file_.html")
report_clusters(ids, labels, "file_.html")