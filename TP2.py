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
from sklearn import metrics


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
def hierarchical_cluster(data, n_cluster):
    labels = AgglomerativeClustering(n_clusters = n_cluster).fit_predict(data)
    
    plt.title("Hierarchical scatter")
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()
    
    return labels
    
def spectral_cluster(data, n_cluster):
    labels = SpectralClustering(n_clusters = n_cluster, assign_labels='cluster_qr').fit_predict(data)
    
    plt.title("Spectral scatter")
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()
    
    return labels
    

def kMeans_cluster(data, n_cluster):
    labels = KMeans(n_clusters = n_cluster).fit_predict(data)
        
    plt.title("Kmeans scatter")
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.show()
    
    return labels

def cluster_predition(data, labels, clusterType):
    performance = [[],[],[],[],[]]
    bestScore = 0
    bestK = 0
    
    #check range
    
    for k in range(2,20):
        if clusterType == 'kMeans':
            predict = KMeans(n_clusters = k).fit_predict(data)
            plot_title = 'K-Means Analysis'
        elif clusterType == 'hierarchical':
            predict = AgglomerativeClustering(n_clusters = k).fit_predict(data)
            plot_title = 'Hierarchical agglomerative Analysis'
        elif clusterType == 'spectral':
            predict = SpectralClustering(n_clusters = k, assign_labels='cluster_qr').fit_predict(data)
            plot_title = 'Spectral Analysis'
        else:
            print("No Cluster type recognized")
            return
        
        tp = tn = fp = fn = 0

        N = labels.shape[0]
    
        for x in range(N):
            for y in range(x+1,N):
                same_cluster = predict[x] == predict[y]
                same_group = labels[x] == labels[y]

                tp += np.sum(np.logical_and(same_cluster, same_group))
                tn += np.sum(np.logical_and(np.logical_not(same_cluster), np.logical_not(same_group)))
                fp += np.sum(np.logical_and(same_cluster, np.logical_not(same_group)))
                fn += np.sum(np.logical_and(np.logical_not(same_cluster), same_group))
        
        #check purity        
        purity = purity_score(labels, predict)
        
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * ( (precision * recall) / (precision + recall))
        rand = (tp+tn)/(N*(N-1)/2)
        
        if(rand > bestScore):
            bestScore = rand
            bestK = k
        
        performance[0].append(purity)
        performance[1].append(precision)
        performance[2].append(recall)
        performance[3].append(f1)
        performance[4].append(rand)
    
    print("Best score for {}: {} and best k {}".format(clusterType, bestScore, bestK))
    kmeans = KMeans(n_clusters=bestK).fit(data)
    predictY = kmeans.predict(data)
    report_clusters(np.array(list(range(0, predictY.shape[0]))), predictY, clusterType+".html")    
    
    plot_clusters(plot_title, 'clusters (k)', range(2,20), performance)

def purity_score(labels, predict):
    contingency_matrix = metrics.cluster.contingency_matrix(labels, predict)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

def plot_clusters(title, xlabel, xaxis, performance):
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.plot(xaxis, performance[0], label='Purity')
    plt.plot(xaxis, performance[1], label='Precision')
    plt.plot(xaxis, performance[2], label='Recall')
    plt.plot(xaxis, performance[3], label='F1 Measure')
    plt.plot(xaxis, performance[4], label='Rand Index')
    
    plt.legend()
    plt.savefig(title, dpi=300)
    plt.show()
    plt.close()
    
#data
def getData(x_pca, x_kPCA, x_iso):
    try:
        data = np.load('features.npz')
        total_features = data['x']
    except IOError:
        total_features = np.append(x_pca, np.append(x_kPCA, x_iso, axis=1), axis=1)
        np.savez('features.npz', x = total_features)
    return total_features

def get_data_set(features, labels):
    labeled_labels = labels[labels[:,1] != 0]
    labeled_features = labeled_data = np.array([features[int(i)] for i in labeled_labels[:,0]])
    return labeled_labels, labeled_features

x = images_as_matrix()
x_pca = fit_pca(x)
x_kPCA = fit_kernel_pca(x)
x_iso = isometric_mapping(x)
  
features = getData(x_pca, x_kPCA, x_iso)
labels = np.loadtxt('labels.txt', delimiter=',', dtype='int')

means = np.mean(features,axis=0)
stdevs = np.std(features,axis=0)
features = (features-means)/stdevs

labeled_labels, labeled_features = get_data_set(features, labels)

#clusters
#hierarchical_cluster(features, 3)

#spectral_cluster(features, 3)

cluster_predition(labeled_features, labeled_labels[:, 1], 'kMeans')

cluster_predition(labeled_features, labeled_labels[:, 1], 'hierarchical')

cluster_predition(labeled_features, labeled_labels[:, 1], 'spectral')