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
    pca.fit(x)
    data = pca.transform(x)
    return data

def fit_kernel_pca(x):
    x = x - x.mean()
    kernel_pca = KernelPCA(n_components=6, kernel='rbf')
    kernel_pca.fit(x)
    data = kernel_pca.transform(x)
    return data
    
def isometric_mapping(x):
    isomap = Isomap(n_components=6)
    isomap.fit(x)
    data = isomap.transform(x)    
    return data
        

def cluster_evaluation(data, labels, clusterType):
    performance = [[],[],[],[],[]]
    bestScore = 0
    bestK = 0
    
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
                tp += np.sum(np.logical_and(predict[x] == predict[y], labels[x] == labels[y]))
                tn += np.sum(np.logical_and(np.logical_not(predict[x] == predict[y]), np.logical_not(labels[x] == labels[y])))
                fp += np.sum(np.logical_and(predict[x] == predict[y], np.logical_not(labels[x] == labels[y])))
                fn += np.sum(np.logical_and(np.logical_not(predict[x] == predict[y]), labels[x] == labels[y]))
               
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
    
    print("\n\nBest k for {}: {}".format(clusterType, bestK))
    print("Best score for {}: {}".format(clusterType, bestScore))
    plot_clusters(plot_title, 'clusters (k)', range(2,20), performance)
    
    return bestK

def bissectingKmeans(data, n):
    performance = []
    ids = np.zeros(data.shape[0])
    for i in range(0, data.shape[0]):
        performance.append([])
        ids[i] = i
    
    maxSize = data
    for i in range(0,n):
        labels = KMeans(n_clusters=2).fit_predict(maxSize)
        for j in range(0,maxSize.shape[0]): 
            performance[int(ids[j])].append(labels[j])
        
            
        n0 = np.count_nonzero(labels==0)
        n1 = np.count_nonzero(labels==1)
        if n0>=n1:
            maxSize = maxSize[labels==0,:]
            ids = ids[labels==0]
        else:
            maxSize = maxSize[labels==1,:]
            ids = ids[labels==1]
            
    return performance

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
    total_features = np.append(x_pca, np.append(x_kPCA, x_iso, axis=1), axis=1)
    return total_features

def get_data_set(features, labels):
    labeled_labels = labels[labels[:,1] != 0]
    labeled_features = labeled_data = np.array([features[int(i)] for i in labeled_labels[:,0]])
    return labeled_labels, labeled_features

#data handling and feature reduction
x = images_as_matrix()
x_pca = fit_pca(x)
x_kPCA = fit_kernel_pca(x)
x_iso = isometric_mapping(x)
  
features = getData(x_pca, x_kPCA, x_iso)
labels = np.loadtxt('labels.txt', delimiter=',', dtype='int')

#standardization
means = np.mean(features,axis=0)
stdevs = np.std(features,axis=0)
features = (features-means)/stdevs

#filtering by labelled data
labeled_labels, labeled_features = get_data_set(features, labels)

f, prob = f_classif(labeled_features, labeled_labels[:,1])
#print(f)
#print(prob)

plt.plot(range(f.shape[0]), f, "x")
delim = 5
plt.plot([0, f.shape[0]], [delim, delim])
plt.savefig("f-test", dpi=200)
plt.show()
plt.close()

kbest = SelectKBest(f_classif, k=4)
x_features = kbest.fit_transform(labeled_features, labeled_labels[:,1])
x_id = kbest.get_support()
features = features[:,x_id]

#best k for each algorithm and performance plotting
bestKmeans = cluster_evaluation(labeled_features[:,x_id], labeled_labels[:, 1], 'kMeans')
bestKhier = cluster_evaluation(labeled_features[:,x_id], labeled_labels[:, 1], 'hierarchical')
bestKspectral = cluster_evaluation(labeled_features[:,x_id], labeled_labels[:, 1], 'spectral')

#html reports
predictY = KMeans(n_clusters=bestKmeans).fit_predict(features)
report_clusters(np.array(list(range(0, predictY.shape[0]))), predictY, "kMeans.html")

predictX = AgglomerativeClustering(n_clusters=bestKhier).fit_predict(features)
report_clusters(np.array(list(range(0, predictX.shape[0]))), predictX, "hierarchical.html")     

predictZ = SpectralClustering(n_clusters=bestKspectral, assign_labels='cluster_qr').fit_predict(features)
report_clusters(np.array(list(range(0, predictZ.shape[0]))), predictZ, "spectral.html") 


label_lists = bissectingKmeans(features, 4)
report_clusters_hierarchical(list(range(0, len(label_lists))), label_lists, "BissectingKMeans.html")