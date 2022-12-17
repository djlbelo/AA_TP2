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

def kmeans_predition(data, labels):
    labels_t = labels[labels>0]
    performance = [[],[],[],[],[]]
    bestScore = 0
    bestK = 0
    
    #check range
    
    for k in range(2,40):
        kmeans = KMeans(n_clusters = k).fit(data)
        predict = kmeans.predict(data)
        labels_pred = predict[labels>0]
        
        tp = tn = fp = fn = 0

        N = labels_t.shape[0]
    
        for x in range(N):
            for y in range(x+1,N):
                same_cluster = labels_pred[x] == labels_pred[y]
                same_group = labels_t[x] == labels_t[y]

                tp += np.sum(np.logical_and(same_cluster, same_group))
                tn += np.sum(np.logical_and(np.logical_not(same_cluster), np.logical_not(same_group)))
                fp += np.sum(np.logical_and(same_cluster, np.logical_not(same_group)))
                fn += np.sum(np.logical_and(np.logical_not(same_cluster), same_group))
        
        #check purity        
        purity = (1/N)*(tp)
        
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
    
    print("Best score {} and best k {}".format(bestScore, bestK))
    kmeans = KMeans(n_clusters=bestK).fit(data)
    predictY = kmeans.predict(data)
    report_clusters(np.array(list(range(0, predictY.shape[0]))), predictY, "kmeans.html")    
    
    plot_clusters('K-Means Analysis', 'clusters (k)', range(2,40), performance)

def plot_clusters(title, xlabel, xaxis, performance):
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.plot(xaxis, perfomance[0], label='Purity')
    plt.plot(xaxis, perfomance[1], label='Precision')
    plt.plot(xaxis, perfomance[2], label='Recall')
    plt.plot(xaxis, perfomance[3], label='F1 Measure')
    plt.plot(xaxis, perfomance[4], label='Rand Index')
    
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

def select_features(total_features, kF):
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
    bestFeatures = SelectKBest(f_classif,k=kF)
    
    mat_idx = bestFeatures.get_support()
    total_features = total_features[:,mat_idx]
    
    return total_features, bestFeatures, labels

def feature_mean_stdv(bestFeatures):
    auxBestFeatures = bestFeatures[:,0:-1:]
    meanFeats = np.mean(auxBestFeatures,0);
    stdevFeats = np.std(auxBestFeatures,0);
    return meanFeats, stdevFeats

x = images_as_matrix()
x_pca = fit_pca(x)
x_kPCA = fit_kernel_pca(x)
x_iso = isometric_mapping(x)
  
features = getData(x_pca, x_kPCA, x_iso)
n_features = int(input("Number of features to select?"))
features, bestFeatures, labels = select_features(features, k=n_features)

#standardized best features
scaler = StandardScaler
bestFeatures = scaler.fit_transform(bestFeatures)

#ids
ids = labels[:,0]
    
#missing preformance and labels
kmeans_predition(bestFeatures, labels[:, 1])

#clusters
hierarchical_cluster(bestFeatures)

spectral_cluster(bestFeatures)

kMeans_cluster(bestFeatures)

# report clusters kMeans
n_cluster = int(input("Number of clusters for KMEAN?"))
kmeans = kMeans_cluster(n_clusters = n_cluster)
labels_kmeans = kmeans.predict(bestFeatures)
report_clusters(ids, labels_kmeans, "KMEANS-" + str(n_cluster) + ".html")

report_clusters(ids, labels, "file_.html")
report_clusters(ids, labels, "file_.html")