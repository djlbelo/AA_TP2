# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 17:34:41 2022

@author: LuísAlmas
"""

from tp2_aux import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap


def fit_pca(x):
    x = x - x.mean()
    pca = PCA(n_components=6)
    #podemos dar só data = pca.fit_transform(x) ?
    pca.fit(x)
    data = pca.transform(x)
    
    #plots
    plt.bar(list(range(len(pca.explained_variance_ratio_))), pca.explained_variance_ratio_)
    plt.yscale('log')
    plt.show()
    print(data.shape)
    
    return data

def fit_kernel_pca(x):
    x = x - x.mean()
    kernel_pca = KernelPCA(n_components=6, kernel='rbf') #Error in n_component
    
    #podemos dar só data = kernel_pca.fit_transform(x)
    kernel_pca.fit(x)
    data = kernel_pca.transform(x)
    print(data.shape)
    return data
    
def isometric_mapping(x):
    isomap = Isomap(n_components=6)
    #podemos dar só data = isomap.fit_transform(x) ?
    isomap.fit(x)
    data = isomap.transform(x)
    print(data.shape)
    
    return data
    
def getData():
    try:
        data = np.load('features.npz')
        total_features = data['x']
    except IOError:
        x = images_as_matrix()
        x_pca = fit_pca(x)
        x_kPCA = fit_kernel_pca(x)
        x_iso = isometric_mapping(x)
        #missing other classes
        
        total_features = np.append(x_pca, np.append(x_kPCA,___,axis=1), axis=1)
        np.savez('features.npz', x = total_features)
    return total_features
    
features = getData()
#= select_features(features)
#data = tp2_aux.images_as_matrix()

fit_pca(data)
#fit_kernel_pca(data)
isometric_mapping(data)

