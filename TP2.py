# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 17:34:41 2022

@author: LuísAlmas
"""

import tp2_aux
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap


def fit_pca(x):
    x = x - x.mean()
    pca = PCA(n_components=6)
    pca.fit(x)
    data = pca.transform(x)
    plt.bar(list(range(len(pca.explained_variance_ratio_))), pca.explained_variance_ratio_)
    plt.yscale('log')
    plt.show()
    print(data.shape)
    
def fit_kernel_pca(x):
    x = x - x.mean()
    kernel_pca = KernelPCA(n_components=18, kernel='rbf') #Error in n_component
    kernel_pca.fit(x)
    data = kernel_pca.transform(x)
    print(data.shape)
    
def isometric_mapping(x):
    isomap = Isomap(n_components=18)
    isomap.fit(x)
    data = isomap.transform(x)
    print(data.shape)
    

data = tp2_aux.images_as_matrix()
fit_pca(data)
#fit_kernel_pca(data)
isometric_mapping(data)

