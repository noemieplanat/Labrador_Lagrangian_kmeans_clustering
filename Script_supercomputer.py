# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:16:30 2022

@author: Noemie Planat and Mathilde Jutras
contact: mathilde.jutras@mail.mcgill.ca
license: GNU General Public License v3.0

Please acknowledge using the following DOI:

This is the script used to perform the unsupervised clustering analysis
on geospatial Lagrangian trajectories detailed in Jutras, Planat & Dufour (2022).

#############################
### Second script to run.
#############################

This portion of the script contains the kernalized k-means model building,
and has necessitates a lot of computational resources. It was ran on a
supercomputer.
"""

import xarray as xr
import numpy as np
import random
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.KernelPCA.html
from sklearn.model_selection import KFold
# https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
#import wpca
# https://github.com/jakevdp/wpca
import time
from sklearn import cluster, datasets
# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
from sklearn.preprocessing import normalize
from sklearn.metrics import silhouette_samples, silhouette_score, calinski_harabasz_score
import os

# Extract the data
def extract(Dataset_path, length_days) :

    ds = xr.open_dataset(Dataset_path, engine='netcdf4')
    lats = ds.lat[:,0:length_days].values # keep only one year of data
    lons = ds.lon[:,0:length_days].values

    return lats, lons

# Translate all the data to one stating point
def translate(lat, lon):
    return lat-np.repeat(lat[:,0][:, np.newaxis], lat.shape[1], axis=1), lon-np.repeat(lon[:,0][:, np.newaxis], lon.shape[1], axis=1)

# Resample so the data all has the same length
def resample(lats, lons, length_days):

    if length_days != None:

        lats_resamp = np.zeros(lats.shape)
        lons_resamp = np.zeros(lats.shape)
        for i in range(lats.shape[0]):
            latsl = lats[i,:]#lats_proj ?
            latsl = latsl[~np.isnan(latsl)]
            xp = np.linspace(0,len(latsl),length_days)
            lats_resamp[i,:] = np.interp(xp, range(len(latsl)), latsl)
            lonsl = lons[i,:] #lons_proj ?
            lonsl = lonsl[~np.isnan(lonsl)]
            lons_resamp[i,:] = np.interp(xp, range(len(lonsl)), lonsl)

    else:
        lats_resamp = lats
        lons_resamp = lons

    return lats_resamp, lons_resamp

def weightPCA(lons, lats, wtype):

    if wtype == 'cos' :
        w = np.cos( np.radians(lats) )
    elif wtype == 'sqrt-cos' :
        w = np.sqrt(abs(np.cos( np.radians(lats) )))
    elif wtype == None:
        w = []

    return w

def PCA_a(Xlat_lon, n_components, copy, whiten, svd_solver, w, kernel, degreek):

    #print('w', w.shape, Xlat_lon.shape)
    # weight PCA
    if len(w) > 0 and kernel != None:
        s = int(Xlat_lon.shape[1]/2)
        Xlat_lon[:,s:] = Xlat_lon[:,s:]*w

    pca = PCA(n_components=n_components, copy=copy, whiten=whiten, svd_solver = svd_solver)
    X_reduced = pca.fit_transform(Xlat_lon)
    N_features_PCA = pca.n_features_
    N_samples_PCA = pca.n_samples_
    N_components_PCA = pca.n_components_
    Explained_variance_ratio_PCA = pca.explained_variance_ratio_
    Singular_values_PCA = pca.singular_values_
    print('The number of initial features was: ', N_features_PCA)
    print('The number of selected features is: ', N_components_PCA)
    print('The number of samples is: ', N_samples_PCA)
    print('The explained variance desired is:', n_components, '%, the obtained variance explained is: ', np.sum(Explained_variance_ratio_PCA), '%')
    Components = pca.components_

    if len(w) > 0 and kernel == None: # the wpca package does not work with kernels
        pca = wpca.WPCA(n_components=N_components_PCA).fit(Xlat_lon, w)
        X_reduced = pca.fit_transform(Xlat_lon, w)
        Components = pca.components_

    if kernel != None:

        pca = KernelPCA(n_components=N_components_PCA, copy_X=copy, eigen_solver=svd_solver, kernel=kernel, degree=degreek, fit_inverse_transform=True)
        X_reduced = pca.fit_transform(Xlat_lon)

    return X_reduced, pca

# k-means
def k_means_X_cv(n_splits, X_reduced, n_clusters, init, nmb_initialisations, max_iter, tol, algorithm, verbose, sample_weight):

    kf = KFold(n_splits=n_splits, shuffle=True)
    avg_silouhette = 0
    i = 0
    for train_index, val_index in kf.split(X_reduced):
        print('fold nmb ', i)
        i+=1
        X_train, X_val = X_reduced[train_index,:], X_reduced[val_index, :]
        k_means = cluster.KMeans(n_clusters=n_clusters, init=init, n_init = nmb_initialisations, max_iter = max_iter, tol = tol, algorithm = algorithm, verbose = verbose)
        k_means.fit(X_train, sample_weight = sample_weight)
        X_centers = k_means.cluster_centers_
        labs_val = predict(X_val, X_centers)
        a_temp = silhouette_score(X_val, labs_val)
        if a_temp>avg_silouhette:
            X_centered_memory = X_centers
            k_means_memory = k_means
            avg_silouhette = a_temp

    return X_centered_memory, k_means_memory

# Main script
def run_script(norm, t, resamp, length_days, wtype, kernel, degreek, lats_valid, lons_valid, lats_train, lons_train, lats_test, lons_test) :

    print('run script')
    # PRE-PROCESSING
    # Apply projection lats_projs/lons_projs are in x/y space, in m. on both Train and Test sets
    lats_train_pp, lons_train_pp = lats_train, lons_train
    lats_valid_pp, lons_valid_pp = lats_valid, lons_valid
    lats_test_pp, lons_test_pp   = lats_test, lons_test
    # Translate
    if t == True :
        lats_train_pp, lons_train_pp = translate(lats_train_pp, lons_train_pp)
        lats_valid_pp, lons_valid_pp = translate(lats_valid_pp, lons_valid_pp)
        lats_test_pp, lons_test_pp   = translate(lats_test_pp, lons_test_pp)
    # Resample
    if resamp == True:
        lats_train_pp, lons_train_pp = resample(lats_train_pp, lons_train_pp, length_days)
        lats_valid_pp, lons_valid_pp = resample(lats_valid_pp, lons_valid_pp, length_days)
        lats_test_pp, lons_test_pp   = resample(lats_test_pp, lons_test_pp, length_days)
    # Weight
    w       = weightPCA(lats_train_pp, lons_train_pp, wtype)
    w_valid = weightPCA(lats_valid_pp, lons_valid_pp, wtype)
    w_test  = weightPCA(lats_test_pp,  lons_test_pp,  wtype)

    # Reshape
    X_lats_lons_train = np.concatenate((lats_train_pp, lons_train_pp), axis = 1) #concatenation of features to make it in the correct shape
    X_lats_lons_valid = np.concatenate((lats_valid_pp, lons_valid_pp), axis = 1)
    X_lats_lons_test  = np.concatenate((lats_test_pp,  lons_test_pp),  axis = 1)

    if resamp == False:
        X_lats_lons_train[np.isnan(X_lats_lons_train)] = 0
        X_lats_lons_valid[np.isnan(X_lats_lons_valid)] = 0
        X_lats_lons_test[np.isnan(X_lats_lons_test)] = 0
    # Normalize
    if norm == True:
        X_lats_lons_train = normalize(X_lats_lons_train, copy=True)
        X_lats_lons_valid = normalize(X_lats_lons_valid, copy=True)
        X_lats_lons_test  = normalize(X_lats_lons_test,  copy=True)
    # weights
    if len(w_valid) > 0:
        s = int(X_lats_lons_valid.shape[1]/2)
        X_lats_lons_valid[:,s:] = X_lats_lons_valid[:,s:]*w_valid
        s = int(X_lats_lons_test.shape[1]/2)
        X_lats_lons_test[:,s:] = X_lats_lons_test[:,s:]*w_test


    print('PCA')
    print(X_lats_lons_train.shape)
    X_reduced_train, pca= PCA_a(X_lats_lons_train, n_components, copy, whiten, svd_solver, w, kernel, degreek)

    X_reduced_valid = pca.transform(X_lats_lons_valid)
    X_reduced_test  = pca.transform(X_lats_lons_test)

    a = pca.alphas_
    l = pca.lambdas_
    dc = pca.dual_coef_

    return X_reduced_train, X_reduced_valid, X_reduced_test, a, l, dc

# Split the data into the test, training and validation sets
def split_sets(lats_all, lons_all, perctest, perctrain):

    s0, s1 = lats_all.shape
    data = np.concatenate((lats_all,lons_all), axis=1)

    # shuffle the dataset before splitting
    random.seed(4)
    random.shuffle(data)

    lats_test = data[:int(perctest*s0), 0:s1]
    lons_test = data[:int(perctest*s0), s1:]

    lats_train = data[int(perctest*s0):int(perctest*s0)+int(perctrain*s0), 0:s1]
    lons_train = data[int(perctest*s0):int(perctest*s0)+int(perctrain*s0), s1:]

    lats_valid = data[int(perctest*s0)+int(perctrain*s0):, 0:s1]
    lons_valid = data[int(perctest*s0)+int(perctrain*s0):, s1:]

    return lats_test, lons_test, lats_train, lons_train, lats_valid, lons_valid


# ------------------------------------------------
# SCRIPT
# ------------------------------------------------

#Set-up

#80% for training, 10% for validation, 10% for test
perctest = 0.1
perctrain = 0.8
length_days = 550
Dataset_path = 'training_25years_100000.nc'

# PCA
n_components = 0.9999 #Either we specify the number of features that we want (n_components=int) either the total variance explainded (inf 1)
copy = True #Do not modify the given vector, rather copy it
svd_solver = 'auto'
whiten = False
kernel = 'cosine'
degreek = 3 # for poly kernel

norm = False # if normalize
t = True # translation
reamp = True # resampling
wtype = None # Type of weight. None if no weight. Otherwise cos or sqrt-cos

lats_all,lons_all = extract(Dataset_path, length_days)
lats_test, lons_test, lats_train, lons_train, lats_valid, lons_valid = split_sets(lats_all, lons_all, perctest, perctrain)

# Run the k-means
X_reduced_train, X_reduced_valid, X_reduced_test, a, l, dc = run_script(norm, t, reamp, length_days, wtype, kernel, degreek, lats_valid, lons_valid, lats_train, lons_train, lats_test, lons_test)

# Save
name = 'cosine_now'

np.savetxt('X_train_kernelpca_%s.csv'%name, X_reduced_train, delimiter=',')
np.savetxt('X_valid_kernelpca_%s.csv'%name, X_reduced_valid, delimiter=',')
np.savetxt('X_test_kernelpca_%s.csv'%name, X_reduced_test, delimiter=',')
np.savetxt('alphas_kernelpca_%s.csv'%name, a, delimiter=',')
np.savetxt('lambdas_kernelpca_%s.csv'%name, l, delimiter=',')
np.savetxt('dualcoef_kernelpca_%s.csv'%name, dc, delimiter=',')
