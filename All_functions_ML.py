# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:16:30 2022

@author: Noemie Planat and Mathilde Jutras
contact: mathilde.jutras@mail.mcgill.ca
license: GNU General Public License v3.0

Please acknowledge using the following DOI:

This is the script used to perform the unsupervised clustering analysis
on geospatial Lagrangian trajectories detailed in Jutras, Planat & Dufour (2022).

This script contains the functions used in script Script_run.py
"""

import xarray as xr
import numpy as np
#import pandas as pd
import os
import random
import pyproj as proj
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import cluster
import wpca


def remove_short(ds):
    l = [len(ds.sel(traj=i).dropna(dim='obs').obs) for i in ds.traj]
    keep = [i for i in range(len(l)) if l[i] > 90] # more than 3 months
    ds2 = ds.sel(traj=xr.DataArray(keep)).rename_dims({'dim_0':'traj'})

    return ds2

def extract(Dataset_path, length_days) :

    ds = xr.open_dataset(Dataset_path)
    lats = ds.lat[:,0:length_days].values # keep only one year of data
    lons = ds.lon[:,0:length_days].values
    temps = ds.temperature[:,0:length_days].values
    sal = ds.salinity[:,0:length_days].values
    return lats, lons, temps, sal


def extract_all_per_year(Dataset_path, yr, length_days, all_filles) :
    file = all_filles[np.where(np.array([int(all_filles[i][-7:-3]) for i in range(len(all_filles))]) == yr)[0][0]]

    ds = xr.open_dataset(file)
    ds = remove_short(ds)
    lats = ds.lat[:,0:length_days].values # keep only one year of data
    lons = ds.lon[:,0:length_days].values
    temps = ds.temperature[:,0:length_days].values
    sal = ds.salinity[:,0:length_days].values
    date = ds.time[:,1].values

    return lats, lons, temps, sal, date

def translate(lat, lon):
    return lat-np.repeat(lat[:,0][:, np.newaxis], lat.shape[1], axis=1), lon-np.repeat(lon[:,0][:, np.newaxis], lon.shape[1], axis=1)

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

def PCA_Kernelized(Xlat_lon, n_components, copy, whiten, svd_solver, w, kernel, degreek):

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
    print('The number of initial features was: ', N_features_PCA)
    print('The number of selected features is: ', N_components_PCA)
    print('The number of samples is: ', N_samples_PCA)
    print('The explained variance desired is:', n_components, '%, the obtained variance explained is: ', np.sum(Explained_variance_ratio_PCA), '%')


    if kernel != None:

        pca = KernelPCA(n_components=N_components_PCA, copy_X=copy, eigen_solver=svd_solver, kernel=kernel, degree=degreek, fit_inverse_transform=True)
        X_reduced = pca.fit_transform(Xlat_lon)
    else:
        print('Kernel is None')
        pca = wpca.WPCA(n_components=N_components_PCA).fit(Xlat_lon, w)
        X_reduced = pca.fit_transform(Xlat_lon, w)
    return X_reduced, pca

def apply_preprocessing(t, la, lo, resamp, length_days, wtype, norm):
    if t == True :
       lats_train_pp, lons_train_pp = translate(la, lo)
    if resamp == True:
        lats_train_pp, lons_train_pp = resample(lats_train_pp, lons_train_pp, length_days)
    w = weightPCA(lats_train_pp, lons_train_pp, wtype)
    # Reshape
    X_lats_lons_train = np.concatenate((lats_train_pp, lons_train_pp), axis = 1) #concatenation of features to make it in the correct shape
    if resamp == False:
        X_lats_lons_train[np.isnan(X_lats_lons_train)] = 0
   # Normalize
    if norm == True:
        X_lats_lons_train = normalize(X_lats_lons_train, copy=True)

    return w, X_lats_lons_train

def pre_processing(norm, t, resamp, length_days, wtype, kernel, degreek, lats_valid, lons_valid, lats_train, lons_train, lats_test, lons_test,  n_components, copy, whiten, svd_solver):
    print('Pre_processing')
    # PRE-PROCESSING
    w, X_lats_lons_train = apply_preprocessing(t, lats_train, lons_train, resamp, length_days, wtype, norm)
    w_test, X_lats_lons_test = apply_preprocessing(t, lats_test, lons_test, resamp, length_days, wtype, norm)
    w_valid, X_lats_lons_valid = apply_preprocessing(t, lats_valid, lons_valid, resamp, length_days, wtype, norm)

    # Apply weights
    if len(w_valid) > 0:
        # the weight for the validation set is applied in function PCA_kernelized
        s = int(X_lats_lons_valid.shape[1]/2)
        X_lats_lons_valid[:,s:] = X_lats_lons_valid[:,s:]*w_valid
        s = int(X_lats_lons_test.shape[1]/2)
        X_lats_lons_test[:,s:] = X_lats_lons_test[:,s:]*w_test

    return X_lats_lons_valid, X_lats_lons_test, X_lats_lons_train, w, w_valid, w_test

def local_PCA(X_lats_lons_valid, X_lats_lons_test, X_lats_lons_train, w, w_valid, w_test,  n_components, copy, whiten, svd_solver, kernel, degreek):
    # Apply the kernelized PCA
    X_reduced_train, pca= PCA_Kernelized(X_lats_lons_train, n_components, copy, whiten, svd_solver, w, kernel, degreek)
    X_reduced_valid = pca.transform(X_lats_lons_valid)
    X_reduced_test  = pca.transform(X_lats_lons_test)
    a = pca.alphas_
    l = pca.lambdas_
    dc = pca.dual_coef_
    return X_reduced_train, X_reduced_valid, X_reduced_test, a, l, dc

def k_means_X_cv(n_splits, X_reduced, n_clusters, init, nmb_initialisations, max_iter, tol, algorithm, verbose, sample_weight):

    kf = KFold(n_splits=n_splits, shuffle=True)
    avg_silouhette = 0
    i = 0
    for train_index, val_index in kf.split(X_reduced):
        print('fold nmb ', i)
        i+=1
        X_train = X_reduced[train_index,:]
        k_means = cluster.KMeans(n_clusters=n_clusters, init=init, n_init = nmb_initialisations, max_iter = max_iter, tol = tol, algorithm = algorithm, verbose = verbose)
        k_means.fit(X_train, sample_weight = sample_weight)
        X_centers = k_means.cluster_centers_
        a_temp = k_means.inertia_
        #a_temp = silhouette_score(X_val, labs_val)
        if a_temp>avg_silouhette:
            X_centered_memory = X_centers
            k_means_memory = k_means
            avg_silouhette = a_temp

    return X_centered_memory, k_means_memory, a_temp

def get_number_centroids(X_centers_train, X_reduced_train):
    N_clusters = X_centers_train.shape[0]
    Liste_number= np.zeros(N_clusters)
    for n in range(N_clusters):
        RV = X_centers_train[n]-X_reduced_train[:,:]
        N_RV = np.linalg.norm(RV, axis = 1)
        Liste_number[n] = np.argmin(N_RV)
    return Liste_number

def split_sets(lats_all, lons_all, temps_all, sals_all, perctest, perctrain):

    s0, s1 = lats_all.shape
    data = np.concatenate((lats_all,lons_all), axis=1)

    random.seed(4)
    random.shuffle(data)
    random.seed(4)
    random.shuffle(temps_all)
    random.seed(4)
    random.shuffle(sals_all)

    lats_test = data[:int(perctest*s0), 0:s1]
    lons_test = data[:int(perctest*s0), s1:]

    lats_train = data[int(perctest*s0):int(perctest*s0)+int(perctrain*s0), 0:s1]
    lons_train = data[int(perctest*s0):int(perctest*s0)+int(perctrain*s0), s1:]

    lats_valid = data[int(perctest*s0)+int(perctrain*s0):, 0:s1]
    lons_valid = data[int(perctest*s0)+int(perctrain*s0):, s1:]

    temps_test = temps_all[:int(perctest*s0),:]
    sals_test = sals_all[:int(perctest*s0),:]

    temps_train =temps_all[int(perctest*s0):int(perctest*s0)+int(perctrain*s0),:]
    sals_train = sals_all[int(perctest*s0):int(perctest*s0)+int(perctrain*s0),:]

    temps_valid = temps_all[int(perctest*s0)+int(perctrain*s0):,:]
    sals_valid = sals_all[int(perctest*s0)+int(perctrain*s0):,:]

    return lats_test, lons_test, lats_train, lons_train, lats_valid, lons_valid, temps_train, temps_test, temps_valid, sals_train, sals_test, sals_valid


def predict(X_reduced_val, X_centers):

    X_reduced_val2 = np.repeat(X_reduced_val[:,:,np.newaxis], X_centers.shape[0], axis = 2)
    X_centers2 = np.repeat(np.transpose(X_centers)[np.newaxis,:,:], X_reduced_val.shape[0], axis = 0)

    return np.argmin(np.linalg.norm(X_reduced_val2-X_centers2, axis=1), axis = 1)

def load_PCA( name, norm, t, resamp, length_days, wtype, n_clusters, kernel, lats_val, lons_val,lats_train, lons_train, path_save, name_kernel_pca, lats_test,lon_test, path_kernelpca) :

    # load the results of the kernalized PCA k-means executed on a supercomputer
    X_reduced_train = np.genfromtxt(path_kernelpca+'X_train_kernelpca_'+name_kernel_pca+'.csv', delimiter=',')
    X_reduced_valid = np.genfromtxt(path_kernelpca+'X_valid_kernelpca_'+name_kernel_pca+'.csv', delimiter=',')
    X_reduced_test = np.genfromtxt(path_kernelpca+'X_test_kernelpca_'+name_kernel_pca+'.csv', delimiter=',')
    return X_reduced_train, X_reduced_valid, X_reduced_test

def kmeans_clustering(X_reduced_train, X_reduced_valid, X_reduced_test, n_split, n_clusters, init, nmb_initialisations, max_iter, tol, algorithm, verbose, sample_weight):
    # Operate the K-MEANS CLUSTERING
    print('Kmeans')
    X_centers_train, k_means_model, a_temp = k_means_X_cv(n_split, X_reduced_train, n_clusters, init, nmb_initialisations, max_iter, tol, algorithm, verbose, sample_weight)
    labels_valid = predict(X_reduced_valid, X_centers_train)
    labels_test = predict(X_reduced_test, X_centers_train)
    print('Get Centroids')
    centroids = get_number_centroids(X_centers_train, X_reduced_train)
    return labels_valid, labels_test, X_reduced_valid, X_reduced_test, centroids, a_temp,X_centers_train

def prediction(X_lats_lons, X_lats_lons_train, Alphas, X_centers_train) :

    Kernel_matrix = cosine_similarity(X_lats_lons, X_lats_lons_train)
    X_reduced = np.matmul(Kernel_matrix, Alphas)
    labels = predict(X_reduced, X_centers_train)
    return labels, X_reduced
