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
### First script to run.
#############################

This portion comprises the pre-processing of the data, as well as
the first steps of the clustering algorithm.
After having run this portion, you will need to run
the script titled Script_supercomputer.py
"""

from All_functions_ML import pre_processing, extract, split_sets, local_PCA, load_PCA, kmeans_clustering

from Config import norm, t, resamp, length_days, wtype, kernel, degreek,   n_components, copy, whiten, svd_solver,\
    Dataset_path, perctest, perctrain, Run_PCA_Local, path, name_PCA_config, path_save_PCA, path_save_clustering, n_clusters, name_Clustering_config,\
    ntest, init, max_iter, algorithm, n_split, verbose, tol, sample_weight, nmb_initialisations, files, init_year,\
    final_year, delta_year, N_particles
import os

if not os.path.exists(path +'/'+name_Clustering_config):
    os.makedirs(path+'/'+name_Clustering_config)

print('1/4-Pre_processing------------------------------------')
# Load the data
lats_all,lons_all, temps_all, sals_all = extract(Dataset_path, length_days)
# Split into the test, training and validation sets
lats_test, lons_test, lats_train, lons_train, lats_valid, lons_valid, temps_train, temps_test, temps_valid, sals_train, sals_test, sals_valid = split_sets(lats_all, lons_all,temps_all, sals_all, perctest, perctrain)

# Apply the preprocessing to the data. The entries for this function are defined in the configuration file
X_lats_lons_valid, X_lats_lons_test, X_lats_lons_train, w_train, w_valid, w_test = pre_processing(norm, t, resamp, length_days, wtype, kernel, degreek, lats_valid, lons_valid, lats_train, lons_train, lats_test, lons_test,  n_components, copy, whiten, svd_solver)

print('2/4-PCA------------------------------------')
if Run_PCA_Local: # if we can run the PCA locally (no kernel)
    if not os.path.exists(path+'/'+name_PCA_config+'/'):
        os.makedirs(path+'/'+name_PCA_config+'/')
        X_reduced_train, X_reduced_valid, X_reduced_test, a, l, dc = local_PCA(X_lats_lons_valid, X_lats_lons_test, X_lats_lons_train, w_train, w_valid, w_test,  n_components, copy, whiten, svd_solver, kernel, degreek)

        np.savetxt(path+'/'+name_PCA_config+'/X_train_kernelpca_%s.csv'%name_PCA_config, X_reduced_train, delimiter=',')
        np.savetxt(path+'/'+name_PCA_config+'/X_valid_kernelpca_%s.csv'%name_PCA_config, X_reduced_valid, delimiter=',')
        np.savetxt(path+'/'+name_PCA_config+'/X_test_kernelpca_%s.csv'%name_PCA_config, X_reduced_test, delimiter=',')
        np.savetxt(path+'/'+name_PCA_config+'/alphas_kernelpca_%s.csv'%name_PCA_config, a, delimiter=',')
        np.savetxt(path+'/'+name_PCA_config+'/lambdas_kernelpca_%s.csv'%name_PCA_config, l, delimiter=',')
        np.savetxt(path+'/'+name_PCA_config+'/dualcoef_kernelpca_%s.csv'%name_PCA_config, dc, delimiter=',')

    else:
        print('This PCA already exists')

if Run_PCA_Local != True:
    print('Run Script_supercomputer on a supercomputer')
    print('Copy the saved files into', path+name_PCA_config+'/ before continuiing')

    X_reduced_train, X_reduced_valid, X_reduced_test =  load_PCA(ntest, norm, t, resamp, length_days, wtype,n_clusters, kernel, lats_valid, lons_valid, lats_train, lons_train, path_save_clustering, name_PCA_config, lats_test, lons_test, path_save_PCA)


print('3/4-Clustering------------------------------------')
labels_valid, labels_test, X_reduced_valid, X_reduced_test, centroids, a_temp,X_centers_train = kmeans_clustering(X_reduced_train, X_reduced_valid, X_reduced_test, n_split, n_clusters, init, nmb_initialisations, max_iter, tol, algorithm, verbose, sample_weight)

# Save the output
os.makedirs(path_save_clustering)
np.save(path_save_clustering+'centroids'+ntest, centroids)
np.save(path_save_clustering+'X_centers_train'+ntest, X_centers_train)
np.save(path_save_clustering+'labels_test'+ntest, labels_test)
np.save(path_save_clustering+'X_reduced_test'+ntest, X_reduced_test)
np.save(path_save_clustering+'lats_test'+ntest, lats_test)
np.save(path_save_clustering+'lons_test'+ntest, lons_test)
np.save(path_save_clustering+'labels_valid'+ntest, labels_valid)
np.save(path_save_clustering+'X_reduced_valid'+ntest, X_reduced_valid)
np.save(path_save_clustering+'lats_valid'+ntest, lats_valid)
np.save(path_save_clustering+'lons_valid'+ntest, lons_valid)
np.save(path_save_clustering+'temps_valid'+ntest, temps_valid)
np.save(path_save_clustering+'temps_test'+ntest, temps_test)
np.save(path_save_clustering+'sals_valid'+ntest, sals_valid)
np.save(path_save_clustering+'sals_test'+ntest, sals_test)
np.save(path_save_clustering+'lats_train'+ntest, lats_train)
np.save(path_save_clustering+'lons_train'+ntest, lons_train)
np.save(path_save_clustering+'temps_train'+ntest, temps_train)
np.save(path_save_clustering+'sals_train'+ntest, sals_train)
