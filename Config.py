# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:57:44 2022

@author: noemie
"""

import glob

# ------------------------------------------------
# Params
# ------------------------------------------------

# global parameters
length_days =550 #length in days of trajectories to use for the classification
perctest = 0.1
perctrain = 0.8

# Preparing training set
path = '/storage/mathilde/MainProject/1_ExternalProcesses/LagrangianTracking/' # path to the trajectories on the local computor
files = glob.glob(path + 'Retroflection/run_continuous_25years_track_*') #liste of all netcdf files containing the trajectories to use
init_year = 1993 #first year to subsample the training trajectories
final_year = 2011 #last year to subsample
delta_year =5 # time step in years for selecting the years in
N_particles = 100000 # number of particles to sub-select to form the set of particles to implement the algorithm on
###name_subset = 'Sub_samp_y_%i'%init_year +'_-_%i'%final_year+'_delta_%i' %delta_year +'_%i'%N_particles+'.nc'
name_subset = '/Clustering/training_25years_goodw_100000.nc'


# Kernel PCA 
Run_PCA_Local = False # Put False if you wanna run the PCA on a super-computer
path_remote = '' # Path on the remote computer
Dataset_path = path + name_subset
Dataset_path_remote = path_remote +name_subset
n_components = 0.9999 #Either we specify the number of features that we want (n_components=int) either the total variance explainded (inf 1)
copy = True #Do not modify the given vector, rather copy it
svd_solver = 'auto' 
whiten = False
kernel = 'cosine' # if None, ran during the Clustering, else, prior to it. 
degreek = 3 # is use poly kernel

norm = False # if normalize
t = True # translation
resamp = False # resampling
wtype = 'cos' # Type of weight for geographic correction. None if no weight. Otherwise cos or sqrt-cos


###name_PCA_config = 'PCA_var_%i'%n_components +'_kernel_'+kernel+'_wtype_'+wtype+'_norm_%i'%norm+'_t_%i'%t+'_resamp_%i'%resamp+'_'+name_subset[:-3]
###path_save_PCA = path+'/'+name_PCA_config+'/' 
name_PCA_config = 'noresamp_goodw'
path_save_PCA = '/storage/mathilde/MainProject/1_ExternalProcesses/LagrangianTracking/Clustering/testnoresamp/'

# Clustering
init = 'k-means++'
nmb_initialisations = 20  # number of initiatilisaton for the k-means++ 
max_iter = 300
tol = 1e-4
algorithm = 'full'
verbose = 0 
sample_weight = None
n_split = 15  # number of iterations for convergence
n_clusters = 30 #number of clusters

ntest='k-'+kernel+'-w-'+wtype+'_%i'%n_clusters+'noresamp'
name_Clustering_config = 'config_'+ntest+name_PCA_config
path_save_clustering = path +'Clustering/Last_Version/'+name_Clustering_config+'/'
