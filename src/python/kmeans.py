import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import pandas as pd
import numpy as np

import time
import csv

import string
from functions import *

from sklearn.cluster import KMeans



######################################################
### INFO ####
######################################################

# Parallel*:      Sequential:      Meaning:                 Dim:
# data            X                reviewer data           (NxD)
# clusters        W                cluster assignments     (Nx1)
# means           A                means                   (KxD)
# clustern        m                number of clusters      (1xK)

# *h_ and d_ prefixes in parallel variable names indicate host vs. device copies

######################################################
### CONFIGURE ####
######################################################

data_fn = "../../data/reviewer-data.csv"
output_dir = "../../analysis/"
d_list = ["cunninlingus_ct_bin","fellatio_ct_bin","intercoursevaginal_ct_bin","kissing_ct_bin","manualpenilestimulation_ct_bin","massage_ct_bin"]

K = 5
limit = 1000 # impose a limit of N on the dataset


######################################################
### RUN K-MEANS ####
######################################################

data, initial_clusters, N, D = prep_data(K, data_fn, d_list, limit)
h_data, h_clusters, h_means, h_distortion = prep_host(data, initial_clusters, K, D)
output = [['algorithm','time','convergence','distortion','n','d','k']]

### Stock k-means ###
kmeans = KMeans(n_clusters=3,n_init=100)
kmeans.fit(data)
print('\n-----Stock K-Means')
print(kmeans.cluster_centers_)


### Sequential ###

start = time.time()
seq_means, seq_clusters, seq_count, seq_distortion = sequential(N, K, D, data, initial_clusters)
stop = time.time()-start
output.append(['sequential',stop, seq_count, seq_distortion, N, D, K])

print('\n-----Sequential output')
print(seq_means)
print(seq_clusters[:10])

### Naive Parallel ###

kernel1, kernel2 = pnaive_mod(N, K, D)
reset_hvars(initial_clusters, h_means, h_distortion, K, D)

start = time.time()
d_data, d_clusters, d_means, d_distortion = prep_device(h_data, h_clusters, h_means, h_distortion)
kernel1(d_data, d_clusters, d_means, block=(K,D,1), grid=(1,1,1))
#kernel2(d_data, d_clusters, d_means, d_distortion, block=(N,1,1), grid=(1,1,1))
cuda.memcpy_dtoh(h_means, d_means)
cuda.memcpy_dtoh(h_clusters, d_clusters)
stop = time.time()-start
output.append(['naive parallel',stop, '?', '?', N, D, K])

print('\n-----Naive Parallel output')
print(h_means)
print(h_clusters[:10])
print('Equals sequential output: %s' % str(np.array_equal(seq_means,h_means)))

### Parallel Improved ###

kernel1, kernel2 = pimproved_mod(N, K, D)
reset_hvars(initial_clusters, h_means, h_distortion, K, D)

start = time.time()
d_data, d_clusters, d_means, d_distortion = prep_device(h_data, h_clusters, h_means, h_distortion)
#kernel1(d_data, d_clusters, d_means, block=(K,D,1), grid=(1,1,1))
#kernel2(d_data, d_clusters, d_means, d_distortion, block=(N,1,1), grid=(1,1,1))
cuda.memcpy_dtoh(h_means, d_means)
cuda.memcpy_dtoh(h_clusters, d_clusters)
stop = time.time()-start
output.append(['improved parallel',stop, '?', '?', N, D, K])

print('\n-----Improved Parallel output')
print(h_means)
print(h_clusters[:10])
print('Equals sequential output: %s' % str(np.array_equal(seq_means,h_means)))

######################################################
### COPY DEVICE DATA BACK TO HOST AND COMPARE ####
######################################################

with open(output_dir + 'times.csv', 'w') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerows(output)
    f.close()

