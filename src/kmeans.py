import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import pandas as pd
import numpy as np

######################################################
### CONFIGURE ####
######################################################

data_fn = "../data/reviewer-data.csv"

K=3

# h_ and d_ indicate if varialbe is on the host or device
# data: reviewer data (NxD) -- previously X
# clusters: cluster assignments for each reviewer (Nx1) -- previously W
# means: means for each cluster and dimension (KxD) -- previously A
# clustern: location of each cluster (1xK) -- previously m

######################################################
### GPU KERNELS (in C) ####
######################################################

mod1 = SourceModule("""
__global__ void newmeans(double *d_data, double *d_clusters, double *d_means, double *d_clustern) {
  int k = blockIdx.x;
  int d = blockIdx.y;
}""")

mod2 = SourceModule("""
__global__ void reassign(double *d_data, double *d_clusters, double *d_means, double *d_clustern, double *d_distortion) {
  int n = blockIdx.x;
}""")

######################################################
### DEFINE VARIALBES ON HOST (CPU) ####
######################################################

# import data file and subset data for k-means
reviewdata = pd.read_csv(data_fn)
acts = ["cunninlingus_ct_bin","fellatio_ct_bin","intercoursevaginal_ct_bin","kissing_ct_bin","manualpenilestimulation_ct_bin","massage_ct_bin"]
h_data = reviewdata[acts][:100].values
h_data = np.ascontiguousarray(h_data, dtype=np.float32)
N,D=h_data.shape

# assign random clusters & shuffle 
h_clusters = np.ascontiguousarray(np.zeros(N,dtype=np.int8, order='C'))
for n in range(N):
    h_clusters[n] = n%K
for i in range(len(h_clusters)-2,-1,-1):
    j= np.random.randint(0,i+1) 
    temp = h_clusters[j]
    h_clusters[j] = h_clusters[i]
    h_clusters[i] = temp
    
# create empty arrays for means
#h_means = np.ascontiguousarray(np.zeros((K,D),dtype=np.float64, order='C'))

######################################################
### ALLOCATE INPUT & COPY DATA TO DEVICE (GPU) ####
######################################################

# Allocate & copy data and cluster assignment variables from host to device
d_data = cuda.mem_alloc(h_data.nbytes)
d_clusters = cuda.mem_alloc(h_clusters.nbytes)
cuda.memcpy_htod(d_data,h_data)
cuda.memcpy_htod(d_clusters,h_clusters)

# Allocate means and clustern variables on device
d_means = cuda.mem_alloc(np.empty((K,D),dtype=np.float64).nbytes)
d_clustern = cuda.mem_alloc(np.empty(K,dtype=np.int8).nbytes)
d_distortion = cuda.mem_alloc(np.empty(1,dtype=np.float64).nbytes)

######################################################
### RUN K-MEANS ############# FIX THIS SECTION ######### 
######################################################

converged = True

while not converged:
    converged = True
    
    #compute means
    kernel1 = mod1.get_function("newmeans")
    kernel1(d_data, d_clusters, d_means, d_clustern, block=(K,D,1), grid=(1,1,1))
    
    for k in range(K):
        for d in range(D):
            d_means[k,d] = 0
        d_clustern[k]=0
            
    for n in range(N):
        for d in range(D):
            d_means[d_clusters[n],d]+=d_data[n,d]
        d_clustern[ W[n] ] +=1
    
    for k in range(K):
        for d in range(D):
            d_means[k,d] = d_means[k,d]/d_clustern[k]
            
    #assign to closest mean
    kernel2 = mod2.get_function("reassign")
    kernel2(d_data, d_clusters, d_means, d_clustern, d_distortion, block=(N,1,1), grid=(1,1,1))
    
    for n in range(N):
        
        min_val = np.inf
        min_ind = -1
        
        for k in range(K):
            temp =0
            for d in range(D):
                temp += (d_data[n,d]-d_means[k,d])**2
            
            if temp < min_val:
                min_val = temp
                min_ind = k
                
        if min_ind != d_clusters[n]:
            d_clusters[n] = min_ind
            converged=False
            
######################################################
### COPY DEVICE DATA BACK TO HOST ####
######################################################

kernel1 = mod1.get_function("newmeans")
kernel1(d_data, d_clusters, d_means, d_clustern, block=(K,D,1), grid=(1,1,1))

cuda.memcpy_dtoh(h_clusters, d_clusters)
