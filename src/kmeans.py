import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import pandas as pd
import numpy as np

######################################################
### INFO ####
######################################################

# Variable*:      Meaning:                 Dim:      Previous name:
# data            reviewer data           (NxD)       X
# clusters        cluster assignments     (Nx1)       W
# means           means                   (KxD)       A
# clustern        number of clusters      (1xK)       M

# *h_ and d_ prefixes in variable names indicate host vs. device copies

######################################################
### CONFIGURE ####
######################################################

data_fn = "../data/reviewer-data.csv"
K = 3

######################################################
### GPU KERNELS (in C) ####
######################################################

mod = SourceModule("""

__global__ void newmeans(int N, int D, int K, double *data, int *clusters, double *means, int *clustern) {
  
  // find the n per cluster with just one lucky thread
  if (threadIdx.x==0 & threadIdx.y==0)
  {
    int l_clustern[tempK];
    //l_clustern = (int*)malloc(sizeof(int) * (*K));
    for(int k=0; k < (K); ++k) l_clustern[k] = 0;
    for (int n=0; n < (N); ++n) l_clustern[clusters[n]]++;
    for(int k =0; k < (K); ++k) clustern[k] = l_clustern[k];
   }
   __syncthreads();
   
   // sum stuff
   
   // divide stuff
   
  }

__global__ void reassign(double *d_data, double *d_clusters, double *d_means, double *d_clustern, double *d_distortion) {
  int n = blockIdx.x;
  }
  
""")

######################################################
### DEFINE VARIALBES ON HOST (CPU) ####
######################################################

# import data file and subset data for k-means
reviewdata = pd.read_csv(data_fn)
acts = ["cunninlingus_ct_bin","fellatio_ct_bin","intercoursevaginal_ct_bin","kissing_ct_bin","manualpenilestimulation_ct_bin","massage_ct_bin"]
h_data = reviewdata[acts][:10].values
h_data = np.ascontiguousarray(h_data, dtype=np.float32)
N,D=h_data.shape

# assign random clusters & shuffle 
h_clusters = np.ascontiguousarray(np.zeros(N,dtype=np.intc, order='C'))
for n in range(N):
    h_clusters[n] = n%K
for i in range(len(h_clusters)-2,-1,-1):
    j= np.random.randint(0,i+1) 
    temp = h_clusters[j]
    h_clusters[j] = h_clusters[i]
    h_clusters[i] = temp
    
# create empty arrays for means, 
h_means = np.ascontiguousarray(np.zeros((K,D),dtype=np.float64, order='C'))
h_distortion = 0

######################################################
### ALLOCATE INPUT & COPY DATA TO DEVICE (GPU) ####
######################################################

# Allocate & copy data and cluster assignment variables from host to device
d_data = cuda.mem_alloc(h_data.nbytes)
d_clusters = cuda.mem_alloc(h_clusters.nbytes)
cuda.memcpy_htod(d_data,h_data)
cuda.memcpy_htod(d_clusters,h_clusters)

# Allocate & copy N, D, and K variables from host to device
#N = np.array(N).astype(np.int32)
#D = np.array(D).astype(np.int32)
#K = np.array(K).astype(np.int32)
d_N = cuda.mem_alloc(4)
d_D = cuda.mem_alloc(4)
d_K = cuda.mem_alloc(4)
cuda.memcpy_htod(d_N,np.array(N).astype(np.int32))
cuda.memcpy_htod(d_D,np.array(D).astype(np.int32))
cuda.memcpy_htod(d_K,K)

# Allocate means and clustern variables on device
d_means = cuda.mem_alloc(h_means.nbytes)
d_clustern = cuda.mem_alloc(np.empty(K,dtype=np.int8).nbytes)
d_distortion = cuda.mem_alloc(4)

print(h_means)
print(h_clusters)

######################################################
### RUN K-MEANS ############# FIX THIS SECTION ######### 
######################################################

converged = True

while not converged:
    converged = True
    
    #compute means
    kernel1 = mod.get_function("newmeans")
    
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
    kernel2 = mod.get_function("reassign")
    
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
### TEST ####
######################################################

kernel1 = mod.get_function("newmeans")
kernel1(d_N, d_D, d_K, d_data, d_clusters, d_means, d_clustern, block=(K,D,1), grid=(1,1,1))

#kernel2 = mod.get_function("reassign")
#kernel2(d_data, d_clusters, d_means, d_clustern, d_distortion, block=(N,1,1), grid=(1,1,1))

######################################################
### COPY DEVICE DATA BACK TO HOST ####
######################################################

cuda.memcpy_dtoh(h_means, d_means)
cuda.memcpy_dtoh(h_clusters, d_clusters)

print('-----')
print(h_means)
print(h_clusters)
print("done")
