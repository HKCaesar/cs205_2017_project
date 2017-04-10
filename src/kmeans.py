import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import pandas as pd
import numpy as np

######################################################
### INFO ####
######################################################

# Variable*:      Meaning:                 Dim:      Sequential name:
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
limit = 50

######################################################
### GPU KERNELS (in C) ####
######################################################

kernel_code_template = ("""

__global__ void newmeans(double *data, int *clusters, double *means) {
  __shared__ int s_clustern[%(K)s];
  int tid = threadIdx.x + (threadIdx.x*threadIdx.y);
  double l_sum = 0;
  int l_clustern;
    
  // find the n per cluster with just one lucky thread
  if (tid==0)
  {
    for(int k=0; k < (%(K)s); ++k) s_clustern[k] = 0;
    for(int n=0; n < (%(N)s); ++n) s_clustern[clusters[n]]++;
   }
   __syncthreads();
   
   // sum stuff  
   for(int n=0; n < (%(N)s); ++n)
   {
     if(clusters[n]==threadIdx.y)
     {
       for(int d=0; d < (%(D)s); ++d)
       {
         l_sum += data[(n*%(D)s)+d];
       }
     }
   }
  
   // divide local sum by the number in that cluster
   means[tid] = l_sum/s_clustern[threadIdx.x];
      
  }

__global__ void reassign(double *d_data, double *d_clusters, double *d_means, double *d_distortion) {
  }
  
""")

######################################################
### DEFINE VARIALBES ON HOST (CPU) ####
######################################################

# import data file and subset data for k-means
reviewdata = pd.read_csv(data_fn)
acts = ["cunninlingus_ct_bin","fellatio_ct_bin","intercoursevaginal_ct_bin","kissing_ct_bin","manualpenilestimulation_ct_bin","massage_ct_bin"]
h_data = reviewdata[acts][:limit].values
h_data = np.ascontiguousarray(h_data, dtype=np.float64)
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
    
# create arrays for means and clusters
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
d_N = cuda.mem_alloc(4)
d_D = cuda.mem_alloc(4)
d_K = cuda.mem_alloc(4)
cuda.memcpy_htod(d_N, np.array(N).astype(np.intc))
cuda.memcpy_htod(d_D, np.array(D).astype(np.intc))
cuda.memcpy_htod(d_K, np.array(K).astype(np.intc))

# Allocate means and clustern variables on device
d_means = cuda.mem_alloc(h_means.nbytes)
d_distortion = cuda.mem_alloc(4)

######################################################
### RUN K-MEANS SEQUENTIALLY ###
######################################################

A = h_means
W = h_clusters
X = h_data
m = np.zeros(K)

converged = False

while not converged:
    converged = True
    
    #compute means
    for k in range(K):
        for d in range(D):
            A[k,d] = 0
        m[k]=0
            
    for n in range(N):
        for d in range(D):
            A[W[n],d]+=X[n,d]
        m[ W[n] ] +=1
    
    for k in range(K):
        for d in range(D):
            A[k,d] = A[k,d]/m[k]
            
    #assign to closest mean
    for n in range(N):
        
        min_val = np.inf
        min_ind = -1
        
        for k in range(K):
            temp =0
            for d in range(D):
                temp += (X[n,d]-A[k,d])**2
            
            if temp < min_val:
                min_val = temp
                min_ind = k
                
        if min_ind != W[n]:
            W[n] = min_ind
            converged=False
            
######################################################
### RUN K-MEANS IN PARALLEL ####
######################################################

kernel_code = kernel_code_template % { 
  'N': N,
  'D': D,
  'K': K,
}
mod = SourceModule(kernel_code)

kernel1 = mod.get_function("newmeans")
kernel1(d_data, d_clusters, d_means, block=(K,D,1), grid=(1,1,1), shared=K)

#kernel2 = mod.get_function("reassign")
#kernel2(d_data, d_clusters, d_means, d_distortion, block=(N,1,1), grid=(1,1,1))

######################################################
### COPY DEVICE DATA BACK TO HOST ####
######################################################

print('-----from CPU')
print(h_means)
print(h_clusters)

cuda.memcpy_dtoh(h_means, d_means)
cuda.memcpy_dtoh(h_clusters, d_clusters)

print('-----GPU output')
print(h_means)
print(h_clusters)
print('-----sequential output')
print(A)
print(W)
print("done")
