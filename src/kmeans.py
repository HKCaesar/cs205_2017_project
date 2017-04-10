import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import pandas as pd
import numpy as np

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

data_fn = "../data/reviewer-data.csv"
K = 3
limit = 1000

######################################################
### GPU KERNELS (in C) ####
######################################################

kernel_code_template = ("""

__global__ void newmeans(double *data, int *clusters, double *means) {
  __shared__ int s_clustern[%(K)s];
  int tid = (%(D)s*threadIdx.x) + threadIdx.y;
  double l_sum = 0;
    
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
     if(clusters[n]==threadIdx.x)
     {
       l_sum += data[(%(D)s*n)+threadIdx.y];
     }
   }
  
   // divide local sum by the number in that cluster
   means[tid] = l_sum/s_clustern[threadIdx.x];
  }

__global__ void reassign(double *d_data, double *d_clusters, double *d_means, double *d_distortion) {
  }
  
""")

######################################################
### DOWNLOAD DATA & ASSIGN INITIAL CLUSTERS ####
######################################################

# import data file and subset data for k-means
reviewdata = pd.read_csv(data_fn)
acts = ["cunninlingus_ct_bin","fellatio_ct_bin","intercoursevaginal_ct_bin","kissing_ct_bin","manualpenilestimulation_ct_bin","massage_ct_bin"]
data = reviewdata[acts][:limit].values
data = np.ascontiguousarray(data, dtype=np.float64)
N,D = data.shape

# assign random clusters & shuffle 
initial_clusters = np.ascontiguousarray(np.zeros(N,dtype=np.intc, order='C'))
for n in range(N):
    initial_clusters[n] = n%K
for i in range(len(initial_clusters)-2,-1,-1):
    j= np.random.randint(0,i+1) 
    temp = initial_clusters[j]
    initial_clusters[j] = initial_clusters[i]
    initial_clusters[i] = temp

######################################################
### RUN K-MEANS SEQUENTIALLY ###
######################################################

A = np.zeros((K,D))
W = initial_clusters
X = data
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
            
print('/n-----sequential output')
print(A)
print(W)
print("done")

######################################################
### ALLOCATE INPUT & COPY DATA TO DEVICE (GPU) ####
######################################################

h_data = data
h_clusters = initial_clusters

# allocate memory & copy data variable from host to device
d_data = cuda.mem_alloc(h_data.nbytes)
cuda.memcpy_htod(d_data,h_data)

# allocate memory & copy clusters variable from host to device
d_clusters = cuda.mem_alloc(h_clusters.nbytes)
cuda.memcpy_htod(d_clusters,h_clusters)

# create & allocate memory for means variable on device
h_means = np.ascontiguousarray(np.zeros((K,D),dtype=np.float64, order='C'))
d_means = cuda.mem_alloc(h_means.nbytes)

# create & allocate memory for distortion variable on device
h_distortion = 0
d_distortion = cuda.mem_alloc(np.array(h_distortion).astype(np.intc).nbytes)

print('/n-----from CPU 1')
print(h_means)
print(h_clusters)

######################################################
### RUN K-MEANS IN PARALLEL ####
######################################################

# define some constants in the kernel code
kernel_code = kernel_code_template % { 
  'N': N,
  'D': D,
  'K': K,
}
mod = SourceModule(kernel_code)

# call the first kernel
kernel1 = mod.get_function("newmeans")
kernel1(d_data, d_clusters, d_means, block=(K,D,1), grid=(1,1,1))

# call the second kernel
#kernel2 = mod.get_function("reassign")
#kernel2(d_data, d_clusters, d_means, d_distortion, block=(N,1,1), grid=(1,1,1))

######################################################
### COPY DEVICE DATA BACK TO HOST AND COMPARE ####
######################################################

print('/n-----GPU output')
cuda.memcpy_dtoh(h_means, d_means)
cuda.memcpy_dtoh(h_clusters, d_clusters)
print(h_means)
print(h_clusters)
print('/n-----Sequential and Parallel means are equal:')
print(np.array_equal(A,h_means))
