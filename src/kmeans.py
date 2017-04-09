import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import pandas as pd
import numpy as np

######################################################
### CONFIGURE ####
######################################################

data_fn = "../data/reviewer-data.csv"
#data_fn = "../data/reviewer-data-sample.csv"

K=3

# h_ and d_ indicate if varialbe is on the host or device
# data: reviewer data (NxD) -- previously X
# clusters: cluster assignments for each reviewer (Nx1) -- previously W
# means: means for each cluster and dimension (KxD) -- previously A
# clusterloc: location of each cluster (1xK) -- previously m

######################################################
### GPU KERNELS (in C) ####
######################################################

mod = SourceModule("""
__global__ void newmeans(int *a, int *b, int *c)  {
  int id = blockIdx.x;
  c[id] = a[id] + b[id];
}""")

mod = SourceModule("""
__global__ void reassign(int *a, int *b, int *c)  {
  int id = blockIdx.x;
  c[id] = a[id] + b[id];
}""")

######################################################
### DEFINE VARIALBES ON HOST (CPU) ####
######################################################

# import data file and subset data for k-means and put into numpy ndarrays
reviewdata = pd.read_csv(data_fn)
acts = ["cunninlingus_ct_bin","fellatio_ct_bin","intercoursevaginal_ct_bin","kissing_ct_bin","manualpenilestimulation_ct_bin","massage_ct_bin"]
h_data = reviewdata[acts][:1000].values

# assign random clusters
N,D=h_data.shape
h_data = np.zeros(N,dtype=np.int)

for n in range(N):
    h_clusters[n] = n%K
    
def shuffle(x,n):
    for i in range(n-2,-1,-1): #from n-2 to 0
        j= np.random.randint(0,i+1) #from 0<=j<=i
        temp = x[j]
        x[j] = x[i]
        x[i] = temp

shuffle(h_clusters,len(h_clusters))

######################################################
### ALLOCATE INPUT & COPY DATA TO DEVICE (GPU) ####
######################################################

# Allocate input on device ######### FIX THIS SECTION ######### 
means = cuda.mem_alloc(np.zeros((K,D)))
clusterloc = cuda.mem_alloc(np.zeros(K))
#c_gpu = cuda.mem_alloc(c.size * c.dtype.itemsize)

# Copy from host to device
cuda.memcpy_htod(h_data, d_data)
cuda.memcpy_htod(h_clusters, d_clusters)

######################################################
### RUN K-MEANS ############# FIX THIS SECTION ######### 
######################################################

converged = True

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
