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
h_data = np.ascontiguousarray(h_data, dtype=np.float64)

# assign random clusters
N,D=h_data.shape
h_clusters = np.zeros(N,dtype=np.int)
h_clusters = np.ascontiguousarray(h_clusters, dtype=np.float64)

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

# Allocate & copy data and cluster assignments from host to device
d_data = cuda.mem_alloc(h_data.nbytes)
d_clusters = cuda.mem_alloc(h_clusters.nbytes)
cuda.memcpy_htod(d_data,h_data)
cuda.memcpy_htod(d_clusters,h_clusters)

# FIX!
means = cuda.mem_alloc(np.zeros((K,D)))
clustern = cuda.mem_alloc(np.zeros(K))

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
        clustern[k]=0
            
    for n in range(N):
        for d in range(D):
            A[W[n],d]+=X[n,d]
        clustern[ W[n] ] +=1
    
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
### TEST ###
######################################################

newmeansf = mod.get_function("newmeans")
a = numpy.array(8)
b = numpy.array(2)
c = numpy.array(0)
a = a.astype(numpy.int32)
b = b.astype(numpy.int32)
c = c.astype(numpy.int32)
a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)
b_gpu = cuda.mem_alloc(b.size * b.dtype.itemsize)
c_gpu = cuda.mem_alloc(c.size * c.dtype.itemsize)
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)
cuda.memcpy_dtoh(c, c_gpu)
print(c)

reassignf = mod.get_function("reassign")
a = numpy.array(9)
b = numpy.array(3)
c = numpy.array(0)
a = a.astype(numpy.int32)
b = b.astype(numpy.int32)
c = c.astype(numpy.int32)
a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)
b_gpu = cuda.mem_alloc(b.size * b.dtype.itemsize)
c_gpu = cuda.mem_alloc(c.size * c.dtype.itemsize)
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)
cuda.memcpy_dtoh(c, c_gpu)
print(c)

print("done")
