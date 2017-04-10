import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import pandas as pd
import numpy as np

import time
import csv

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
output_dir = "../analysis/"
d_list = ["cunninlingus_ct_bin","fellatio_ct_bin","intercoursevaginal_ct_bin","kissing_ct_bin","manualpenilestimulation_ct_bin","massage_ct_bin"]

K = 3
limit = 1000 # impose a limit of N on the dataset

######################################################
### DEFINE SEQUENTIAL K-MEANS FUNCTiON ###
######################################################

def sequential(data, initial_clusters):
  
  A = np.zeros((K,D))
  W = initial_clusters
  X = data
  m = np.zeros(K)
  count = 0
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
      count +=1
  
  distortion = '?'
  return A, W, count, distortion

######################################################
### DEFINE "NAIVE" PARALLEL K-MEANS KERNEL FOR GPU ####
######################################################

def pnaive_mod():
  
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
  
  kernel_code = kernel_code_template % {'N': N, 'D': D, 'K': K}
  return SourceModule(kernel_code)

######################################################
### DEFINE "IMPROVED" PARALLEL K-MEANS KERNEL FOR GPU ####
######################################################

def pimproved_mod():
  
  kernel_code_template = ("""

  __global__ void newmeans(double *data, int *clusters, double *means) {
    }

  __global__ void reassign(double *d_data, double *d_clusters, double *d_means, double *d_distortion) {
    }

  """)
  
  kernel_code = kernel_code_template % {'N': N, 'D': D, 'K': K}
  return SourceModule(kernel_code)

######################################################
### DEFINTE FUNCTIONS TO MANAGE DATA ####
######################################################

# download data and assign initial clusters
def prep_data():
  # import data file and subset data for k-means
  reviewdata = pd.read_csv(data_fn)
  data = reviewdata[d_list][:limit].values
  data = np.ascontiguousarray(data, dtype=np.float64)

  # assign random clusters & shuffle 
  initial_clusters = np.ascontiguousarray(np.zeros(N,dtype=np.intc, order='C'))
  for n in range(N):
      initial_clusters[n] = n%K
  for i in range(len(initial_clusters)-2,-1,-1):
      j= np.random.randint(0,i+1) 
      temp = initial_clusters[j]
      initial_clusters[j] = initial_clusters[i]
      initial_clusters[i] = temp
  
  return data, intital_clusters

# define h_vars on host
def prep_host():
  global h_data, h_clusters, hmeans, h_distortion
  h_data = data
  h_clusters = initial_clusters
  h_means = np.ascontiguousarray(np.zeros((K,D),dtype=np.float64, order='C'))
  h_distortion = 0
  return

# allocate memory and copy data to d_vars on device
def prep_device():
  global d_data, d_clusters, d_means, d_distortion
  d_data = cuda.mem_alloc(h_data.nbytes)
  d_clusters = cuda.mem_alloc(h_clusters.nbytes)
  d_means = cuda.mem_alloc(h_means.nbytes)
  d_distortion = cuda.mem_alloc(np.array(h_distortion).astype(np.intc).nbytes)
  cuda.memcpy_htod(d_data,h_data)
  cuda.memcpy_htod(d_clusters,h_clusters)
  return

# reset h_vars
def reset_hvars(): 
  global h_clusters, hmeans, h_distortion, d_clusters
  h_clusters = initial_clusters
  h_means = np.ascontiguousarray(np.zeros((K,D),dtype=np.float64, order='C'))
  h_distortion = 0
  cuda.memcpy_htod(d_clusters,h_clusters)
  return

######################################################
### RUN K-MEANS ####
######################################################

header = ['algorithm','time','convergence','distortion','n','d','k']
output = [header]

data, intital_clusters = prep_data()
N,D=data.shape
prep_host()
prep_device()

### Sequential ###
start = time.time()
seq_means, seq_clusters, seq_count, seq_distortion = sequential(data, initial_clusters)
output.append(['sequential',time.time()-start, seq_count, seq_distortion, N, D, K])

### Naive Parallel ###

mod = pnaive_mod()
kernel1 = mod.get_function("newmeans")
#kernel2 = mod.get_function("reassign")
reset_hvars()

start = time.time()
kernel1(d_data, d_clusters, d_means, block=(K,D,1), grid=(1,1,1))
#kernel2(d_data, d_clusters, d_means, d_distortion, block=(N,1,1), grid=(1,1,1))
cuda.memcpy_dtoh(h_means, d_means)
cuda.memcpy_dtoh(h_clusters, d_clusters)
times.append(time.time()-start)
output.append(['naive parallel',time.time()-start, '?', '?', N, D, K])

print('\n-----Naive Parallel output')
cuda.memcpy_dtoh(h_means, d_means)
cuda.memcpy_dtoh(h_clusters, d_clusters)
print(h_means)
print(h_clusters[:10])
print('Equals sequential output: %s' % str(np.array_equal(seq_means,h_means)))

### Parallel Improved ###

mod = pimproved_mod()
#kernel1 = mod.get_function("newmeans")
#kernel2 = mod.get_function("reassign")
reset_hvars()

start = time.time()
#kernel1(d_data, d_clusters, d_means, block=(K,D,1), grid=(1,1,1))
#kernel2(d_data, d_clusters, d_means, d_distortion, block=(N,1,1), grid=(1,1,1))
cuda.memcpy_dtoh(h_means, d_means)
cuda.memcpy_dtoh(h_clusters, d_clusters)
output.append(['improved parallel',time.time()-start, '?', '?', N, D, K])

print('\n-----Improved Parallel output')
cuda.memcpy_dtoh(h_means, d_means)
cuda.memcpy_dtoh(h_clusters, d_clusters)
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
