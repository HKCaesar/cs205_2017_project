import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import time
import string

######################################################
### PREP DATA & INITIAL LABELS ####
######################################################

def prep_data(data_fn, d_list, N, D, K):

  # import data file and subset data for k-means
  reviewdata = pd.read_csv(data_fn)
  print(reviewdata.shape)
  data = reviewdata[d_list[:D]][:N].values
  data = np.ascontiguousarray(data, dtype=np.float64)

  # assign random clusters & shuffle 
  initial_labels = np.ascontiguousarray(np.zeros(N,dtype=np.intc, order='C'))
  for n in range(N):
      initial_labels[n] = n%K
  for i in range(len(initial_labels)-2,-1,-1):
      j= np.random.randint(0,i+1) 
      temp = initial_labels[j]
      initial_labels[j] = initial_labels[i]
      initial_labels[i] = temp
  
  return data, initial_labels

######################################################
### STOCK K-MEANS ###
######################################################

def stock(data, K, limit):
  
    start = time.time()
    stockmeans = KMeans(n_clusters=K,n_init=limit)
    stockmeans.fit(data)
    runtime = time.time()-start
    
    return stockmeans.cluster_centers_, stockmeans.labels_, stockmeans.inertia_, runtime

######################################################
### SEQUENTIAL K-MEANS ###
######################################################

def sequential(data, initial_labels, N, D, K, limit):
  
  A = np.zeros((K,D))
  W = initial_labels.copy()
  X = data
  m = np.zeros(K)
  count = 0
  converged = False
  start = time.time()
  
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
      if count==1:
        A1 = A.copy()
        W1 = W.copy()  
      if count==limit: break
        
  runtime = time.time()-start
  
  return A, W, count, runtime, A1, W1

######################################################
### pyCUDA K-MEANS  ####
######################################################

# define h_vars on host
def prep_host(data, initial_labels, K, D):
  
  h_data = data.copy()
  h_labels = initial_labels.copy()
  h_means = np.ascontiguousarray(np.empty((K,D),dtype=np.float64, order='C'))
  h_converged_array = np.zeros((data.shape[0]),dtype=np.intc)
  return h_data, h_labels, h_means, h_converged_array

# allocate memory and copy data to d_vars on device
def prep_device(h_data, h_labels, h_means, h_converged_array, h_converged):
  
  d_data = cuda.mem_alloc(h_data.nbytes)
  d_labels = cuda.mem_alloc(h_labels.nbytes)
  d_means = cuda.mem_alloc(h_means.nbytes)
  d_converged_array = cuda.mem_alloc(h_converged_array.nbytes)
  d_converged = cuda.mem_alloc(h_converged.nbytes)
  cuda.memcpy_htod(d_data,h_data)
  cuda.memcpy_htod(d_labels,h_labels)
  cuda.memcpy_htod(d_converged_array, h_converged_array)
  cuda.memcpy_htod(d_converged, h_converged)
  
  return d_data, d_labels, d_means, d_converged_array, d_converged

# define kernels
def parallel_mod(kernel_fn, N, K, D):
    
    template = string.Template(open(kernel_fn, "r").read())
    code = template.substitute(N = N, K = K, D = D)
    mod = SourceModule(code)
    kernel1 = mod.get_function("newMeans")
    kernel2 = mod.get_function("reassign")
    
    return kernel1, kernel2
  
def pyCUDA(data, initial_labels, kernel_fn, N, K, D, limit):
    
    h_converged = np.zeros((1),dtype=np.intc)
    count = 0
    kernel1, kernel2 = parallel_mod(kernel_fn, N, K, D)
    h_data, h_labels, h_means, h_converged_array = prep_host(data, initial_labels, K, D)

    start = time.time()
    d_data, d_labels, d_means, d_converged_array, d_converged = prep_device( h_data, h_labels, h_means, h_converged_array, h_converged)

    while not h_converged:
        kernel1(d_data, d_labels, d_means, block=(K,D,1), grid=(1,1,1))
        kernel2(d_data, d_labels, d_means, d_converged_array, d_converged, block=(K,D,1), grid=(N,1,1))
        cuda.memcpy_dtoh(h_converged, d_converged)
        count +=1
        if count==limit: break
          
    cuda.memcpy_dtoh(h_means, d_means)
    cuda.memcpy_dtoh(h_labels, d_labels)
    runtime = time.time()-start
    
    return h_means, h_labels, count, runtime

######################################################
### mpi4py K-MEANS  ####
######################################################

def mpi4py(data, initial_labels, kernel_fn, N, K, D, limit):
  
  start = time.time()
  count = 0
  h_means = np.ascontiguousarray(np.empty((K,D),dtype=np.float64, order='C'))
  h_labels = np.ascontiguousarray(np.empty(initial_labels.shape,dtype=np.intc, order='C'))
  runtime = time.time()-start
  
  return h_means, h_labels, count, runtime

######################################################
### hybrid K-MEANS  ####
######################################################

def hybrid(data, initial_labels, kernel_fn, N, K, D, limit):
  
  start = time.time()
  count = 0
  h_means = np.ascontiguousarray(np.empty((K,D),dtype=np.float64, order='C'))
  h_labels = np.ascontiguousarray(np.empty(initial_labels.shape,dtype=np.intc, order='C'))
  runtime = time.time()-start
  
  return h_means, h_labels, count, runtime
