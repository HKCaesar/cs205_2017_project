import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from sklearn.cluster import KMeans
import string
import pandas as pd
import numpy as np
import time
import csv


######################################################
### DEFINE SEQUENTIAL K-MEANS FUNCTiON ###
######################################################


def sequential(N, K, D, data, initial_clusters):
  
  A = np.zeros((K,D))
  W = initial_clusters.copy()
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

def pnaive_mod(N, K, D):
  
    template = string.Template(open("../cuda/pycumean.c", "r").read())
    code = template.substitute(N = N,
                           K = K,
                           D = D)
    mod = SourceModule(code)
    kernel1 = mod.get_function("newMeans")
    kernel2 = mod.get_function("reassign")
  
    return kernel1, kernel2

######################################################
### DEFINE "IMPROVED" PARALLEL K-MEANS KERNEL FOR GPU ####
######################################################

def pimproved_mod(N, K, D):
  
  template = string.Template(open("../cuda/pycumean.c", "r").read())
  code = template.substitute(N = N,
                                 K = K,
                                 D = D)
  mod = SourceModule(code)
  kernel1 = mod.get_function("newMeans")
  kernel2 = mod.get_function("reassign")
  
  return kernel1, kernel2



######################################################
### DEFINTE FUNCTIONS TO MANAGE DATA ####
######################################################

# download data and assign initial clusters
def prep_data(K, data_fn, d_list, limit):

  # import data file and subset data for k-means
  reviewdata = pd.read_csv(data_fn)
  data = reviewdata[d_list][:limit].values
  data = np.ascontiguousarray(data, dtype=np.float64)
  N,D=data.shape

  # assign random clusters & shuffle 
  initial_clusters = np.ascontiguousarray(np.zeros(N,dtype=np.intc, order='C'))
  for n in range(N):
      initial_clusters[n] = n%K
  for i in range(len(initial_clusters)-2,-1,-1):
      j= np.random.randint(0,i+1) 
      temp = initial_clusters[j]
      initial_clusters[j] = initial_clusters[i]
      initial_clusters[i] = temp
  
  return data, initial_clusters, N, D

# define h_vars on host
def prep_host(data, initial_clusters, K, D):
  h_data = data
  h_clusters = initial_clusters.copy()
  h_means = np.ascontiguousarray(np.zeros((K,D),dtype=np.float64, order='C'))
  h_distortion = 0
  return h_data, h_clusters, h_means, h_distortion

# allocate memory and copy data to d_vars on device
def prep_device(h_data, h_clusters, h_means, h_distortion):
  d_data = cuda.mem_alloc(h_data.nbytes)
  d_clusters = cuda.mem_alloc(h_clusters.nbytes)
  d_means = cuda.mem_alloc(h_means.nbytes)
  d_distortion = cuda.mem_alloc(np.array(h_distortion).astype(np.intc).nbytes)
  cuda.memcpy_htod(d_data,h_data)
  cuda.memcpy_htod(d_clusters,h_clusters)
  return d_data, d_clusters, d_means, d_distortion

# reset h_vars
def reset_hvars(initial_clusters, h_means, h_distortion, K, D):
  
  h_clusters = initial_clusters.copy()
  h_means = np.ascontiguousarray(np.zeros((K,D),dtype=np.float64, order='C'))
  h_distortion = 0
  return


