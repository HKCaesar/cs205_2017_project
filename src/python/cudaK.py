import numpy as np
import time
import string

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

######################################################
### pyCUDA K-MEANS  ####
######################################################

# define h_vars on host
def prep_host(data, initial_labels, K, D, blockDimX):
    N = data.shape[0]
    h_data = data.copy()
    h_data = np.asfortranarray(h_data,dtype=np.float64)
    h_labels = initial_labels.copy()
    h_centers = np.ascontiguousarray(np.empty((K, D), dtype=np.float64, order='C'))
    h_converged_array = np.zeros((blockDimX), dtype=np.intc)
    h_dist = np.zeros((N,K), dtype=np.float64, order='F')
    h_clustern = np.zeros((K), dtype=np.intc)
    return h_data, h_labels, h_centers, h_converged_array, h_dist, h_clustern

# allocate memory and copy data to d_vars on device
def prep_device(h_data, h_labels, h_centers, h_converged_array, h_dist, h_clustern):
    d_data = cuda.mem_alloc(h_data.nbytes)
    d_labels = cuda.mem_alloc(h_labels.nbytes)
    d_centers = cuda.mem_alloc(h_centers.nbytes)
    d_converged_array = cuda.mem_alloc(h_converged_array.nbytes)
    d_dist = cuda.mem_alloc(h_dist.nbytes)
    d_clustern = cuda.mem_alloc(h_clustern.nbytes)
    
    cuda.memcpy_htod(d_data, h_data)
    cuda.memcpy_htod(d_labels, h_labels)
    cuda.memcpy_htod(d_converged_array, h_converged_array)
    cuda.memcpy_htod(d_dist, h_dist)
    cuda.memcpy_htod(d_clustern, h_clustern)
    
    return d_data, d_labels, d_centers, d_converged_array, d_dist, d_clustern

# define kernels
def parallel_mod(kernel_fn, N, K, D, numThreads):
    template = string.Template(open(kernel_fn, "r").read())
    code = template.substitute(N=N, K=K, D=D, numThreads=numThreads)
    mod = SourceModule(code)
    dist = mod.get_function("dist")
    reassign = mod.get_function("reassign")
    countCluster = mod.get_function("countCluster")
    newMeans = mod.get_function("newMeans")
    
    return dist, reassign, countCluster, newMeans

# run pyCUDA
def cudakmeans(data, initial_labels, kernel_fn, N, K, D, numThreads, limit, standardize_count):
#    count = 0
    blockDimX = N/numThreads
    if (blockDimX*numThreads < N) : blockDimX += 1
    
    dist, reassign, countCluster, newMeans = parallel_mod(kernel_fn, N, K, D, numThreads)
    h_data, h_labels, h_centers, h_converged_array, h_dist, h_clustern = prep_host(data, initial_labels, K, D, blockDimX)
    
    start = time.time()
    d_data, d_labels, d_centers, d_converged_array, d_dist, d_clustern = prep_device(h_data, h_labels, h_centers, h_converged_array, h_dist, h_clustern)
    if standardize_count>0: loop_limit = standardize_count
    else: loop_limit=limit
    
    for i in range(loop_limit):
        countCluster(d_labels, d_clustern, block=(numThreads, 1, 1), grid=(1, K, 1))
        newMeans(d_data, d_labels, d_centers, d_clustern, block=(numThreads, 1, 1), grid=(K, D, 1))
        dist(d_data, d_centers, d_dist, block=(numThreads, 1, 1), grid=(blockDimX, K, 1))
        reassign(d_dist, d_labels, d_converged_array, block=(numThreads, 1, 1), grid=(blockDimX, 1, 1))
        
        cuda.memcpy_dtoh(h_converged_array, d_converged_array)
        #        count += 1
        if standardize_count == 0:
            if np.mean(h_converged_array) == 1: break
    
    cuda.memcpy_dtoh(h_centers, d_centers)
    cuda.memcpy_dtoh(h_labels, d_labels)
    
    runtime = time.time() - start
    return h_centers, h_labels, i, runtime
