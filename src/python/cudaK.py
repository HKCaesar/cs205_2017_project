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
def prep_host(data, initial_labels, K, D):
    h_data = data.copy()
    h_labels = initial_labels.copy()
    h_centers = np.ascontiguousarray(np.empty((K, D), dtype=np.float64, order='C'))
    h_converged_array = np.zeros((data.shape[0]), dtype=np.intc)
    return h_data, h_labels, h_centers, h_converged_array

# allocate memory and copy data to d_vars on device
def prep_device(h_data, h_labels, h_centers, h_converged_array):
    d_data = cuda.mem_alloc(h_data.nbytes)
    d_labels = cuda.mem_alloc(h_labels.nbytes)
    d_centers = cuda.mem_alloc(h_centers.nbytes)
    d_converged_array = cuda.mem_alloc(h_converged_array.nbytes)
    cuda.memcpy_htod(d_data, h_data)
    cuda.memcpy_htod(d_labels, h_labels)
    cuda.memcpy_htod(d_converged_array, h_converged_array)

    return d_data, d_labels, d_centers, d_converged_array

# define kernels
def parallel_mod(kernel_fn, N, K, D):
    template = string.Template(open(kernel_fn, "r").read())
    code = template.substitute(N=N, K=K, D=D)
    mod = SourceModule(code)
    kernel1 = mod.get_function("newCenters")
    kernel2 = mod.get_function("reassign")

    return kernel1, kernel2

# run pyCUDA
def cudakmeans(data, initial_labels, kernel_fn, N, K, D, limit, standardize_count):
    count = 0
    kernel1, kernel2 = parallel_mod(kernel_fn, N, K, D)
    h_data, h_labels, h_centers, h_converged_array = prep_host(data, initial_labels, K, D)

    start = time.time()
    d_data, d_labels, d_centers, d_converged_array = prep_device(h_data, h_labels, h_centers, h_converged_array)
    if standardize_count>0: loop_limit = standardize_count
    else: loop_limit=limit

    for i in range(loop_limit):
        kernel1(d_data, d_labels, d_centers, block=(K, D, 1), grid=(1, 1, 1))
        kernel2(d_data, d_labels, d_centers, d_converged_array, block=(K, D, 1), grid=(N, 1, 1))
        cuda.memcpy_dtoh(h_converged_array, d_converged_array)
        count += 1
        if standardize_count == 0:
            if np.sum(h_converged_array) == 0: break

    cuda.memcpy_dtoh(h_centers, d_centers)
    cuda.memcpy_dtoh(h_labels, d_labels)

    runtime = time.time() - start
    ai = 0 * count
    distortion = 0

    return h_centers, h_labels, count, runtime, distortion, ai
