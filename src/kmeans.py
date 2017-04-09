import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import pandas as pd

####################################
### CONFIGURE ####
####################################

data_fn = "../data/reviewer-data.csv"
#data_fn = "../data/reviewer-data-sample.csv"

####################################
### KERNELS ####
####################################

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

####################################
### GRAB DATA ####
####################################

# import data
data = pd.read_csv(data_fn)
data=data[:1000]
print(len(data))

####################################
### ALLOCATE INPUT ####
####################################

# Allocate input on host
a = numpy.array(8)
b = numpy.array(2)
c = numpy.array(0)

a = a.astype(numpy.int32)
b = b.astype(numpy.int32)
c = c.astype(numpy.int32)

####################################
### ALLOCATE INPUT & COPY TO GPU ####
####################################

# Allocate on device
a_gpu = cuda.mem_alloc(a.size * a.dtype.itemsize)
b_gpu = cuda.mem_alloc(b.size * b.dtype.itemsize)
c_gpu = cuda.mem_alloc(c.size * c.dtype.itemsize)

# Copy from host to device
cuda.memcpy_htod(a_gpu, a)
cuda.memcpy_htod(b_gpu, b)

####################################
### RUN K-MEANS ####
####################################


