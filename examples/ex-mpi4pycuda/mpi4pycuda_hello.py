#!/usr/bin/env python

"""
Demo of how to pass GPU memory managed by pycuda to mpi4py.
Notes
-----
This code can be used to perform peer-to-peer communication of data via
NVIDIA's GPUDirect technology if mpi4py has been built against a
CUDA-enabled MPI implementation.
"""

import atexit
import sys

# PyCUDA 2014.1 and later have built-in support for wrapping GPU memory with a
# buffer interface:
import pycuda
if pycuda.VERSION >= (2014, 1):
    bufint = lambda arr: arr.gpudata.as_buffer(arr.nbytes)
else:
    import cffi
    ffi = cffi.FFI()
    bufint = lambda arr: ffi.buffer(ffi.cast('void *', arr.ptr), arr.nbytes)

import numpy as np
from mpi4py import MPI

import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
drv.init()

def dtype_to_mpi(t):
    if hasattr(MPI, '_typedict'):
        mpi_type = MPI._typedict[np.dtype(t).char]
    elif hasattr(MPI, '__TypeDict__'):
        mpi_type = MPI.__TypeDict__[np.dtype(t).char]
    else:
        raise ValueError('cannot convert type')
    return mpi_type

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

N_gpu = drv.Device(0).count()
if N_gpu < 2:
    sys.stdout.write('at least 2 GPUs required')
else:
    dev = drv.Device(rank)
    ctx = dev.make_context()
    atexit.register(ctx.pop)
    atexit.register(MPI.Finalize)

    if rank == 0:
        x_gpu = gpuarray.arange(100, 200, 10, dtype=np.double)
        print ('before (%i): ' % rank)+str(x_gpu)
        comm.Send([bufint(x_gpu), dtype_to_mpi(x_gpu.dtype)], dest=1)
        print 'sent'
        print ('after  (%i): ' % rank)+str(x_gpu)
    elif rank == 1:
        x_gpu = gpuarray.zeros(10, dtype=np.double)
        print ('before (%i): ' % rank)+str(x_gpu)
        comm.Recv([bufint(x_gpu), dtype_to_mpi(x_gpu.dtype)], source=0)
        print 'received'
        print ('after  (%i): ' % rank)+str(x_gpu)
