#!python

from cython.parallel cimport parallel, prange, threadid

import numpy as np
cimport numpy as np
np.import_array()

cdef extern from "src/kmeans.h":
     double sumSq (double *x, double *c, int D,  int N,  int n,  int k)
     void selectCluster (double *x, double *c,  int *assign,  int N,  int D,  int K,  int *conv)
     void clusterCenter (double *x, double *c,  int *assign,  int N,  int K,  int D,  int *count)
     void kMeans (double *x, double *c,  int *assign,  int N,  int K,  int D, double cC)


cpdef void kM(double[:,:] x, double[:,:] c, int[:] assign):
    cdef  int N = x.shape[0]
    cdef  int D = x.shape[1]
    cdef  int K = c.shape[0]

    if assign.shape[0] != N:
        print("Warning!")

    cdef np.ndarray[np.float64_t, ndim=2, mode = 'c'] x_buff = np.ascontiguousarray(x, dtype = np.float64)
    cdef np.ndarray[np.float64_t, ndim=2, mode = 'c'] c_buff = np.ascontiguousarray(c, dtype = np.float64)
    cdef np.ndarray[np.int_t, ndim=1, mode = 'c'] assign_buff = np.ascontiguousarray(assign, dtype = np.int)

    cdef  double * xC = <double*> x_buff.data
    cdef  double * cC = <double*> c_buff.data
    cdef  int * assignC = <int*> assign_buff.data
    cdef double cd = c.data

    kMeans( xC, cC, assignC, N, K, D, cd)

    c = cd[::]


cpdef SS(double[:,:] x, double[:,:] c):
    cdef  int N = x.shape[0]
    cdef  int D = x.shape[1]
    cdef  int K = c.shape[0]

    cdef np.ndarray[np.float64_t, ndim=2, mode = 'c'] x_buff = np.ascontiguousarray(x, dtype = np.float64)
    cdef np.ndarray[np.float64_t, ndim=2, mode = 'c'] c_buff = np.ascontiguousarray(c, dtype = np.float64)

    cdef  double * xC = <double*> x_buff.data
    cdef  double * cC = <double*> c_buff.data
    cdef  int n
    cdef  int k

    for n in range(N):
        print(x_buff[n,0])
    print(sumSq(xC, cC, D, N, 0,  0))



#    for n in range(N):
#        for k in range(K):
#            print(sumSq(&x[0,0], &c[0,0], D, N, n,  k))


