#!python

from cython.parallel cimport parallel, prange, threadid
from libc.stdlib cimport malloc, free

cdef extern from "src/kmeans.h":
    extern void sumSq (double *x, double *c, double *ss,  int D,  int N,  int n,  int k)
    extern void selectCluster (double *x, double *c,  int *assign,  int N,  int D,  int K,  int *conv, double *dist)
    extern void clusterCenter (double *x, double *c,  int *assign,  int N,  int K,  int D,  int *count)
    extern void kMeans (double *x, double *c,  int *assign,  int N,  int K,  int D)


cpdef int kM(double[:,:] x, double[:,:] c, int[:] assign):
    cdef  int N = x.shape[0]
    cdef  int D = x.shape[1]
    cdef  int K = c.shape[0]

    cdef  double[::,::1] xC = x.copy()
    cdef  double[::,::1] cC = c.copy()
    cdef  int[::1] assignC = assign.copy()

    kMeans(&xC[0,0], &cC[0,0], &assignC[0], N, K, D)


    assign = assignC
    c = cC

    return 0
