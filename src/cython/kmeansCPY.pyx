#!python

from cython.parallel cimport parallel, prange, threadid
from libc.stdlib cimport malloc, free

cdef extern from "../kmeans.h":
    extern void sumSq (float *x, float *c, float *ss,  int D,  int N,  int K,  int n,  int k)
    extern void selectCluster (float *x, float *c,  int *assign,  int N,  int D,  int K,  int *conv, float *dist)
    extern void clusterCenter (float *x, float *c,  int *assign,  int N,  int K,  int D,  int *count)
    extern void kMeans (float *x, float *c,  int *assign,  int N,  int K,  int D)


cpdef int kM(float[::,::1] x, float[::,::1] c, int[::1] assign):
    cdef  int N = x.shape[0]
    cdef  int D = x.shape[1]
    cdef  int K = c.shape[0]
    kMeans(&x[0,0], &c[0,0], &assign[0], N, K, D)

    return 0
