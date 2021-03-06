
import numpy as np
import time
import cython
from cython.parallel cimport prange, parallel

@cython.boundscheck(False)
def cython_kmeans(double [:,:] data, int [:] initial_labels, int N, int D, int K, int limit, int standardize_count):
    cdef double [:,:] centers = np.empty((K, D))
    cdef int [:] labels = initial_labels.copy()
    cdef int [:] clustern = np.empty(K)
    cdef int count = 0

    if standardize_count>0: loop_limit = standardize_count
    else: loop_limit=limit
    start = time.time()

    cdef int converged

    cdef double temp

    cdef double min_val
    cdef int min_ind

    for i in range(loop_limit):

        converged = 1

        # compute centers
        for k in range(K):
            for d in range(D):
                centers[k, d] = 0
            clustern[k] = 0
        for n in range(N):
            for d in range(D):
                centers[labels[n], d] += data[n, d]
            clustern[labels[n]] += 1
        for k in range(K):
            for d in range(D):
                centers[k, d] = centers[k, d] / clustern[k]

        # assign to closest center
        for n in range(N):
            min_val = np.inf
            min_ind = -1
            for k in range(K):
                temp = 0
                for d in range(D):
                    temp += (data[n, d] - centers[k, d]) ** 2

                if temp < min_val:
                    min_val = temp
                    min_ind = k
            if min_ind != labels[n]:
                labels[n] = min_ind
                converged = 0

        count += 1
        if standardize_count == 0:
            if converged: break

    runtime = time.time() - start
    return centers, labels, count, runtime
