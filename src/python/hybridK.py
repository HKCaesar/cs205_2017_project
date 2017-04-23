import time
import numpy as np

######################################################
### hybrid K-MEANS  ####
######################################################

def hybrid(data, initial_labels, kernel_fn, N, K, D, limit):
    start = time.time()
    count = 0
    h_centers = np.ascontiguousarray(np.empty((K, D), dtype=np.float64, order='C'))
    h_labels = np.ascontiguousarray(np.empty(initial_labels.shape, dtype=np.intc, order='C'))
    runtime = time.time() - start
    ai = 500 * count
    distortion = 100

    return h_centers, h_labels, count, runtime, distortion, ai