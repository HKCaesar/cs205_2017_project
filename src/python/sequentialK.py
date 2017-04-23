from sklearn.cluster import KMeans
import numpy as np
import time

######################################################
### SEQUENTIAL K-MEANS ###
######################################################

def sequential(data, initial_labels, N, D, K, limit):
    centers = np.empty((K, D))
    labels = initial_labels.copy()
    clustern = np.empty(K)
    count = 0
    converged = False
    start = time.time()

    while not converged:

        converged = True

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
                converged = False

        count += 1
        if count == limit: break

    runtime = time.time() - start
    ai = 200 * count
    distortion = 100

    return centers, labels, count, runtime, distortion, ai


######################################################
### STOCK K-MEANS ###
######################################################

def stock(data, K, count):
    start = time.time()
    stockmeans = KMeans(n_clusters=K, n_init=count)
    stockmeans.fit(data)
    runtime = time.time() - start
    ai = 100 * count
    return stockmeans.cluster_centers_, stockmeans.labels_, count, runtime, stockmeans.inertia_, ai