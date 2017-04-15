from mpi4py import MPI
from kmeans.utilities import compute_means, reassign_labels, generate_initial_assignment, partition, distortion
import numpy as np
import time
import sys

def kmeans_sequential(data, n_clusters,max_iter=100):

    start = time.time()

    n_data, n_dimensions = data.shape
    centers = np.zeros((n_clusters,n_dimensions))
    labels = generate_initial_assignment(n_data,n_clusters)

    for k in range(max_iter):
        compute_means(labels,centers,data)
        converged = reassign_labels(labels,centers,data)
        if converged: break

    timing = time.time()-start

    return centers, labels, timing

def mpi_kmeans(data, n_clusters,max_iter=100):

    all_data = data

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start = time.time()

    print('started timing')

    n_data, n_dimensions = data.shape

    print("dim: ",n_data, n_dimensions)

    centers = np.zeros((n_clusters,n_dimensions))
    labels = generate_initial_assignment(n_data,n_clusters)
    allocations,labels = partition(labels,size)
    labels = comm.scatter(labels, root=0)

    data = data[labels]

    for k in range(max_iter):

        compute_means(labels,centers,data,sum_values=True)

        centers = comm.gather(centers, root=0)

        if rank==0:
            print(type(centers))
            centers = np.sum(centers,axis=0)/n_data
            print(k, distortion(labels,centers,data))

        centers = comm.bcast(centers, root=0)

        converged = reassign_labels(labels,centers,all_data)

        converged = comm.gather(converged,root=0)

        converged = np.all(converged)

        if converged: break


    if rank==0:
        timing = time.time()-start
        print(timing)
        return [centers,labels,timing]