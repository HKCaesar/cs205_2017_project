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

        print("1. rank: %d, mean: %r",rank, centers.shape )

        centers = comm.gather(centers, root=0)

        if rank==0:
            print(type(centers))
            centers = np.sum(centers,axis=0)/n_data
            print("2. rank: %d, mean: %r",rank, centers.shape)
            centers = comm.bcast(centers, root=0)

        print("3. rank: %d, mean: %r",rank, centers.shape)

        converged = reassign_labels(labels,centers,data)

        converged = comm.gather(converged,root=0)

        if rank == 0:
            print("%d. "%k,distortion(labels, centers, data)  )

            if np.all(converged): break





    timing = time.time()-start

    print(timing)

    return [centers,labels,timing]