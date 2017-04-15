from mpi4py import MPI
from kmeans.utilities import compute_means, reassign_labels, generate_initial_assignment, partition, distortion
import numpy as np
import time

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
    size = MPI.Get_size()

    start = time.time()

    n_data, n_dimensions = data.shape
    centers = np.zeros((n_clusters,n_dimensions))

    labels = generate_initial_assignment(n_data,n_clusters)
    allocations,labels = partition(labels,size-1)
    labels = comm.scatter(labels, root=0)

    if rank !=0:
        data   = data[labels]
        n_data = allocations[rank]

    for k in range(max_iter):
        if rank !=0:
            compute_means(labels,centers,sum_values=True)

        centers = comm.gather(centers, root=0)

        if rank==0:
            centers = np.mean(centers)

        centers = comm.bcast(data, root=0)

        if rank != 0:
            converged = reassign_labels(labels,centers,data)

        converged = comm.gather(converged,root=0)

        if rank == 0:
            print( distortion(labels, centers, data)  )

            if np.all(converged): break





    timing = time.time()-start

    print(timing)

    return [centers,labels,timing]