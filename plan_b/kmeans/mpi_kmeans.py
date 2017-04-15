from mpi4py import MPI
from kmeans.utilities import compute_means, reassign_labels, generate_initial_assignment, partition, distortion, allotment_to_indices
import numpy as np
import time
from itertools import chain
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

    n_data, n_dimensions = data.shape

    centers = np.zeros((n_clusters,n_dimensions))
    labels = generate_initial_assignment(n_data,n_clusters)
    all_labels = labels

    if rank==0:
        compute_means(labels,centers,data)
        print("first mean:", distortion(labels,centers,data))
        print(centers)

    allocations,labels = partition(labels,size)

    labels = labels[rank]

    indices = allotment_to_indices(allocations)

    data = data[indices[rank][0]:indices[rank][1]]


    for k in range(max_iter):

        compute_means(labels,centers,data,sum_values=True)

        centers = comm.gather(centers, root=0)

        if rank==0:

            temp = np.zeros((n_clusters,n_dimensions))

            for center in centers:
                temp+=center

            centers = temp


        collected_labels = comm.gather(labels, root=0)

        if rank == 0:
            collected_labels = np.array(list(chain(*collected_labels)))

            for j in range(n_clusters) :
                total = np.sum(collected_labels==j)
                centers[j,:] = centers[j,:]/total


            if k==0:
                print("first p means:")
                print(centers)

            print(k, distortion(all_labels,centers,all_data))



        centers = comm.bcast(centers, root=0)

        converged = reassign_labels(labels,centers,data)


        converged = comm.gather(converged,root=0)

        print("2.", labels)

        converged = np.all(converged)

        if converged: break


    labels = comm.gather( [rank, labels] ,root=0)

    comm.Finalize()

    if rank==0:
        labels = np.array(list(chain(*labels)))
        timing = time.time()-start
        return [centers,labels,timing]