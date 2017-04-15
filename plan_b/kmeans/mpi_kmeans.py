from mpi4py import MPI
from kmeans.utilities import compute_means, reassign_labels, generate_initial_assignment, partition, distortion
import numpy as np
import time
from itertools import chain

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

    centers = np.zeros((n_clusters,n_dimensions))
    labels = generate_initial_assignment(n_data,n_clusters)
    all_labels = labels
    allocations,labels = partition(labels,size)
    labels = comm.scatter(labels, root=0)

    print("1.",rank, labels)

    data = data[labels]

    for k in range(max_iter):

        compute_means(labels,centers,data,sum_values=True)

        centers = comm.gather(centers, root=0)

        if rank==0:
            print( len(centers) )

            temp = np.zeros((n_clusters,n_dimensions))

            for center in centers:
                temp+=center

            centers=temp/n_data
            print(k, distortion(all_labels,centers,all_data))

        centers = comm.bcast(centers, root=0)

        converged = reassign_labels(labels,centers,data)



        #print(rank, labels)

        converged = comm.gather(converged,root=0)

        print("2.")

        converged = np.all(converged)

        if converged: break


    labels = comm.gather( [rank, labels] ,root=0)

    if rank==0:
        print(labels)

    #labels = list(chain(*labels))

    #comm.Finalize()

    if rank==0:
        timing = time.time()-start
        #return [centers,labels,timing]