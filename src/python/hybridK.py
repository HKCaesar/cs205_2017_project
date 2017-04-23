import time
import numpy as np
from itertools import chain
from mpiK import *
from cudaK import *

def hybridkmeans(data, initial_labels, kernel_fn, N, K, D, limit, comm):

    size = comm.Get_size()
    rank = comm.Get_rank()
    count = 0
    runtime = 0
    centers = np.empty((K, D))
    kernel1, kernel2 = parallel_mod(kernel_fn, N, K, D)
    start = time.time()

    # break up labels and data into roughly equal groups for each CPU in MPI.COMM_WORlD
    allocations = partition(N, size)
    indices = allotment_to_indices(allocations)
    index = comm.scatter(indices, root=0)
    data_chunk = data[index[0]:index[1]]
    labels_chunk = initial_labels[index[0]:index[1]]

    # prep CUDA stuff

    h_data, h_labels, h_centers, h_converged_array = prep_host(data_chunk, labels_chunk, K, D)
    d_data, d_labels, d_centers, d_converged_array = prep_device(h_data, h_labels, h_centers, h_converged_array)

    for k in range(limit):

        compute_centers(labels_chunk,centers,data_chunk)
        centers = comm.gather(centers, root=0)
        collected_labels = comm.gather(labels_chunk, root=0)

        if rank==0:
            count += 1
            temp_centers = np.empty((K, D))
            for center in centers:
                temp_centers+=center
            collected_labels = np.array(list(chain(*collected_labels)))
            for j in range(K):
                total = np.sum(collected_labels==j)
                temp_centers[j,:] = temp_centers[j,:]/total
            centers = temp_centers

        centers = comm.bcast(centers, root=0)
        converged = reassign_labels(labels_chunk,centers,data_chunk)

        converged = comm.allgather(converged)
        converged = np.all(converged)
        if converged: break

    labels = comm.gather(labels_chunk,root=0)
    if rank==0: labels = np.array(list(chain(*labels)))
    labels = comm.bcast(labels, root=0)
    count = comm.bcast(count, root=0)
    distortion = 100
    ai = 600*count
    if rank==0: runtime = time.time() - start
    runtime = comm.bcast(runtime, root=0)

    return centers, labels, count, runtime, distortion, ai