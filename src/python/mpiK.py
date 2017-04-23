import time
import numpy as np
from mpi4py import MPI
from itertools import chain
import sys

def allotment_to_indices(allotments):
    indices = np.cumsum(allotments)
    indices=np.append(indices,[0])
    indices=np.sort(indices)
    indices=np.column_stack([indices[:-1],indices[1:]])
    return(indices)

def partition(sequence, n_chunks):
    N = len(sequence)
    chunk_size = int(N/n_chunks)
    left_over = N-n_chunks*chunk_size
    allocations = np.array([chunk_size]*n_chunks)
    left_over=([1]*left_over)+([0]*(n_chunks-left_over))
    np.random.shuffle(left_over)
    allocations = left_over+allocations
    indexes = allotment_to_indices(allocations)
    return allocations, [sequence[index[0]:index[1]]  for index in indexes]

def compute_centers(labels, centers, data):
    K,D=centers.shape
    print(centers)
    for k in range(K):
        centers[k,:] = np.sum(data[labels==k],axis=0)
    return centers

def reassign_labels(labels,centers,data):
    old_labels = labels.copy()
    def minimize(x):
        return np.argmin(np.sum((centers-x)**2,axis=1)) #finds closest cluster
    labels[:] = np.apply_along_axis(minimize,1,data)
    return np.array_equal(labels,old_labels)

def mpikmeans(data, initial_labels, K, D, limit):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start = time.time()
    centers = np.empty((K, D))

    # break up labels and data into roughly equal groups for each CPU in MPI.COMM_WORlD
    allocations,labels = partition(initial_labels.copy(),size)
    indices = allotment_to_indices(allocations)
    indices,labels = comm.scatter(zip(indices, labels), root=0)
    data = data[indices[0]:indices[1]]

    for k in range(limit):

        compute_centers(labels,centers,data)
        centers = comm.gather(centers, root=0)
        collected_labels = comm.gather(labels, root=0)

        if rank==0:
            temp_centers = np.empty((K, D))
            for center in centers:
                temp_centers+=center
            collected_labels = np.array(list(chain(*collected_labels)))
            for j in range(K):
                total = np.sum(collected_labels==j)
                temp_centers[j,:] = temp_centers[j,:]/total
            centers = temp_centers

        centers = comm.bcast(centers, root=0)
        converged = reassign_labels(labels,centers,data)
        converged = comm.allgather(converged)
        print(converged)
        converged = np.all(converged)
        print(converged)

        if converged: break

    labels = comm.gather(labels,root=0)

    if rank==0:
        labels = np.array(list(chain(*labels)))
        timing = time.time()-start
        return [centers,labels,timing]
    else:
        sys.exit(0)
