import time
import numpy as np
from itertools import chain

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
    for k in range(K):
        centers[k,:] = np.sum(data[labels==k],axis=0)
    return centers

def reassign_labels(labels,centers,data):
    old_labels = labels.copy()
    def minimize(x):
        return np.argmin(np.sum((centers-x)**2,axis=1)) #finds closest cluster
    labels[:] = np.apply_along_axis(minimize,1,data)
    return np.array_equal(labels,old_labels)

def mpikmeans(data, initial_labels, K, D, limit, comm):

    size = comm.Get_size()
    rank = comm.Get_rank()
    start = time.time()
    count = 0
    runtime = 0
    centers = np.empty((K, D))

    # break up labels and data into roughly equal groups for each CPU in MPI.COMM_WORlD
    print(size)
    allocations,labels = partition(initial_labels.copy(),size)
    indices = allotment_to_indices(allocations)
    indices,labels = comm.scatter(zip(indices, labels), root=0)
    data = data[indices[0]:indices[1]]

    for k in range(limit):

        compute_centers(labels,centers,data)
        centers = comm.gather(centers, root=0)
        collected_labels = comm.gather(labels, root=0)

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
        converged = reassign_labels(labels,centers,data)

        converged = comm.allgather(converged)
        converged = np.all(converged)
        if converged: break

    labels = comm.gather(labels,root=0)
    if rank==0: labels = np.array(list(chain(*labels)))
    labels = comm.bcast(labels, root=0)
    count = comm.bcast(count, root=0)
    distortion = 100
    ai = 600*count
    if rank==0: runtime = time.time() - start
    runtime = comm.bcast(runtime, root=0)

    return centers, labels, count, runtime, distortion, ai

