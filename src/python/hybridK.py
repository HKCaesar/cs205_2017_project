from mpiK import *
from cudaK import *

def hybridkmeans(data, initial_labels, kernel_fn, N, K, D, numThreads, limit, standardize_count, comm):
    
    size = comm.Get_size()
    rank = comm.Get_rank()
    count = 0
    runtime = 0
    if standardize_count>0: loop_limit = standardize_count
    else: loop_limit=limit
    start = time.time()
    
    # break up labels and data into roughly equal groups for each CPU in MPI.COMM_WORlD
    allocations = partition(N, size)
    indices = allotment_to_indices(allocations)
    index = comm.scatter(indices, root=0)
    data_chunk = data[index[0]:index[1]]
    labels_chunk = initial_labels[index[0]:index[1]]
    
    # prep CUDA stuff
    
    print(cuda.mem_get_info())
    blockDimX = data_chunk.shape[0]/numThreads
    if (blockDimX*numThreads < data_chunk.shape[0]) : blockDimX += 1
    dist, reassign, countCluster, newMeans = parallel_mod(kernel_fn, data_chunk.shape[0], K, D, numThreads)
    h_data, h_labels, h_centers, h_converged_array, h_dist, h_clustern = prep_host(data_chunk, labels_chunk, K, D, blockDimX)
    d_data, d_labels, d_centers, d_converged_array, d_dist, d_clustern = prep_device(h_data, h_labels, h_centers, h_converged_array, h_dist, h_clustern)
    
    for k in range(loop_limit):
        
        countCluster(d_labels, d_clustern, block=(numThreads, 1, 1), grid=(1, K, 1))
        newMeans(d_data, d_labels, d_centers, d_clustern, block=(numThreads, 1, 1), grid=(K, D, 1))
        comm.Barrier()
        cuda.memcpy_dtoh(h_centers, d_centers)
        #cuda.memcpy_dtoh(h_labels, d_labels) # if use cluster count, don't need labels
        cuda.memcpy_dtoh(h_clustern, d_clustern)
        centers = comm.gather(h_centers, root=0)
        #collected_labels = comm.gather(h_labels, root=0)
        collected_count = comm.gather(h_clustern, root=0)
        
        if rank==0:
            count += 1
            temp_centers = np.empty((K, D))
            temp_count = np.empty(K)
            for i in range(len(centers)):
                temp_sum = centers[i]
                for k in range(K) : temp_sum[k,:] *= float(collected_count[i][k])
                temp_centers += temp_sum
                temp_count   += collected_count[i]
            for j in range(K):
                if temp_count[j] != 0:
                    temp_centers[j,:] = temp_centers[j,:]/float(temp_count[j])
                else:
                    temp_centers[j,:] = 0 #hacky way to prevent 0 division
            h_centers = temp_centers
        
        h_centers = comm.bcast(h_centers, root=0)
        cuda.memcpy_htod(d_centers, h_centers)
        dist(d_data, d_centers, d_dist, block=(numThreads, 1, 1), grid=(blockDimX, K, 1))
        reassign(d_dist, d_labels, d_converged_array, block=(numThreads, 1, 1), grid=(blockDimX, 1, 1))
        cuda.memcpy_dtoh(h_converged_array, d_converged_array)
        converged = np.all(h_converged_array)
        converged = comm.allgather(converged)
        converged = np.all(converged)
        if standardize_count == 0:
            if converged: break
    
    cuda.memcpy_dtoh(h_labels, d_labels)
    print(cuda.mem_get_info())
    
    labels = comm.gather(h_labels,root=0)
    if rank==0: labels = np.array(list(chain(*labels)))
    labels = comm.bcast(labels, root=0)
    count = comm.bcast(count, root=0)
    if rank==0: runtime = time.time() - start
    runtime = comm.bcast(runtime, root=0)
    
    return h_centers, labels, count, runtime
