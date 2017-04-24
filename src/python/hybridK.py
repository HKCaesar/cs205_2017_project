from mpiK import *
from cudaK import *

def hybridkmeans(data, initial_labels, kernel_fn, N, K, D, limit, standardize_count, comm):

    size = comm.Get_size()
    rank = comm.Get_rank()
    count = 0
    runtime = 0
    if standardize_count>0: loop_limit = standardize_count
    else: loop_limit=limit
    kernel1, kernel2 = parallel_mod(kernel_fn, N, K, D)
    start = time.time()

    # break up labels and data into roughly equal groups for each CPU in MPI.COMM_WORlD
    allocations = partition(N, size)
    indices = allotment_to_indices(allocations)
    index = comm.scatter(indices, root=0)
    data_chunk = data[index[0]:index[1]]
    labels_chunk = initial_labels[index[0]:index[1]]

    # prep CUDA stuff

    print(cuda.mem_get_info())
    h_data, h_labels, h_centers, h_converged_array = prep_host(data_chunk, labels_chunk, K, D)
    d_data, d_labels, d_centers, d_converged_array = prep_device(h_data, h_labels, h_centers, h_converged_array)

    for k in range(loop_limit):

        kernel1(d_data, d_labels, d_centers, block=(K, D, 1), grid=(1, 1, 1))
        comm.Barrier()
        cuda.memcpy_dtoh(h_centers, d_centers)
        cuda.memcpy_dtoh(h_labels, d_labels)
        centers = comm.gather(h_centers, root=0)
        collected_labels = comm.gather(h_labels, root=0)

        if rank==0:
            count += 1
            temp_centers = np.empty((K, D))
            for center in centers:
                temp_centers+=center
            collected_labels = np.array(list(chain(*collected_labels)))
            for j in range(K):
                total = np.sum(collected_labels==j)
                temp_centers[j,:] = temp_centers[j,:]/total
            h_centers = temp_centers

        h_centers = comm.bcast(h_centers, root=0)
        cuda.memcpy_htod(d_centers, h_centers)
        kernel2(d_data, d_labels, d_centers, d_converged_array, block=(K, D, 1), grid=(N, 1, 1))
        cuda.memcpy_dtoh(h_converged_array, d_converged_array)
        if np.sum(h_converged_array) == 0:
            converged = True
        else:
            converged = False
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
    distortion = 100
    ai = 600*count
    if rank==0: runtime = time.time() - start
    runtime = comm.bcast(runtime, root=0)

    return h_centers, labels, count, runtime, distortion, ai