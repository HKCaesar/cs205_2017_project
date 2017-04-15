from mpi4py import MPI

def mpi_kmeans(data=None, n_clusters=None,n_init=None,max_iter=None):

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = MPI.

    if rank == 0:


    else:



    return [centers,labels,timing]