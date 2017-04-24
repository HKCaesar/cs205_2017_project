from functions import *
from sequentialK import *
from mpiK import *
from cudaK import *
from hybridK import *

from mpi4py import MPI
import itertools
import sys

######################################################
### INFO ####
######################################################

# Parallel*:      Sequential:      Meaning:                 Dim:
# data            X                reviewer data           (NxD)
# labels          W                cluster assignments     (Nx1)
# centers           A              cluster centers         (KxD)
# clustern        m                number per cluster      (1xK)

# *h_ and d_ prefixes in parallel variable names indicate host vs. device copies

######################################################
### CONFIGURE ####
######################################################

data_fn = "../../data/reviewer-data.csv"
d_list = ["cunninlingus_ct_bin","fellatio_ct_bin","intercoursevaginal_ct_bin","kissing_ct_bin","manualpenilestimulation_ct_bin","massage_ct_bin"]

kernel_fn = "pycuda.c"
output_fn = "../../analysis/output.csv"

Ks = [3]
Ns = [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]                # max N for review data is ANYTHING (can be over 118684)
Ds = [6]                    # max D for review data is 6 (we could increase this actually)

limit = 10                  # max number of times the k-means loop can run (even if it doesn't converge)
erase=True                  # start with a blank output file
standardize_count = 1       # use the same count for all k-means regardless of conversion

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if erase==True: blank_output_file(output_fn)
if standardize_count==1: ref_count = 0

for N, D, K in [x for x in list(itertools.product(Ns, Ds, Ks))]:

    ###################################
    ### PREP DATA & INITIAL LABELS ####
    output = []
    data, initial_labels = prep_data(data_fn, d_list, N, D, K)
    comm.Barrier()

    ############################################################################################################
    if rank == 0:
        print('\n\n\n----- N:%d D:%d K:%d -----' % (N, D, K))

        ###############################
        ### RUN SEQUENTIAL K-MEANS ####
        centers, labels, count, runtime, distortion, ai = seqkmeans(data, initial_labels, N, D, K, limit)
        output.append(['sequential',runtime, count, distortion, ai, N, D, K, centers])
        ref_centers=centers
        ref_count=count
        print_output(output[-1], ref_centers, ref_count)

        ##########################
        ### RUN STOCK K-MEANS ####
        centers, labels, count, runtime, distortion, ai = stockkmeans(data, K, ref_count)
        output.append(['stock', runtime, count, distortion, ai, N, D, K, centers])
        print_output(output[-1], ref_centers, ref_count)

        ###########################
        ### RUN pyCUDA K-MEANS ####
        if standardize_count > 0: standardize_count = ref_count
        centers, labels, count, runtime, distortion, ai = cudakmeans(data, initial_labels, kernel_fn, N, K, D, limit, standardize_count)
        output.append(['pyCUDA', runtime, count, distortion, ai, N, D, K, centers])
        print_output(output[-1], ref_centers, ref_count)

    ###########################
    ### RUN mpi4py K-MEANS ####
    comm.Barrier()
    if standardize_count > 0: standardize_count = comm.bcast(ref_count, root=0)
    centers, labels, count, runtime, distortion, ai = mpikmeans(data, initial_labels, N, K, D, limit, standardize_count, comm)
    comm.Barrier()
    if rank == 0:
        output.append(['mpi4py',runtime, count, distortion, ai, N, D, K, centers])
        print_output(output[-1], ref_centers, ref_count)

    ###########################
    ### RUN hybrid K-MEANS ####
    comm.Barrier()
    centers, labels, count, runtime, distortion, ai = hybridkmeans(data, initial_labels, kernel_fn, N, K, D, limit, standardize_count, comm)
    comm.Barrier()
    if rank == 0:
        output.append(['hybrid',runtime, count, distortion, ai, N, D, K, centers])
        print_output(output[-1], ref_centers, ref_count)

  ######################################################
  ### WRITE OUTPUT TO CSV ONCE PER LOOP ####
    comm.Barrier()
    if rank ==0:
        write_output(output, output_fn)
    comm.Barrier()

    ### RESET LOOP VARS ####
    data = None
    initial_labels = None
    comm.Barrier()

if rank != 0:
    sys.exit(0)