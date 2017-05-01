from functions import *
from sequentialK import *
from mpiK import *
from cudaK import *
from hybridK import *
import os

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

### get environment variables from SLURM
try:
    arrind = int(os.environ["SLURM_ARRAY_TASK_ID"])
except KeyError:
    try:
        arrind= int(os.environ["PARAM1"])
    except:
        arrind = 0

try:
    jobid = int(os.environ["SLURM_ARRAY_JOB_ID"])
except KeyError:
    try:
        jobid = int(os.environ["SLURM_JOB_ID"])
    except KeyError:
        jobid=0

try:
    numnode = int(os.environ["SLURM_JOB_NUM_NODES"])
except KeyError:
    try:
        numnode = int(os.environ["PARAM2"])
    except:
        numnode = 0

try:
    ntask = int(os.environ["SLURM_NTASKS"])
except KeyError:
    try:
        numnode = int(os.environ["TASK"])
    except:
        numnode = 0

#data_fn = "../../data/reviewer-data.csv"
data_fn = "../../data/fittedVals_logit.csv"
#d_list = ["cunninlingus_ct_bin","fellatio_ct_bin","intercoursevaginal_ct_bin","kissing_ct_bin","manualpenilestimulation_ct_bin","massage_ct_bin", "avgspph_avg", "app_reviewer_avg", "perf_reviewer_avg", "rptescorts_reviewers", "rvwduration_years_reviewer", "reviewno_escort_avg", "appgap_reviewer_avg", "age_cont2_avg", "height_cont2_avg", "breastsize_cont2_avg", "breastcup_cont_avg", "hairlength_cont_avg", "smokes_bin_avg", "avgpph_avg", "unqescorts_reviewers", "ethnicity_n", "build_n", "haircolor_n", "reviewno_escort_max", "reviewno_escort_min"]
d_list = ['analoral_ct_bin', 'cuddling_ct_bin', 'cunninlingus_ct_bin', 'ejaculationonbody_ct_bin', 'fellatio_ct_bin', 'intercourseanal_ct_bin', 'intercoursevaginal_ct_bin', 'kissing_ct_bin', 'manualanalstimulation_ct_bin', 'manualpenilestimulation_ct_bin', 'manualvaginalstimulation_ct_bin', 'massage_ct_bin', 'sm_ct_bin', 'testiclestimulation_ct_bin', 'threesome_ct_bin', 'ethnicity_mode', 'build_mode', 'breastappearance_mode', 'haircolor_mode', 'hairtype_mode', 'service_mode']

kernel_fn = "../cuda/pycumean.c"

output_fn = "../../analysis/output/output_jobid_"+str(jobid) + "_array_" + str(arrind) + "_time_" + time.strftime('%Y%m%d_%H%M%S') +"_processors_" + str(ntask)  +  ".csv" #will save csv with number of processors used, jobid, etc

erase=True                  # start with a blank output file
if erase==True: blank_output_file(output_fn)
env_vars = [numnode,ntask,ntask]  # list N, n, and GPUs (to put in the output.csv)

######################################################
### SET RUNTIME VARIABLES ####
######################################################



# Dimensions of Problems
#Ns = [1000,10000]
Ns = [1024, 1024 * 5, 1024 *10, 1024 * 20, 1024 * 30, 1024 * 40, 1024 * 50, 1024 *60, 1024 *70, 1024 *80, 1024 *90, 1024 *100, 1024 *500, 1024 *1000]
Ds = [5, 25, 50, 100, 1000] # max D can be anything

Ks = [3,4,5,10,25,50,100]

numThreads = 256

limit = 1000                  # max number of times the k-means loop can run (even if it doesn't converge)
standardize_count = 0       # use the same count for all k-means regardless of conversion

algorithms_to_run = ["hybrid","mpi","cuda"]   # indicate which algorithms should run
#algorithms_to_run = ["cuda"]
#algorithms_to_run = ["mpi"]
#algorithms_to_run = ["hybrid"]


######################################################
### MAIN LOOP TO RUN 5 K-MEANS ALGORITHMS ####
######################################################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
P = comm.Get_size()


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
        centers, labels, count, runtime = seqkmeans(data, initial_labels, N, D, K, limit, standardize_count)
        output.append(['sequential',runtime, count, N, D, K] + env_vars + [centers] )
        ref_centers=centers
        ref_count=count
        print_output(output[-1], ref_centers, ref_count)
        
        ##########################
        ### RUN STOCK K-MEANS ####
        if standardize_count > 0: loop_limit = standardize_count
        else: loop_limit = limit
        centers, labels, count, runtime = stockkmeans(data, K, loop_limit)
        output.append(['stock', runtime, count, N, D, K] + env_vars + [centers] )
        print_output(output[-1], ref_centers, ref_count)
        
        ###########################
        ### RUN pyCUDA K-MEANS ####
        if "cuda" in algorithms_to_run:
            centers, labels, count, runtime = cudakmeans(data, initial_labels, kernel_fn, N, K, D, numThreads, limit, standardize_count)
            output.append(['pyCUDA', runtime, count, N, D, K] + env_vars + [centers] )
            print_output(output[-1], ref_centers, ref_count)

    ###########################
    ### RUN mpi4py K-MEANS ####
    comm.Barrier()
    if "mpi" in algorithms_to_run:
        centers, labels, count, runtime = mpikmeans(data, initial_labels, N, K, D, limit, standardize_count, comm)
        comm.Barrier()
        if rank == 0:
            output.append(['mpi4py',runtime, count, N, D, K] + env_vars + [centers] )
            print_output(output[-1], ref_centers, ref_count)

    ###########################
    ### RUN hybrid K-MEANS ####
    comm.Barrier()
    if "hybrid" in algorithms_to_run:
        centers, labels, count, runtime = hybridkmeans(data, initial_labels, kernel_fn, N, K, D, numThreads, limit, standardize_count, comm)
        comm.Barrier()
        if rank == 0:
            output.append(['hybrid',runtime, count, N, D, K] + env_vars + [centers] )
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
