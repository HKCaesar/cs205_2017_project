from functions import *
from sequentialK import *
from mpiK import *
from cudaK import *
from hybridK import *

import itertools

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

erase=True
if erase==True: blank_output_file(output_fn)

limit = 10
    
Ks = [3]
Ns = [100]     # max N for review data is 118684
Ds = [6]       # max D for review data is 6 (we could increase this actually)

for N, D, K in [x for x in list(itertools.product(Ns, Ds, Ks))]:
    
  output = []

  ######################################################
  ### PREP DATA & INITIAL LABELS ####
  ######################################################

  data, initial_labels = prep_data(data_fn, d_list, N, D, K)

  ######################################################
  ### RUN SEQUENTIAL K-MEANS ####
  ######################################################

  centers, labels, count, runtime, distortion, ai = seqkmeans(data, initial_labels, N, D, K, limit)
  output.append(['sequential',runtime, count, distortion, ai, N, D, K, centers])
  ref_centers=centers
  ref_count=count
  print_output(output[-1], ref_centers, ref_count)

  ######################################################
  ### RUN pyCUDA K-MEANS ####
  ######################################################

  centers, labels, count, runtime, distortion, ai = cudakmeans(data, initial_labels, kernel_fn, N, K, D, limit)
  output.append(['pyCUDA', runtime, count, distortion, ai, N, D, K, centers])
  print_output(output[-1], ref_centers, ref_count)

  ######################################################
  ### RUN mpi4py K-MEANS ####
  ######################################################


  centers, labels, runtime = mpikmeans(data=data, n_clusters=K, max_iter=limit)
  distortion = 0
  ai = 0
  count = 0
  output.append(['mpi4py',runtime, count, distortion, ai, N, D, K, centers])
  print_output(output[-1], ref_centers, ref_count)

  ######################################################
  ### RUN hybrid K-MEANS ####
  ######################################################

  centers, labels, count, runtime, distortion, ai = hybrid(data, initial_labels, kernel_fn, N, K, D, limit)
  output.append(['hybrid',runtime, count, distortion, ai, N, D, K, centers])
  print_output(output[-1], ref_centers, ref_count)

  ######################################################
  ### RUN STOCK K-MEANS ####
  ######################################################

  centers, labels, count, runtime, distortion, ai = stockkmeans(data, K, ref_count)
  output.append(['stock', runtime, count, distortion, ai, N, D, K, centers])
  print_output(output, ref_centers, ref_count)

  ######################################################
  ### WRITE OUTPUT TO CSV ####
  ######################################################

  write_output(output, output_fn)
