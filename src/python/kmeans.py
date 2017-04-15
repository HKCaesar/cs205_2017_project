from functions import *

######################################################
### INFO ####
######################################################

# Parallel*:      Sequential:      Meaning:                 Dim:
# data            X                reviewer data           (NxD)
# labels          W                cluster assignments     (Nx1)
# means           A                means                   (KxD)
# clustern        m                number per cluster      (1xK)

# *h_ and d_ prefixes in parallel variable names indicate host vs. device copies

######################################################
### CONFIGURE ####
######################################################

data_fn = "../../data/reviewer-data.csv"
d_list = ["cunninlingus_ct_bin","fellatio_ct_bin","intercoursevaginal_ct_bin","kissing_ct_bin","manualpenilestimulation_ct_bin","massage_ct_bin"]

kernel_fn = "../cuda/pycumean.c"

output_dir = "../../analysis/"

K = 4
N = 1024 # max N for review data is 118684
D = 4 # max D for review data is 6 (we could increase this actually)
limit = 10

output = [['algorithm','time','convergence','distortion','n','d','k']]

######################################################
### PREP DATA & INITIAL LABELS ####
######################################################

data, initial_labels = prep_data(data_fn, d_list, N, D, K)

######################################################
### RUN SEQUENTIAL K-MEANS ####
######################################################

means, labels, count, runtime, distortion, means1, labels1 = sequential(data, initial_labels, N, D, K, limit)
output.append(['sequential',runtime, count, '', N, D, K, means])

print('\n-----sequential output count == 1') # will eventually delete this
print(means1)
print(label1[:10])
ref_means = means1

######################################################
### RUN STOCK K-MEANS ####
######################################################

means, labels, distortion, runtime, distortion = stock(data, K, count)
output.append(['stock', runtime, '', distortion, N, D, K, means])

######################################################
### RUN pyCUDA K-MEANS ####
######################################################

means, labels, count, runtime, distortion = pyCUDA(data, initial_labels, kernel_fn, N, K, D, limit)
output.append(['pyCUDA', runtime, count, distortion, N, D, K, means])

######################################################
### RUN mpi4py K-MEANS ####
######################################################

means, labels, count, runtime, distortion = mpi4py(data, initial_labels, kernel_fn, N, K, D, limit)
output.append(['mpi4py',runtime, count, distortion, N, D, K, means])

######################################################
### RUN hybrid K-MEANS ####
######################################################

means, labels, count, runtime, distortion = hybrid(data, initial_labels, kernel_fn, N, K, D, limit)
output.append(['hybrid',runtime, count, distortion, N, D, K, means])

######################################################
### MAKE GRAPHS & WRITE OUTPUT TO CSV ####
######################################################

process_output(output, output_dir, ref_means)
