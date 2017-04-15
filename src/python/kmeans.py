from functions import *
import csv

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
### RUN STOCK K-MEANS ####
######################################################

means, labels, distortion, runtime, distortion = stock(data, K, limit)

output.append(['stock',runtime, '', distortion, N, D, K])
print('\n-----stock output')
print(means)
print(labels[:10])
print(distortion)

######################################################
### RUN SEQUENTIAL K-MEANS ####
######################################################

means, labels, count, runtime, distortion, A1, W1 = sequential(data, initial_labels, N, D, K, limit)

output.append(['sequential',runtime, count, '', N, D, K])
print('\n-----sequential output')
print(means)
print(labels[:10])
ref_means = means

print('\n-----sequential output count == 1') # will eventually delete this
print(A1)
print(W1[:10])
ref_means = A1

######################################################
### RUN pyCUDA K-MEANS ####
######################################################

means, labels, count, runtime, distortion = pyCUDA(data, initial_labels, kernel_fn, N, K, D, limit)

output.append(['pyCUDA',runtime, count, '', N, D, K])
print('\n-----pyCUDA output')
print(means)
print(labels[:10])
print('Equals stock means: %s' % str(np.array_equal(ref_means,means)))

######################################################
### RUN mpi4py K-MEANS ####
######################################################

means, labels, count, runtime, distortion = mpi4py(data, initial_labels, kernel_fn, N, K, D, limit)

output.append(['mpi4py',runtime, count, '', N, D, K])
print('\n-----mpi4py output')
print(means)
print(labels[:10])
print('Equals stock means: %s' % str(np.array_equal(ref_means,means)))

######################################################
### RUN hybrid K-MEANS ####
######################################################

means, labels, count, runtime, distortion = pyCUDA(data, initial_labels, kernel_fn, N, K, D, limit)

output.append(['hybrid',runtime, count, '', N, D, K])
print('\n-----hybrid output')
print(means)
print(labels[:10])
print('Equals stock means: %s' % str(np.array_equal(ref_means,means)))

######################################################
### WRITE DATA TO CSV ####
######################################################

with open(output_dir + 'times.csv', 'w') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerows(output)
    f.close()
