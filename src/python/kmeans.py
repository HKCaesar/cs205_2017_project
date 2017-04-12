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

K = 3
N = 1000 # max N for review data is 118684
D = 6 # max D for review data is 6 (we could increase this actually)
limit = 10

output = [['algorithm','time','convergence','distortion','n','d','k']]

######################################################
### PREP DATA & INITIAL LABELS ####
######################################################

data, initial_labels = prep_data(data_fn, d_list, N, D, K)

######################################################
### RUN STOCK K-MEANS ####
######################################################

means, labels, distortion, runtime = stock(data, K, limit)

output.append(['stock',runtime, '', distortion, N, D, K])
print('\n-----Stock output')
print(means)
print(labels[:10])
print(distortion)

######################################################
### RUN SEQUENTIAL K-MEANS ####
######################################################

means, labels, count, runtime, A1, W1 = sequential(data, initial_labels, N, D, K, limit)

output.append(['sequential',runtime, count, '', N, D, K])
print('\n-----Sequential output')
print(means)
print(labels[:10])
ref_means = means

print('\n-----Sequential output count == 1') # will eventually delete this
print(A1)
print(W1[:10])
ref_means = A1

######################################################
### RUN PARALLEL K-MEANS ####
######################################################

means, labels, count, runtime = parallel(data, initial_labels, kernel_fn, N, K, D, limit)

output.append(['parallel',runtime, count, '', N, D, K])
print('\n-----Parallel output')
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
