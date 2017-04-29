import pandas as pd
import numpy as np
import csv

######################################################
### PREP DATA & INITIAL LABELS ####
######################################################

def prep_data(data_fn, d_list, N, D, K):
    # import data file and subset data for k-means
    reviewdata = pd.read_csv(data_fn)
    
    M = 118684 # actual matrix size of real data
    total_iter = int(N/M) + 1
    data = np.empty((M*total_iter, D))
    
    for n in range(int(N/M)):
        data[(n * M):(n * M + M),] = reviewdata[d_list[:D]]
        print(len(data))
    data = data[d_list[:D]][:N].values
    data = np.ascontiguousarray(data, dtype=np.float64)

    # assign random clusters & shuffle
    initial_labels = np.ascontiguousarray(np.zeros(N,dtype=np.intc, order='C'))
    for n in range(N):
        initial_labels[n] = n%K
    for i in range(len(initial_labels)-2,-1,-1):
        j= np.random.randint(0,i+1)
        temp = initial_labels[j]
        initial_labels[j] = initial_labels[i]
        initial_labels[i] = temp

    return data, initial_labels

######################################################
### OUTPUT FILE FUNCTIONS ###
#####################################################

# create a blank output file with header
def blank_output_file(output_fn):
    with open(output_fn, 'w') as f:
        writer = csv.writer(f, delimiter = ',')
        writer.writerow(['algorithm','time','convergence', 'n','d','k', 'Nodes N', 'Nodes n', 'GPUs'])
        f.close()
    return

 # write output to csv in append mode
def write_output(output, output_fn):
    with open(output_fn, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows([o[:-1] for o in output])
        f.close()
    return

######################################################
### PRINT OUTPUT ###
######################################################

def print_output(o, ref_means, ref_count):
  
    # print some stuff
    print('\n-----' + o[0])
    for p in o: print(p)
    if o[0][0]!='s':
        print('Equals reference (sequential) means: %s' % str(np.array_equal(ref_means,o[-1])))
        print('Equals reference (sequential) count: %s' % str(np.array_equal(ref_count,o[2])))
    return
