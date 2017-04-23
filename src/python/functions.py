import pandas as pd
import numpy as np
import csv

######################################################
### PREP DATA & INITIAL LABELS ####
######################################################

def prep_data(data_fn, d_list, N, D, K):

  # import data file and subset data for k-means
  reviewdata = pd.read_csv(data_fn)
  print(reviewdata.shape)
  data = reviewdata[d_list[:D]][:N].values
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
### CALCULATE DISTORTION ###
######################################################

def distortion(data, labels, means):
    #temp=np.sum((means[labels:]-data)**2) <---- FIX!!!!
    return 100

######################################################
### MAKE GRAPHS ###
######################################################

def process_output(output, output_fn, ref_means, ref_count):
  
  # print some stuff
  for o in output:
    print('\n-----'+o[0])
    if o[0][0]!='s': 
      print('Equals reference (sequential) means: %s' % str(np.array_equal(ref_means,o[-1])))
      print('Equals reference (sequential) count: %s' % str(np.array_equal(ref_count,o[2])))
    for p in o: print(p)
  
  # write to csv
  with open(output_fn, 'a') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerows([o[:8] for o in output])
    f.close()
  
  return
