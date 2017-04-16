import itertools
from mpi4py import MPI
from itertools import chain
import sys
from scipy.stats import gaussian_kde

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import time
import string
import csv

######################################################
### PREP DATA & INITIAL LABELS ####
######################################################

def prep_data(reviewdata, d_list, N, D, K):

  # import data file and subset data for k-means
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
### SEQUENTIAL K-MEANS ###
######################################################

def sequential(data, initial_labels, N, D, K, limit):

  means = np.empty((K,D))
  labels = initial_labels.copy()
  clustern = np.empty(K)
  count = 0
  converged = False
  start = time.time()

  while not converged:

      converged = True

      #compute means
      for k in range(K):
          for d in range(D):
              means[k,d] = 0
          clustern[k]=0
      for n in range(N):
          for d in range(D):
              means[labels[n],d]+=data[n,d]
          clustern[labels[n] ] +=1
      for k in range(K):
          for d in range(D):
              means[k,d] = means[k,d]/clustern[k]

      #assign to closest mean
      for n in range(N):
          min_val = np.inf
          min_ind = -1
          for k in range(K):
              temp =0
              for d in range(D):
                  temp += (data[n,d]-means[k,d])**2

              if temp < min_val:
                  min_val = temp
                  min_ind = k
          if min_ind != labels[n]:
              labels[n] = min_ind
              converged=False

      count +=1
      if count==limit: break

  runtime = time.time()-start
  ai = 200 * count
  distortion = 50

  return means, labels, count, runtime, distortion, ai

######################################################
### pyCUDA K-MEANS  ####
######################################################

# define h_vars on host
def prep_host(data, initial_labels, K, D):

  h_data = data.copy()
  h_labels = initial_labels.copy()
  h_means = np.ascontiguousarray(np.empty((K,D),dtype=np.float64, order='C'))
  h_converged_array = np.zeros((data.shape[0]),dtype=np.intc)
  return h_data, h_labels, h_means, h_converged_array

# allocate memory and copy data to d_vars on device
def prep_device(h_data, h_labels, h_means, h_converged_array):

  d_data = cuda.mem_alloc(h_data.nbytes)
  d_labels = cuda.mem_alloc(h_labels.nbytes)
  d_means = cuda.mem_alloc(h_means.nbytes)
  d_converged_array = cuda.mem_alloc(h_converged_array.nbytes)
  cuda.memcpy_htod(d_data,h_data)
  cuda.memcpy_htod(d_labels,h_labels)
  cuda.memcpy_htod(d_converged_array, h_converged_array)

  return d_data, d_labels, d_means, d_converged_array

# define kernels
def parallel_mod(kernel_fn, N, K, D):

    template = string.Template(open(kernel_fn, "r").read())
    code = template.substitute(N = N, K = K, D = D)
    mod = SourceModule(code)
    kernel1 = mod.get_function("newMeans")
    kernel2 = mod.get_function("reassign")

    return kernel1, kernel2

def pyCUDA(data, initial_labels, kernel_fn, N, K, D, limit):

    count = 0
    kernel1, kernel2 = parallel_mod(kernel_fn, N, K, D)
    h_data, h_labels, h_means, h_converged_array = prep_host(data, initial_labels, K, D)

    start = time.time()
    d_data, d_labels, d_means, d_converged_array = prep_device( h_data, h_labels, h_means, h_converged_array)

    while count<limit:
        kernel1(d_data, d_labels, d_means, block=(K,D,1), grid=(1,1,1))
        kernel2(d_data, d_labels, d_means, d_converged_array, block=(K,D,1), grid=(N,1,1))
        cuda.memcpy_dtoh(h_converged_array, d_converged_array)
        count +=1
        if np.sum(h_converged_array)==0: break

    cuda.memcpy_dtoh(h_means, d_means)
    cuda.memcpy_dtoh(h_labels, d_labels)
    runtime = time.time()-start
    ai = 300 * count
    distortion = 50

    return h_means, h_labels, count, runtime, distortion, ai

######################################################
### mpi4py K-MEANS  ####
######################################################

def mpi4py(data, initial_labels, kernel_fn, N, K, D, limit):

  start = time.time()
  count = 0
  h_means = np.ascontiguousarray(np.empty((K,D),dtype=np.float64, order='C'))
  h_labels = np.ascontiguousarray(np.empty(initial_labels.shape,dtype=np.intc, order='C'))
  runtime = time.time()-start
  ai = 400 * count
  distortion = 50

  return h_means, h_labels, count, runtime, distortion, ai

######################################################
### hybrid K-MEANS  ####
######################################################

def hybrid(data, initial_labels, kernel_fn, N, K, D, limit):

  start = time.time()
  count = 0
  h_means = np.ascontiguousarray(np.empty((K,D),dtype=np.float64, order='C'))
  h_labels = np.ascontiguousarray(np.empty(initial_labels.shape,dtype=np.intc, order='C'))
  runtime = time.time()-start
  ai = 500 * count
  distortion = 50

  return h_means, h_labels, count, runtime, distortion, ai

######################################################
### STOCK K-MEANS ###
######################################################

def stock(data, K, count):

    start = time.time()
    stockmeans = KMeans(n_clusters=K,n_init=count)
    stockmeans.fit(data)
    runtime = time.time()-start
    ai = 100 * count
    return stockmeans.cluster_centers_, stockmeans.labels_, count, runtime, stockmeans.inertia_, ai

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


def allotment_to_indices(allotments):
    indices = np.cumsum(allotments)
    indices=np.append(indices,[0])
    indices=np.sort(indices)
    indices=np.column_stack([indices[:-1],indices[1:]])
    return(indices)

def generate_random_subset(df, subset_size):
    n = len(df)
    indices = np.arange(n)
    np.random.shuffle(indices)

    if subset_size <= 1: subset_size = int(subset_size*n)

    np.sort(indices)

    indices = indices[:subset_size]

    if isinstance(df,pd.core.frame.DataFrame):
        return df.iloc[indices,:]

    return indices, df[indices]

def partition(sequence, n_chunks):
    N = len(sequence)
    chunk_size = int(N/n_chunks)
    left_over = N-n_chunks*chunk_size
    allocations = np.array([chunk_size]*n_chunks)
    left_over=([1]*left_over)+([0]*(n_chunks-left_over))
    np.random.shuffle(left_over)
    allocations = left_over+allocations


    indexes = allotment_to_indices(allocations)

    return allocations, [sequence[index[0]:index[1]]  for index in indexes]



def generate_initial_assignment(N,K):
    W = np.empty(N,dtype=np.int)
    for k in range(N): W[k] = k%K
    np.random.shuffle(W)
    return W


def compute_means(labels, centers, data, sum_values=False):
    N,D=data.shape
    K,D=centers.shape

    for k in range(K):
        if sum_values==False:
            centers[k,:] = np.mean(data[labels==k],axis=0)
        else:
            centers[k,:] = np.sum(data[labels==k],axis=0)


    return centers

def reassign_labels(labels,centers,data):
    old_labels = labels.copy()

    def minimize(x):
        return np.argmin(np.sum((centers-x)**2,axis=1)) #finds closest cluster

    labels[:] = np.apply_along_axis(minimize,1,data)

    return np.array_equal(labels,old_labels)


def mpi_kmeans(data, n_clusters, max_iter):

    all_data = data

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start = time.time()

    n_data, n_dimensions = data.shape

    centers = np.zeros((n_clusters,n_dimensions))
    labels = generate_initial_assignment(n_data,n_clusters)


    allocations,labels = partition(labels,size)
    indices = allotment_to_indices(allocations)

    indices,labels = comm.scatter(zip(indices, labels) , root=0)

    data = data[indices[0]:indices[1]]


    for k in range(max_iter):

        compute_means(labels,centers,data,sum_values=True)

        centers = comm.gather(centers, root=0)

        if rank==0:

            temp = np.zeros((n_clusters,n_dimensions))

            for center in centers:
                temp+=center

            centers = temp

        collected_labels = comm.gather(labels, root=0)


        if rank == 0:

            collected_labels = np.array(list(chain(*collected_labels)))

            for j in range(n_clusters) :
                total = np.sum(collected_labels==j)
                centers[j,:] = centers[j,:]/total


        centers = comm.bcast(centers, root=0)

        converged = reassign_labels(labels,centers,data)

        converged = comm.allgather(converged)



        converged = np.all(converged)

        if converged: break


    labels = comm.gather(labels,root=0)

    if rank==0:
        labels = np.array(list(chain(*labels)))
        timing = time.time()-start
        return [centers,labels,timing]
    else:
        sys.exit(0)


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
reviewdata = pd.read_csv(data_fn)

kernel_fn = "../cuda/pycumean.c"

output_fn = "../../analysis/output.csv"
with open(output_fn, 'w') as f:
    writer = csv.writer(f, delimiter = ',')
    writer.writerow(['algorithm','time','convergence','distortion', 'arithmetic intensity', 'n','d','k'])
    f.close()

limit = 10

Ks = [5]
Ns = [1000]     # max N for review data is 118684
Ds = [6]        # max D for review data is 6 (we could increase this actually)

for N, D, K in [x for x in list(itertools.product(Ns, Ds, Ks))]:

  output = []

  ######################################################
  ### PREP DATA & INITIAL LABELS ####
  ######################################################

  data, initial_labels = prep_data(reviewdata, d_list, N, D, K)

  ######################################################
  ### RUN SEQUENTIAL K-MEANS ####
  ######################################################

  means, labels, count, runtime, distortion, ai = sequential(data, initial_labels, N, D, K, limit)
  output.append(['sequential',runtime, count, distortion, ai, N, D, K, means])
  ref_means=means
  ref_count=count

  ######################################################
  ### RUN pyCUDA K-MEANS ####
  ######################################################

  means, labels, count, runtime, distortion, ai = pyCUDA(data, initial_labels, kernel_fn, N, K, D, limit)
  output.append(['pyCUDA', runtime, count, distortion, ai, N, D, K, means])

  ######################################################
  ### RUN mpi4py K-MEANS ####
  ######################################################

  means, labels, runtime = mpi_kmeans(data, K, limit)
  distortion = 0
  ai = 0
  count = 0
  output.append(['mpi4py',runtime, count, distortion, ai, N, D, K, means])

  ######################################################
  ### RUN hybrid K-MEANS ####
  ######################################################

  means, labels, count, runtime, distortion, ai = hybrid(data, initial_labels, kernel_fn, N, K, D, limit)
  output.append(['hybrid',runtime, count, distortion, ai, N, D, K, means])

  ######################################################
  ### RUN STOCK K-MEANS ####
  ######################################################

  means, labels, count, runtime, distortion, ai = stock(data, K, ref_count)
  output.append(['stock', runtime, count, distortion, ai, N, D, K, means])

  ######################################################
  ### MAKE GRAPHS & WRITE OUTPUT TO CSV ####
  ######################################################

  process_output(output, output_fn, ref_means, ref_count)
