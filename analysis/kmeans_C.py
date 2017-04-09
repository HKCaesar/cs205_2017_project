from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import kM

#matplotlib inline


data = pd.read_csv('../data/yeast.csv', header=None).values
#k-means witk sklearn

K=3
N,D=data.shape

A = np.zeros((K,D), dtype=np.float64)
W = np.zeros(N,dtype=np.intc)
X = data
m = np.zeros(K)

#randomization where we make sure there are points in every cluster
for n in range(N):
    W[n] = n%K


def shuffle(x,n):
    for i in range(n-2,-1,-1): #from n-2 to 0
        j= np.random.randint(0,i+1) #from 0<=j<=i
        temp = x[j]
        x[j] = x[i]
        x[i] = temp


shuffle(W,len(W))




converged = False

while not converged:
    converged = True
    
    #compute means
    for k in range(K):
        for d in range(D):
            A[k,d] = 0
        m[k]=0
            
    for n in range(N):
        for d in range(D):
            A[W[n],d]+=X[n,d]
        m[ W[n] ] +=1
    
    for k in range(K):
        for d in range(D):
            A[k,d] = A[k,d]/m[k]
            
    #assign to closest mean
    for n in range(N):
        
        min_val = np.inf
        min_ind = -1
        
        for k in range(K):
            temp =0
            for d in range(D):
                temp += (X[n,d]-A[k,d])**2
            
            if temp < min_val:
                min_val = temp
                min_ind = k
                
        if min_ind != W[n]:
            W[n] = min_ind
            converged=False


c = np.zeros((K,D))
x = data
WW = np.zeros(N,dtype=np.int32)

def shuffle(x,n):
    for i in range(n-2,-1,-1): #from n-2 to 0
        j= np.random.randint(0,i+1) #from 0<=j<=i
        temp = x[j]
        x[j] = x[i]
        x[i] = temp


shuffle(WW,len(WW))


c = c.copy(order='C')
WW = WW.copy(order='C')
x = x.copy(order='C')


X = X.copy(order='C')
A = A.copy(order='C')


kM.kMeans(x, c, WW)

print("Stock K-mean equal to Eric's clusters?  ", end=" ")
print(np.array_equal(kmeans.cluster_centers_,c))
print(c)
print(kmeans.cluster_centers_)

print("Stock K-mean equal to Kareem's clusters?  ", end=" ")
print(np.array_equal(kmeans.cluster_centers_,A))
print(A)
print(kmeans.cluster_centers_)

print("Did Kareem's slow code do at least as well as Eric's?  ", end=" ")
print(np.array_equal(W, WW))

print("Kareem v. Eric Clusters", end=" ")
print(np.array_equal(c,A))

exit()


gcc -std=c11 -c src/kmeans.c kmeans_wrap.c \
    -I/usr/local/include/python2.1
