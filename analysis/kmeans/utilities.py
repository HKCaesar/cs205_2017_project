import numpy as np
from sklearn.decomposition import PCA, FastICA
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
from sklearn.manifold import TSNE

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

def kmeans_plot(labels, centers, data, print_distortion=None, print_best=None):
    K,D = centers.shape

    cut_off = 20000
    if len(data) > cut_off:
        indices, data = generate_random_subset(data, cut_off)
        labels = labels[indices]

    pca = PCA(n_components=2, svd_solver='full')

    #pca = FastICA(n_components=3)

    #pca = TSNE(n_components=2, random_state=0)



    #pca.fit(data)

    #pcs   = pca.transform(data)
    #cntrs = pca.transform(centers)

    result = pca.fit_transform( np.vstack( (centers,data) ) )

    cntrs = result[:K]
    pcs   = result[K:]

    #blues = sns.color_palette("deep",2*K)

    blues = sns.color_palette("colorblind",K)

    #blues = sns.color_palette("Set2",K)

    greys = sns.color_palette("Greys", 16)
    reds = sns.color_palette("Reds", 8)

    x,y=pcs[:,0],pcs[:,1]

    # Calculate the point density
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    fig = plt.figure(figsize=(8,8),dpi=100)
    ax = fig.add_subplot(111)

    for k in range(K):
        ax.scatter(x[labels==k], y[labels==k], s=30, edgecolor='black',c= blues[k] )

    a=ax.scatter(cntrs[:,0],cntrs[:,1],facecolor=greys[13],edgecolor='white',s=120,alpha=0.95)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')

    if print_distortion!=None:
        distortion_text = "%.2f" % print_distortion
        distortion_text = distortion_text.rjust(10)

        ax.text(0.81, 0.96,"Distortion: %s" % distortion_text,
                horizontalalignment='center',
                verticalalignment='center',
                transform = ax.transAxes,family='monospace')

        if print_best!=None:

                ax.text(0.81, 0.93,"Best:       "+("%.2f"%print_best).rjust(10),
                horizontalalignment='center',
                verticalalignment='center',
                transform = ax.transAxes,family='monospace',color=reds[7])

    return fig


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


def distortion(labels,centers,data):
    return np.sum((centers[labels,:]-data)**2)


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
