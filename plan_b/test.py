import pandas as pd
from kmeans.utilities import distortion

data = pd.read_csv('../data/yeast.csv', header=None).values


from kmeans.mpi_kmeans import mpi_kmeans
from kmeans.kmeans_stock import kmeans_stock

centers,labels,timing=mpi_kmeans(data=data, n_clusters=2)
print(distortion(labels,centers,data))

centers,labels,timing= kmeans_stock(data=data,n_clusters=2)
print(distortion(labels,centers,data))

