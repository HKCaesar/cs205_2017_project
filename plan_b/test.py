import pandas as pd
from kmeans.utilities import distortion

data = pd.read_csv('../data/yeast.csv', header=None).values


from kmeans.mpi_kmeans import mpi_kmeans
from kmeans.kmeans_stock import kmeans_stock

print("")

centers,labels,timing=mpi_kmeans(data=data, n_clusters=2)
print("mpi distortion: %f time: %f" % (distortion(labels,centers,data),timing))

centers,labels,timing= kmeans_stock(data=data,n_clusters=2)
print("stock distortion: %f time: %f" % (distortion(labels,centers,data),timing))

