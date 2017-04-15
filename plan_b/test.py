import pandas as pd
from kmeans.utilities import distortion

data = pd.read_csv('../data/yeast.csv', header=None).values


from kmeans.mpi_kmeans import mpi_kmeans

centers,labels,timing=mpi_kmeans(data=data, n_clusters=3)

#print(distortion(labels,centers,data))

