import pandas as pd
from kmeans.utilities import distortion

#data = pd.read_csv('../data/yeast.csv', header=None).values

columns=["cunninlingus_ct_bin","fellatio_ct_bin",
          "intercoursevaginal_ct_bin","kissing_ct_bin",
          "manualpenilestimulation_ct_bin","massage_ct_bin"]

data = pd.read_csv('../data/reviewer-data.csv')[columns].values


from kmeans.mpi_kmeans import mpi_kmeans
from kmeans.kmeans_stock import kmeans_stock

centers,labels,timing=mpi_kmeans(data=data, n_clusters=2)
print("")
print("mpi distortion: %f time: %f s" % (distortion(labels,centers,data),timing))

centers,labels,timing= kmeans_stock(data=data,n_clusters=2)
print("stock distortion: %f time: %f s" % (distortion(labels,centers,data),timing))

