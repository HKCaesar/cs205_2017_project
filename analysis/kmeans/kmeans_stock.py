import time
from sklearn.cluster import KMeans

def kmeans_stock(data, n_clusters,max_iter=100):
    start = time.time()
    stock = KMeans(n_clusters=n_clusters,n_init=1,max_iter=max_iter)
    stock.fit(data)
    timing = time.time()-start
    return stock.cluster_centers_,stock.labels_,timing