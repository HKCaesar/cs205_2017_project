//
//  pycumean.c
//  
//
//  Created by Eric Dunipace on 3/30/17.
//
//

//include "pycumean.h"

#define  N $N
#define  K $K
#define  D $D

//double sumSq (double *x, double *c, int N,  int D,  int n,  int k)
//{ //calculates the squared distance
//    
//    double ss = 0;
//    
//    for(int d = 0; d < D; ++d)
//    {
//        ss += (x[d + D * n] - c[d + k * D]) * (x[d + D * n] - c[d + k * D]);
//    }
//    return ss;
//}
//
//void selectCluster (double *x, double *c,  int *assign,  int N,  int K,  int D,  int * conv)
//{//selects the cluster and calculates the distance (we may want to separate these actions)
//    double min;
//    int min_idx;
//    int convCheck = 1;
//    double dist;
//
//    
//    for (int n = 0; n < N; ++n)
//    {
//        min = 1.0/0.0;
//        min_idx = -1;
//        
//        for (int k = 0; k < K; ++k)
//        {
//            dist = sumSq (x, c, N, D, n, k);
//            if (min > dist)
//            {
//                min = dist;
//                min_idx = k;
//            }
//        }
//        if(convCheck)
//        {
//            (*conv) = (min_idx == assign[n] ? 1 : 0);
//            if(!conv) convCheck = 0;
//        }
//        
//        assign[n] = min_idx;
//        
//    }
//}


//__global__  void kMeans (double *d_data, double *d_means,  int *d_clusters,  const int N,  const int K,  const int D)
//{ //runs the k means algorithm
//    __shared__  int conv = 0;
//    __shared__  int clustersn[K];
//    
//    
//    while(!conv)
//    {
//        newMeans(N, D, K, d_data, d_clusters, d_means, clustern);
//        reassign(d_data, d_clusters, d_means,d_clustern,);
//        
//    }
//}

//__global__ void newmeans(double *data, int *clusters, double *means) {
//  __shared__ int s_clustern[%(K)s];
//  int tid = (%(D)s*threadIdx.x) + threadIdx.y;
//  double l_sum = 0;
//    
//  // find the n per cluster with just one lucky thread
//  if (tid==0)
//  {
//    for(int k=0; k < (%(K)s); ++k) s_clustern[k] = 0;
//    for(int n=0; n < (%(N)s); ++n) s_clustern[clusters[n]]++;
//   }
//   __syncthreads();
//   
//   // sum stuff  
//   for(int n=0; n < (%(N)s); ++n)
//   {
//     if(clusters[n]==threadIdx.x)
//     {
//       l_sum += data[(%(D)s*n)+threadIdx.y];
//     }
//   }
//  
//   // divide local sum by the number in that cluster
//   means[tid] = l_sum/s_clustern[threadIdx.x];
//  }

__global__ void newMeans(double *data, int *clusters, double *means, int *clustern)
{
    int row = threadIdx.x;
    int col = threadIdx.y;
    int tid = row + col * D;
    int l_clust;
    double l_sum = 0.0;

    // find the n per cluster with K lucky threads
    if (tid < K) {
        int s_clust = 0;
        for (int n = 0; n < N; ++n){
            if(clusters[n] == tid) s_clust++;
        }
        clustern[tid] = s_clust;
    }
    __syncthreads();

    l_clust = clustern[row];

    for (int n = 0; n < N; ++n)
    {
        if(clusters[n] == row){
            l_sum += data[n * D +  col];
        }
    }
    means[tid] = l_sum/l_clust;

}

__global__ void reassign(double *d_data, double *d_clusters, double *d_means, double *d_clustern) {
    
}

