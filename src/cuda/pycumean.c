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

__global__ void newMeans(double *data, int *labels, double *means)
{
    __shared__ int s_clustern[K];
    int k = threadIdx.x;
    int d = threadIdx.y;
    int tid = d + k * D;
    double l_sum = 0.0;

    // find the n per cluster with K lucky threads
    if (tid < K) {
        int l_clustern = 0;
        for (int n = 0; n < N; ++n){
            if(labels[n] == tid) l_clustern++;
        }
        s_clustern[tid] = l_clustern;
    }
    __syncthreads();
    
    // find KxD local sums
    for (int n = 0; n < N; ++n)
    {
        if(labels[n] == k){
            l_sum += data[d + n * D];
        }
    }
    
    // find KxD means
    means[tid] = l_sum/s_clustern[k];

}

  __global__ void reassign(double *data, double *labels, double *means) {
    
    __shared__ int s_squares[K*D];
    __shared__ int s_sums[K];
    int k = threadIdx.x;
    int d = threadIdx.y;
    int n = blockIdx.x;
    int tid = d + k * D;
    int dataid = d + n * D;
    double min;
    int min_idx;
    
    // get KxD squares
    s_squares[tid] = (data[dataid] - means[tid]) * (data[dataid] - means[tid]);
    __syncthreads();
    
    // add KxD squares to get K sums using K lucky threads
    if (tid < K) {
      for(int d = 0; d < D; ++d) {
        s_sums[k] += s_squares[d + k * D];
      }
    }
    __syncthreads();
    
    // check for the minimum of the K sums using 1 lucky thread
    if (tid == 0) {
      min = s_sums[0];
      min_idx = 0;
      for(int k = 1; k < K; ++k) {
        if (s_sums[k] < min) {
          min = s_sums[k];
          min_idx = k;
        }
      }
      if (labels[n] ! = min_idx) {
        labels[n] = min_idx;
      }
    }
  }
