//
//  pycumean.c
//  
//
//  Created by Eric Dunipace on 3/30/17.
//
//

//include "pycumean.h"

#include <stdio.h>

#define  N $N
#define  K $K
#define  D $D

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

__global__ void reassign(double *data, int *labels, double *means, int *converged)
{
    __shared__ double s_squares[K*D];
    __shared__ double s_sums[K];
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
        if (k < K && d == 0) {
          for(int dd = 0; dd < D; ++dd) {
            s_sums[k] += s_squares[dd + k * D];
          }
        }
    __syncthreads();
    
    // check for the minimum of the K sums using 1 lucky thread per block
        if (tid == 0) {
          converged[n] = 0;
          min = s_sums[0];
          min_idx = 0;
          for(int kk = 1; kk < K; ++kk) {
            if (s_sums[kk] < min) {
              min = s_sums[kk];
              min_idx = kk;
            }
          }          
          if (labels[n] != min_idx) {
            converged[n] = 1;
            labels[n] = min_idx;
          }
        }
    __syncthreads();

}
