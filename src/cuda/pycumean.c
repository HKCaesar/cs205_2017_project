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

__global__ void reassign(double *data, double *labels, double *means, int *conv_array, int * conv)
{
    __shared__ int s_squares[K*D];
    __shared__ int s_sums[K];
    int k = threadIdx.x;
    int d = threadIdx.y;
    int n = blockIdx.x;
    int wInBlockid = d + k * D;
    int tid = d + k * D + n * K*D;
    int dataid = d + n * D;
    double min;
    int min_idx;
    __shared__ int s_conv[K*D];
    
    if(tid == 0) (*conv) = 1;
    
    // get KxD squares
        s_squares[wInBlockid] = (data[dataid] - means[wInBlockid]) * (data[dataid] - means[wInBlockid]);
        __syncthreads();
    
    // add KxD squares to get K sums using K lucky threads
        if (k < K && d == 0) {
          for(int dd = 0; dd < D; ++dd) {
            s_sums[k] += s_squares[dd + k * D];
          }
        }
    __syncthreads();
    
    for(int ii=0;ii<k;ii++){
        printf(s_sums[ii]); printf(" ");
    }
    
    // check for the minimum of the K sums using 1 lucky thread per block
        if (wInBlockid == 0) {
          min = 1.0/0.0;
          min_idx = -1;
          for(int kk = 0; kk < K; ++kk) {
            if (s_sums[kk] < min) {
              min = s_sums[kk];
              min_idx = kk;
            }
          }
          if (labels[n] != min_idx) {
            conv_array[n] = 0;
            labels[n] = min_idx;
          }
        }
    __syncthreads();

    //check convergence
    if(tid < N){
        s_conv[wInBlockid] = conv[tid];
    }
        __syncthreads();
    
    if(tid < N){
        for(int kk = (K*D)/2; kk > 0; kk>>=1)
        {
            double temp = s_conv[wInBlockid + kk];
            if(temp <  s_conv[wInBlockid]) s_conv[wInBlockid] = temp;
            __syncthreads();
        }
        if(wInBlockid == 0) conv_array[n] = s_conv[0];
    }
    __syncthreads();
    if(tid == 0){
        for(int kk = 0; kk < N/(K*D); kk++){
            (*conv) = ((*conv) > conv_array[n] ? 0 : 1);
            if(!(*conv)) break;
        }
    }
     __syncthreads();
}
