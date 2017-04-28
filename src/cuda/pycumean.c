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
#define  numThreads  $numThreads



__device__ int isPowerOfTwo(int x)
{
    return ((x & (x - 1)) == 0);
}
__device__ double maxd(double X, double Y)
{
    return (((X) > (Y)) ? (X) : (Y));
}

__device__ int mini(int X, int Y)
{
    return (((X) < (Y)) ? (X) : (Y));
}

__device__ void reduceDouble(double *data, int start, int width)
{
    int tid = threadIdx.x;
    int idx = (tid + start);
    int end = (start + width - 1);
    int redThd = width/2;
    
    if (width > 32 && isPowerOfTwo(width)) {

        while(redThd > 32)
        {
            if (tid < redThd)
            {
                data[idx] += data[idx + redThd];
            }
            __syncthreads();
            
            redThd>>=1;
        }
        
        if (idx < 32)
        {
            data[idx] += data[idx + 32];
            __syncthreads();
            
            data[idx] += data[idx + 16];
            __syncthreads();
            
            data[idx] += data[idx + 8];
            __syncthreads();
            
            data[idx] += data[idx + 4];
            __syncthreads();
            
            data[idx] += data[idx + 2];
            __syncthreads();
            
            data[idx] += data[idx + 1];
            __syncthreads();
        }
    } else {
        
        if((width %2) !=0)
        {
            if(idx == end)
            {
                data[end - 1] += data[end];
            }
            redThd = width/2;
        }
        __syncthreads();

        while(redThd > 0)
        {
            if (tid < redThd)
            {
                data[idx] += data[idx + redThd];
            }
            
            __syncthreads();
            
            if((redThd %2) != 0)
            {
                if(tid == (redThd-1)) data[idx - 1] += data[idx];
            }
            redThd>>=1;
            __syncthreads();
        }
        __syncthreads();
    }
}

__device__ void reduceInt(int *data, int start, int width)
{
    int tid = threadIdx.x;
    int idx = (tid + start);
    int end = (start + width - 1);
    int redThd = width/2;
    
    if (width > 32 && isPowerOfTwo(width)) {
        
        while(redThd > 32)
        {
            if (tid < redThd)
            {
                data[idx] += data[idx + redThd];
            }
            __syncthreads();
            
            redThd>>=1;
        }
        
        if (idx < 32)
        {
            data[idx] += data[idx + 32];
            __syncthreads();
            
            data[idx] += data[idx + 16];
            __syncthreads();
            
            data[idx] += data[idx + 8];
            __syncthreads();
            
            data[idx] += data[idx + 4];
            __syncthreads();
            
            data[idx] += data[idx + 2];
            __syncthreads();
            
            data[idx] += data[idx + 1];
            __syncthreads();
        }
    } else {
        
        if((width %2) !=0)
        {
            if(idx == end)
            {
                data[end - 1] += data[end];
            }
            redThd = width/2;
        }
        __syncthreads();
        
        while(redThd > 0)
        {
            if (tid < redThd)
            {
                data[idx] += data[idx + redThd];
            }
            
            __syncthreads();
            
            if((redThd %2) != 0)
            {
                if(tid == (redThd-1)) data[idx - 1] += data[idx];
            }
            redThd>>=1;
            __syncthreads();
        }
        __syncthreads();
    }
}

__global__ void dist(double *data, double *means, double *dist)
{
    int activeThd = mini(numThreads, (N - (numThreads * blockIdx.x))); //in case last piece of data doesn't need all warps have the min function
    __shared__ double s_means[D]; //equivalent to a D vector
    __shared__ double s_data[numThreads];//equivalent to an N_subset vector


    int bid = threadIdx.x; //within block index
    int k = blockIdx.y; //which K mean are we talking about here
    int dataid = bid + blockIdx.x * numThreads; //gets specific individual
    
    if(bid < activeThd) //for last threadblock, not enough rows may be left to cover all threads
    {
        int meanCol = 0;
        double temp = 0;

        s_data[bid] = 0;
        if(bid<D) s_means[bid] = means[k*D + bid];
        __syncthreads();
        
        
        for(int n = dataid;
            (n < (N*D)) && (meanCol < D);
            n += N)
        {//reads data and doubles and sums should iterate through columns (column major format)
            temp = data[n] - s_means[meanCol];
            s_data[bid] += temp * temp;
            meanCol++;
        }
        dist[dataid + k * N] = s_data[bid]; //saves to global memory
    }
    __syncthreads();

}

__global__ void reassign(double *dist, int *label, int *conv_array)
{
    int activeThd = mini(numThreads, (N - (numThreads * blockIdx.x))); //in case last piece of data doesn't need all warps have the min function
    __shared__ int s_label[numThreads]; //equivalent to a D vector
//    __shared__ double s_min[numThreads];//equivalent to an N_subset vector
    __shared__ int s_conv[numThreads];//equivalent to an N_subset vector


    int bid = threadIdx.x; //within block index

    if(bid < activeThd)
    {
        int dataid = bid + blockIdx.x * numThreads; //gets specific individual
        int meanCol = 0;
        double temp;
        double l_min = 1.0/0.0; //set min to infinity! and beyond
        
        for(int n = bid + blockIdx.x * numThreads;
            (n < (N*K)) && (meanCol < K);
            n += N)
        {
            temp = dist[n];
            if(l_min > temp){
                l_min = temp;
                s_label[bid] = meanCol;
            }
            meanCol++;
        }
        s_conv[bid] = ((s_label[bid] == label[dataid]) ? 1 : 0);
        __syncthreads();

        reduceInt(&s_conv[0], 0 , activeThd);
        __syncthreads();

        
        if(bid==0) conv_array[blockIdx.x] = s_conv[0]/activeThd;
        label[dataid] = s_label[bid];
        __syncthreads();

    }
}

__global__ void countCluster(int *labels,int *clustern)
{
    __shared__ int s_labels[numThreads];
    
    int activeThd = mini(numThreads, (N - (numThreads * blockIdx.x))); //in case last piece of data doesn't need all warps have the min function
    int bid = threadIdx.x; //within block index
    int k =  blockIdx.y; //which K mean are we talking about here
    
    
    if (bid < activeThd)
    {//make sure used threads not stretch beyond data
        s_labels[bid] = 0;
        
        for(int n = bid; n < N; n += numThreads)
        {
            s_labels[bid] += (labels[n] == k);
        }
        __syncthreads();
        
        reduceInt(&s_labels[0], 0, numThreads);
        __syncthreads();
        
        if(bid == 0) clustern[k] = s_labels[0];
    }
    
    
}

__global__ void newMeans(double *data, int *labels, double *means, int *clustern)
{
    //FYI, max shared memory is 49,152 bytes or 6,144 doubles or 12,288 ints
    
    //loads data into COLUMN MAJOR format.

    /*loads a portion of a column of the data into memory, then conditional
     sums across data equaling the correct label */
    
    __shared__ double s_data[numThreads];
    
    int bid = threadIdx.x; //within block index
    int k = blockIdx.x; //which K mean are we talking about here
    int d = blockIdx.y; //which column of the K means matrix
//    int dataid = bid + blockIdx.x * numThreads; //gets specific individual
    
    // grid strided for loop
    s_data[bid] = 0;
//    s_labels[bid] = 0;
    
    for(int n = bid;
         n < N;
         n += numThreads)
    {
        s_data[bid] += ((labels[n] == k) ? data[n + d * N] : 0);
    }
    __syncthreads();
    
    reduceDouble(&s_data[0], 0, numThreads);
    __syncthreads();

       // store final result into the K x D matrix of means
    
    
    if(bid == 0) {
        double temp = ((double)clustern[k]);
        
        if(temp ==0) {
            means[k * D + d]  = 0;
        } else {
            means[k * D + d]  = s_data[0]/temp;
        }
    }
    
}
