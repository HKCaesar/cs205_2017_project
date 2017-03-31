//
//  kmeans.c
//  
//
//  Created by Eric Dunipace on 3/30/17.
//
//

#include "kmeans.h"


void inline sumSq (float *x, float *c, float *ss, int *D, int *N, int *K, int *n, int *k )
{
    float sum;
    
    sum = 0;
    
    for(size_t d = 0; d < D; ++n)
    {
        sum += (*x[d + D*n] - *c[d + k*K]) * (*x[d + D*n] - *c[d + k*K]);
    }
    ss = sum;
}

void inline selectCluster (float *x, float *c, int *assign, int *N, int *D, int *K, int *conv)
{
    float *dist;
    float min;
    int min_idx;
    int convCheck = 1;
    
    dist = (float*) malloc(sizeof(float))
    
    for (size_t n = 0; n < N; ++n)
    {
        min = INFINITY;
        min_idx = -1;
        
        for (size_t k = 0; k < K; ++k)
        {
            sumSq (x, c, dist, D, N, K, n, k);
            if (min > dist)
            {
                min = dist;
                min_idx = k;
            }
        }
        if(convCheck)
        {
            min_idx == assign[n] ? conv = 1 : conv = 0;
            if(!conv) convCheck = 0;
        }
        
        assign[n] = min_idx;
        
    }
}

void inline clusterCenter (float *x, float *c, int *assign, int *N, int *K, int *D)
{
    int *count;
    
    count = (int*) malloc(sizeof(int) * K);
    
    for (size_t t; t < (K*D); t++)
    {
        c[t] = 0;
    }
    
    for (size_t n = 0; n < N; ++n)
    {
        for (size_t d = 0; d < D; ++d)
        {
            c[assign[n] * K + d] += x[n * D +  d];
            ++count[assign[n]];
        }
        
    }
    
    for(size_t k = 0; k < K; ++k){
        for (size_t d = 0; d < D; ++d) {
            c[k*K + d] /= count[k];
        }
    }
}

void inline allTrue (int *same, int *conv, int *N)
{
    size_t n = 0;
    
    while(n < N && conv) {
        same[n] ? conv = 1 : conv = 0;
        ++n;
    }
    
}

void inline kMeans (float *x, float *c, int *assign, int *N, int *K, int *D)
{
    int conv;
    
    conv = 1;
    
    while(!conv)
    {
        clusterCenter(x, c, assign, N, D, K);
        selectCluster(x, c, assign, N, D, K, conv);
    }
}
