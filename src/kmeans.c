//
//  kmeans.c
//  
//
//  Created by Eric Dunipace on 3/30/17.
//
//

#include "kmeans.h"


void inline sumSq (float *x, float *c, float *ss, int D, int N, int K, int n, int k)
{ //calculates the squared distance
    
    float sum = 0;
    
    for(size_t d = 0; d < D; ++d)
    {
        sum += (x[d + D * n] - c[d + k * K]) * (x[d + D * n] - c[d + k * K]);
    }
    (*ss) = sum;
}

void inline selectCluster (float *x, float *c, int *assign, int N, int D, int K, int *conv, float *dist)
{//selects the cluster and calculates the distance (we may want to separate these actions)
    float min;
    int min_idx;
    int convCheck = 1;

    
    for (size_t n = 0; n < N; ++n)
    {
        min = INFINITY;
        min_idx = -1;
        
        for (size_t k = 0; k < K; ++k)
        {
            sumSq (x, c, dist, D, N, K, n, k);
            if (min > (*dist))
            {
                min = (*dist);
                min_idx = k;
            }
        }
        if(convCheck)
        {
            (*conv) = (min_idx == assign[n] ? 1 : 0);
            if(!conv) convCheck = 0;
        }
        
        assign[n] = min_idx;
        
    }
}

void inline clusterCenter (float *x, float *c, int *assign, int N, int K, int D, int *count)
{//calculates the center of the cluster
    
    
    for (size_t t; t < (K * D); t++)
    {
        c[t] = 0;
    }
    
    for (size_t n = 0; n < N; ++n)
    {
        for (size_t d = 0; d < D; ++d)
        {
            c[assign[n] * K + d] += x[n * D +  d];
            count[assign[n]]++;
        }
        
    }
    
    for(size_t k = 0; k < K; ++k){
        for (size_t d = 0; d < D; ++d) {
            c[k * K + d] /= count[k];
        }
    }
}

void inline allTrue (int *same, int *conv, int N)
{ //not needed at the moment
    size_t n = 0;
    
    while(n < N && conv) {
        (*conv) = (same[n] ?  1 : 0);
        ++n;
    }
    
}

void inline kMeans (float *x, float *c, int *assign, int N, int K, int D)
{ //runs the k means algorithm
    int *conv;
    int *count;
    float *dist;

    
    count = (int*) malloc(sizeof(int) * K);
    conv = (int*) malloc(sizeof(int));
    dist = (float*) malloc(sizeof(float));
    
    (*conv) = 0;
    
    while(!(*conv))
    {
        for(size_t k = 0; k < K; ++k) count[k] = 0;
        
        clusterCenter(x, c, assign, N, D, K, count);
        selectCluster(x, c, assign, N, D, K, conv, dist);
        
    }

}
