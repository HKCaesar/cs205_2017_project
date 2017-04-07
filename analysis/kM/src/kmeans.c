//
//  kmeans.c
//  
//
//  Created by Eric Dunipace on 3/30/17.
//
//

#include "kmeans.h"


extern void sumSq (double *x, double *c, double *ss,  int D,  int N,  int n,  int k)
{ //calculates the squared distance
    
    double sum = 0;
    
    for(size_t d = 0; d < D; ++d)
    {
        sum += (x[d + D * n] - c[d + k * D]) * (x[d + D * n] - c[d + k * D]);
    }
    (*ss) = sum;
}

extern void selectCluster (double *x, double *c,  int *assign,  int N,  int D,  int K,  int *conv, double *dist)
{//selects the cluster and calculates the distance (we may want to separate these actions)
    double min;
     int min_idx;
     int convCheck = 1;

    
    for (size_t n = 0; n < N; ++n)
    {
        min = INFINITY;
        min_idx = -1;
        
        for (size_t k = 0; k < K; ++k)
        {
            sumSq (x, c, dist, D, N, n, k);
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

extern void clusterCenter (double *x, double *c,  int *assign,  int N,  int K,  int D,  int *count)
{//calculates the center of the cluster
    
    
    for (size_t k = 0; k < K; k++)
    {
        for(size_t d = 0; d < D; ++d) c[k * D + d];
        count[k] = 0;
    }
    
    for (size_t n = 0; n < N; ++n)
    {
        for (size_t d = 0; d < D; ++d)
        {
            c[assign[n] * D + d] += x[n * D +  d];
            count[assign[n]]++;
        }
        
    }
    
    for(size_t k = 0; k < K; ++k){
        for (size_t d = 0; d < D; ++d) {
            c[k * D + d] /= count[k];
        }
    }
}

//extern void allTrue ( int *same,  int *conv,  int N)
//{ //not needed at the moment
//    size_t n = 0;
//    
//    while(n < N && conv) {
//        (*conv) = (same[n] ?  1 : 0);
//        ++n;
//    }
//    
//}

extern void kMeans (double *x, double *c,  int *assign,  int N,  int K,  int D)
{ //runs the k means algorithm
    int *conv = ( int*) malloc(sizeof( int));
    int *count = ( int*) malloc(sizeof( int) * K);
    double *dist = (double*) malloc(sizeof(double));

    (*conv) = 0;
    
    while(!(*conv))
    {
        
        clusterCenter(x, c, assign, N, D, K, count);
        selectCluster(x, c, assign, N, D, K, conv, dist);
        
    }
    
    free(count);
    free(conv);
    free(dist);

}
