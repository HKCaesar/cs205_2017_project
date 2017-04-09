//
//  kmeans.c
//  
//
//  Created by Eric Dunipace on 3/30/17.
//
//

#include "kmeans.h"


double sumSq (double *x, double *c, int N,  int D,  int n,  int k)
{ //calculates the squared distance
    
    double ss = 0;
    
    for(int d = 0; d < D; ++d)
    {
        ss += (x[d + D * n] - c[d + k * D]) * (x[d + D * n] - c[d + k * D]);
    }
    return ss;
}

void selectCluster (double *x, double *c,  int *assign,  int N,  int K,  int D,  int * conv)
{//selects the cluster and calculates the distance (we may want to separate these actions)
    double min;
    int min_idx;
    int convCheck = 1;
    double dist;

    
    for (int n = 0; n < N; ++n)
    {
        min = 1.0/0.0;
        min_idx = -1;
        
        for (int k = 0; k < K; ++k)
        {
            dist = sumSq (x, c, N, D, n, k);
            if (min > dist)
            {
                min = dist;
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

void clusterCenter (double *x, double *c,  int *assign,  int N,  int K,  int D,  int *count)
{//calculates the center of the cluster
    
    
    for (int k = 0; k < K; ++k)
    {
        for(int d = 0; d < D; ++d){
            c[k * D + d] = 0;
        }
        count[k] = 0;
    }
    for (int n = 0; n < N; ++n)
    {
        for (int d = 0; d < D; ++d)
        {
            c[assign[n] * D + d] += x[n * D +  d];
        }
        count[assign[n]]++;

    }

    for(int k = 0; k < K; k++)
    {
        for (int d = 0; d < D; ++d)
        {
            c[k * D + d] /= count[k];
        }
    }
}

//void allTrue ( int *same,  int *conv,  int N)
//{ //not needed at the moment
//    int n = 0;
//    
//    while(n < N && conv) {
//        (*conv) = (same[n] ?  1 : 0);
//        ++n;
//    }
//    
//}

void kMeans (double *x, double *c,  int *assign,  int N,  int K,  int D, double cC)
{ //runs the k means algorithm
    int conv = 0;
    int count[K];
    
    while(!conv)
    {
    
        clusterCenter(x, c, assign, N, K, D, count);
        selectCluster(x, c, assign, N, K, D, &conv);
        
    }
    
    cC = (*c);
}
