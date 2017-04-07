//
//  kmeans.h
//  
//
//  Created by Eric Dunipace on 3/30/17.
//
//

#ifndef kmeans_h
#define kmeans_h

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

extern void sumSq (double *x, double *c, double *ss,  int D,  int N, int n,  int k);
extern void selectCluster (double *x, double *c,  int *assign,  int N,  int D,  int K,  int *conv, double *dist);
extern void clusterCenter (double *x, double *c,  int *assign,  int N,  int K,  int D,  int *count);
//extern void allTrue ( int *same,  int *conv,  int N);
extern void kMeans (double *x, double *c,  int *assign,  int N,  int K,  int D);


#endif /* kmeans_h */
