//
//  kmeans.h
//  
//
//  Created by Eric Dunipace on 3/30/17.
//
//

#ifndef pycumean_h
#define pycumean_h

#include <stdio.h>
#include <stdlib.h>

//double sumSq (double *x, double *c, int N, int D,  int n,  int k);
//void selectCluster (double *x, double *c,  int *assign,  int N,  int K,  int D,  int * conv);
//void clusterCenter (double *x, double *c,  int *assign,  int N,  int K,  int D,  int * count);
//void allTrue ( int *same,  int *conv,  int N);
void newMeans(const int N, const int D, const int K, double *data, int *clusters, double *means, int *clustern)


#endif /* pycumean_h */
