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

void inline sumSq (float *x, float *c, float *ss, int *D, int *N, int *K, int *n, int *k );
void inline selectCluster (float *x, float *c, int *assign, int *N, int *D, int *K, int *conv);
void inline clusterCenter (float *x, float *c, int *assign, int *N, int *K, int *D);
void inline allTrue (int *same, int *conv, int *N);
void inline kMeans (float *x, float *c, int *assign, int *N, int *K, int *D);



#endif /* kmeans_h */
