//
//  k_run.c
//  
//
//  Created by Eric Dunipace on 3/30/17.
//
//

#include "kmeans.h"
#include <time.h>


int main(){
    float *x;
    float *c;
    int *assign;
    int N = 100;
    int K = 5;
    int D = 5;
    
    x = (float*) malloc(sizeof(float) * N * D);
    c = (float*) malloc(sizeof(float) * K * D);
    assign = (int*) malloc(sizeof(int) * N);
    
    srand ( time(NULL) );
    
    for(int i = 0; i < (N*D); ++i) x[i] = rand();
    for(int i = 0; i < (K*D); ++i) c[i] = rand();
    for(int i = 0; i < (N/K); ++i) for(int kk=0; kk < 5; ++kk) assign[kk + i * K] = kk;
    
    for(int n =0; n < N; n++) printf("%d", assign[n]);
//    for(int i = 0; i < (N*D); ++i){
//        printf("%f", x[i]);
//        printf(", ");
//    }
    printf("\n");
    
    kMeans (x, c, assign, N, K, D);
    
    for(int n =0; n < N; n++) printf("%d", assign[n]);
    
    return 0;
}
