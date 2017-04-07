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
    double *x;
    double *c;
    int *assign;
    int N = 100;
    int K = 5;
    int D = 5;
    
    x = (double*) malloc(sizeof(double) * N * D);
    c = (double*) malloc(sizeof(double) * K * D);
    assign = (int*) malloc(sizeof(int) * N);
    
    srand ( time(NULL) );
    
    for(int i = 0; i < (N*D); ++i) x[i] = rand();
    for(int i = 0; i < (K*D); ++i) c[i] = 0;
    for(int i = 0; i < (N/K); ++i) for(int kk=0; kk < 5; ++kk) assign[kk + i * K] = kk;
    
//    for(int n =0; n < N; n++) printf("%d", assign[n]);
//    for(int i = 0; i < (N*D); ++i){
//        printf("%f", x[i]);
//        printf(", ");
//    }
//    printf("\n");
    
    kMeans (x, c, assign, N, K, D);
    
    for(int n =0; n < (K*D); n++){
        printf("%d", c[n]);
        printf("\n");
    }
    
    free(x);
    free(c);
    free(assign);
    return 0;
}
