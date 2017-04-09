//
//  k_run.c
//  
//
//  Created by Eric Dunipace on 3/30/17.
//
//

#include "kmeans.h"
#include <time.h>
#include <math.h>
#include<stdio.h>
#include<stdlib.h>

double sampleNormal() {
    double u = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double) rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1) return sampleNormal();
    double c = sqrt(-2 * log(r) / r);
    return u * c;
}


int main(){
    double *x;
    double *c;
    int *assign;
    int N = 100;
    int K = 3;
    int D = 6;
    int count = 0;
    
    x = (double*) malloc(sizeof(double) * N * D);
    c = (double*) malloc(sizeof(double) * K * D);
    assign = (int*) malloc(sizeof(int) * N);
    
    srand ( time(NULL) );
    
    for(int i = 0; i < (D*K); ++i){
        c[i] = count;
        if(i % 6 ==0) count += 5;
    }
    count = 0;
    
    for(int n =0; n < N; ++n) {
        if(n < 33) assign[n] = 0;
        if(n >= 33 && n < 66) assign[n] = 1;
        if(n >= 66) assign[n]=2;
    }
    
    printf("true clusters: ");
    for(int n =0; n < N; n++){
        printf("%d", assign[n]);
        printf(", ");
    }
    printf("\n");
    
    for(int i = 0; i < (N*D); ++i){
        if(i % D == 0) count++;

        x[i] = sampleNormal() + c[assign[count]*D];
    }
    
    printf("data: ");
    for(int n =0; n < (N*D); n++) {
        printf("%f", x[n]);
        printf(", ");
    }
    printf("\n");

    printf("\n");


    for(int i = 0; i < (K*D); ++i) c[i] = 0;
    for(int i = 0; i < (N/K); ++i) for(int kk=0; kk < K; ++kk) assign[kk + i * K] = kk;
    
//    for(int n =0; n < N; n++) printf("%d", assign[n]);
//    for(int i = 0; i < (N*D); ++i){
//        printf("%f", x[i]);
//        printf(", ");
//    }
//    printf("\n");
    
//    for(int n =0; n < (K*D); n++){
//        printf("%f", c[n]);
//        printf("\n");
//    }
//    for(int n = 0; n < N; n++){
//        for(int k = 0; k < K; k++){
//            printf("%f", sumSq(x, c, D,  N,  n,  k));
//            printf("\n");
//        }
//    }
//    printf("\n");
//    printf("\n");


    kMeans (x, c, assign, N, K, D);
    
    printf("calculated clusters: ");
    for(int n =0; n < (K*D); n++){
        printf("%f", c[n]);
        printf(", ");
    }
    printf("\n");
    printf("\n");
//
//    printf("\n");
//    printf("\n");
//    printf("\n");
//
    for(int n =0; n < N; n++){
        printf("%d", assign[n]);
        printf("\n");
    }
//
    free(x);
    free(c);
    free(assign);
    return 0;
}
