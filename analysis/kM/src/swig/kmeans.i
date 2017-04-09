%module simple
%{
  #define SWIG_FILE_WITH_INIT
  #include "../kmeans.h"
%}

%include "numpy.i"
%init %{
import_array();
%}


%apply (float* IN_ARRAY2, int DIM1, int DIM2) {(float *x, int N, int D), (float *c,int K, int D)};
%apply (int* IN_ARRAY1, int DIM1) {(int *assign, int N),(int *count, int K)};

%include "../kmeans.h"
