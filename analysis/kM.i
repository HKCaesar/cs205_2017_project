%module kM
%{
  #define SWIG_FILE_WITH_INIT
  #include "src/kmeans.h"
%}

%include "numpy.i"
%init %{
import_array();
%}


%apply (double *IN_ARRAY2, int DIM1, int DIM2) {(double *x, int N, int D),(double *c, int K, int DD)};
%apply (int *IN_ARRAY1, int DIM1) {(int *assign, int NN)};

%rename (kMeans) kM_wrap;
%exception kM_wrap {
    $action
    if (PyErr_Occurred()) SWIG_fail;
}
%inline %{
    void kM_wrap(double *x, int N, int D, double *c, int K, int DD, int *assign, int NN) {
        if (N != NN) {
            PyErr_Format(PyExc_ValueError,
                         "Arrays of lengths (%d,%d) given",
                         N, NN);
        }
        kMeans(x, c, assign, N, K, D);
    }
%}


%include "src/kmeans.h"
