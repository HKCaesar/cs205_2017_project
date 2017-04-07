#!/bin/bash

#swig -python -o wrap.c kmeans.i
#gcc -O2 -std=c11 -fPIC -c ../kmeans.c

python setup.py build_ext --inplace
