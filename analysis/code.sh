#!/bin/bash

#swig -python kM.i
#gcc -O2 -std=c11 -fPIC -c src/kmeans.c kM_wrap.c \
#    -I/Users/eifer/anaconda/include -arch x86_64 \
#    -I/Users/eifer/anaconda/lib/python3.5/site-packages/numpy/core/include \
#    -I. -I/Users/eifer/anaconda/include/python3.5m
#
#gcc -std=c11 -fpic -I/Users/eifer/anaconda/include -arch x86_64 \
#-I/Users/eifer/anaconda/lib/python3.5/site-packages/numpy/core/include \
#-I. -I/Users/eifer/anaconda/include/python3.5m \
#-c src/kmeans.c kM_wrap.c
#
#swig -python kM.i
#gcc -c `python-config --cflags` src/kmeans.c kM_wrap.c
#gcc -bundle `python-config --ldflags` kmeans.o kM_wrap.o -o _kM.so
python setup.py build_ext --inplace
