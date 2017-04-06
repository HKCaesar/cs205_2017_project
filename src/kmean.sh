#!/bin/bash

org="kmean_test"

# Load required modules
# module load python/2.7.11-fasrc01
#source activate pycuda
#module load cuda/7.5-fasrc02
#pip install pycuda
#module load gcc/5.2.0-fasrc01 openmpi/1.10.4-fasrc01
#pip install mpi4py

module load gcc/5.2.0-fasrc01

# Pull data from git
git fetch
git checkout $org
git pull origin $org

# Make appropriate files
make

# Run target files
./x.kCPU > kcpu.txt

