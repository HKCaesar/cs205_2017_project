#!/bin/bash
#SBATCH -J mpi4pycuda_hello
#SBATCH -o mpi4pycuda_hello.out
#SBATCH -e mpi4pycuda_hello.err
#SBATCH -p holyseasgpu
#SBATCH -n 2
#SBATCH --gres=gpu:2
#SBATCH -t 0-00:30
#SBATCH --mem-per-cpu=500
#SBATCH --constraint=cuda-7.5

module load python/2.7.11-fasrc01
module load cuda/7.5-fasrc02
module load gcc/4.8.2-fasrc01 openmpi/1.10.2-fasrc01
source activate pycuda

srun -n $SLURM_NTASKS --mpi=pmi2 python mpi4pycuda_hello.py
