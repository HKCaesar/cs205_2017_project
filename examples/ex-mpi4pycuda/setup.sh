module load python/2.7.11-fasrc01
source activate pycuda
module load cuda/7.5-fasrc02
pip install pycuda
module load gcc/4.9.0-fasrc01 openmpi/1.10.4-fasrc01
pip install mpi4py
