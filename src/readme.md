# mpi4py + pyCUDA environment

### set-up (do once): 
```
module load python/2.7.11-fasrc01
conda create -n pycuda --clone $PYTHON_HOME
srun -p gpu --pty --mem 2000 --gres gpu:1 -t 1200 /bin/bash
module load python/2.7.11-fasrc01
source activate pycuda
module load cuda/7.5-fasrc02
pip install pycuda
module load gcc/4.8.2-fasrc01 openmpi/1.10.2-fasrc01
pip install mpi4py
```

## command-line code:

From the login node: 
```
sbatch sbatch.run
```
[need to call python file from within sbatch.run]
