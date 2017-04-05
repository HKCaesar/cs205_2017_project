# mpi4py + pyCUDA environment

### set-up (do once): 
```
module load python/2.7.11-fasrc01
conda create -n pycuda --clone $PYTHON_HOME
pip install pycuda
pip install mpi4py
```

## command-line code:

From the login node: 
```
sbatch sbatch.run
```
[need to call python file from within sbatch.run]
