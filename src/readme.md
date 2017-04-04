# mpi4py + pyCUDA environment

### set-up (do once): 
```
module load python/2.7.11-fasrc01
conda create -n pycuda --clone $PYTHON_HOME
```

## run code

Step 1. Load modules (onto the login node).

Step 2. Run python code via sbatch.run

```
source setup.sh
sbatch sbatch.run
```
