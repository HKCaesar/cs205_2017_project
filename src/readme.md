# setting up the mpi4py + pycuda environment

### Pre-requisite (only needs to be done once): 
```
module load python/2.7.11-fasrc01
conda create -n pycuda --clone $PYTHON_HOME
```

### Steps

Step 1. Start an interactive session:

Step 2. Load modules.

Step 3. Run python code via sbatch.run

```
srun -p gpu -n 1 --mem-per-cpu=1000 --gres=gpu:1 --constraint=cuda-7.5 --pty -t 0-01:00 /bin/bash
source setup.sh
sbatch sbatch.run
```
