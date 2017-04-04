# setting up the mpi4py + pycuda environment

Pre-requisite (only needs to be done once): 
```
module load python/2.7.11-fasrc01
conda create -n pycuda --clone $PYTHON_HOME
```

Step 1. Start an interactive session:

Step 2. Load modules.

Step 3. Run python code via sbatch.run

```
srun -p gpu --pty --mem 1000 --gres gpu:1 -t 500 /bin/bash
source setup.sh
sbatch sbatch.run
```
