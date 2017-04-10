# submit batch job with kmeans.py

From the login node: 
```
sbatch sbatch.run
```

## run interactive session with kmeans.py (doesn't run mpi4py, just pyCUDA)

From the login node:
```
srun -p gpu -n 1 --mem-per-cpu=500 --gres=gpu:1 --constraint=cuda-7.5 --pty -t 0-01:00 /bin/bash
module load python/2.7.11-fasrc01
module load cuda/7.5-fasrc02
module load gcc/4.8.2-fasrc01 openmpi/1.10.2-fasrc01
source activate pycuda
git pull; python kmeans.py
```

## mpi4py + pyCUDA environment set-up (do once): 
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
