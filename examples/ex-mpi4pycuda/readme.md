# mpi4pycuda Hello World

### set-up (do once): 
```
module load python/2.7.11-fasrc01
conda create -n pycuda --clone $PYTHON_HOME
```

## command-line code:

Step 1. Start an interactive session:

Step 2. Load modules.

Step 3. Run python code via sbatch.run

Step 4. Check output.

```
srun -p gpu -n 1 --mem-per-cpu=1000 --gres=gpu:1 --constraint=cuda-7.5 --pty -t 0-01:00 /bin/bash
source setup.sh
sbatch sbatch.run
cat mpi4pycuda_hello.out
cat mpi4pycuda_hello.err
```

Code from: https://gist.github.com/lebedov/8514d3456a94a6c73e6d
