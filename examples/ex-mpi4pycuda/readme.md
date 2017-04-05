# mpi4pycuda Hello World

### set-up (do once): 
```
module load python/2.7.11-fasrc01
conda create -n pycuda --clone $PYTHON_HOME
module load cuda/7.5-fasrc02
pip install pycuda
module load gcc/4.8.2-fasrc01 openmpi/1.10.2-fasrc01
pip install mpi4py
```

## command-line code:

Step 1. From the login node: 
```
sbatch sbatch.run
```

Step 2. Check output:
```
cat mpi4pycuda_hello.out
cat mpi4pycuda_hello.err
```

Code from: https://gist.github.com/lebedov/8514d3456a94a6c73e6d
