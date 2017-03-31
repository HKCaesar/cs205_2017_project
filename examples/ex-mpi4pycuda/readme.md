# mpi4pycuda Hello World

Step 1. Load modules. *only necessary one time*

Step 2. Make file. *only necessary if changes were made to Python code*

Step 3. Nothing.

Step 4. Check output. *.out file should contain a line printed by each core*

```
source setup.sh
sbatch sbatch.run
cat mpi4pycuda_hello.out
cat mpi4pycuda_hello.err
```

## currently throwing a segmentation fault error :-(

Code from: https://gist.github.com/lebedov/8514d3456a94a6c73e6d
