# mpi4pycuda Hello World

Step 1. Load modules. *only necessary one time*

Step 2. Run the sbatch.run file. 

[wait]

Step 3. Check output. *.out file should contain a line printed by each core*

```
source setup.sh
sbatch sbatch.run
cat mpi4pycuda_hello.out
cat mpi4pycuda_hello.err
```

## currently throwing a segmentation fault error and running indefinitely :-( maybe this is relevant? https://rc.fas.harvard.edu/resources/documentation/gpgpu-computing-on-odyssey/

Code from: https://gist.github.com/lebedov/8514d3456a94a6c73e6d
