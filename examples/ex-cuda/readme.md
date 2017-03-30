# CUDA Hello World

Step 1. Load modules. *only necessary one time*

Step 2. Make file. *only necessary if changes were made to the C code*

Step 3. Submit job. *you can manipulate the -n paramater in the sbatch.run file to change the number of cores*

Step 4. Check output. *.out file should contain a line printed by each core*

```
source setup.sh
nvcc hello.cu -o hello
sbatch sbatch.run
cat cuda_hello.out
```
Code from Ingemar Ragnemalm: https://www.pdc.kth.se/resources/computers/historical-computers/zorn/how-to/how-to-compile-and-run-a-simple-cuda-hello-world