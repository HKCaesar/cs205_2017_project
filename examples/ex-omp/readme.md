# OpenMP Hello World

Step 1. Load modules. *only necessary one time*

Step 2. Make file. *only necessary if changes were made to the C code*

Step 3. Submit job. *you can manipulate the -c paramater in the sbatch.run file to change the number of cores*

Step 4. Check output. *.out file should contain a line printed by each core*

```
source setup.sh
make
sbatch sbatch.run
cat omp_hello.out
```
Code from Odyssey OpenMP resources: https://github.com/fasrc/User_Codes/tree/master/CS205/OpenMP
