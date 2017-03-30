Odyssey MPI resources: https://github.com/fasrc/User_Codes/tree/master/CS205/MPI

```
Step 1. Load modules: [only necessary one time]

source setup.sh

Step 2. Make file: [only necessary if changes were made to mpi_hello_world.c]

make
```

```
Step 3. Submit job:

sbatch sbatch.run

[you can manipulate the -c paramater in the sbatch.run file to change the number of cores]
```

```
Step 4. Check output: 

cat mpi_hello.out
```
