Odyssey MPI resources: https://github.com/fasrc/User_Codes/tree/master/CS205/MPI

```
srun --mem-per-cpu=1000 -p gpu -n 1 --gres=gpu:1 --constraint=cuda-7.5 --pty -t 0-01:00 /bin/bash
source new-modules.sh
module load intel/15.0.0-fasrc01
module load openmpi/1.10.2-fasrc01
```

```
make
sbatch sbatch.run
cat mpi_hello.out
```

[you can manipulate the -c paramater in the sbatch.run file to change the number of cores]
