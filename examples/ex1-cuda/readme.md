```
srun --mem-per-cpu=1000 -p gpu -n 1 --gres=gpu:1 --constraint=cuda-7.5 --pty -t 0-01:00 /bin/bash
cd ~/cs205_2017_project/examples/ex1-cuda

source new-modules.sh
module load cuda/7.5-fasrc01

nvcc hello.cu -o hello.out
sbatch sbatch.run 
[change n in the sbatch.run file to change the number of cores]

cat slurm-#.out 
[ls to find the job# of the file]
```
