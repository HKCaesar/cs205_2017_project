```
srun --mem-per-cpu=1000 -p gpu -n 1 --gres=gpu:1 --constraint=cuda-7.5 --pty -t 0-01:00 /bin/bash

source new-modules.sh
module load cuda/7.5-fasrc01

nvcc hello.cu -o hello.out
sbatch sbatch.run
```
