source new-modules.sh
module load python/2.7.6-fasrc01
module load pycuda/2015.1.3-fasrc01
module load intel/15.0.0-fasrc01
module load cuda/7.5-fasrc01
srun --mem-per-cpu=1000 -p gpu -n 1 --gres=gpu:2 --constraint=cuda-7.5 --pty -t 0-01:00 /bin/bash
