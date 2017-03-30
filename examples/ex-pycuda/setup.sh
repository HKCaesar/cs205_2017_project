source new-modules.sh
module load pycuda/2015.1.3-fasrc01
srun --mem-per-cpu=1000 -p gpu -n 1 --gres=gpu:1 --constraint=cuda-7.5 --pty -t 0-01:00 /bin/bash