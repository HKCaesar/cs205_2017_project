source new-modules.sh
module load cuda/7.5-fasrc01

rm -f cuda_hello.out
rm -f cuda_hello.err
rm -f cuda_square.out
rm -f cuda_square.err

srun -p gpu -n 1 --mem-per-cpu=1000 --gres=gpu:1 --constraint=cuda-7.5 --pty -t 0-01:00 /bin/bash
