# submit batch job with kmeans.py

From the login node: 

```
cd ~/cs205_2017_project/src

for PARAM1 in $(seq 1 10); do
 for PARAM2 in 1 2; do
  for PARAM3 in 1 2; do
  #  #
  echo "${PARAM1}, ${PARAM2}, ${PARAM3}"
  export PARAM1 PARAM2 PARAM3
  #
  sbatch -N ${PARAM2} -n ${PARAM3} --gres=gpu:${PARAM3} \
    sbatch.run
   #
   sleep 1 # pause to be kind to the scheduler
  done
 done
done

cat kmeans.out
cat kmeans.err
```

or

```
srun -p holyseasgpu -n 2 --mem-per-cpu=2500 --gres=gpu:2 --constraint=cuda-7.5 --mpi=pmi2 --pty -t 0-04:00 /bin/bash
cd ~/cs205_2017_project/src/python
module load python/2.7.11-fasrc01
module load cuda/7.5-fasrc02
module load gcc/4.8.2-fasrc01 openmpi/1.10.2-fasrc01
source activate pycuda
git pull; mpiexec -n 2 python run.py
```

## mpi4py + pyCUDA environment set-up (do once): 
```
module load python/2.7.11-fasrc01
conda create -n pycuda --clone $PYTHON_HOME
srun -p gpu --pty --mem 2000 --gres gpu:1 -t 1200 /bin/bash
module load python/2.7.11-fasrc01
source activate pycuda
module load cuda/7.5-fasrc02
pip install pycuda
module load gcc/4.8.2-fasrc01 openmpi/1.10.2-fasrc01
pip install mpi4py
```
