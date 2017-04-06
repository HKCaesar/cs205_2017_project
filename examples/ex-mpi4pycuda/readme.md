# mpi4pycuda Hello World

### set-up (do once): 
```
module load python/2.7.11-fasrc01
conda create -n pycuda --clone $PYTHON_HOME
pip install pycuda
pip install mpi4py
```
_Note: mpi4py didn't compile for me on login node. Maybe it needs some other options (Eric)_

_Update: this may work with this compiler ```module load gcc/5.2.0-fasrc01 openmpi/1.10.4-fasrc01```, but haven't tested_

## command-line code:

Step 1. From the login node: 
```
sbatch sbatch.run
```

Step 2. Check output:
```
cat mpi4pycuda_hello.out
cat mpi4pycuda_hello.err
```

Code from: https://gist.github.com/lebedov/8514d3456a94a6c73e6d


## Per full commands from email:
#### From Login
```
module load python/2.7.11-fasrc01
conda create -n pycuda --clone $PYTHON_HOME
srun -p gpu --pty --mem 2000 --gres gpu:1 -t 1200 /bin/bash
```


#### Then on GPU:
```
module load python/2.7.11-fasrc01
source activate pycuda
module load cuda/7.5-fasrc02
pip install pycuda
module load gcc/5.2.0-fasrc01 openmpi/1.10.4-fasrc01
pip install mpi4py

python #if you want python now for some reason
```

### Testing
```
sbatch sbatch.run
```
