# Handy Slurm

https://slurm.schedmd.com/quickstart.html

```
scontrol show partition
scontrol show node holyseasgpu01
scontrol show job ####

nvidia-smi
```
### 1 CPU

```
$ srun -p gpu -n 1 --pty --mem 100 -t 500 /bin/bash
$ scontrol show job 85402636

JobId=85402636 JobName=bash
   NumNodes=1 NumCPUs=2 NumTasks=1 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=2,mem=100M,node=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryNode=100M MinTmpDiskNode=0
   Features=(null) Gres=(null) Reservation=(null)
```  

### 1 GPU
#### NumNodes=1 NumCPUs=2 NumTasks=1 CPUs/Task=1
#### Features=(null) Gres=gpu:1 Reservation=(null)

```
$ srun -p gpu --pty --mem 100 --gres gpu:1 -t 500 /bin/bash
$ scontrol show job 85402645

JobId=85402645 JobName=bash
   NumNodes=1 NumCPUs=2 NumTasks=1 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=2,mem=100M,node=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryNode=100M MinTmpDiskNode=0
   Features=(null) Gres=gpu:1 Reservation=(null)
```  


### 1 CPU, 1 GPU

```
$ srun -p gpu -n 1 --pty --mem 100 --gres gpu:1 -t 500 /bin/bash
$ scontrol show job 85402509

JobId=85402509 JobName=bash
   NumNodes=1 NumCPUs=2 NumTasks=1 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=2,mem=100M,node=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryNode=100M MinTmpDiskNode=0
   Features=(null) Gres=gpu:1 Reservation=(null)
```  

### 2 CPUs, 1 GPU

```
$ srun -p gpu -n 2 --pty --mem 100 --gres gpu:1 -t 500 /bin/bash
$ scontrol show job 85402335

JobId=85402335 JobName=bash
   NumNodes=1 NumCPUs=2 NumTasks=2 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=2,mem=100M,node=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryNode=100M MinTmpDiskNode=0
   Features=(null) Gres=gpu:1 Reservation=(null)

```   

### 1 CPU, 2 GPUs

```
$ srun -p gpu -n 1 --pty --mem 100 --gres gpu:2 -t 500 /bin/bash
$ scontrol show job 85402220

JobId=85402220 JobName=bash
   NumNodes=1 NumCPUs=2 NumTasks=1 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=2,mem=1000M,node=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryNode=1000M MinTmpDiskNode=0
   Features=(null) Gres=gpu:2 Reservation=(null)
```

### 1 CPU, 5 GPUs

```
$ srun -p gpu -n 1 --pty --mem 1000 --gres gpu:5 -t 500 /bin/bash
$ scontrol show job 85401770

JobId=85401770 JobName=bash
   NumNodes=1 NumCPUs=2 NumTasks=1 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=2,mem=1000M,node=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryNode=1000M MinTmpDiskNode=0
   Features=(null) Gres=gpu:5 Reservation=(null)
```

### 1 CPU, 6 GPUs

```
$ srun -p gpu -n 1 --pty --mem 100 --gres gpu:6 -t 500 /bin/bash
$ scontrol show job 85401966

JobId=85401966 JobName=bash
   NumNodes=1 NumCPUs=2 NumTasks=1 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=2,mem=100M,node=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryNode=100M MinTmpDiskNode=0
   Features=(null) Gres=gpu:6 Reservation=(null)
```

### 2 CPUs, 2 GPUs

```
$ srun -p gpu -n 2 --pty --mem 1000 --gres gpu:2 -t 500 /bin/bash
$ scontrol show job 85362472
JobId=85362472 JobName=bash
   NumNodes=1 NumCPUs=2 NumTasks=2 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=2,mem=1000M,node=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryNode=1000M MinTmpDiskNode=0
   Features=(null) Gres=gpu:2 Reservation=(null)
```
