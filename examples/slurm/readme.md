# Handy Slurm

https://slurm.schedmd.com/quickstart.html

```
scontrol show partition
scontrol show node gpu
scontrol show job ####

nvidia-smi
```

### 2 CPUs, 1 GPU
srun -p gpu -n 2 --pty --mem 1000 --gres gpu:1 -t 500 /bin/bash
scontrol show job 85362170

JobId=85362170 JobName=bash
   NumNodes=1 NumCPUs=2 NumTasks=2 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=2,mem=1000M,node=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryNode=1000M MinTmpDiskNode=0
   Features=(null) Gres=gpu:1 Reservation=(null)

### 1 CPU, 2 GPUs
srun -p gpu -n 1 --pty --mem 1000 --gres gpu:2 -t 500 /bin/bash
scontrol show job 85362494

JobId=85362494 JobName=bash
   NumNodes=1 NumCPUs=2 NumTasks=1 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=2,mem=1000M,node=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryNode=1000M MinTmpDiskNode=0
   Features=(null) Gres=gpu:2 Reservation=(null)
  
### 2 CPUs, 2 GPUs
srun -p gpu -n 2 --pty --mem 1000 --gres gpu:2 -t 500 /bin/bash
scontrol show job 85362472
JobId=85362472 JobName=bash
   NumNodes=1 NumCPUs=2 NumTasks=2 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=2,mem=1000M,node=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryNode=1000M MinTmpDiskNode=0
   Features=(null) Gres=gpu:2 Reservation=(null)
