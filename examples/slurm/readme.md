srun -p gpu -n 2 --pty --mem 1000 --gres gpu:1 -t 500 /bin/bash
scontrol show job 85362170

JobId=85362170 JobName=bash
   UserId=cs205u1715(10752) GroupId=cs205(402877) MCS_label=N/A
   Priority=19906065 Nice=0 Account=cs205 QOS=normal
   JobState=RUNNING Reason=None Dependency=(null)
   Requeue=1 Restarts=0 BatchFlag=0 Reboot=0 ExitCode=0:0
   RunTime=00:16:34 TimeLimit=08:20:00 TimeMin=N/A
   SubmitTime=2017-04-06T15:12:22 EligibleTime=2017-04-06T15:12:22
   StartTime=2017-04-06T15:12:22 EndTime=2017-04-06T23:32:23 Deadline=N/A
   PreemptTime=None SuspendTime=None SecsPreSuspend=0
   Partition=gpu AllocNode:Sid=rclogin09:19349
   ReqNodeList=(null) ExcNodeList=(null)
   NodeList=supermicgpu01
   BatchHost=supermicgpu01
   NumNodes=1 NumCPUs=2 NumTasks=2 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=2,mem=1000M,node=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryNode=1000M MinTmpDiskNode=0
   Features=(null) Gres=gpu:1 Reservation=(null)
   OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
   Command=/bin/bash
   WorkDir=/n/home02/cs205u1715
   Power=

srun -p gpu -n 2 --pty --mem 1000 --gres gpu:2 -t 500 /bin/bash
scontrol show job 85362472
JobId=85362472 JobName=bash
   UserId=cs205u1715(10752) GroupId=cs205(402877) MCS_label=N/A
   Priority=19889074 Nice=0 Account=cs205 QOS=normal
   JobState=RUNNING Reason=None Dependency=(null)
   Requeue=1 Restarts=0 BatchFlag=0 Reboot=0 ExitCode=0:0
   RunTime=00:01:21 TimeLimit=08:20:00 TimeMin=N/A
   SubmitTime=2017-04-06T15:32:16 EligibleTime=2017-04-06T15:32:16
   StartTime=2017-04-06T15:32:16 EndTime=2017-04-06T23:52:16 Deadline=N/A
   PreemptTime=None SuspendTime=None SecsPreSuspend=0
   Partition=gpu AllocNode:Sid=rclogin09:27094
   ReqNodeList=(null) ExcNodeList=(null)
   NodeList=supermicgpu01
   BatchHost=supermicgpu01
   NumNodes=1 NumCPUs=2 NumTasks=2 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=2,mem=1000M,node=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryNode=1000M MinTmpDiskNode=0
   Features=(null) Gres=gpu:2 Reservation=(null)
   OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
   Command=/bin/bash
   WorkDir=/n/home02/cs205u1715
   Power=

srun -p gpu -n 1 --pty --mem 1000 --gres gpu:2 -t 500 /bin/bash
scontrol show job 85362494

JobId=85362494 JobName=bash
   UserId=cs205u1715(10752) GroupId=cs205(402877) MCS_label=N/A
   Priority=19884775 Nice=0 Account=cs205 QOS=normal
   JobState=RUNNING Reason=None Dependency=(null)
   Requeue=1 Restarts=0 BatchFlag=0 Reboot=0 ExitCode=0:0
   RunTime=00:00:38 TimeLimit=08:20:00 TimeMin=N/A
   SubmitTime=2017-04-06T15:35:22 EligibleTime=2017-04-06T15:35:22
   StartTime=2017-04-06T15:35:25 EndTime=2017-04-06T23:55:31 Deadline=N/A
   PreemptTime=None SuspendTime=None SecsPreSuspend=0
   Partition=gpu AllocNode:Sid=rclogin09:27094
   ReqNodeList=(null) ExcNodeList=(null)
   NodeList=supermicgpu01
   BatchHost=supermicgpu01
   NumNodes=1 NumCPUs=2 NumTasks=1 CPUs/Task=1 ReqB:S:C:T=0:0:*:*
   TRES=cpu=2,mem=1000M,node=1
   Socks/Node=* NtasksPerN:B:S:C=0:0:*:* CoreSpec=*
   MinCPUsNode=1 MinMemoryNode=1000M MinTmpDiskNode=0
   Features=(null) Gres=gpu:2 Reservation=(null)
   OverSubscribe=OK Contiguous=0 Licenses=(null) Network=(null)
   Command=/bin/bash
   WorkDir=/n/home02/cs205u1715
   Power=
 