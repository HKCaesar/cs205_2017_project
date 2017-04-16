import os
import glob

time_file_list = glo

for filename in glob.glob("*.slurm"):
    cmd = "sbatch %s" % filename
    os.system(cmd)
