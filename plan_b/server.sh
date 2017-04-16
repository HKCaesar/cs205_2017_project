#!/bin/bash
srun -n 16 -p interact --mem 2000 -t 0-04:00 --mpi=pmi2 --pty /bin/bash

