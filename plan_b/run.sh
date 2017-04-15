#!/bin/bash
git pull
mpiexec -n 16 python test.py
