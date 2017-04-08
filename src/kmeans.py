import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

import pandas as pd

##################
### CONFIGURE ####
##################

data_fn = "../data/reviewer-data-sample.csv"

##################
### SETUP ####
##################

# import data
data = pd.read_csv(data_fn)
print(data)
