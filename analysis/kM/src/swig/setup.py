from distutils.core import setup, Extension
import numpy
import os

os.environ['CC'] = 'gcc';
setup(name='kmeans', version='1.0', ext_modules =[Extension('_kmeans',
 ['../kmeans.c', 'kmeans.i'], include_dirs = [numpy.get_include(),'.'], extra_compile_args = ["-std=c11"])])
