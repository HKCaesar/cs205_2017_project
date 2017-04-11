from distutils.core import setup, Extension
import numpy
import os

os.environ['CC'] = 'gcc';
setup(name='kM', version='1.0', ext_modules =[Extension('_kM',
 ['src/kmeans.c', 'kM.i'], include_dirs = [numpy.get_include(),'.'], extra_compile_args = ["-std=c11"])])
