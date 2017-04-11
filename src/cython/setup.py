from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

### list all .pyx files

our_modules = [
    Extension("kmeansCPY", ["kmeansCPY.pyx", "../C/kmeans.c"], language="c", extra_compile_args=['-O3'], extra_link_args=['-std=c11'], include_dirs=[numpy.get_include()])
]

### apparently equivalent ways to do setup

setup(name = 'kmeansCPY', ext_modules=cythonize(our_modules))
#setup(name = 'hw1', cmdclass = {'build_ext': build_ext}, ext_modules = our_modules)
