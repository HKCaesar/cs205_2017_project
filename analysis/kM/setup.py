from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

### list all .pyx files

our_modules = [
    Extension("kmeansCPY", ["kmeansCPY.pyx", "src/kmeans.c"], language="c", extra_compile_args=['-O3'], extra_link_args=['-std=c11'])
]

### apparently equivalent ways to do setup

setup(name = 'kmeansCPY', ext_modules=cythonize(our_modules))
#setup(name = 'hw1', cmdclass = {'build_ext': build_ext}, ext_modules = our_modules)
