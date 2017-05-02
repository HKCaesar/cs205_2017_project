
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy

### list all .pyx files

our_modules = [
    Extension("cython_kmeans", ["cython_kmeans.pyx", "../C/cython_kmeans.c"],
    language="c",
    extra_link_args=['-fopenmp'], include_dirs=[numpy.get_include()])
]


setup(name = 'cython_kmeans', ext_modules=cythonize(our_modules))
