from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("LAP", ["mexLap.c", "LAP.pyx"],include_dirs=[get_include()])]
)
