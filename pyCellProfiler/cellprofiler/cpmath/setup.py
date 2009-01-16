"""setup.py - setup to build C modules for cpmath

"""
__version__="$Revision$"

from distutils.core import setup,Extension
from numpy import get_include
import os

def configuration():
    extension = Extension(name="_cpmorphology",
                          sources=["src/cpmorphology.c"],
                          include_dirs=['src']+[get_include()],
                          extra_compile_args=['-O3'])
    dict = { "name":"cpmath",
             "description":"algorithms for CellProfiler",
             "maintainer":"Lee Kamentsky",
             "maintainer_email":"leek@broad.mit.edu",
             "ext_modules": [extension]
            }
    return dict

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    setup(**configuration())
    

