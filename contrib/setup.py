from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from numpy import get_include
import sys
import os

def configuration():
    extension = [Extension("LAP", ["mexLap.c", "LAP.pyx"],include_dirs=[get_include()])]
    cmdclass = {'build_ext': build_ext}
    d = dict(name="contrib",
             description="Externally developed code",
             maintainer="Lee Kamentsky",
             maintainer_email="leek@broadinstitute.org",
             cmdclass=cmdclass,
             ext_modules=extension)
    if (sys.platform.startswith("win") and
        os.environ["PROCESSOR_ARCHITECTURE"] == "AMD64"):
        d["include_dirs"] = ["include"]
    return d

if __name__ == '__main__':
    if '/' in __file__:
        os.chdir(os.path.dirname(__file__))
    setup(**configuration())

