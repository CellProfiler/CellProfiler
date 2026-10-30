"""Extension module declarations for centriole.

All package metadata lives in pyproject.toml. This file exists only because
setuptools still needs a setup.py to declare C/Cython extension modules.
"""

import glob
import os.path
import sys

import numpy
import setuptools
from setuptools.command.build_ext import build_ext


class BuildExt(build_ext):
    def build_extensions(self):
        if sys.platform == "darwin":
            prefix_flags = {
                flag
                for prefix in (sys.prefix, sys.base_prefix)
                for flag in (
                    "-L" + os.path.join(prefix, "lib"),
                    "-Wl,-rpath," + os.path.join(prefix, "lib"),
                )
            }
            for name in ("linker_so", "linker_so_cxx"):
                command = getattr(self.compiler, name, None)
                if command is not None:
                    self.compiler.set_executable(
                        name, [arg for arg in command if arg not in prefix_flags]
                    )
        super().build_extensions()


try:
    import Cython.Build

    __cython = True
except ImportError:
    __cython = False


if __cython:
    __suffix = "pyx"
    __extkwargs = {
        "language": "c++",
        # TODO: needed for cython 3.0
        # "define_macros": [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    }
else:
    __suffix = "cpp"
    __extkwargs = {}

__extensions = [
    setuptools.Extension(
        name="centriole._propagate",
        sources=[
            "centriole/_propagate.{}".format("c" if __suffix == "cpp" else __suffix)
        ],
        # TODO: needed for cython 3.0
        # define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")] if __suffix == "pyx" else None,
        include_dirs=["centriole/include", numpy.get_include()],
    )
]

for pyxfile in glob.glob(os.path.join("centriole", "*.pyx")):
    name = os.path.splitext(os.path.basename(pyxfile))[0]

    if name == "_propagate":
        continue

    __extensions += [
        setuptools.Extension(
            name="centriole.{}".format(name),
            sources=["centriole/{}.{}".format(name, __suffix)],
            include_dirs=["centriole/include", numpy.get_include()],
            **__extkwargs
        )
    ]

if __suffix == "pyx":
    __extensions = Cython.Build.cythonize(
        __extensions, compiler_directives={"language_level": "3"}
    )

setuptools.setup(ext_modules=__extensions, cmdclass={"build_ext": BuildExt})
