import os
import sys

import setuptools.dist


class Test(setuptools.Command):
    user_options = [
        ("pytest-args=", "a", "arguments to pass to py.test")
    ]

    def initialize_options(self):
        self.pytest_args = []

    def finalize_options(self):
        pass

    def run(self):
        try:
            import pytest
            import unittest
        except ImportError:
            raise ImportError

        import cellprofiler.utilities.cpjvm

        #
        # Monkey-patch pytest.Function
        # See https://github.com/pytest-dev/pytest/issues/1169
        #
        try:
            from _pytest.unittest import TestCaseFunction

            def runtest(self):
                setattr(self._testcase, "__name__", self.name)
                self._testcase(result=self)

            TestCaseFunction.runtest = runtest
        except:
            pass

        import cellprofiler.preferences

        cellprofiler.preferences.set_headless()

        cellprofiler.utilities.cpjvm.cp_start_vm()

        errno = pytest.main(self.pytest_args)

        cellprofiler.__main__.stop_cellprofiler()

        sys.exit(errno)


setuptools.setup(
        author="CellProfiler contributors",
        author_email="cellprofiler-dev@broadinstitute.org",
        classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 2.7",
            "Topic :: Scientific/Engineering :: Bio-Informatics",
            "Topic :: Scientific/Engineering :: Image Recognition",
            "Topic :: Scientific/Engineering"
        ],
        cmdclass={
            "test": Test
        },
        entry_points={
            "console_scripts": [
                "cellprofiler=cellprofiler.__main__:main"
            ]
        },
        install_requires=[
            "centrosome",
            "h5py",
            "inflect",
            "javabridge",
            "joblib",
            "mahotas",
            "matplotlib",
            "MySQL-python",
            "numpy",
            "prokaryote",
            "python-bioformats",
            "pyzmq",
            "raven",
            "requests",
            "scikit-image",
            "scikit-learn",
            "scipy"
        ],
        license="BSD",
        name="CellProfiler",
        package_data={
            "images": os.path.join("data", "images", "*")
        },
        packages=setuptools.find_packages(exclude=[
            "tests",
        ]),
        python_requires=">=2.7, <3",
        setup_requires=[
            "pytest"
        ],
        url="http://cellprofiler.org",
        version="3.0.0rc2"
)
