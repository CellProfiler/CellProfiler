import os
import setuptools

version_file = open(os.path.join(os.path.dirname(__file__), "cellprofiler", "VERSION"))
version = version_file.read().strip()

setuptools.setup(
    author="cellprofiler-dev",
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
    entry_points={
        "console_scripts": [
            "cellprofiler=cellprofiler.__main__:main"
        ]
    },
    extras_require={
        "application": [

        ],
        "test": [
            "pytest"
        ]
    },
    include_package_data=True,
    install_requires=[
        "cellh5",
        "centrosome",
        "h5py",
        "inflect",
        "javabridge",
        "libtiff",
        "mahotas",
        "matplotlib<2.0.0",
        "MySQL-python",
        "numpy",
        "prokaryote>=1.0.11",
        "pyamg==3.1.1",
        "pytest",
        "python-bioformats",
        "pyzmq",
        "scikit-image",
        "scipy"
    ],
    license="BSD",
    name="CellProfiler",
    package_data={
        "cellprofiler": [
            "data/*.png"
        ],
    },
    packages=setuptools.find_packages(
        exclude=[
            "tests"
        ]
    ),
    url="https://github.com/CellProfiler/CellProfiler",
    version="3.0.0rc1"
)
