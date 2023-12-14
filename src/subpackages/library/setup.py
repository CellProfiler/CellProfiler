import setuptools

setuptools.setup(
    author="Nodar Gogoberidze",
    author_email="ngogober@broadinstitute.org",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
    extras_require={
        "dev": ["pyinstaller", "setuptools", "wheel", "twine"],
        "test": ["pytest~=7.4.1"],
    },
    install_requires=[
        "numpy~=1.24.4",
        "scikit-image~=0.20.0",
        "scipy>=1.9.1,<1.11",
        "mahotas~=1.4.13",
        "centrosome~=1.2.2",
        "matplotlib~=3.1.3",
    ],
    license="BSD",
    name="cellprofiler-library",
    package_data={"cellprofiler_library": ["py.typed"]},
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3.8",
    url="https://github.com/CellProfiler/",
    version="5.0.0.dev1"
)
