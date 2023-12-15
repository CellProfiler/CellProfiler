import setuptools

setuptools.setup(
    author="Allen Goodman",
    author_email="allen.goodman@icloud.com",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
    extras_require={
        "dev": [
            "click>=7.1.2",
            "sphinx>=3.1.2",
            "twine==3.1.1",
        ],
        "test": ["pytest~=7.4.1", "pytest-timeout~=2.1.0"],
        "wx": ["wxPython==4.2.0"],
    },
    install_requires=[
        "cellprofiler-library>=5.dev",
        "boto3>=1.12.28",
        "centrosome~=1.2.2",
        "docutils==0.15.2",
        "future>=0.18.2",
        "fsspec>=2021.11.0",
        "h5py~=3.6.0",
        "lxml>=4.6.4",
        "matplotlib~=3.1.3",
        "numpy~=1.24.4",
        "psutil>=5.9.5",
        "pyzmq~=22.3.0",
        "scikit-image~=0.20.0",
        "scipy>=1.9.1,<1.11",
        "scyjava>=1.9.1",
        "zarr~=2.16.1",
        "google-cloud-storage~=2.10.0",
    ],
    license="BSD",
    name="cellprofiler-core",
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3.8",
    url="https://github.com/CellProfiler",
    version="5.0.0.dev"
)
