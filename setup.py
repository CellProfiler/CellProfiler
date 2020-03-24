import setuptools

setuptools.setup(
    author="Allen Goodman",
    author_email="allen.goodman@icloud.com",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    extras_require={
        "dev": ["black==19.3b0", "pre-commit==1.18.3"],
        "test": ["pytest==5.1.3"],
        "wx": ["wxPython==4.0.6"],
    },
    install_requires=[
        "boto3==1.9.233",
        "centrosome==1.1.6",
        "docutils==0.15.2",
        "h5py==2.10.0",
        "javabridge@https://github.com/CellProfiler/python-javabridge/tarball/master",
        "matplotlib==3.1.1",
        "numpy==1.17.2",
        "prokaryote==2.4.1",
        "python-bioformats==1.5.2",
        "scikit-image==0.15.0",
        "scipy==1.3.1",
    ],
    license="BSD",
    name="cellprofiler-core",
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3.7, <4",
    url="https://github.com/CellProfiler/core",
    version="4.0.0rc1"
)
