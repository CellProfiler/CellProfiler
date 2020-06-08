import setuptools

setuptools.setup(
    author="Allen Goodman",
    author_email="allen.goodman@icloud.com",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    extras_require={
        "dev": ["black==19.10b0", "pre-commit==2.2.0", "twine==3.1.1"],
        "test": ["pytest==5.4.1"],
        "wx": ["wxPython==4.0.7.post2"],
    },
    install_requires=[
        "boto3>=1.12.28",
        "centrosome==1.1.7",
        "docutils==0.15.2",
        "h5py>=2.10.0",
        "python-javabridge==4.0.0rc1",
        "matplotlib>=3.1.3",
        "numpy>=1.18.2",
        "prokaryote==2.4.1",
        "python-bioformats>=1.5.2",
        "scikit-image>=0.16.2",
        "scipy>=1.4.1",
    ],
    license="BSD",
    name="cellprofiler-core",
    packages=setuptools.find_packages(exclude=["tests"]),
    python_requires=">=3.8, <4",
    url="https://github.com/CellProfiler/core",
    version="4.0.0rc7",
)
