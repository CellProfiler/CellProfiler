# coding=utf-8

from setuptools import setup

setup(
    name='cellprofiler',
    version='2.1.2',
    description='…',
    url='http://cellprofiler.org',
    author='Broad Institute of MIT and Harvard, Imaging Platform',
    license='GPL-2.0',
    classifiers=[
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
    ],
    keywords='',
    packages=[
        'cellprofiler',
        'contrib',
        'imagej',
    ],
    install_requires=[
        'h5py',
        'javabridge',
        'lxml',
        'matplotlib',
        'matplotlib',
        'mysqldbda',
        'numpy',
        'pandas',
        'python-bioformats',
        'scikit-image',
        'scikit-learn',
        'scipy',
    ],
    package_data={
        'javabridge': [
            'jars/*.jar',
        ],
    },
    data_files={

    },
    entry_points={
        'console_scripts': [
            'CellProfiler=CellProfiler:main',
        ],
    }
)
