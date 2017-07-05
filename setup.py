import os

import setuptools

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
        url="http://cellprofiler.org",
        version="3.0.0rc2"
)
