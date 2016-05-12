<img src="http://i.imgur.com/WMFG0fo.png">

[![Build Status](https://img.shields.io/travis/CellProfiler/CellProfiler/master.svg)](https://travis-ci.org/CellProfiler/CellProfiler) [![Build Status](https://img.shields.io/pypi/v/cellprofiler.svg)](https://pypi.python.org/pypi/cellprofiler) [![Build Status](https://img.shields.io/pypi/dm/cellprofiler.svg)](https://pypi.python.org/pypi/cellprofiler)

**CellProfiler** is a free open-source software designed to enable biologists without training in computer vision or programming to quantitatively measure phenotypes from thousands of images automatically.

## Installation

Compiled releases for Linux, OS X, and Windows are available from [cellprofiler.org/releases](http://cellprofiler.org/releases/).

### Install from the Python Package Index (PyPI)

*CellProfiler requires Python 2.*

```sh
$ pip install --editable git+git@github.com:CellProfiler/CellProfiler.git#egg=cellprofiler
```

### Install from Source

####  Prerequisites

[NumPy](http://www.numpy.org/) is the fundamental package for scientific computing with Python:

```sh
$ pip install numpy
```

#### Installation

```sh
$ pip install --editable .
```
