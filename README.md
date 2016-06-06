<img src="http://i.imgur.com/WMFG0fo.png">

[![Build Status](https://img.shields.io/travis/CellProfiler/CellProfiler/master.svg)](https://travis-ci.org/CellProfiler/CellProfiler) [![Build Status](https://img.shields.io/pypi/v/cellprofiler.svg)](https://pypi.python.org/pypi/cellprofiler) [![Build Status](https://img.shields.io/pypi/dm/cellprofiler.svg)](https://pypi.python.org/pypi/cellprofiler)

**CellProfiler** is a free open-source software designed to enable biologists without training in computer vision or programming to quantitatively measure phenotypes from thousands of images automatically. More information can be found in the [CellProfiler Wiki](https://github.com/CellProfiler/CellProfiler/wiki).

## Binary installation

Compiled releases for CentOS 6, OS X, and Windows are available from [cellprofiler.org/releases](http://cellprofiler.org/releases/).

## Source installation

CellProfiler requires Python 2 and [NumPy](http://www.numpy.org/), the fundamental package for scientific computing with Python. More information about installation from source can be found in the wiki, i.e. for [Linux](https://github.com/CellProfiler/CellProfiler/wiki/Source-installation-(Linux)), [OS X](https://github.com/CellProfiler/CellProfiler/wiki/Source-installation-(OS-X)) and [Windows PC](https://github.com/CellProfiler/CellProfiler/wiki/Source-installation-(PC)).

### Install from the Python Package Index (PyPI)

```sh
$ pip install --editable git+git@github.com:CellProfiler/CellProfiler.git#egg=cellprofiler
```

### Install manually

```sh
$ pip install numpy
$ git clone https://github.com/CellProfiler/CellProfiler.git
$ cd CellProfiler
$ pip install --editable .
```
