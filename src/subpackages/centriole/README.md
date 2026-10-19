# centriole

This subpackage contains the code to build the `centriole` package, an open source image processing library. It provides the low-level numerical and morphological routines — thresholding, propagation, Zernike moments, Haralick textures, convex hulls, the Jonker-Volgenant assignment solver, and more — that the [library](https://github.com/CellProfiler/CellProfiler/tree/main/src/subpackages/library), [core](https://github.com/CellProfiler/CellProfiler/tree/main/src/subpackages/core), and [frontend](https://github.com/CellProfiler/CellProfiler/tree/main/src/frontend) subpackages build on. It can be used in isolation, separate from the `cellprofiler` packages.

Several modules are Cython/C++ extensions, so this subpackage requires a compiler to build.

It was previously developed as a standalone package named `centrosome` at https://github.com/CellProfiler/centrosome, whose history is preserved here. Releases made before the move into this monorepo are tagged `centrosome/<version>`.
