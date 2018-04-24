"""FastEMD - python wrapper of the FastEMD algorithm

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2015 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org

FastEMD is a library generously contributed to the open-source community
under a BSD license by it's author, Ofir Pele. Please see the c++ header files
for their copyright. The following papers should be used when citing this
library's implementation of the earth mover's distance:

A Linear Time Histogram Metric for Improved SIFT Matching
 Ofir Pele, Michael Werman
 ECCV 2008
bibTex:
@INPROCEEDINGS{Pele-eccv2008,
author = {Ofir Pele and Michael Werman},
title = {A Linear Time Histogram Metric for Improved SIFT Matching},
booktitle = {ECCV},
year = {2008}
}
 Fast and Robust Earth Mover's Distances
 Ofir Pele, Michael Werman
 ICCV 2009
@INPROCEEDINGS{Pele-iccv2009,
author = {Ofir Pele and Michael Werman},
title = {Fast and Robust Earth Mover's Distances},
booktitle = {ICCV},
year = {2009}
}
"""
from __future__ import absolute_import
from ._fastemd import EMD_NO_FLOW, EMD_WITHOUT_TRANSHIPMENT_FLOW,\
     EMD_WITHOUT_EXTRA_MASS_FLOW, emd_hat_int32
