"""princomp.py - find the principal components of a vector of measurements.

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org

"""
__version__="$Revision$"

import numpy

def princomp(x):
    """Determine the principal components of a vector of measurements
    
    Determine the principal components of a vector of measurements
    x should be a M x N numpy array composed of M observations of n variables
    The output is:
    coeffs - the NxN correlation matrix that can be used to transform x into its components
    
    The code for this function is based on "A Tutorial on Principal Component
    Analysis", Shlens, 2005 http://www.snl.salk.edu/~shlens/pub/notes/pca.pdf
    (unpublished)
    """
    
    (M,N)  = x.shape
    Mean   = x.mean(0)
    y      = x - Mean
    cov    = numpy.dot(y.transpose(),y) / (M-1)
    (V,PC) = numpy.linalg.eig(cov)
    order  = (-V).argsort()
    coeff  = PC[:,order]
    return coeff
    
    
