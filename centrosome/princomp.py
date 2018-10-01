from __future__ import absolute_import
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
    
    
