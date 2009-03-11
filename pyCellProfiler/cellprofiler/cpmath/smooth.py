"""smooth.py - smoothing of images

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import numpy

def smooth_with_noise(image, bits):
    """Smooth the image with a per-pixel random multiplier
    
    image - the image to perturb
    bits - the noise is this many bits below the pixel value
    
    The noise is random with normal distribution, so the individual pixels
    get either multiplied or divided by a normally distributed # of bits
    """
    
    numpy.random.seed(0)
    r = numpy.random.normal(size=image.shape)
    image_copy = image.copy()
    image_copy[image_copy==0]= pow(2.0,-bits)
    result = numpy.exp(numpy.log(image_copy)+ 0.5*r *
                       (-numpy.log2(image_copy)/bits))
    result[result>1] = 1
    result[result<0] = 0
    return result
    