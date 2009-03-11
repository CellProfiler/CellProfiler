"""rankorder.py - convert an image of any type to an image of ints whose
pixels have an identical rank order compared to the original image

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__ = "$Revision$"
import numpy

def rank_order(image):
    """Return an image of the same shape where each pixel has the
    rank-order value of the corresponding pixel in the image.
    The returned image's elements are of type numpy.uint32 which
    simplifies processing in C code.
    """
    flat_image = image.flatten()
    sort_order = flat_image.argsort()
    sort_rank  = numpy.zeros(sort_order.shape,numpy.uint32)
    sort_rank[1:] = numpy.cumsum(flat_image[sort_order[:-1]] != 
                                 flat_image[sort_order[1:]])
    int_image = numpy.zeros(sort_order.shape,numpy.uint32)
    int_image[sort_order] = sort_rank
    return int_image.reshape(image.shape)
    
    