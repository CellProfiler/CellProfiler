# imagej.imageprocessor.py - utilities for image processors
#
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
#
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
#
# Website: http://www.cellprofiler.org
#
import numpy as np
import bioformats
import cellprofiler.utilities.jutil as J

def get_image(imageprocessor_obj, do_scaling = False):
    '''Retrieve the image from an ImageProcessor
    
    Returns the image as a numpy float array.
    '''
    #
    # The strategy is:
    # * Make a TypeConverter
    # * Ask the TypeConverter for a float ImageProcessor
    # * Get the pixels - should be a float array
    #
    type_converter = J.make_instance(
        'ij/process/TypeConverter', '(Lij/process/ImageProcessor;Z)V',
        imageprocessor_obj, do_scaling)
    float_processor = J.call(
        type_converter, 'convertToFloat', '([F)Lij/process/ImageProcessor;',
        None)
    jpixels = J.call(
        float_processor, 'getPixels', '()Ljava/lang/Object;')
    pixels = J.get_env().get_float_array_elements(jpixels)
    height = J.call(imageprocessor_obj, 'getHeight', '()I')
    width = J.call(imageprocessor_obj, 'getWidth', '()I')
    pixels.shape = (height, width)
    return pixels

def make_image_processor(array):
    '''Create an image processor from the given image
    
    array - an array that will be cast to double. Values should be
            between 0 and 255
    '''
    return J.make_instance(
        'ij/process/FloatProcessor', '(II[D)V', 
        array.shape[1], array.shape[0], array)
