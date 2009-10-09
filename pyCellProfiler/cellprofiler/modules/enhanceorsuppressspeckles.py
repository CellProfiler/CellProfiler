'''<b>Enhance Or Suppress Speckles</b> enhances or suppresses the contrast of 
speckle pixels with respect to the rest of the image
<hr>
This module enhances or suppresses the intensity of speckle pixels
using the white top-hat or opening morphological operations. Opening
suppresses speckles. It applies a grayscale erosion to reduce everything
within a given radius to the lowest value within that radius, then uses
a grayscale dilation to restore objects larger than the radius to an
approximation of their former shape. The white top-hat filter enhances 
speckles by subtracting the effects of opening from the original image.

Do you want to enhance or suppress speckles?
Choose "Enhance" to get an image whose intensity is largely composed of
the speckles. Choose "Suppress" to get an image with the speckles
removed.

What is the size of the speckles?


'''
#CellProfiler is distributed under the GNU General Public License.
#See the accompanying file LICENSE for details.
#
#Developed by the Broad Institute
#Copyright 2003-2009
#
#Please see the AUTHORS file for credits.
#
#Website: http://www.cellprofiler.org

__version__="$Revision$"

import numpy as np

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps
from cellprofiler.cpmath.cpmorphology import opening, white_tophat

ENHANCE = 'Enhance'
SUPPRESS = 'Suppress'

class EnhanceOrSuppressSpeckles(cpm.CPModule):

    category = "Image Processing"
    variable_revision_number = 1
    
    def create_settings(self):
        self.module_name = 'EnhanceOrSuppressSpeckles'
        self.image_name = cps.ImageNameSubscriber('Select the input image',
                                                  'None')
        self.filtered_image_name = cps.ImageNameProvider('Name the output image',
                                                         'FilteredBlue')
        self.method = cps.Choice('Do you want to enhance or suppress speckles?',
                                 [ ENHANCE, SUPPRESS],doc="""
            Choose <i>Enhance</i> to get an image whose intensity is largely composed of
            the speckles. Choose <i>Suppress</i> to get an image with the speckles
            removed.""")
        self.object_size = cps.Integer('What is the speckle size?',
                                       10,1,doc="""
            This is the diameter of the largest speckle to be enhanced or suppressed.""")

    def settings(self):
        return [ self.image_name, self.filtered_image_name,
                self.method, self.object_size]


    def run(self, workspace):
        image = workspace.image_set.get_image(self.image_name.value,
                                              must_be_grayscale = True)
        radius = int((self.object_size.value+1) / 2)
        if self.method == ENHANCE:
            if image.has_mask:
                result = white_tophat(image.pixel_data, radius, image.mask)
            else:
                result = white_tophat(image.pixel_data, radius)
        elif self.method == SUPPRESS:
            if image.has_mask:
                result = opening(image.pixel_data, radius, image.mask)
            else:
                result = opening(image.pixel_data, radius)
        else:
            raise ValueError("Unknown filtering method: %s"%self.method)
        result_image = cpi.Image(result, parent_image=image)
        workspace.image_set.add(self.filtered_image_name.value, result_image)
        
        if not workspace.frame is None:
            figure = workspace.create_or_find_figure(subplots=(1,2))
            figure.subplot_imshow_grayscale(0, 0, image.pixel_data,
                                            "Original: %s" % 
                                            self.image_name.value)
            figure.subplot_imshow_grayscale(0, 1, result,
                                            "Filtered: %s" %
                                            self.filtered_image_name.value)
