'''<b>ImageConvexHull</b> finds the convex hull of a binary image
<hr>
Useful for defining edges (or corners) of wells after thresholding when
image objecst of interest remain.
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

__version__="$Revision: 8566 $"

import numpy as np
import scipy.ndimage as scind

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage as cpi

from cellprofiler.cpmath.cpmorphology import convex_hull_image

class ImageConvexHull(cpm.CPModule):
    
    module_name = 'ImageConvexHull'
    category = "Image Processing"
    variable_revision_number = 1
     
    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber('Select the input image (binary)','None')
        self.convex_hull_image_name = cps.ImageNameProvider('Name the output image','ConvexHull')


    def settings(self):
        return [self.image_name, self.convex_hull_image_name]

    def visible_settings(self):
        result = [self.image_name,  self.convex_hull_image_name]
        return result

    def run(self, workspace):
        image = workspace.image_set.get_image(self.image_name.value,
                                              must_be_binary=True)
       # output_image = ~covex_hull_image(~image)

        pixel_data = image.pixel_data
        output_pixels = convex_hull_image(~pixel_data) 
        
        output_image = cpi.Image(output_pixels, parent_image = image)
        workspace.image_set.add(self.convex_hull_image_name.value,
                                output_image)
        workspace.display_data.pixel_data = pixel_data
        workspace.display_data.output_pixels = output_pixels

    def is_interactive(self):
        return False

    def display(self, workspace):
        figure = workspace.create_or_find_figure(subplots=(1,2))
        figure.subplot_imshow_bw(0, 0, 
                                        workspace.display_data.pixel_data,
                                        "Original: %s" % 
                                        self.image_name.value)
        figure.subplot_imshow_bw(0, 1, 
                                        workspace.display_data.output_pixels,
                                        "Convex Hull: %s" % 
                                        self.image_name.value)
    
    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
        return setting_values, variable_revision_number, from_matlab

