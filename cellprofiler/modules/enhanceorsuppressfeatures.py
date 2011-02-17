'''<b>Enhance Or Suppress Features</b> enhances or suppresses certain image features (such as speckles, ring shapes, and neurites), which can improve subsequent identification of objects
<hr>

This module enhances or suppresses the intensity of certain pixels relative
to the rest of the image, by applying image processing filters to the image. It produces a grayscale image in which objects can be identified using an <b>Identify</b> module.
'''

# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

__version__="$Revision$"

import numpy as np

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps
from cellprofiler.cpmath.cpmorphology import opening, closing, white_tophat
from cellprofiler.cpmath.filter import enhance_dark_holes, circular_hough
from cellprofiler.gui.help import HELP_ON_PIXEL_INTENSITIES

ENHANCE = 'Enhance'
SUPPRESS = 'Suppress'

E_SPECKLES = 'Speckles'
E_NEURITES = 'Neurites'
E_DARK_HOLES = 'Dark holes'
E_CIRCLES = 'Circles'

class EnhanceOrSuppressFeatures(cpm.CPModule):

    module_name = 'EnhanceOrSuppressFeatures'
    category = "Image Processing"
    variable_revision_number = 2
    
    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber('Select the input image',
                                                'None',doc="""
            What did you call the image with features to be enhanced or suppressed?""")
        
        self.filtered_image_name = cps.ImageNameProvider('Name the output image',
                                        'FilteredBlue',doc="""
                                        What do you want to call the feature-enhanced or suppressed image?""")
        
        self.method = cps.Choice('Select the operation',
                                        [ ENHANCE, SUPPRESS],doc="""
                                        Do you want to enhance or suppress the features you designated?
                                        <ul><li><i>Enhance</i> produces an image whose intensity is largely composed of
                                        the features of interest. <li>Choose <i>Suppress</i> to produce an image with the features largely
                                        removed.</li></ul>""")
        
        self.enhance_method = cps.Choice('Feature type',
                                        [E_SPECKLES, E_NEURITES, E_DARK_HOLES, E_CIRCLES],
                                        doc="""<i>(Used only if Enhance is selected)</i><br>
                                        This module can enhance three kinds of image intensity features:
                                        <ul><li><i>Speckles</i>: A speckle is an area of enhanced intensity
                                        relative to its immediate neighborhood. The module enhances
                                        speckles using a white tophat filter, which is the image minus the
                                        morphological grayscale opening of the image. The opening operation
                                        first suppresses the speckles by applying a grayscale erosion to reduce everything
                                        within a given radius to the lowest value within that radius, then uses
                                        a grayscale dilation to restore objects larger than the radius to an
                                        approximation of their former shape. The white tophat filter enhances 
                                        speckles by subtracting the effects of opening from the original image.
                                        </li>
                                        <li><i>Neurites</i>: Neurites are taken to be long, thin features
                                        of enhanced intensity. The module takes the difference of the
                                        white and black tophat filters (a black tophat filter is the 
                                        morphological grayscale opening of the image minus the image itself). 
                                        The effect is to enhance lines whose width is the "feature size".</li>
                                        <li><i>Dark holes</i>: The module uses morphological reconstruction 
                                        (the rolling-ball algorithm) to identify dark holes within brighter
                                        areas, or brighter ring shapes. The image is inverted so that the dark holes turn into
                                        bright peaks. The image is successively eroded and the eroded image
                                        is reconstructed at each step, resulting in an image which is
                                        missing the peaks. Finally, the reconstructed image is subtracted
                                        from the previous reconstructed image. This leaves circular bright
                                        spots with a radius equal to the number of iterations performed.
                                        </li>
                                        <li><i>Circles</i>: The module calculates the circular Hough transform of
                                        the image at the diameter given by the feature size. The Hough transform
                                        will have the highest intensity at points that are centered within a ring
                                        of high intensity pixels where the ring diameter is the feature size. You
                                        may want to use the <b>EnhanceEdges</b> module to find the edges of your
                                        circular object and then process the output by enhancing circles. You can
                                        use <b>IdentifyPrimaryObjects</b> to find the circle centers and then use
                                        these centers as seeds in <b>IdentifySecondaryObjects</b> to find whole,
                                        circular objects using a watershed.</li>
                                        </ul>
                                        In addition, this module enables you to suppress certain features (such as speckles)
                                        by specifying the feature size.""")
        
        self.object_size = cps.Integer('Feature size',
                                        10,2,doc="""
                                        <i>(Used only if circles, speckles or neurites are selected, or if suppressing features)</i><br>
                                        What is the feature size? 
                                        The diameter of the largest speckle, the width of the circle, or the width of the neurites to be enhanced or suppressed, which
                                        will be used to calculate an adequate filter size. %(HELP_ON_PIXEL_INTENSITIES)s"""%globals())
        
        self.hole_size = cps.IntegerRange('Range of hole sizes',
                                        value=(1,10),minval=1, doc="""
                                        <i>(Used only if dark hole detection is selected)</i><br>
                                        The range of hole sizes to be enhanced. The algorithm will
                                        identify only holes whose diameters fall between these two 
                                        values""")

    def settings(self):
        return [ self.image_name, self.filtered_image_name,
                self.method, self.object_size, self.enhance_method,
                self.hole_size]


    def visible_settings(self):
        result = [self.image_name, self.filtered_image_name,
                  self.method]
        if self.method == ENHANCE:
            result += [self.enhance_method]
            result += [self.hole_size if self.enhance_method == E_DARK_HOLES
                       else self.object_size]
        else:
            result += [self.object_size]
        return result
            
    def run(self, workspace):
        image = workspace.image_set.get_image(self.image_name.value,
                                              must_be_grayscale = True)
        #
        # Match against Matlab's strel('disk') operation.
        #
        radius = (float(self.object_size.value)-1.0) / 2.0
        mask = image.mask if image.has_mask else None
        pixel_data = image.pixel_data
        if self.method == ENHANCE:
            if self.enhance_method == E_SPECKLES:
                result = white_tophat(pixel_data, radius, mask)
            elif self.enhance_method == E_NEURITES:
                #
                # white_tophat = img - opening
                # black_tophat = closing - img
                # desired effect = img + white_tophat - black_tophat
                #                = img + img - opening - closing + img
                #                = 3*img - opening - closing
                result = (3 * pixel_data - 
                          opening(pixel_data, radius, mask) -
                          closing(pixel_data, radius, mask))
                result[result > 1] = 1
                result[result < 0] = 0
                if image.has_mask:
                    result[~mask] = pixel_data[~mask]
            elif self.enhance_method == E_DARK_HOLES:
                min_radius = max(1,int(self.hole_size.min / 2))
                max_radius = int((self.hole_size.max+1)/2)
                result = enhance_dark_holes(pixel_data, min_radius,
                                            max_radius, mask)
            elif self.enhance_method == E_CIRCLES:
                result = circular_hough(pixel_data, radius + .5, mask=mask)
            else:
                raise NotImplementedError("Unimplemented enhance method: %s"%
                                          self.enhance_method.value)
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
            figure = workspace.create_or_find_figure(title="EnhanceOrSuppressFeatures, image cycle #%d"%(
                workspace.measurements.image_set_number),subplots=(2,1))
            figure.subplot_imshow_grayscale(0, 0, image.pixel_data,
                                            "Original: %s" % 
                                            self.image_name.value)
            figure.subplot_imshow_grayscale(1, 0, result,
                                            "Filtered: %s" %
                                            self.filtered_image_name.value,
                                            sharex = figure.subplot(0,0),
                                            sharey = figure.subplot(0,0))
        
    def upgrade_settings(self, setting_values, variable_revision_number,
                             module_name, from_matlab):
        '''Adjust setting values if they came from a previous revision
        
        setting_values - a sequence of strings representing the settings
                         for the module as stored in the pipeline
        variable_revision_number - the variable revision number of the
                         module at the time the pipeline was saved. Use this
                         to determine how the incoming setting values map
                         to those of the current module version.
        module_name - the name of the module that did the saving. This can be
                      used to import the settings from another module if
                      that module was merged into the current module
        from_matlab - True if the settings came from a Matlab pipeline, False
                      if the settings are from a CellProfiler 2.0 pipeline.
        
        Overriding modules should return a tuple of setting_values,
        variable_revision_number and True if upgraded to CP 2.0, otherwise
        they should leave things as-is so that the caller can report
        an error.
        '''
        if not from_matlab and variable_revision_number == 1:
            #
            # V1 -> V2, added enhance method and hole size
            #
            setting_values = setting_values + [E_SPECKLES, "1,10"]
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab

EnhanceOrSuppressSpeckles = EnhanceOrSuppressFeatures
