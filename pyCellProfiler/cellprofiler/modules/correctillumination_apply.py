"""correctillumination_apply.py - apply image correction

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import numpy as np

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage  as cpi

DOS_DIVIDE = "Divide"
DOS_SUBTRACT = "Subtract"
RE_NONE = "No rescaling"
RE_STRETCH = "Stretch 0 to 1"
RE_MATCH = "Match maximums"

class CorrectIllumination_Apply(cpm.CPModule):
    """Help for the Correct Illumination Apply module:
Category: Image Processing

SHORT DESCRIPTION:
Applies an illumination function, created by
CorrectIllumination_Calculate, to an image in order to correct for uneven
illumination (uneven shading).
*************************************************************************

This module corrects for uneven illumination of each image. An
illumination function image that represents the variation in
illumination across the field of view is either made by a previous
module or loaded by a previous module in the pipeline.  This module
then applies the illumination function to each image coming through
the pipeline to produce the corrected image.

Settings:

Divide or Subtract:
This module either divides each image by the illumination function,
or the illumination function is subtracted from each image. The
choice depends on how the illumination function was calculated and
on your physical model of how illumination variation affects the
background of images relative to the objects in images. If the
background is significant relative to the real signal coming from
cells (a somewhat empirical decision), then the Subtract option may be
preferable. If, in contrast, the signal to background ratio is quite
high (the cells are stained strongly), then the Divide option is
probably preferable. Typically, Subtract is used if the illumination
function was calculated using the background option in the
CORRECTILLUMINATION_CALCULATE module and divide is used if the
illumination function was calculated using the regular option.

Rescaling:
If subtracting the illumination function, any pixels that end up
negative are set to zero, so no rescaling of the corrected image is
necessary. If dividing, the resulting corrected image may be in a
very different range of intensity values relative to the original,
depending on the values of the illumination function. If you are not
rescaling, you should confirm that the illumination function is in a
reasonable range (e.g. 1 to some number), so that the resulting
image is in a reasonable range (0 to 1). Otherwise, you have two
options to rescale the resulting image: either stretch the image
so that the minimum is zero and the maximum is one, or match the
maximum of the corrected image to the the maximum of the original.
Either of these options has the potential to disturb the brightness
of images relative to other images in the set, so caution should be
used in interpreting intensity measurements from images that have
been rescaled. See the help for the Rescale Intensity module for details.

See also CorrectIllumination_Calculate, RescaleIntensity.
"""
    category = "Image Processing"
    variable_revision_number = 1
    
    def create_settings(self):
        """Make settings here (and set the module name)"""
        self.module_name = "CorrectIllumination_Apply"
        self.image_name = cps.ImageNameSubscriber("What did you call the image to be corrected?","None")
        self.corrected_image_name = cps.ImageNameProvider("What do you want to call the corrected image?","CorrBlue")
        self.illum_correct_function_image_name = cps.ImageNameSubscriber("What did you call the illumination correction function image to be used to carry out the correction (produced by another module or loaded as a .mat format image using Load Single Image)?","None")
        self.divide_or_subtract = cps.Choice("How do you want to apply the illumination correction function?",
                                             [DOS_DIVIDE, DOS_SUBTRACT])
        self.rescale_option = cps.Choice("Choose rescaling method",
                                         [RE_NONE, RE_STRETCH, RE_MATCH])

    def settings(self):
        """Return the settings in the file load/save order"""
        return [self.image_name, self.corrected_image_name,
                self.illum_correct_function_image_name, self.divide_or_subtract,
                self.rescale_option]

    def backwards_compatibilize(self, setting_values, variable_revision_number, 
                                module_name, from_matlab):
        """Adjust settings based on revision # of save file
        
        setting_values - sequence of string values as they appear in the
                         saved pipeline
        variable_revision_number - the variable revision number of the module
                                   at the time of saving
        module_name - the name of the module that did the saving
        from_matlab - True if saved in CP Matlab, False if saved in pyCP
        
        returns the updated setting_values, revision # and matlab flag
        """
        # No SVN records of revisions 1 & 2
        if from_matlab and variable_revision_number == 3:
            # Same order as pyCP
            from_matlab = False
            variable_revision_number = 1
        return setting_values, variable_revision_number, from_matlab

    def visible_settings(self):
        """Return the list of displayed settings
        
        Only display the rescale option when dividing
        """
        result = [self.image_name, self.corrected_image_name,
                  self.illum_correct_function_image_name, 
                  self.divide_or_subtract]
        if self.divide_or_subtract == DOS_DIVIDE:
            result.append(self.rescale_option)
        return result

    def run(self, workspace):
        orig_image     = workspace.image_set.get_image(self.image_name.value,
                                                       must_be_grayscale=True)
        illum_function = workspace.image_set.get_image(self.illum_correct_function_image_name.value,
                                                       must_be_grayscale=True)
        
        if self.divide_or_subtract == DOS_DIVIDE:
            output_pixels = orig_image.pixel_data / illum_function.pixel_data
            output_pixels = self.rescale(output_pixels,
                                         orig_image.pixel_data)
        elif self.divide_or_subtract == DOS_SUBTRACT:
            output_pixels = orig_image.pixel_data - illum_function.pixel_data
            output_pixels[output_pixels < 0] = 0
        else:
            raise ValueError("Unhandled option for divide or subtract: %s"%(self.divide_or_subtract.value))
        
        if workspace.frame != None:
            self.display(workspace,
                         orig_image.pixel_data,
                         illum_function.pixel_data,
                         output_pixels)
        output_image = cpi.Image(output_pixels, parent_image = orig_image) 
        workspace.image_set.add(self.corrected_image_name.value,
                                output_image)
    
    def rescale(self, pixel_data, orig_pixel_data):
        """Rescale according to the rescale option setting"""
        if self.rescale_option == RE_NONE:
            return pixel_data
        elif self.rescale_option == RE_STRETCH:
            pmin = pixel_data.min()
            pmax = pixel_data.max()
            if pmin==pmax:
                return np.ones(pixel_data.shape)
            return (pixel_data-pmin)/(pmax-pmin)
        elif self.rescale_option == RE_MATCH:
            pmax = pixel_data.max()
            omax = orig_pixel_data.max()
            if pmax == 0:
                return np.ones(orig_pixel_data.shape) * omax
            else:
                return pixel_data * omax /pmax
        else:
            raise ValueError("Unhandled option for rescaling: %s"%(self.rescale_option))

    def display(self, workspace, orig_pixels, illum_pixels, output_pixels):
        figure = workspace.create_or_find_figure(subplots=(2,2))
        figure.subplot_imshow_grayscale(0, 0, orig_pixels,
                                        "Original image: %s"%(self.image_name.value))
        figure.subplot_imshow_grayscale(0, 1, illum_pixels,
                                        "Illumination function: %s"%(self.illum_correct_function_image_name.value))
        figure.subplot_imshow_grayscale(1, 0, output_pixels,
                                        "Final image: %s"%(self.corrected_image_name.value))
        statistics = [ ["Min value", round(output_pixels.min(),4)],
                      [ "Max value", round(output_pixels.max(),4)]]
        figure.subplot_table(1, 1, statistics, ratio=(.6,.4))
