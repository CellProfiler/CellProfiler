'''<b>CorrectIllumination_Apply:</b> Applies an illumination function, created by
CorrectIllumination_Calculate, to an image in order to correct for uneven
illumination (uneven shading).
<hr>

This module applies a previously created illumination correction function,
either loaded by <b>LoadSingleImage</b> or created by <b>CorrectIllumination_Calculate</b>.
This module corrects each image in the pipeline using the function you specify. 

See also <b>CorrectIllumination_Calculate, RescaleIntensity</b>.'''

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
import cellprofiler.settings as cps
import cellprofiler.cpimage  as cpi

DOS_DIVIDE = "Divide"
DOS_SUBTRACT = "Subtract"
RE_NONE = "No rescaling"
RE_STRETCH = "Stretch 0 to 1"
RE_MATCH = "Match maximums"

class CorrectIllumination_Apply(cpm.CPModule):

    category = "Image Processing"
    variable_revision_number = 1
    
    def create_settings(self):
        """Make settings here (and set the module name)"""
        self.module_name = "CorrectIllumination_Apply"
        self.image_name = cps.ImageNameSubscriber("What did you call the image to be corrected?","None")
        self.corrected_image_name = cps.ImageNameProvider("What do you want to call the corrected image?","CorrBlue")
        self.illum_correct_function_image_name = cps.ImageNameSubscriber("What did you call the illumination correction function image to be used to carry out the correction (produced by another module or loaded as a .mat format image using Load Single Image)?","None")
        self.divide_or_subtract = cps.Choice("How do you want to apply the illumination correction function?",
                                             [DOS_DIVIDE, DOS_SUBTRACT], doc = '''This choice depends on how the illumination function was calculated
                                             and on your physical model of how illumination variation affects the background of images relative to 
                                             the objects in images. <ul><li>Subtract: Use <i>Subtract</i> if the background signal is significant relative to the real signal
                                             coming from the cells (a somewhat empirical decision).  If you created the illumination correction function using <i>Background</i>,
                                             then you will want to choose <i>Subtract</i> here.</li><li>Divide: Use <i>Divide</i> if the the signal to background ratio 
                                             is quite high (the cells are stained very strongly).  If you created the illumination correction function using <i>Regular</i>,
                                             then you will want to choose <i>Divide</i> here.</ul>''')
        self.rescale_option = cps.Choice("Choose rescaling method",
                                         [RE_NONE, RE_STRETCH, RE_MATCH], doc = '''<ul><li>Subtract: Any pixels that end up negative are set to zero, so no rescaling is necessary.
                                         <li>Divide: The resulting image may be in a very different range of intensity values relative to the original image.
                                         If the illumination correction function is in the range 1 to infinity, <i>Divide</i> will usually yield an image in a reasonable
                                         range (0 to 1).  However, if the image is not in this range, or the intensity gradient within the image is still very great,
                                         you may want to rescale the image.  There are two:<ul><li>Stretch the image from 0 to 1.<li>Match the maximum of the corrected image
                                         to the maximum of the original image.</ul></ul>''')

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
        statistics = [ ["Min value", round(illum_pixels.min(),4)],
                      [ "Max value", round(illum_pixels.max(),4)]]
        figure.subplot_table(1, 1, statistics, ratio=(.6,.4))
