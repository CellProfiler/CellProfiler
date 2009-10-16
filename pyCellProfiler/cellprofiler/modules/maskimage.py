"""<b>MaskImage</b>:
Masks an image and saves it for future use.
<hr>

This module masks an image and saves it in the handles structure for
future use. The masked image is based on the original image and the
object selected. 

Note that the image saved for further processing downstream is grayscale.
If a binary mask is desired in subsequent modules, you might be able to 
access the image's crop mask, or simply use the ApplyThreshold module
instead of MaskImage.

See also IdentifyPrimAutomatic, IdentifyPrimManual.

"""
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

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

class MaskImage(cpm.CPModule):

    module_name = "MaskImage"
    category = "Image Processing"
    variable_revision_number = 1
    
    def create_settings(self):
        """Create the settings here and set the module name (initialization)
        
        """
        self.object_name = cps.ObjectNameSubscriber("Select object for mask:","None",
                                                    doc = '''From which object would you like to make a mask?''')
        self.image_name = cps.ImageNameSubscriber("Select input image:","None", doc = '''Which image do you want to mask?''')
        self.masked_image_name = cps.ImageNameProvider("Name output image:",
                                                       "MaskBlue", doc = '''What do you want to call the masked image?''')
        self.invert_mask = cps.Binary("Do you want to invert the mask?",False)

    def settings(self):
        """Return the settings in the order that they will be saved or loaded
        
        Note: the settings are also the visible settings in this case, so
              they also control the display order. Implement visible_settings
              for a different display order.
        """
        return [self.object_name, self.image_name,
                self.masked_image_name, self.invert_mask]

    def run(self, workspace):
        objects = workspace.get_objects(self.object_name.value)
        labels = objects.segmented
        if self.invert_mask.value:
            mask = labels == 0
        else:
            mask = labels > 0
        orig_image = workspace.image_set.get_image(self.image_name.value,
                                                   must_be_grayscale = True)
        if orig_image.has_mask:
            mask = np.logical_and(mask, orig_image.mask)
        masked_pixels = orig_image.pixel_data.copy()
        masked_pixels[np.logical_not(mask)] = 0
        masked_image = cpi.Image(masked_pixels,mask=mask,
                                 parent_image = orig_image,
                                 masking_objects = objects)
        
        if workspace.frame:
            figure = workspace.create_or_find_figure(subplots=(2,1))
            figure.subplot_imshow_grayscale(0,0,orig_image.pixel_data,
                                            "Original image: %s"%(self.image_name.value))
            figure.subplot_imshow_grayscale(1,0,masked_pixels,
                                            "Masked image: %s"%(self.masked_image_name.value))
        workspace.image_set.add(self.masked_image_name.value, masked_image)
    
    def backwards_compatibilize(self, setting_values, 
                                variable_revision_number, 
                                module_name, from_matlab):
        """Adjust the setting_values to upgrade from a previous version
        
        """
        if from_matlab and variable_revision_number == 3:
            from_matlab = False
            variable_revision_number = 1
        return setting_values, variable_revision_number, from_matlab

