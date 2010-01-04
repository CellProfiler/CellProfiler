"""<b>Mask Image</b> hides certain portions of an image (based on previously identified objects or a binary image) so they are ignored by subsequent mask-respecting modules in the pipeline
<hr>

This module masks an image and saves it in the handles structure for
future use. The masked image is based on the original image and the
masking object or image that is selected. 

Note that the image that is created by this module for further processing downstream is grayscale. If a binary mask is desired in subsequent modules, you might be able to access the image's crop mask, or simply use the <b>ApplyThreshold</b> module instead of <b>MaskImage</b>.

See also <b>ApplyThreshold</b>, <b>IdentifyPrimAutomatic</b>, <b>IdentifyPrimManual</b>.

"""
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

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

IO_IMAGE = "Image"
IO_OBJECTS = "Objects"

class MaskImage(cpm.CPModule):

    module_name = "MaskImage"
    category = "Image Processing"
    variable_revision_number = 2
    
    def create_settings(self):
        """Create the settings here and set the module name (initialization)
        
        """
        self.source_choice=cps.Choice(
            "Do you want to mask using objects or an image?",
            [IO_OBJECTS, IO_IMAGE],
            doc="""You can mask an image in two ways:<br>
            <ul><li><b>Objects</b>: Here, you use objects created by another
            module (for instance <b>IdentifyPrimAutomatic</b>). The module
            will mask out all parts of the image that are not within one
            of the objects (unless you invert the mask).</li>
            <li><b>Image</b>: Here, you use a binary image as the mask, where black 
            portions of the image (false or zero-value pixels) will be masked out.
            If the image is not binary, the module will use
            all pixels whose intensity is greater than .5 as the mask's
            foreground (white area). You may instead use <b>ApplyThreshold</b> to create a binary
            image with finer control over the intensity choice.</li></ul>""")
        self.object_name = cps.ObjectNameSubscriber("Select object for mask","None",
                                                    doc = '''<i>(Only used if mask is to be made from objects)</i> <br> Which objects would you like to use to mask the input image?''')
        self.masking_image_name = cps.ImageNameSubscriber(
            "Select image for mask","None",
            doc = """<i>(Only used if mask is to be made from an image)</i> <br> Which image would you like to use to mask the input image?""")
        self.image_name = cps.ImageNameSubscriber("Select the input image","None", doc = '''Which image do you want to mask?''')
        self.masked_image_name = cps.ImageNameProvider("Name the output image",
                                                       "MaskBlue", doc = '''What do you want to call the masked image?''')
        self.invert_mask = cps.Binary("Invert the mask?",False)

    def settings(self):
        """Return the settings in the order that they will be saved or loaded
        
        Note: the settings are also the visible settings in this case, so
              they also control the display order. Implement visible_settings
              for a different display order.
        """
        return [self.object_name, self.image_name,
                self.masked_image_name, self.invert_mask,
                self.source_choice, self.masking_image_name]
    
    def visible_settings(self):
        """Return the settings as displayed in the user interface"""
        return [self.source_choice,
                self.object_name if self.source_choice == IO_OBJECTS
                else self.masking_image_name,
                self.image_name, self.masked_image_name,
                self.invert_mask]

    def run(self, workspace):
        image_set = workspace.image_set
        if self.source_choice == IO_OBJECTS:
            objects = workspace.get_objects(self.object_name.value)
            labels = objects.segmented
            if self.invert_mask.value:
                mask = labels == 0
            else:
                mask = labels > 0
        else:
            objects = None
            try:
                mask = image_set.get_image(self.masking_image_name.value,
                                           must_be_binary=True).pixel_data
            except ValueError:
                mask = image_set.get_image(self.masking_image_name.value,
                                           must_be_grayscale=True).pixel_data
                mask = mask > .5
        orig_image = image_set.get_image(self.image_name.value,
                                         must_be_grayscale = True)
        if mask.shape != orig_image.pixel_data.shape:
            tmp = np.zeros(orig_image.pixel_data.shape, mask.dtype)
            tmp[mask] = True
            mask = tmp
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
        image_set.add(self.masked_image_name.value, masked_image)
    
    def upgrade_settings(self, setting_values, 
                         variable_revision_number, 
                         module_name, from_matlab):
        """Adjust the setting_values to upgrade from a previous version
        
        """
        if from_matlab and variable_revision_number == 3:
            from_matlab = False
            variable_revision_number = 1
        if (not from_matlab) and variable_revision_number == 1:
            #
            # Added ability to select an image
            #
            setting_values = setting_values + [IO_OBJECTS, "None"]
        return setting_values, variable_revision_number, from_matlab

