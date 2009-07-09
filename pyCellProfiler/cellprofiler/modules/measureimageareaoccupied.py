"""measureimageareaoccupied.py - measure the area of an image occupied by objects

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__ = "$Revision$"

import numpy as np

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps

'''Measurement feature name format for the AreaOccupied measurement'''
F_AREA_OCCUPIED = "AreaOccupied_AreaOccupied_%s"

'''Measure feature name format for the TotalArea measurement'''
F_TOTAL_AREA = "AreaOccupied_TotalArea_%s"

class MeasureImageAreaOccupied(cpm.CPModule):
    """% SHORT DESCRIPTION:
Measures total area occupied by objects within an image
*************************************************************************
This module reports the sum of the areas of the objects segmented by one
of the Identify modules (IdentifyPrimAutomatic, IdentifyPrimSecondary, etc.).
The numbers it reports are those of the filtered image, so they may exclude
objects touching the image boundary and objects filtered out because of
their size. Both the area occupied and the total image area will respect
the masking, if any, of the primary image used by the Identify module.

You may want to threshold an image and then measure the number of pixels
above that threshold. This can be done by using IdentifyPrimAutomatic as
your thresholder. If you use it for this purpose, you may want to turn off
features that either filter out images (Discard objects touching the border,
Discard too-small objects).

Features measured:
AreaOccupied
TotalImageArea

Settings:

What is the name of the object in which you want to measure the image area?
This is the object name that is specified in one of the Identify modules
from earlier in the pipeline.

Do you want to save the object labels as a binary image?
You can check the checkbox for this option and fill in a name in the text
box that appears below the checkbox if you want to create a binary image
of the labeled pixels being processed by this  module.
"""

    category = "Measurement"
    variable_revision_number = 1
    
    def create_settings(self):
        """Create the settings variables here and name the module
        
        """
        self.module_name = "MeasureImageAreaOccupied"
        self.object_name = cps.ObjectNameSubscriber("What is the name of the object in which you want to measure the image area?","None")
        self.should_save_image = cps.Binary("Do you want to save the object labels as a binary image?",False)
        self.image_name = cps.ImageNameProvider("What do you want to call the output binary image showing the object area occupied? (StainName)","Stain")

    def backwards_compatibilize(self, setting_values, variable_revision_number, 
                                module_name, from_matlab):
        """Account for the save-format of previous versions of this module
        
        We check for the Matlab version which did the thresholding as well
        as the measurement; this duplicated the functionality in the Identify
        modules.
        """
        if from_matlab:
            raise NotImplementedError("The MeasureImageArea module has changed substantially. \n"
                                      "You should threshold your image using IdentifyPrimAutomatic\n"
                                      "and then measure the resulting objects' area using this module.")
        return setting_values, variable_revision_number, from_matlab

    def settings(self):
        """The settings as saved in the pipeline file
        
        """
        return [self.object_name, self.should_save_image, self.image_name]

    def visible_settings(self):
        """The settings, in the order that they should be displayed in the UI
        
        """
        result = [self.object_name, self.should_save_image]
        if self.should_save_image.value:
            result += [self.image_name]
        return result


    def run(self, workspace):
        objects = workspace.get_objects(self.object_name.value)
        if objects.has_parent_image:
            area_occupied = np.sum(objects.segmented[objects.parent_image.mask]>0)
            total_area = np.sum(objects.parent_image.mask)
        else:
            area_occupied = np.sum(objects.segmented > 0)
            total_area = np.product(objects.segmented.shape)
        if workspace.frame != None:
            figure = workspace.create_or_find_figure(subplots=(2,1))
            statistics = (("Area occupied","%d pixels"%(area_occupied)),
                          ("Total area", "%d pixels"%(total_area)))
            figure.subplot_imshow_labels(0,0,objects.segmented,
                                         title="Object labels: %s"%(self.object_name.value))
            figure.subplot_table(1,0,statistics)
        
        m = workspace.measurements
        m.add_image_measurement(F_AREA_OCCUPIED%(self.object_name.value),
                                np.array([area_occupied], dtype=float ))
        m.add_image_measurement(F_TOTAL_AREA%(self.object_name.value),
                                np.array([total_area], dtype=float))
        if self.should_save_image.value:
            binary_pixels = objects.segmented > 0
            output_image = cpi.Image(binary_pixels,
                                     parent_image = objects.parent_image)
            workspace.image_set.add(self.image_name.value,
                                    output_image)
    
    def get_measurement_columns(self, pipeline):
        '''Return column definitions for measurements made by this module'''
        return [(cpmeas.IMAGE,
                 F_AREA_OCCUPIED % (self.object_name.value),
                 cpmeas.COLTYPE_FLOAT),
                (cpmeas.IMAGE,
                 F_TOTAL_AREA % (self.object_name.value),
                 cpmeas.COLTYPE_FLOAT)
                 ]

    def get_categories(self, pipeline, object_name):
        """The categories output by this module for the given object (or Image)
        
        """
        if object_name == "Image":
            return ["AreaOccupied"]

        return []

    def get_measurements(self, pipeline, object_name, category):
        """The measurements available for a given category"""
        if object_name == "Image" and category == "AreaOccupied":
            return ["AreaOccupied", "TotalArea"]

    def get_measurement_objects(self, pipeline, object_name, category, 
                                measurement):
        """The objects measured for a particular measurement
        
        """
        if (object_name == "Image" and category == "AreaOccupied" and
            measurement in ("AreaOccupied","TotalArea")):
            return [ self.object_name.value ]
        return []
