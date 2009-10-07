"""<b> Measure Image Area Occupied</b>
measures total image area occupied by objects
<hr>
This module reports the sum of the areas of the objects defined by one
of the Identify modules (<b>IdentifyPrimAutomatic</b>, <b>IdentifyPrimSecondary</b>, etc.).
Both the area occupied and the total image area will respect
the masking, if any, of the primary image used by the Identify module.

If you want to threshold an image and then measure the number of pixels
above that threshold, this can be done by using this module together with IdentifyPrimAutomatic as
your thresholder.
<br>
<br>
Features that can be measured by this module:
<ul>
<li>AreaOccupied
<li>TotalImageArea
</ul>
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
    category = "Measurement"
    variable_revision_number = 1
    
    def create_settings(self):
        """Create the settings variables here and name the module
        
        """
        self.module_name = "MeasureImageAreaOccupied"
        self.object_name = cps.ObjectNameSubscriber("Select the object name","None",
            doc='''What is the name of the object for which you want to measure the occupied image area? 
            This is the object name that was specified in one of the Identify modules from earlier 
            in the pipeline.''')
        self.should_save_image = cps.Binary("Save object labels as a binary image?",False,
            doc='''You can check the checkbox for this option and fill in a name in the text box that 
            appears below the checkbox if you want to create a binary image of the labeled pixels being 
            processed by this  module.''')
        self.image_name = cps.ImageNameProvider("Name the output binary image",
            "Stain",doc='''What do you want to call the output binary image showing the area occupied by objects?''')

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
        return []

    def get_measurement_objects(self, pipeline, object_name, category, 
                                measurement):
        """The objects measured for a particular measurement
        
        """
        if (object_name == "Image" and category == "AreaOccupied" and
            measurement in ("AreaOccupied","TotalArea")):
            return [ self.object_name.value ]
        return []
