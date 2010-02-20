"""<b> Measure Image Area Occupied</b> measures the total area in an image that is occupied by objects
<hr>
This module reports the sum of the areas of the objects defined by one
of the Identify modules (<b>IdentifyPrimaryObjects</b>, <b>IdentifySecondaryObjects</b>, etc.).
Both the area occupied and the total image area will respect
the masking, if any, of the primary image used by the Identify module.

If you want to measure the number of pixels
above a threshold, this can be done by using this module preceded by
thresholding performed by <b>IdentifyPrimaryObjects</b>.
<br>
<br>
Features that can be measured by this module:
<ul>
<li>AreaOccupied
<li>TotalImageArea
</ul>
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

# The number of settings per image
IMAGE_SETTING_COUNT = 3


class MeasureImageAreaOccupied(cpm.CPModule):
    module_name = "MeasureImageAreaOccupied"
    category = "Measurement"
    variable_revision_number = 2
    
    def create_settings(self):
        """Create the settings variables here and name the module
        
        """
        self.divider_top = cps.Divider(line=False)
        self.objects = []
        self.add_object(can_remove = False)
        self.add_button = cps.DoSomething("","Add another object", self.add_object, True)
        self.divider_bottom = cps.Divider(line=False)
        
    def add_object(self, can_remove=True):
        # The text for these settings will be replaced in renumber_settings()
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=True))
        
        group.append("object_name", cps.ObjectNameSubscriber("Select objects to measure", "None"))
        
        group.append("should_save_image", cps.Binary("Retain a binary image of the object regions for use later in the pipeline (for example, in SaveImages)?", False))
        
        group.append("image_name", cps.ImageNameProvider("Name the output binary image", "Stain",doc="""
                                        <i>(Used only if the binary image of the objects is to be retained for later use in the pipeline)</i> <br> 
                                        Choose a name, which will allow the binary image of the objects to be selected later in the pipeline."""))
        if can_remove:
            group.append("remover", cps.RemoveSettingButton("", "Remove this object", self.objects, group))
        
        self.objects.append(group)        

    def settings(self):
        """The settings as saved in the pipeline file
        
        """
        result = []
        for object in self.objects:
            result += [object.object_name, object.should_save_image,object.image_name]
        return result

    def visible_settings(self):
        """The settings, in the order that they should be displayed in the UI
        
        """
        result = []
        for index, object in enumerate(self.objects):
            result += object.visible_settings()
        result += [self.add_button]
        return result
    
    def prepare_settings(self, setting_values):
        value_count = len(setting_values)
        assert value_count % IMAGE_SETTING_COUNT == 0
        object_count = value_count / IMAGE_SETTING_COUNT
        # always keep the first object
        del self.objects[1:]
        while len(self.objects) < object_count:
            self.add_object()
            
    def get_non_redundant_object_measurements(self):
        '''Return a non-redundant sequence of object measurement objects'''
        dict = {}
        for object in self.objects:
            key = ((object.object_name, object.image_name) if object.should_save_image.value
                   else (object.object_name,))
            dict[key] = object
        return dict.values()

    def run(self, workspace):
        statistics = [["Area occupied","Total area"]]
        for object in self.get_non_redundant_object_measurements():
            statistics += self.measure(object,workspace)
#        if workspace.frame != None:
#            figure = workspace.create_or_find_figure(subplots=(2,1))
#            figure.subplot_imshow_labels(0,0,objects.segmented,
#                                         title="Object labels: %s"%(object))
#            figure.subplot_table(1,0,statistics)
            
    def measure(self, object, workspace):
        '''Performs the measurements on the requested objects'''
        objects = workspace.get_objects(object.object_name.value)
        if objects.has_parent_image:
            area_occupied = np.sum(objects.segmented[objects.parent_image.mask]>0)
            total_area = np.sum(objects.parent_image.mask)
        else:
            area_occupied = np.sum(objects.segmented > 0)
            total_area = np.product(objects.segmented.shape)
        
        
        m = workspace.measurements
        m.add_image_measurement(F_AREA_OCCUPIED%(object.object_name.value),
                                np.array([area_occupied], dtype=float ))
        m.add_image_measurement(F_TOTAL_AREA%(object.object_name.value),
                                np.array([total_area], dtype=float))
        if object.should_save_image.value:
            binary_pixels = objects.segmented > 0
            output_image = cpi.Image(binary_pixels,
                                     parent_image = objects.parent_image)
            workspace.image_set.add(object.image_name.value,
                                    output_image)
        return[[object.object_name.value, object.image_name.value if object.should_save_image.value else "",
                feature_name, str(value)]
                for feature_name, value in (('Area Occupied', area_occupied),
                                            ('Total Area', total_area))]
    
    def get_measurement_columns(self, pipeline):
        '''Return column definitions for measurements made by this module'''
        columns = []
        for object in self.get_non_redundant_object_measurements():
            for feature, coltype in ((F_AREA_OCCUPIED, cpmeas.COLTYPE_FLOAT),
                                     (F_TOTAL_AREA, cpmeas.COLTYPE_FLOAT)):
                columns.append((cpmeas.IMAGE, 
                                feature % object.object_name.value, 
                                coltype))
        return columns
       
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
            return [ object.object_name.value for object in self.objects ]
        return []
    
    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
        """Account for the save-format of previous versions of this module
        
        We check for the Matlab version which did the thresholding as well
        as the measurement; this duplicated the functionality in the Identify
        modules.
        """
        if from_matlab:
            raise NotImplementedError("The MeasureImageAreaOccupied module has changed substantially. \n"
                                      "You should threshold your image using IdentifyPrimaryObjects\n"
                                      "and then measure the resulting objects' area using this module.")
        if variable_revision_number == 1:
            # We added the ability to process multiple objects in v2, but
            # the settings for v1 miraculously map to v2
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab

