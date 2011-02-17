"""<b> Measure Image Area Occupied</b> measures the total area in an image that is occupied by objects
<hr>

This module reports the sum of the areas of the objects defined by one
of the <b>Identify</b> modules, or the area of the foreground in a binary
image.
Both the area occupied and the total image area will respect
the masking, if any, of the primary image used by the Identify module.

<p>You can use this module to measure the number of pixels above a given threshold 
if you precede it with thresholding performed by <b>ApplyThreshold</b>, and then select the binary image
output by <b>ApplyThreshold</b> to be measured by this module.</p>

<h4>Available measurements</h4>
<ul>
<li><i>AreaOccupied:</i> The total area occupied by the input objects.</li>
<li><i>TotalImageArea:</i> The total pixel area of the image.</li>
</ul>

See also <b>IdentifyPrimaryObjects</b>, <b>IdentifySecondaryObjects</b>, <b>IdentifyTertiaryObjects</b>
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

# The number of settings per image or object group
IMAGE_SETTING_COUNT = 1

OBJECT_SETTING_COUNT = 3


class MeasureImageAreaOccupied(cpm.CPModule):
    module_name = "MeasureImageAreaOccupied"
    category = "Measurement"
    variable_revision_number = 3
    
    def create_settings(self):
        """Create the settings variables here and name the module
        
        """

        self.operands = []
        self.count = cps.HiddenCount(self.operands)
        self.add_operand(can_remove=False)
        self.add_operand_button = cps.DoSomething("", "Add", self.add_operand)
        self.remover = cps.DoSomething("", "Remove", self.remove)
     
    def add_operand(self, can_remove=True):
        class Operand(object):
            def __init__(self):
                self.__spacer = cps.Divider(line=True)
                self.__operand_choice = cps.Choice("Measure the area occupied in a binary image, or in objects?", ["Binary Image", "Objects"], doc = """You can either measure the area occupied by previously-identified objects, or the area occupied by the foreground in a binary (black and white) image.""")
                self.__operand_objects = cps.ObjectNameSubscriber("Select objects to measure","None", doc = """<i>(Used only if 'Objects' are to be measured)</i> Select the previously identified objects you would like to measure. """)
                self.__should_save_image = cps.Binary("Retain a binary image of the object regions?", False, doc="""<i>(Used only if 'Objects' are to be measured)</i> This setting is helpful if you would like to use a binary image later in the pipeline, for example in SaveImages.  The image will display the object area that you have measured as the foreground in white and the background in black. """)
                self.__image_name = cps.ImageNameProvider("Name the output binary image", "Stain",doc="""
                                        <i>(Used only if the binary image of the objects is to be retained for later use in the pipeline)</i> <br> 
                                        Specify a name that will allow the binary image of the objects to be selected later in the pipeline.""")
                self.__binary_name = cps.ImageNameSubscriber("Select a binary image to measure", "None", doc="""<i>(Used only if 'Binary image' is to be measured)</i> This is a binary image created earlier in the pipeline, where you would like to measure the area occupied by the foreground in the image.""")
            
            @property
            def spacer(self):
                return self.__spacer
    
            @property
            def operand_choice(self):
                return self.__operand_choice

            @property
            def operand_objects(self):
                return self.__operand_objects

            @property
            def should_save_image(self):
                return self.__should_save_image
            
            @property
            def image_name(self):
                return self.__image_name

            @property
            def binary_name(self):
                return self.__binary_name
        
            @property
            def remover(self):
                return self.__remover

            @property
            def object(self):
                if self.operand_choice == "Binary Image":
                    return self.binary_name.value
                else:
                    return self.operand_objects.value
        self.operands += [Operand()]
            
    
    def remove(self):
        del self.operands[-1]
        return self.operands

    def settings(self):
        result = [self.count]
        for op in self.operands:                         
            result += [op.operand_choice, op.operand_objects, op.should_save_image,op.image_name, op.binary_name]
        return result

    def prepare_settings(self, setting_values):
        count = int(setting_values[0])
        sequence = self.operands
        del sequence[count:]
        while len(sequence) < count:
            self.add_operand()
            sequence = self.operands

    def visible_settings(self):
        result = []
        for op in self.operands:
            result += [op.spacer]
            result += [op.operand_choice]        
            result += ([op.operand_objects, op.should_save_image] if op.operand_choice == "Objects" else [op.binary_name])
            if op.should_save_image:
                result.append(op.image_name)
        result.append(self.add_operand_button)
        result.append(self.remover)
        return result

   
    def is_interactive(self):
        return False


            
    def run(self, workspace):
        m = workspace.measurements
        statistics = [["Objects or Image", "Area Occupied", "Total Area"]]
        for op in self.operands:
            if op.operand_choice  == "Objects":
                statistics += self.measure_objects(op,workspace) 
            if op.operand_choice == "Binary Image":
                statistics += self.measure_images(op,workspace)
        if workspace.frame is not None:
            workspace.display_data.statistics = statistics
    
    def display(self, workspace):
        figure = workspace.create_or_find_figure(title="MeasureImageAreaOccupied, image cycle #%d"%(
                workspace.measurements.image_set_number),subplots=(1,1))
        figure.subplot_table(0, 0, workspace.display_data.statistics,
                             ratio=(.25,.25,.25))
        
    def measure_objects(self, operand, workspace):
        '''Performs the measurements on the requested objects'''
        objects = workspace.get_objects(operand.operand_objects.value)
        if objects.has_parent_image:
            area_occupied = np.sum(objects.segmented[objects.parent_image.mask]>0)
            total_area = np.sum(objects.parent_image.mask)
        else:
            area_occupied = np.sum(objects.segmented > 0)
            total_area = np.product(objects.segmented.shape)
        m = workspace.measurements
        m.add_image_measurement(F_AREA_OCCUPIED%(operand.operand_objects.value),
                                np.array([area_occupied], dtype=float ))
        m.add_image_measurement(F_TOTAL_AREA%(operand.operand_objects.value),
                                np.array([total_area], dtype=float))
        if operand.should_save_image.value:
            binary_pixels = objects.segmented > 0
            output_image = cpi.Image(binary_pixels,
                                     parent_image = objects.parent_image)
            workspace.image_set.add(operand.image_name.value,
                                    output_image)
        return[[operand.operand_objects.value,
                str(area_occupied),str(total_area)]]

    def measure_images(self,operand,workspace):
        '''Performs measurements on the requested images'''
        image = workspace.image_set.get_image(operand.binary_name.value, must_be_binary = True)
        area_occupied = np.sum(image.pixel_data > 0)
        total_area = np.prod(np.shape(image.pixel_data))
        m = workspace.measurements
        m.add_image_measurement(F_AREA_OCCUPIED%(operand.binary_name.value),
                                np.array([area_occupied], dtype=float))
        m.add_image_measurement(F_TOTAL_AREA%(operand.binary_name.value),
                                np.array([total_area], dtype=float))
        return [[operand.binary_name.value, str(area_occupied), str(total_area)]]


    def get_measurement_columns(self, pipeline):
        '''Return column definitions for measurements made by this module'''
        columns = []
        for op in self.operands:
            for feature, coltype in ((F_AREA_OCCUPIED, cpmeas.COLTYPE_FLOAT),
                                     (F_TOTAL_AREA, cpmeas.COLTYPE_FLOAT)):
                columns.append((cpmeas.IMAGE, 
                                feature % (op.operand_objects.value if op.operand_choice == "Objects" else op.binary_name.value), 
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
            return [ op.operand_objects.value for op in self.operands ]
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
                                      "You should use this module by either:\n"
                                      "(1) Thresholding your image using an Identify module\n"
                                      "and then measure the resulting objects' area; or\n"
                                      "(2) Create a binary image with ApplyThreshold and then measure the\n"
                                      "resulting foreground image area.")
        if variable_revision_number == 1:
            # We added the ability to process multiple objects in v2, but
            # the settings for v1 miraculously map to v2
            variable_revision_number = 2
        if variable_revision_number == 2:
            # Permits choice of binary image or objects to measure from
            count = len(setting_values) / 3
            new_setting_values = [str(count)]
            for i in range(0, count):
                new_setting_values += ['Objects', setting_values[(i*3)], setting_values[(i*3)+1], setting_values[(i*3)+2], 'None']
            setting_values = new_setting_values
            variable_revision_number = 3
        return setting_values, variable_revision_number, from_matlab

