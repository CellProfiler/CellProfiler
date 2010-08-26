'''<b>Measure Image Intensity</b> measures the total intensity in an image by summing all of the pixel intensities (excluding masked pixels)
<hr>

This module will sum all pixel values to measure the total image
intensity. The user can measure all pixels in the image or can restrict
the measurement to pixels within objects. If the image has a mask, only
unmasked pixels will be measured.
                                     
<h4>Available measurements</h4>
<ul>
<li><i>TotalIntensity:</i> Sum of all pixel intensity values.</li>
<li><i>MeanIntensity, MedianIntensity:</i> Mean and median of pixel intensity values.</li>
<li><i>StdIntensity, MADIntensity:</i> Standard deviation and median absolute deviation (MAD) of pixel intensity values.</li>
<li><i>MinIntensity, MaxIntensity:</i> Minimum and maximum of pixel intensity values.</li>
<li><i>TotalArea:</i> Number of pixels measured.</li>
</ul>
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
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas

'''Number of settings saved/loaded per image measured'''
SETTINGS_PER_IMAGE = 3

'''Measurement feature name format for the TotalIntensity measurement'''
F_TOTAL_INTENSITY = "Intensity_TotalIntensity_%s"

'''Measurement feature name format for the MeanIntensity measurement'''
F_MEAN_INTENSITY = 'Intensity_MeanIntensity_%s'

'''Measurement feature name format for the MeanIntensity measurement'''
F_MEDIAN_INTENSITY = 'Intensity_MedianIntensity_%s'

'''Measurement feature name format for the StdIntensity measurement'''
F_STD_INTENSITY = 'Intensity_StdIntensity_%s'

'''Measurement feature name format for the MedAbsDevIntensity measurement'''
F_MAD_INTENSITY = 'Intensity_MADIntensity_%s'

'''Measurement feature name format for the MaxIntensity measurement'''
F_MAX_INTENSITY = 'Intensity_MaxIntensity_%s'

'''Measurement feature name format for the MinIntensity measurement'''
F_MIN_INTENSITY = 'Intensity_MinIntensity_%s'

'''Measurement feature name format for the TotalArea measurement'''
F_TOTAL_AREA = 'Intensity_TotalArea_%s'

ALL_MEASUREMENTS = ["TotalIntensity", "MeanIntensity", "StdIntensity", "MADIntensity", "MedianIntensity",
                    "MinIntensity",  "MaxIntensity", "TotalArea"]

class MeasureImageIntensity(cpm.CPModule):

    module_name = 'MeasureImageIntensity'
    category = "Measurement"
    variable_revision_number = 2
    
    def create_settings(self):
        '''Create the settings & name the module'''
        self.divider_top = cps.Divider(line=False)
        self.images = []
        self.add_image_measurement(can_remove = False)
        self.add_button = cps.DoSomething("", "Add another image",
                                          self.add_image_measurement)
        self.divider_bottom = cps.Divider(line=False)
    
    def add_image_measurement(self, can_remove = True):
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider())
        
        group.append("image_name", cps.ImageNameSubscriber("Select the image to measure",
                                                            "None", doc = '''What did you call the images whose intensity you want to measure? Choose an image name from the drop-down menu to calculate intensity for that
                                                            image. Use the <i>Add another image</i> button below to add additional images which will be
                                                            measured. You can add the same image multiple times if you want to measure
                                                            the intensity within several different objects.'''))
        
        group.append("wants_objects", cps.Binary("Measure the intensity only from areas enclosed by objects?",
                                                  False, doc = "Check this option to restrict the pixels being measured to those within the boundaries of an object."))
        
        group.append("object_name",cps.ObjectNameSubscriber("Select the input objects","None", 
                                                           doc = '''<i>(Used only when measuring intensity from area enclosed by objects)</i><br>What is the name of the objects to use? The intensity measurement will be restricted to within these objects.'''))
        if can_remove:
            group.append("remover", cps.RemoveSettingButton("", 
                                                            "Remove this image", self.images, group))
        self.images.append(group)
                    
    def settings(self):
        result = []
        for image in self.images:
            result += [image.image_name, image.wants_objects, image.object_name]
        return result
            
    def visible_settings(self):
        result = []
        for index, image in enumerate(self.images):
            temp = image.visible_settings()
            if not image.wants_objects:
                temp.remove(image.object_name)
            result += temp
        result += [self.add_button]
        return result
    
    def prepare_settings(self, setting_values):
        assert len(setting_values) % SETTINGS_PER_IMAGE == 0
        image_count = len(setting_values) / SETTINGS_PER_IMAGE
        while image_count > len(self.images):
            self.add_image_measurement()
        while image_count < len(self.images):
            self.remove_image_measurement(self.images[-1].key)

    def get_non_redundant_image_measurements(self):
        '''Return a non-redundant sequence of image measurement objects'''
        dict = {}
        for im in self.images:
            key = ((im.image_name, im.object_name) if im.wants_objects.value
                   else (im.image_name,))
            dict[key] = im
        return dict.values()
        
    def is_interactive(self):
        return False

    def run(self, workspace):
        '''Perform the measurements on the imageset'''
        #
        # Then measure each
        #
        statistics = [["Image","Masking object","Feature","Value"]]
        for im in self.get_non_redundant_image_measurements():
            statistics += self.measure(im, workspace)
        workspace.display_data.statistics = statistics

    def display(self, workspace):
        figure = workspace.create_or_find_figure(title="MeasureImageIntensity, image cycle #%d"%(
                workspace.measurements.image_set_number),subplots=(1,1))
        figure.subplot_table(0, 0, workspace.display_data.statistics, 
                             ratio=(.25,.25,.25,.25))
    
    def measure(self, im, workspace):
        '''Perform measurements according to the image measurement in im
        
        im - image measurement info (see ImageMeasurement class above)
        workspace - has all the details for current image set
        '''
        image = workspace.image_set.get_image(im.image_name.value,
                                              must_be_grayscale=True)
        pixels = image.pixel_data

        measurement_name = im.image_name.value
        if im.wants_objects.value:
            measurement_name += "_" + im.object_name.value
            objects = workspace.get_objects(im.object_name.value)
            if image.has_mask:
                pixels = pixels[np.logical_and(objects.segmented != 0,
                                               image.mask)]
            else:
                pixels = pixels[objects.segmented != 0]
        elif image.has_mask:
            pixels = pixels[image.mask]


        pixel_count = np.product(pixels.shape)
        if pixel_count == 0:
            pixel_sum = 0
            pixel_mean = 0
            pixel_std = 0
            pixel_mad = 0
            pixel_median = 0
            pixel_min = 0
            pixel_max = 0
        else:
            pixel_sum = np.sum(pixels)
            pixel_mean = pixel_sum/float(pixel_count)
            pixel_std = np.std(pixels)
            pixel_median = np.median(pixels)
            pixel_mad = np.median(np.abs(pixels - pixel_median))
            pixel_min = np.min(pixels)
            pixel_max = np.max(pixels)
        m = workspace.measurements
        m.add_image_measurement(F_TOTAL_INTENSITY%(measurement_name), pixel_sum)
        m.add_image_measurement(F_MEAN_INTENSITY%(measurement_name), pixel_mean)
        m.add_image_measurement(F_MEDIAN_INTENSITY%(measurement_name), pixel_median)
        m.add_image_measurement(F_STD_INTENSITY%(measurement_name), pixel_std)
        m.add_image_measurement(F_MAD_INTENSITY%(measurement_name), pixel_mad)
        m.add_image_measurement(F_MAX_INTENSITY%(measurement_name), pixel_max)
        m.add_image_measurement(F_MIN_INTENSITY%(measurement_name), pixel_min)
        m.add_image_measurement(F_TOTAL_AREA%(measurement_name), pixel_count)
        return [[im.image_name.value, 
                 im.object_name.value if im.wants_objects.value else "",
                 feature_name, str(value)]
                for feature_name, value in (('Total intensity', pixel_sum),
                                            ('Mean intensity', pixel_mean),
                                            ('Median intensity', pixel_median),
                                            ('Std intensity', pixel_std),
                                            ('MAD intensity', pixel_mad),
                                            ('Min intensity', pixel_min),
                                            ('Max intensity', pixel_max),
                                            ('Total area', pixel_count))]
    
    def get_measurement_columns(self, pipeline):
        '''Return column definitions for measurements made by this module'''
        columns = []
        for im in self.get_non_redundant_image_measurements():
            for feature, coltype in ((F_TOTAL_INTENSITY, cpmeas.COLTYPE_FLOAT),
                                     (F_MEAN_INTENSITY, cpmeas.COLTYPE_FLOAT),
                                     (F_MEDIAN_INTENSITY, cpmeas.COLTYPE_FLOAT),
                                     (F_STD_INTENSITY, cpmeas.COLTYPE_FLOAT),
                                     (F_MAD_INTENSITY, cpmeas.COLTYPE_FLOAT),
                                     (F_MIN_INTENSITY, cpmeas.COLTYPE_FLOAT),
                                     (F_MAX_INTENSITY, cpmeas.COLTYPE_FLOAT),
                                     (F_TOTAL_AREA, cpmeas.COLTYPE_INTEGER)):
                measurement_name = im.image_name.value + (("_" + im.object_name.value) if im.wants_objects.value else "")
                columns.append((cpmeas.IMAGE, feature % measurement_name, coltype))
        return columns
                        
    def get_categories(self, pipeline, object_name):
        if object_name == cpmeas.IMAGE:
            return ["Intensity"]
        else:
            return []

    def get_measurements(self, pipeline, object_name, category):
        if (object_name == cpmeas.IMAGE and
            category == "Intensity"):
            return ALL_MEASUREMENTS
        return []

    def get_measurement_images(self, pipeline, object_name, 
                               category, measurement):
        if (object_name == cpmeas.IMAGE and
            category == "Intensity" and
            measurement in ALL_MEASUREMENTS):
            result = []
            for im in self.images:
                image_name = im.image_name.value 
                if im.wants_objects:
                    image_name += "_" + im.object_name.value
                result += [image_name]
            return result
        return []
    
    def upgrade_settings(self, setting_values, 
                         variable_revision_number, 
                         module_name, from_matlab):
        '''Account for prior versions when loading
        
        We handle Matlab revision # 2 here. We don't support thresholding
        because it was generally unused. The first setting is the image name.
        '''
        if from_matlab and variable_revision_number == 2:
            setting_values = [setting_values[0], # image name
                              cps.NO,            # wants objects
                              "None" ]           # object name
            variable_revision_number = 1
            from_matlab = False
        if variable_revision_number == 1:
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab


