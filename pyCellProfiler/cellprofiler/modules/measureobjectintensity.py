"""measureobjectintensity - take intensity measurements on an object set

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import numpy as np
import scipy.ndimage as nd
import uuid

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
import cellprofiler.cpmath.outline as cpmo
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix

INTENSITY = 'Intensity'
INTEGRATED_INTENSITY = 'IntegratedIntensity'
MEAN_INTENSITY = 'MeanIntensity'
STD_INTENSITY = 'StdIntensity'
MIN_INTENSITY = 'MinIntensity'
MAX_INTENSITY = 'MaxIntensity'
INTEGRATED_INTENSITY_EDGE = 'IntegratedIntensityEdge'
MEAN_INTENSITY_EDGE = 'MeanIntensityEdge'
STD_INTENSITY_EDGE = 'StdIntensityEdge'
MIN_INTENSITY_EDGE = 'MinIntensityEdge'
MAX_INTENSITY_EDGE = 'MaxIntensityEdge'
MASS_DISPLACEMENT = 'MassDisplacement'
LOWER_QUARTILE_INTENSITY = 'LowerQuartileIntensity'
MEDIAN_INTENSITY = 'MedianIntensity'
UPPER_QUARTILE_INTENSITY = 'UpperQuartileIntensity'

ALL_MEASUREMENTS = [INTEGRATED_INTENSITY, MEAN_INTENSITY, STD_INTENSITY,
                        MIN_INTENSITY, MAX_INTENSITY, INTEGRATED_INTENSITY_EDGE,
                        MEAN_INTENSITY_EDGE, STD_INTENSITY_EDGE,
                        MIN_INTENSITY_EDGE, MAX_INTENSITY_EDGE, 
                        MASS_DISPLACEMENT, LOWER_QUARTILE_INTENSITY,
                        MEDIAN_INTENSITY, UPPER_QUARTILE_INTENSITY]

class MeasureObjectIntensity(cpm.CPModule):
    """% SHORT DESCRIPTION:
Measures several intensity features for identified objects.
*************************************************************************

Given an image with objects identified (e.g. nuclei or cells), this
module extracts intensity features for each object based on a
corresponding grayscale image. Measurements are recorded for each object.

Features measured:       Feature Number:
IntegratedIntensity     |       1
MeanIntensity           |       2
StdIntensity            |       3
MinIntensity            |       4
MaxIntensity            |       5
IntegratedIntensityEdge |       6
MeanIntensityEdge       |       7
StdIntensityEdge        |       8
MinIntensityEdge        |       9
MaxIntensityEdge        |      10
MassDisplacement        |      11
LowerQuartileIntensity  |      12
MedianIntensity         |      13
UpperQuartileIntensity  |      14

How it works:
Retrieves objects in label matrix format and a corresponding original
grayscale image and makes measurements of the objects. The label matrix
image should be "compacted": that is, each number should correspond to an
object, with no numbers skipped. So, if some objects were discarded from
the label matrix image, the image should be converted to binary and
re-made into a label matrix image before feeding it to this module.

Intensity Measurement descriptions:

* IntegratedIntensity - The sum of the pixel intensities within an
 object.
* MeanIntensity - The average pixel intensity within an object.
* StdIntensity - The standard deviation of the pixel intensities within
 an object.
* MaxIntensity - The maximal pixel intensity within an object.
* MinIntensity - The minimal pixel intensity within an object.
* IntegratedIntensityEdge - The sum of the edge pixel intensities of an
 object.
* MeanIntensityEdge - The average edge pixel intensity of an object.
* StdIntensityEdge - The standard deviation of the edge pixel intensities
 of an object.
* MaxIntensityEdge - The maximal edge pixel intensity of an object.
* MinIntensityEdge - The minimal edge pixel intensity of an object.
* MassDisplacement - The distance between the centers of gravity in the
 gray-level representation of the object and the binary representation of
 the object.
* LowerQuartileIntensity - the intensity value of the pixel for which 25%
 of the pixels in the object have lower values.
* MedianIntensity - the median intensity value within the object
* UpperQuartileIntensity - the intensity value of the pixel for which 75%
 of the pixels in the object have lower values.

For publication purposes, it is important to note that the units of
intensity from microscopy images are usually described as "Intensity
units" or "Arbitrary intensity units" since microscopes are not 
callibrated to an absolute scale. Also, it is important to note whether 
you are reporting either the mean or the integrated intensity, so specify
"Mean intensity units" or "Integrated intensity units" accordingly.

See also MeasureImageIntensity.
"""
    variable_revision_number = 2
    category = "Measurement"
    
    def create_settings(self):
        self.module_name = "MeasureObjectIntensity"
        self.image_keys =[uuid.uuid1()]
        self.image_names = [cps.ImageNameSubscriber("What did you call the grayscale images you want to process?","None")]
        self.image_names_remove_buttons = [cps.DoSomething("Remove above image",
                                                           "Remove",
                                                           self.remove_image_cb,
                                                           self.image_keys[0])]
        self.image_name_add_button = cps.DoSomething("Add another image",
                                                     "Add image", 
                                                     self.add_image_cb)
        self.image_divider = cps.Text("Marks end of images list",cps.DO_NOT_USE)
        self.object_keys = [uuid.uuid1()]
        self.object_names = [cps.ObjectNameSubscriber("What did you call the objects that you want to measure?","None")]
        self.object_name_remove_buttons = [cps.DoSomething("Remove this object",
                                                           "Remove",
                                                           self.remove_cb,
                                                           self.object_keys[0])]
        self.object_name_add_button = cps.DoSomething("Add another object","Add object",self.add_cb)
    
    def remove_cb(self, id):
        index = self.object_keys.index(id)
        del self.object_keys[index]
        del self.object_names[index]
        del self.object_name_remove_buttons[index]
    
    def add_cb(self):
        new_uuid = uuid.uuid1()
        self.object_keys.append(new_uuid)
        self.object_names.append(cps.ObjectNameSubscriber("What did you call the objects that you want to measure?","None"))
        self.object_name_remove_buttons.append(cps.DoSomething("Remove this object","Remove",self.remove_cb,new_uuid))

    def remove_image_cb(self, id):
        index = self.image_keys.index(id)
        del self.image_keys[index]
        del self.image_names[index]
        del self.image_names_remove_buttons[index]
    
    def add_image_cb(self):
        new_uuid = uuid.uuid1()
        self.image_keys.append(new_uuid)
        self.image_names.append(cps.ImageNameSubscriber("What did you call the grayscale images you want to process?","None"))
        self.image_names_remove_buttons.append(cps.DoSomething("Remove above image","Remove",self.remove_image_cb, new_uuid))
        
    def visible_settings(self):
        result = []
        for image_name,remove_button in zip(self.image_names,
                                            self.image_names_remove_buttons):
            result.extend((image_name, remove_button))
        result.append(self.image_name_add_button)
        for object_name,remove_button in zip(self.object_names, 
                                             self.object_name_remove_buttons):
            result.extend((object_name,remove_button))
        result.append(self.object_name_add_button)
        return result
    
    def settings(self):
        result = list(self.image_names)
        result.append(self.image_divider)
        result.extend(self.object_names)
        return result

    def set_setting_values(self,setting_values,variable_revision_number,module_name):
        """Adjust the number of object_names according to the number of
        setting values, then call the superclass to set the object names
        to the variable values
        """
        if variable_revision_number ==1:
            raise NotImplementedError("Can't read version # 1")
        if module_name != self.module_class():
            # Old matlab-style. Erase any setting values that are
            # "Do not use"
            new_setting_values = [setting_values[0],cps.DO_NOT_USE]
            for setting_value in setting_values[1:]:
                if setting_value != cps.DO_NOT_USE:
                    new_setting_values.append(setting_value)
            setting_values = new_setting_values
            module_name = self.module_class()
        
        # Count the # of settings up to the first, "DO_NOT_USE"
        for i in range(len(setting_values)):
            if setting_values[i]==cps.DO_NOT_USE:
                break
        
        image_count = i
        while len(self.image_names) > image_count:
            self.remove_image_cb(len(self.image_names)-1)
        while len(self.image_names) < image_count:
            self.add_image_cb()
        
        object_count = len(setting_values)-image_count -1
        while len(self.object_names) > object_count:
            self.remove_cb(len(self.object_names)-1)
        while len(self.object_names) < object_count:
            self.add_cb()
        super(MeasureObjectIntensity,self).set_setting_values(setting_values, 
                                                              variable_revision_number, 
                                                              module_name)

    def get_measurement_columns(self, pipeline):
        '''Return the column definitions for measurements made by this module'''
        columns = []
        for image_name in self.image_names:
            for object_name in self.object_names:
                for feature in (ALL_MEASUREMENTS):
                    columns.append((object_name.value,
                                    "%s_%s_%s"%(INTENSITY, feature,
                                                image_name.value),
                                    cpmeas.COLTYPE_FLOAT))
        return columns
            
    def get_categories(self,pipeline, object_name):
        """Get the categories of measurements supplied for the given object name
        
        pipeline - pipeline being run
        object_name - name of labels in question (or 'Images')
        returns a list of category names
        """
        for object_name_variable in self.object_names:
            if object_name_variable == object_name:
                return [INTENSITY]
        return []
    
    def get_measurements(self, pipeline, object_name, category):
        """Get the measurements made on the given object in the given category"""
        if category != INTENSITY:
            return []
        for object_name_variable in self.object_names:
            if object_name_variable == object_name:
                return ALL_MEASUREMENTS
    
    def get_measurement_images(self, pipeline,object_name, category, measurement):
        """Get the images used to make the given measurement in the given category on the given object"""
        if category != INTENSITY:
            return []
        if measurement not in ALL_MEASUREMENTS:
            return []
        for object_name_variable in self.object_names:
            if object_name_variable == object_name:
                return [image_name.value for image_name in self.image_names]
        return []
    
    def run(self, workspace):
        for image_name in self.image_names:
            image = workspace.image_set.get_image(image_name.value,
                                                  must_be_grayscale=True)
            img = image.pixel_data
            if image.has_mask:
                masked_image = img.copy()
                masked_image[image.mask] = 0
                img = img[image.mask]
            else:
                masked_image = img
            for object_name in self.object_names:
                objects = workspace.object_set.get_objects(object_name.value)
                labels   = objects.segmented
                nobjects = int(np.max(labels))
                outlines = cpmo.outline(labels)
                
                if image.has_mask:
                    masked_labels = labels.copy()
                    masked_labels[image.mask] = 0
                    labels = labels[image.mask]
                    masked_outlines = outlines.copy()
                    masked_outlines[image.mask] = 0
                    outlines = outlines[image.mask]
                else:
                    masked_labels = labels
                    masked_outlines = outlines
                
                if nobjects > 0:
                    lindexes = np.arange(nobjects)+1
                    integrated_intensity = fix(nd.sum(img, labels, lindexes))
                    integrated_intensity_edge = fix(nd.sum(img, outlines,
                                                           lindexes))
                    mean_intensity = fix(nd.mean(img, labels, lindexes))
                    mean_intensity_edge = fix(nd.mean(img, outlines, lindexes))
                    std_intensity = fix(nd.standard_deviation(img, labels, 
                                                              lindexes))
                    std_intensity_edge = fix(nd.standard_deviation(img, outlines, 
                                                                   lindexes))
                    min_intensity = fix(nd.minimum(img, labels, lindexes))
                    min_intensity_edge = fix(nd.minimum(img, outlines,
                                                        lindexes))
                    max_intensity = fix(nd.maximum(img, labels, lindexes))
                    max_intensity_edge = fix(nd.maximum(img, outlines,
                                                        lindexes))
                else:
                    integrated_intensity = np.zeros((0,))
                    integrated_intensity_edge = np.zeros((0,))
                    mean_intensity = np.zeros((0,))
                    mean_intensity_edge = np.zeros((0,))
                    std_intensity = np.zeros((0,))
                    std_intensity_edge = np.zeros((0,))
                    min_intensity = np.zeros((0,))
                    min_intensity_edge = np.zeros((0,))
                    max_intensity = np.zeros((0,))
                    max_intensity_edge = np.zeros((0,))
                    
                # The mass displacement is the distance between the center
                # of mass of the binary image and of the intensity image. The
                # center of mass is the average X or Y for the binary image
                # and the sum of X or Y * intensity / integrated intensity
                if nobjects > 0:
                    mesh_x, mesh_y = np.meshgrid(range(masked_image.shape[1]),
                                                 range(masked_image.shape[0]))
                    cm_x = fix(nd.mean(mesh_x, masked_labels, lindexes))
                    cm_y = fix(nd.mean(mesh_y, masked_labels, lindexes))
                    
                    i_x = fix(nd.sum(mesh_x * masked_image,masked_labels,
                                          lindexes))
                    i_y = fix(nd.sum(mesh_y * masked_image,masked_labels,
                                          lindexes))
                    cmi_x = i_x / integrated_intensity
                    cmi_y = i_y / integrated_intensity
                    diff_x = cm_x - cmi_x
                    diff_y = cm_y - cmi_y
                    mass_displacement = np.sqrt(diff_x * diff_x+diff_y*diff_y)
                else:
                    mass_displacement = np.zeros((0,))
                
                # We do the quantile measurements using an indexing trick:
                # given a label integer L and the intensity at that label, I
                # L+I is a number between L and L+1. So if you add the label
                # matrix to the intensity matrix and sort, you'll get the
                # intensities in order by label, then by magnitude. If you
                # do a cumsum of areas of labels, you'll get indices into
                # the ordered array and you can read out quantiles pretty easily
                
                if nobjects > 0:
                    assert img.min() >= 0,"Algorithm requires image values between 0 and 1"
                    assert img.max() <= 1,"Algorithm requires image values between 0 and 1"
                    areas = fix(nd.sum(np.ones(labels.shape,int),
                                       labels, np.arange(nobjects+1)))
                    areas = areas.astype(int)
                    indices = np.cumsum(areas)[:-1]
                    ordered_image = img + labels.astype(float)
                    ordered_image = ordered_image.flatten()
                    ordered_image.sort()
                    max_indices = (indices + areas[1:] - 1).astype(int)
                    indices_25  = indices+(areas[1:]+2)/4
                    indices_50  = indices+(areas[1:]+1)/2
                    indices_75  = indices+3*(areas[1:]+2) / 4
                    #
                    # Check for round-up overflow
                    #
                    for indices in (indices_25, indices_50, indices_75):
                        imask = indices > max_indices
                        indices[imask] = max_indices[imask]
                    lower_quartile_intensity = (ordered_image[indices_25] - 
                                                range(1,nobjects+1))
                    median_intensity         = (ordered_image[indices_50] - 
                                                range(1,nobjects+1))
                    upper_quartile_intensity = (ordered_image[indices_75] - 
                                                range(1,nobjects+1))
                else:
                    lower_quartile_intensity = np.zeros((0,))
                    median_intensity = np.zeros((0,))
                    upper_quartile_intensity = np.zeros((0,))
                
                m = workspace.measurements
                for feature_name, measurement in \
                    ((INTEGRATED_INTENSITY, integrated_intensity),
                     (MEAN_INTENSITY, mean_intensity),
                     (STD_INTENSITY, std_intensity),
                     (MIN_INTENSITY, min_intensity),
                     (MAX_INTENSITY, max_intensity),
                     (INTEGRATED_INTENSITY_EDGE, integrated_intensity_edge),
                     (MEAN_INTENSITY_EDGE, mean_intensity_edge),
                     (STD_INTENSITY_EDGE, std_intensity_edge),
                     (MIN_INTENSITY_EDGE, min_intensity_edge),
                     (MAX_INTENSITY_EDGE, max_intensity_edge),
                     (MASS_DISPLACEMENT, mass_displacement),
                     (LOWER_QUARTILE_INTENSITY, lower_quartile_intensity),
                     (MEDIAN_INTENSITY, median_intensity),
                     (UPPER_QUARTILE_INTENSITY, upper_quartile_intensity)):
                    measurement_name = "%s_%s_%s"%(INTENSITY,feature_name,
                                                   image_name.value)
                    m.add_measurement(object_name.value,measurement_name, 
                                      measurement)
