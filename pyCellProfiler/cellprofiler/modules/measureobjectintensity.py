"""<b>MeasureObjectIntensity</b> - Measures several intensity features for identified objects.
<hr>
Given an image with objects identified (e.g. nuclei or cells), this
module extracts intensity features for each object based on a
corresponding grayscale image. Measurements are recorded for each object.

Retrieves objects in label matrix format and a corresponding original
grayscale image and makes measurements of the objects. The label matrix
image should be "compacted": that is, each number should correspond to an
object, with no numbers skipped. So, if some objects were discarded from
the label matrix image, the image should be converted to binary and
re-made into a label matrix image before feeding it to this module.

Features that can be measured by this module:
<ul><li><i>IntegratedIntensity:</i> The sum of the pixel intensities within an
 object.</li>
<li><i>MeanIntensity:</i> The average pixel intensity within an object.</li>
<li><i>StdIntensity:</i> The standard deviation of the pixel intensities within
 an object.</li>
<li><i>MaxIntensity:</i> The maximal pixel intensity within an object.</li>
<li><i>MinIntensity:</i> The minimal pixel intensity within an object.</li>
<li><i>IntegratedIntensityEdge:</i> The sum of the edge pixel intensities of an
 object.</li>
<li><i>MeanIntensityEdge:</i> The average edge pixel intensity of an object.</li>
<li><i>StdIntensityEdge:</i> The standard deviation of the edge pixel intensities
 of an object.</li>
<li><i>MaxIntensityEdge:</i> The maximal edge pixel intensity of an object.</li>
<li><i>MinIntensityEdge:</i> The minimal edge pixel intensity of an object.</li>
<li><i>MassDisplacement:</i> The distance between the centers of gravity in the
 gray-level representation of the object and the binary representation of
 the object.</li>
<li><i>LowerQuartileIntensity:</i> The intensity value of the pixel for which 25%
 of the pixels in the object have lower values.</li>
<li><i>MedianIntensity:</i> The median intensity value within the object</li>
<li><i>UpperQuartileIntensity:</i> The intensity value of the pixel for which 75%
 of the pixels in the object have lower values.</li></ul>

For publication purposes, it is important to note that the units of
intensity from microscopy images are usually described as "Intensity
units" or "Arbitrary intensity units" since microscopes are not 
callibrated to an absolute scale. Also, it is important to note whether 
you are reporting either the mean or the integrated intensity, so specify
"Mean intensity units" or "Integrated intensity units" accordingly.

See also <b>MeasureImageIntensity</b>.
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
import scipy.ndimage as nd

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
import cellprofiler.cpmath.outline as cpmo
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix
from cellprofiler.cpmath.filter import stretch

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

    module_name = "MeasureObjectIntensity"
    variable_revision_number = 3
    category = "Measurement"
    
    def create_settings(self):
        self.images = []
        self.add_image()
        self.image_count = cps.HiddenCount(self.images)
        self.add_image_button = cps.DoSomething("", "Add image", self.add_image)
        self.divider = cps.Divider()
        self.objects = []
        self.add_object()
        self.add_object_button = cps.DoSomething("", "Add object", self.add_object)

    def add_image(self):
        group = cps.SettingsGroup()
        group.append("name", cps.ImageNameSubscriber("Select input image:","None", doc = 
                                                     """What did you call the grayscale images you want to process?"""))
        group.append("remover", cps.RemoveSettingButton("", "Remove above image", self.images, group))
        self.images.append(group)

    def add_object(self):
        group = cps.SettingsGroup()
        group.append("name", cps.ObjectNameSubscriber("Select objects to measure:","None", doc = 
                                                          """What did you call the objects that you want to measure?"""))
        group.append("remover", cps.RemoveSettingButton("", "Remove above image", self.images, group))
        self.objects.append(group)

    def settings(self):
        result = [self.image_count]
        result += [im.name for im in self.images]
        result += [obj.name for obj in self.objects]
        return result

    def visible_settings(self):
        result = []
        for im in self.images:
            result += im.unpack_group()
        result += [self.add_image_button, self.divider]
        for im in self.objects:
            result += im.unpack_group()
        result += [self.add_object_button]
        return result
        
    def upgrade_settings(self,setting_values,variable_revision_number,
                         module_name,from_matlab):
        '''Adjust setting values if they came from a previous revision
        
        setting_values - a sequence of strings representing the settings
                         for the module as stored in the pipeline
        variable_revision_number - the variable revision number of the
                         module at the time the pipeline was saved. Use this
                         to determine how the incoming setting values map
                         to those of the current module version.
        module_name - the name of the module that did the saving. This can be
                      used to import the settings from another module if
                      that module was merged into the current module
        from_matlab - True if the settings came from a Matlab pipeline, False
                      if the settings are from a CellProfiler 2.0 pipeline.
        
        Overriding modules should return a tuple of setting_values,
        variable_revision_number and True if upgraded to CP 2.0, otherwise
        they should leave things as-is so that the caller can report
        an error.
        '''
        if from_matlab and variable_revision_number == 2:
            # Old matlab-style. Erase any setting values that are
            # "Do not use"
            new_setting_values = [setting_values[0],cps.DO_NOT_USE]
            for setting_value in setting_values[1:]:
                if setting_value != cps.DO_NOT_USE:
                    new_setting_values.append(setting_value)
            setting_values = new_setting_values
            from_matlab = False
            variable_revision_number = 2
        if variable_revision_number == 2:
            assert not from_matlab
            num_imgs = setting_values.index(cps.DO_NOT_USE)
            setting_values = [str(num_imgs)] + setting_values[:num_imgs] + setting_values[num_imgs+1:]
            variable_revision_number = 3
        return setting_values, variable_revision_number, from_matlab

    def prepare_settings(self,setting_values):
        """Do any sort of adjustment to the settings required for the given values
        
        setting_values - the values for the settings just prior to mapping
                         as done by set_settings_from_values
        This method allows a module to specialize itself according to
        the number of settings and their value. For instance, a module that
        takes a variable number of images or objects can increase or decrease
        the number of relevant settings so they map correctly to the values.
        
        See cellprofiler.modules.measureobjectareashape for an example.
        """
        #
        # The settings have two parts - images, then objects
        # The parts are divided by the string, cps.DO_NOT_USE
        #
        image_count = int(setting_values[0])
        object_count = len(setting_values) - image_count - 1
        del self.images[image_count:]
        while len(self.images) < image_count:
            self.add_image()
        del self.objects[object_count:]
        while len(self.objects) < object_count:
            self.add_object()

    def get_measurement_columns(self, pipeline):
        '''Return the column definitions for measurements made by this module'''
        columns = []
        for image_name in [im.name for im in self.images]:
            for object_name in [obj.name for obj in self.objects]:
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
        for object_name_variable in [obj.name for obj in self.objects]:
            if object_name_variable.value == object_name:
                return [INTENSITY]
        return []
    
    def get_measurements(self, pipeline, object_name, category):
        """Get the measurements made on the given object in the given category"""
        if category != INTENSITY:
            return []
        for object_name_variable in [obj.name for obj in self.objects]:
            if object_name_variable.value == object_name:
                return ALL_MEASUREMENTS
        return []
    
    def get_measurement_images(self, pipeline,object_name, category, measurement):
        """Get the images used to make the given measurement in the given category on the given object"""
        if category != INTENSITY:
            return []
        if measurement not in ALL_MEASUREMENTS:
            return []
        for object_name_variable in [obj.name for obj in self.objects]:
            if object_name_variable == object_name:
                return [image.name.value for image in self.images]
        return []
    
    def is_interactive(self):
        return False
    
    def run(self, workspace):
        if workspace.frame is not None:
            statistics = [("Image","Object","Feature","Mean","Median","STD")]
            workspace.display_data.statistics = statistics
        for image_name in [img.name for img in self.images]:
            image = workspace.image_set.get_image(image_name.value,
                                                  must_be_grayscale=True)
            img = image.pixel_data
            if image.has_mask:
                masked_image = img.copy()
                masked_image[image.mask] = 0
                img = img[image.mask]
            else:
                masked_image = img
            for object_name in [obj.name for obj in self.objects]:
                objects = workspace.object_set.get_objects(object_name.value)
                labels   = objects.segmented
                nobjects = np.int32(np.max(labels))
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
                    stretched_img = stretch(img) * .99
                    flat_img = img.flatten()
                    areas = fix(nd.sum(np.ones(labels.shape,int),
                                       labels, np.arange(nobjects+1)))
                    areas = areas.astype(int)
                    indices = np.cumsum(areas)[:-1]
                    ordered_image = stretched_img + labels.astype(float)
                    ordered_image = ordered_image.flatten()
                    image_idx = np.argsort(ordered_image)
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
                    lower_quartile_intensity = flat_img[image_idx[indices_25]]
                    median_intensity         = flat_img[image_idx[indices_50]]
                    upper_quartile_intensity = flat_img[image_idx[indices_75]]
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
                    if workspace.frame is not None and len(measurement) > 0:
                        statistics.append((image_name.value, object_name.value, 
                                           feature_name,
                                           np.round(np.mean(measurement),3),
                                           np.round(np.median(measurement),3),
                                           np.round(np.std(measurement),3)))
        
    def display(self, workspace):
        figure = workspace.create_or_find_figure(subplots=(1,1))
        figure.subplot_table(0,0,workspace.display_data.statistics,
                             ratio=(.2,.2,.3,.1,.1,.1))
