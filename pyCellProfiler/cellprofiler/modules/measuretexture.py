"""
Measure texture features for an object.

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

__version__="$Revision: 1 $"

import numpy as np
import scipy.ndimage as scind
import uuid

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix
from cellprofiler.cpmath.haralick import Haralick
from cellprofiler.cpmath.filter import gabor

"""The category of the per-object measurements made by this module"""
TEXTURE = 'Texture'

"""The "name" slot in the object group dictionary entry"""
OG_NAME = 'name'
"""The "remove"slot in the object group dictionary entry"""
OG_REMOVE = 'remove'

F_HARALICK = """AngularSecondMoment Contrast Correlation Variance 
InverseDifferenceMoment SumAverage SumVariance SumEntropy Entropy
DifferenceVariance DifferenceEntropy InfoMeas1 InfoMeas2""".split()

F_GABOR = "Gabor"

class MeasureTexture(cpm.CPModule):
    """SHORT DESCRIPTION
    The MeasureTexture module measures the degree and nature of textures
    within objects using several different metrics.
    *************************************************************
    Haralick Features:
    Haralick texture features are derived from the co-occurrence matrix, 
    which contains information about how image intensities in pixels with a 
    certain position in relation to each other occur together. For example, 
    how often does a pixel with intensity 0.12 have a neighbor 2 pixels to 
    the right with intensity 0.15? The current implementation in CellProfiler
    uses a shift of 1 pixel to the right for calculating the co-occurence 
    matrix. A different set of measurements is obtained for larger shifts, 
    measuring texture on a larger scale. The original reference for the 
    Haralick features is Haralick et al. (1973) Textural Features for Image
    Classification. IEEE Transaction on Systems Man, Cybernetics,
    SMC-3(6):610-621, where 14 features are described:
    H1. Angular Second Moment
    H2. Contrast
    H3. Correlation
    H4. Sum of Squares: Variation
    H5. Inverse Difference Moment
    H6. Sum Average
    H7. Sum Variance
    H8. Sum Entropy
    H9. Entropy
    H10. Difference Variance
    H11. Difference Entropy
    H12. Information Measure of Correlation 1
    H13. Information Measure of Correlation 2
    
    Gabor "wavelet" features:
    These features are similar to wavelet features, and they are obtained by
    applying so-called Gabor filters to the image. The Gabor filters measure
    the frequency content in different orientations. They are very similar to
    wavelets, and in the current context they work exactly as wavelets, but
    they are not wavelets by a strict mathematical definition. The Gabor
    features detect correlated bands of intensities, for instance, images of
    Venetian blinds would have high scores in the horizontal orientation.
    
    MeasureTexture performs the following algorithm to compute a score
    at each scale using the Gabor filter:
    * Divide the half-circle from 0 to 180 degrees by the number of desired
      angles. For instance, if the user choses two angles, MeasureTexture
      uses 0 degrees and 90 degrees (horizontal and vertical) for the filter
      orientations. This is the Theta value from the reference paper.
    * For each angle, compute the Gabor filter for each object in the image
      at two phases separated by 90 degrees in order to account for texture
      features whose peaks fall on even or odd quarter-wavelengths.
    * Multiply the image times each Gabor filter and sum over the pixels
      in each object.
    * Take the square-root of the sum of the squares of the two filter scores.
      This results in one score per Theta.
    * Save the maximum score over all Theta as the score at the desired scale.
    
    The original reference is Gabor, D. (1946). "Theory of communication" 
    Journal of the Institute of Electrical Engineers, 93:429-441.

    Settings:
    Enter at least one image, at least one object and at least one scale. The
    scale is the distance between correlated intensities in the image in 
    pixels. Enter the number of angles to compute for Gabor. The default is
    four which detects bands in the horizontal, vertical and diagonal
    orientations.
    """

    variable_revision_number = 1
    category = 'Measurement'

    def create_settings(self):
        """Create the settings for the module at startup.
        
        The module allows for an unlimited number of measured objects, each
        of which has an entry in self.object_groups.
        """ 
        self.module_name = "MeasureTexture"
        self.image_groups = []
        self.object_groups = []
        self.scale_groups = []
        self.image_count = cps.HiddenCount(self.image_groups)
        self.object_count = cps.HiddenCount(self.object_groups)
        self.scale_count = cps.HiddenCount(self.scale_groups)
        self.add_image_cb()
        self.add_images = cps.DoSomething("Add another image", "Add",
                                          self.add_image_cb)
        self.add_object_cb()
        self.add_objects = cps.DoSomething("Add another object", "Add",
                                           self.add_object_cb)
        self.add_scale_cb()
        self.add_scales = cps.DoSomething("Add another scale", "Add",
                                          self.add_scale_cb)
        self.gabor_angles = cps.Integer("How many angles do you want to use for each Gabor measurement?",4,2)

    def settings(self):
        """The settings as they appear in the save file."""
        result = [self.image_count, self.object_count, self.scale_count]
        for groups in (self.image_groups, self.object_groups, 
                       self.scale_groups):
            for group in groups:
                result += group.settings()
        result += [self.gabor_angles]
        return result

    def backwards_compatibilize(self,setting_values,variable_revision_number,
                                module_name,from_matlab):
        """Adjust the setting_values for older save file versions
        
        setting_values - a list of strings representing the settings for
                         this module.
        variable_revision_number - the variable revision number of the module
                                   that saved the settings
        module_name - the name of the module that saved the settings
        from_matlab - true if it was a Matlab module that saved the settings
        
        returns the modified settings, revision number and "from_matlab" flag
        """
        if from_matlab and variable_revision_number == 2:
            #
            # The first 3 settings are:
            # image count (1 for legacy)
            # object count (calculated)
            # scale_count (calculated)
            #
            object_names = [name for name in setting_values[1:7]
                            if name.upper() != cps.DO_NOT_USE.upper()] 
            scales = setting_values[7].split(',')
            setting_values = ([ "1", str(len(object_names)), str(len(scales)),
                               setting_values[0]] + object_names + scales +
                              ["4"])
            variable_revision_number = 1
            from_matlab = False
        return setting_values, variable_revision_number, from_matlab

    def prepare_to_set_values(self,setting_values):
        """Adjust the number of object groups based on the number of
        setting_values"""
        for count, sequence, fn in\
            ((int(setting_values[0]), self.image_groups, self.add_image_cb),
             (int(setting_values[1]), self.object_groups, self.add_object_cb),
             (int(setting_values[2]), self.scale_groups, self.add_scale_cb)):
            while len(sequence) > count:
                del sequence[count]
            while len(sequence) < count:
                fn()
        
    def visible_settings(self):
        """The settings as they appear in the module viewer"""
        result = []
        for groups, add_button in ((self.image_groups, self.add_images),
                                   (self.object_groups, self.add_objects),
                                   (self.scale_groups, self.add_scales)):
            for group in groups:
                result += group.visible_settings()
            result += [add_button]
        result += [self.gabor_angles]
        return result

    def add_image_cb(self):
        """Add a slot for another image"""
        class ImageSettings(object):
            def __init__(self, sequence):
                self.key = uuid.uuid4()
                def remove(sequence=sequence, key=self.key):
                    index = [x.key for x in sequence].index(key)
                    del sequence[index]
                
                self.image_name = cps.ImageNameSubscriber("What did you call the images you want to measure?","None")
                self.remove_button = cps.DoSomething("Remove the above image",
                                                     "Remove",remove)
            def settings(self):
                return [self.image_name]
            def visible_settings(self):
                return [self.image_name, self.remove_button]
        self.image_groups.append(ImageSettings(self.image_groups))
        
    def add_object_cb(self):
        """Add a slot for another object"""
        class ObjectSettings(object):
            def __init__(self, sequence):
                self.key = uuid.uuid4()
                def remove(sequence=sequence, key=self.key):
                    index = [x.key for x in sequence].index(key)
                    del sequence[index]
                
                self.object_name = cps.ObjectNameSubscriber("What did you call the objects you want to measure?","None")
                self.remove_button = cps.DoSomething("Remove the above objects",
                                                     "Remove",remove)
            def settings(self):
                return [self.object_name]
            def visible_settings(self):
                return [self.object_name, self.remove_button]
            
        self.object_groups.append(ObjectSettings(self.object_groups))

    def add_scale_cb(self):
        '''Add another scale to be measured'''
        class ScaleSettings(object):
            def __init__(self, sequence):
                self.key = uuid.uuid4()
                def remove(sequence=sequence, key=self.key):
                    index = [x.key for x in sequence].index(key)
                    del sequence[index]
                
                self.scale = cps.Integer("What is the scale of the texture?",
                                         len(sequence)+3)
                self.remove_button = cps.DoSomething("Remove the above scale",
                                                     "Remove",remove)
            def settings(self):
                return [self.scale]
            def visible_settings(self):
                return [self.scale, self.remove_button]
            
        self.scale_groups.append(ScaleSettings(self.scale_groups))

    def get_categories(self,pipeline, object_name):
        """Get the measurement categories supplied for the given object name.
        
        pipeline - pipeline being run
        object_name - name of labels in question (or 'Images')
        returns a list of category names
        """
        if any([object_name == og.object_name for og in self.object_groups]):
            return [TEXTURE]
        else:
            return []

    def get_measurements(self, pipeline, object_name, category):
        '''Get the measurements made on the given object in the given category
        
        pipeline - pipeline being run
        object_name - name of objects being measured
        category - measurement category
        '''
        if len(self.get_categories(pipeline, object_name)) > 0:
            return F_HARALICK
        return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        '''Get the list of images measured
        
        pipeline - pipeline being run
        object_name - name of objects being measured
        category - measurement category
        measurement - measurement made on images
        '''
        measurements = self.get_measurements(pipeline, object_name, category)
        if measurement in measurements:
            return [x.image_name.value for x in self.image_groups]
        return []

    def get_measurement_scales(self, pipeline, object_name, category, 
                               measurement, image_name):
        '''Get the list of scales at which the measurement was taken

        pipeline - pipeline being run
        object_name - name of objects being measured
        category - measurement category
        measurement - name of measurement made
        image_name - name of image that was measured
        '''
        if len(self.get_measurement_images(pipeline, object_name, category,
                                           measurement)) > 0:
            return [x.scale.value for x in self.scale_groups]
        return []
    
    def get_measurement_columns(self, pipeline):
        '''Get column names output for each measurement.'''
        cols = []
        for feature in F_HARALICK+[F_GABOR]:
            for im in self.image_groups:
                for sg in self.scale_groups:
                    cols += [('Image',
                              '%s_%s_%s_%d'%(TEXTURE, feature, im.image_name.value, sg.scale.value),
                              'float')]
                   
        for ob in self.object_groups:
            for feature in F_HARALICK+[F_GABOR]:
                for im in self.image_groups:
                    for sg in self.scale_groups:
                        cols += [(ob.object_name.value,
                                  "%s_%s_%s_%d"%(TEXTURE, feature, im.image_name.value, sg.scale.value),
                                  'float')]
        return cols

    def run(self, workspace):
        """Run, computing the area measurements for the objects"""
        
        statistics = [["Image","Object","Measurement","Scale", "Value"]]
        for image_group in self.image_groups:
            image_name = image_group.image_name.value
            for scale_group in self.scale_groups:
                scale = scale_group.scale.value
                statistics += self.run_image(image_name, scale, workspace)
                statistics += self.run_image_gabor(image_name, scale, workspace)
                for object_group in self.object_groups:
                    object_name = object_group.object_name.value
                    statistics += self.run_one(image_name, 
                                               object_name,
                                               scale, workspace)
                    statistics += self.run_one_gabor(image_name, 
                                                     object_name, 
                                                     scale,
                                                     workspace)
        if not workspace.frame is None:
            figure = workspace.create_or_find_figure(subplots=(1,1))
            figure.subplot_table(0,0,statistics,ratio=(.20,.20,.20,.20,.20))
    
    def run_one(self, image_name, object_name, scale, workspace):
        """Run, computing the area measurements for a single map of objects"""
        statistics = []
        image = workspace.image_set.get_image(image_name,
                                              must_be_grayscale=True)
        objects = workspace.get_objects(object_name)
        pixel_data = image.pixel_data
        labels = objects.segmented
        pixel_data = objects.crop_image_similarly(pixel_data)
        for name, value in zip(F_HARALICK, Haralick(pixel_data,
                                                    labels,
                                                    scale).all()):
            statistics += self.record_measurement(workspace, 
                                                  image_name, 
                                                  object_name, 
                                                  scale,
                                                  name, 
                                                  value)
        return statistics

    def run_image(self, image_name, scale, workspace):
        '''Run measurements on image'''
        statistics = []
        image = workspace.image_set.get_image(image_name,
                                              must_be_grayscale=True)
        pixel_data = image.pixel_data
        image_labels = np.ones(pixel_data.shape, int)
        if image.has_mask:
            image_labels[~ image.mask] = 0
        for name, value in zip(F_HARALICK, Haralick(pixel_data,
                                                    image_labels,
                                                    scale).all()):
            statistics += self.record_image_measurement(workspace, 
                                                        image_name, 
                                                        scale,
                                                        name, 
                                                        value)
        return statistics
        
    def run_one_gabor(self, image_name, object_name, scale, workspace):
        objects = workspace.get_objects(object_name)
        labels = objects.segmented
        object_count = np.max(labels)
        if object_count > 0:
            image = workspace.image_set.get_image(image_name,
                                                  must_be_grayscale=True)
            pixel_data = image.pixel_data
            pixel_data = objects.crop_image_similarly(pixel_data)
            best_score = np.zeros((object_count,))
            for angle in range(self.gabor_angles.value):
                theta = np.pi * angle / self.gabor_angles.value
                g = gabor(pixel_data, labels, scale, theta)
                score_r = fix(scind.sum(g.real, labels,
                                         np.arange(object_count)+ 1))
                score_i = fix(scind.sum(g.imag, labels,
                                         np.arange(object_count)+ 1))
                score = np.sqrt(score_r**2+score_i**2)
                best_score = np.maximum(best_score, score)
        else:
            best_score = np.zeros((0,))
        statistics = self.record_measurement(workspace, 
                                             image_name, 
                                             object_name, 
                                             scale,
                                             F_GABOR, 
                                             best_score)
        return statistics
            
    def run_image_gabor(self, image_name, scale, workspace):
        image = workspace.image_set.get_image(image_name,
                                              must_be_grayscale=True)
        pixel_data = image.pixel_data
        labels = np.ones(pixel_data.shape, int)
        if image.has_mask:
            labels[~image.mask] = 0
        best_score = 0
        for angle in range(self.gabor_angles.value):
            theta = np.pi * angle / self.gabor_angles.value
            g = gabor(pixel_data, labels, scale, theta)
            score_r = np.sum(g.real)
            score_i = np.sum(g.imag)
            score = np.sqrt(score_r**2+score_i**2)
            best_score = max(best_score, score)
        statistics = self.record_image_measurement(workspace, 
                                                   image_name, 
                                                   scale,
                                                   F_GABOR, 
                                                   best_score)
        return statistics

    def record_measurement(self, workspace,  
                           image_name, object_name, scale,
                           feature_name, result):
        """Record the result of a measurement in the workspace's
        measurements"""
        data = fix(result)
        data[~np.isfinite(data)] = 0
        workspace.add_measurement(object_name, 
                                  "%s_%s_%s_%d"%
                                  (TEXTURE, feature_name,image_name, scale), 
                                  data)
        statistics = [[image_name, object_name, 
                       "%s %s"%(aggregate_name, feature_name), scale, 
                       "%.2f"%fn(data)]
                       for aggregate_name, fn in (("min",np.min),
                                                  ("max",np.max),
                                                  ("mean",np.mean),
                                                  ("median",np.median),
                                                  ("std dev",np.std))]
        return statistics

    def record_image_measurement(self, workspace,  
                                 image_name, scale,
                                 feature_name, result):
        """Record the result of a measurement in the workspace's
        measurements"""
        if not np.isfinite(result):
            result = 0
        workspace.measurements.add_image_measurement("%s_%s_%s_%d"%
                                                     (TEXTURE, feature_name,
                                                      image_name, scale), 
                                                     result)
        statistics = [[image_name, "-", 
                       feature_name, scale, 
                       "%.2f"%(result)]]
        return statistics
