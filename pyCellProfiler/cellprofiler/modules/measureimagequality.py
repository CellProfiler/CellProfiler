'''measureimagequality.py - Measurements of saturation and blur

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

__version__="$Revision$"

import numpy as np
import scipy.ndimage as scind
import uuid

from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
import cellprofiler.cpmath.threshold as cpthresh

IMAGE_QUALITY = 'ImageQuality'
FOCUS_SCORE = 'FocusScore'
LOCAL_FOCUS_SCORE = 'LocalFocusScore'
PERCENT_SATURATION = 'PercentSaturation'
PERCENT_MAXIMAL = 'PercentMaximal'
THRESHOLD = 'Threshold'
MEAN_THRESH_ALL_IMAGES = 'MeanThresh_AllImages'
MEDIAN_THRESH_ALL_IMAGES = 'MedianThresh_AllImages'
STD_THRESH_ALL_IMAGES = 'StdThresh_AllImages'
SETTINGS_PER_GROUP = 7
class MeasureImageQuality(cpm.CPModule):
    '''SHORT DESCRIPTION:
Measures features that indicate image quality.
*************************************************************************

This module measures features that indicate image quality. This includes the
percentage of pixels in the image that are saturated. Measurements of blur 
(poor focus) are also calculated.

Features measured:   
FocusScore           a measure of the intensity variance across image
LocalFocusScore      a measure of the intensity variance between image parts
PercentSaturation    percent of pixels with a value of 1
PercentMaximal       percent of pixels at the maximum intensity value
Threshold            calculated threshold for image

More about Saturation and PercentMaximal: Saturation means that 
the pixel's intensity value is equal to the maximum possible intensity value 
for that image type). Sometimes images have undergone some kind of transform-
ation such that no pixels ever reach the maximum possible intensity value of 
the image type. For this reason, the percentage of pixels at that *individual*
image's maximum intensity value is also calculated. Given noise in images, 
this should typically be a low percentage but if the images were saturated
during imaging, a higher than usual PercentMaximal will be observed.

More about Blur: The module can also measure blur by calculating a focus score
(higher = better focus). This calculation is slow, so it is optional. The score 
is calculated using the normalized variance. We used this algorithm because it
was ranked best in this paper:
Sun, Y., Duthaler, S., Nelson, B. "Autofocusing in Computer Microscopy:
   Selecting the optimals focus algorithm." Microscopy Research and
   Technique 65:139-149 (2004)

The calculation of the focus score is as follows:
[m,n] = size(Image);
MeanImageValue = mean(Image(:));
SquaredNormalizedImage = (Image-MeanImageValue).^2;
FocusScore{ImageNumber} = ...
   sum(SquaredNormalizedImage(:))/(m*n*MeanImageValue);

The above score is to measure a relative score given a focus setting of 
a certain microscope. Using this, one can calibrate the microscope's
focus setting. However it doesn't necessarily tell you how well an image
was focused when taken. That means these scores obtained from many different
images probably taken in different situations and with different cell
contents can not be used for focus comparison.

The LocalFocusScore is a local version of the original FocusScore. 
LocalFocusScore was just named after the original one to be consistent 
with naming. Note that these focus scores do not necessarily 
represent the qualities of focusing between different images. 
LocalFocusScore was added to differentiate good segmentation and bad 
segmentation images in the cases when bad segmentation images usually 
contain no cell objects with high background noise.

Example Output:

Percent of pixels that are Saturated:
RescaledOrig:     0.002763

Percent of pixels that are in the Maximal
Intensity:
RescaledOrig:     0.0002763


Focus Score:
RescaledOrig: 0.016144

Suggested Threshold:
Orig: 0.0022854

Settings:
You may specify any number of images by using the "Add" button to add
more images to the list. Each image can have its focus, saturation and/or
threshold measured.

The local focus score measures the variance between areas of the image by
dividing the image up into windows. If you choose to calculate blur / focus,
you'll have an opportunity to set the window size - the window size should
be twice the maximum size of the objects you are trying to segment. You
can measure the local focus score over multiple windows by adding an image
to the list more than once and by setting different window sizes for
each image.

If you choose to measure the threshold, you will be able to select the
thresholding method from a list. Please see IdentifyPrimAutomatic for
a description of thresholding methods. Only the global threshold is calculated.
The mixture of Gaussians method uses the expected fraction of pixels that
are foreground to determine whether to assign the pixels in the middle
Gaussian distribution to foreground or background. You'll be asked to specify
fraction in the question, "What fraction of the image is composed of objects?" 
'''

    category = "Measurement"
    variable_revision_number = 1

    def create_settings(self):
        self.module_name = "MeasureImageQuality"
        self.image_groups = []
        self.add_image_group()
        self.add_button = cps.DoSomething("Add another image for measurement:",
                                          "Add", self.add_image_group)
    
    def add_image_group(self):
        class ImageGroup(object):
            def __init__(self, image_groups):
                self.__key = uuid.uuid4() 
                self.__image_name = cps.ImageNameSubscriber("What did you call the grayscale images whose quality you want to measure?","None")
                self.__check_blur = cps.Binary("Would you like to check for blur?",
                                               True)
                self.__window_size = cps.Integer("The local focus score is measured within an NxN pixel window applied to the image. What value of N would you like to use? A suggested value is twice the typical object diameter.",
                                                 20, minval =1)
                self.__check_saturation = cps.Binary("Would you like to check for saturation?",
                                                     True)
                self.__calculate_threshold = cps.Binary("Would you like to calculate a suggested threshold?",
                                                        True)
                self.__threshold_method = cps.Choice("What thresholding method would you like to use?",
                                                     cpthresh.TM_GLOBAL_METHODS,
                                                     cpthresh.TM_OTSU_GLOBAL)
                self.__object_fraction = cps.Float("What fraction of the image is composed of objects?",
                                                   0.1,0,1)
                self.__remove_button = cps.DoSomething("Remove this image:",
                                                       "Remove",
                                                       self.remove, 
                                                       image_groups)
            def str(self):
                return "ImageGroup: %s, key=%s"%(self.image_name.value, 
                                                 str(self.key))
            @property
            def key(self):
                '''The unique key for this image group'''
                return self.__key
            
            @property
            def image_name(self):
                '''The name setting of the image to be measured'''
                return self.__image_name
            
            @property
            def check_blur(self):
                '''The setting for turning blur checking on and off'''
                return self.__check_blur
            
            @property
            def window_size(self):
                '''The setting for the focus score window dimensions'''
                return self.__window_size
            
            @property
            def check_saturation(self):
                '''The setting for turning saturation checking on and off'''
                return self.__check_saturation
            
            @property
            def calculate_threshold(self):
                '''The setting for turning threshold calculation on and off'''
                return self.__calculate_threshold
            
            @property
            def threshold_method(self):
                '''The setting for choosing the threshold method'''
                return self.__threshold_method
            
            @property
            def threshold_algorithm(self):
                '''The thresholding algorithm to run'''
                return self.threshold_method.value.split(' ')[0]
            
            @property
            def threshold_feature_name(self):
                '''The feature name of the threshold measurement generated'''
                return "%s_%s%s_%s"%(IMAGE_QUALITY, THRESHOLD, 
                                     self.threshold_algorithm,
                                     self.image_name.value)
                    
            @property
            def object_fraction(self):
                '''The setting for specifying the amount of foreground pixels'''
                return self.__object_fraction
             
            def remove(self, image_groups):
                '''Remove ourself from the list of image groups'''
                index = [x.key for x in image_groups].index(self.key)
                assert index != -1, "%s is no longer present in its list"%self
                del image_groups[index]
            
            def settings(self):
                '''The settings in the order that they are loaded and saved'''
                return [self.image_name, self.check_blur, self.window_size,
                        self.check_saturation, self.calculate_threshold,
                        self.threshold_method, self.object_fraction]
            
            def visible_settings(self):
                '''The settings as displayed to the user'''
                result = [self.image_name, self.check_blur ]
                if self.check_blur.value:
                    result.append(self.window_size)
                result += [self.check_saturation, self.calculate_threshold]
                if self.calculate_threshold.value:
                    result.append(self.threshold_method)
                    if self.threshold_method == cpthresh.TM_MOG_GLOBAL:
                        result.append(self.object_fraction)
                result.append(self.__remove_button)
                return result
        image_group = ImageGroup(self.image_groups)
        self.image_groups.append(image_group)

    def backwards_compatibilize(self, setting_values, variable_revision_number, 
                                module_name, from_matlab):
        '''Upgrade from previous versions of setting formats'''
        
        if from_matlab and variable_revision_number == 1:
            # Slot 0 asked if blur should be checked on all images
            # Slot 1 had the window size for all images
            # Slots 2-4, 5-7, 8-10, 11-13 contain triples of:
            # image name for blur and saturation
            # image name for threshold calculation
            # threshold method
            #
            # So here, we save the answer to the blur question and 
            # the window size and apply those to every image. We
            # collect images in dictionaries that tell how the image
            # should be checked.
            #
            d = {}
            check_blur = setting_values[0]
            window_size = setting_values[1]
            for i in range(2,14,3):
                saturation_image = setting_values[i]
                threshold_image = setting_values[i+1]
                threshold_method = setting_values[i+2]
                if saturation_image != cps.DO_NOT_USE: 
                    if not d.has_key(saturation_image):
                        d[saturation_image] = {"check_blur":check_blur,
                                               "check_saturation":cps.YES,
                                               "check_threshold":cps.NO,
                                               "threshold_method":threshold_method}
                    else:
                        d[saturation_image]["check_blur"] = check_blur
                        d[saturation_image]["check_saturation"] = cps.YES
                if threshold_image != cps.DO_NOT_USE:
                    if not d.has_key(threshold_image):
                        d[threshold_image] = {"check_blur":cps.NO,
                                               "check_saturation":cps.NO,
                                               "check_threshold":cps.YES,
                                               "threshold_method":threshold_method}
                    else:
                        d[threshold_image]["check_threshold"] = cps.YES
                        d[threshold_image]["threshold_method"]= threshold_method
            setting_values = []
            for image_name in d.keys():
                dd = d[image_name]
                setting_values += [image_name, dd["check_blur"], window_size,
                                   dd["check_saturation"], 
                                   dd["check_threshold"],
                                   dd["threshold_method"],
                                   ".10"]
            from_matlab = False
            variable_revision_number = 1
        return setting_values, variable_revision_number, from_matlab

    def prepare_to_set_values(self, setting_values):
        '''Adjust self.image_groups to account for the expected # of images'''
        assert len(setting_values) % SETTINGS_PER_GROUP == 0
        group_count = len(setting_values) / SETTINGS_PER_GROUP
        while len(self.image_groups) > group_count:
            self.image_groups[-1].remove()
        while len(self.image_groups) < group_count:
            self.add_image_group()

    def settings(self):
        '''The settings in the save / load order'''
        result = []
        for image_group in self.image_groups:
            result += image_group.settings()
        return result

    def visible_settings(self):
        '''The settings as displayed to the user'''
        result = []
        for image_group in self.image_groups:
            result += image_group.visible_settings()
        result.append(self.add_button)
        return result

    def test_valid(self, pipeline):
        '''Make sure settings are compatible
        
        In particular, we make sure that no measurements are duplicated
        '''
        d = {}
        for image_group in self.image_groups:
            image_name = image_group.image_name.value
            if not d.has_key(image_name):
                d[image_name] = {}
            dd = d[image_name]
            for key_name, value in \
                (("check_blur_%d"%image_group.window_size.value, image_group.check_blur.value),
                 ("check_saturation", image_group.check_saturation.value),
                 ("calculate_threshold_%s", (image_group.calculate_threshold.value,
                                             image_group.threshold_algorithm))):
                if value:
                    if dd.has_key(key_name):
                        raise cps.ValidationError("%s image specified twice"%
                                                  image_name, 
                                                  image_group.image_name)
                else:
                    dd[key_name] = True
                    
        return cpm.CPModule.test_valid(self, pipeline)

    @property
    def any_threshold(self):
        '''True if some image has its threshold calculated'''
        return any([ig.calculate_threshold.value 
                    for ig in self.image_groups])
    
    @property
    def any_saturation(self):
        '''True if some image has its saturation calculated'''
        return any([ig.check_saturation.value
                     for ig in self.image_groups])
    
    @property
    def any_blur(self):
        '''True if some image has its blur calculated'''
        return any([ig.check_blur.value
                    for ig in self.image_groups])
    
    def get_measurement_columns(self):
        '''Return column definitions for all measurements'''
        columns = []
        for ig in self.image_groups:
            if ig.check_blur.value:
                for feature in (FOCUS_SCORE,LOCAL_FOCUS_SCORE):
                    columns.append((cpmeas.IMAGE,
                                    '%s_%s_%s_%d'%(IMAGE_QUALITY, feature,
                                                   ig.image_name.value,
                                                   ig.window_size.value),
                                    cpmeas.COLTYPE_FLOAT))
            if ig.check_saturation.value:
                for feature in (PERCENT_SATURATION, PERCENT_MAXIMAL):
                    columns.append((cpmeas.IMAGE,
                                    '%s_%s_%s'%(IMAGE_QUALITY, feature,
                                                ig.image_name.value),
                                    cpmeas.COLTYPE_FLOAT))
            if ig.calculate_threshold.value:
                feature = ig.threshold_feature_name
                columns.append((cpmeas.IMAGE, feature, cpmeas.COLTYPE_FLOAT))
        return columns
            
    def get_categories(self, pipeline, object_name):
        if object_name == cpmeas.IMAGE:
            return [IMAGE_QUALITY]
        elif (object_name == cpmeas.EXPERIMENT and self.any_threshold):
            return [IMAGE_QUALITY]
        return [] 

    def get_measurements(self, pipeline, object_name, category):
        if object_name == cpmeas.IMAGE and category == IMAGE_QUALITY:
            result = []
            if self.any_blur:
                result += [FOCUS_SCORE, LOCAL_FOCUS_SCORE]
            if self.any_saturation:
                result += [PERCENT_SATURATION, PERCENT_MAXIMAL]
            thresholds = set([THRESHOLD+ig.threshold_algorithm 
                              for ig in self.image_groups
                              if ig.calculate_threshold.value])
            thresholds = list(thresholds)
            thresholds.sort()
            result += thresholds
            return result
        elif object_name == cpmeas.EXPERIMENT and category == IMAGE_QUALITY:
            return [MEAN_THRESH_ALL_IMAGES, MEDIAN_THRESH_ALL_IMAGES,
                    STD_THRESH_ALL_IMAGES]
        return []

    def get_measurement_images(self, pipeline, object_name, category, 
                               measurement):
        if object_name != cpmeas.IMAGE or category != IMAGE_QUALITY:
            return []
        if measurement in (FOCUS_SCORE, LOCAL_FOCUS_SCORE):
            return [ig.image_name.value for ig in self.image_groups
                    if ig.check_blur.value]
        if measurement in (PERCENT_MAXIMAL, PERCENT_SATURATION):
            return [ig.image_name.value for ig in self.image_groups
                    if ig.check_saturation.value]
        if measurement.startswith(THRESHOLD):
            return [ig.image_name.value for ig in self.image_groups
                    if (ig.calculate_threshold.value and
                        measurement == THRESHOLD+ig.threshold_algorithm)]
    
    def get_measurement_scales(self, pipeline, object_name, category, 
                               measurement, image_name):
        '''Get the scales (window_sizes) for the given measurement'''
        if (object_name == cpmeas.IMAGE and
            category == IMAGE_QUALITY and
            measurement in (FOCUS_SCORE, LOCAL_FOCUS_SCORE)):
            return [ig.window_size for ig in self.image_groups
                    if ig.image_name == image_name]
        return []

    def run(self, workspace):
        '''Calculate statistics over all image groups'''
        statistics = []
        for image_group in self.image_groups:
            statistics += self.run_on_image_group(image_group, workspace)

        if not workspace.frame is None:
            figure = workspace.create_or_find_figure(subplots=(1,1))
            figure.subplot_table(0,0,statistics)
    
    def post_run(self, workspace):
        '''Calculate the experiment statistics at the end of a run'''
        statistics = []
        for image_group in self.image_groups:
            statistics += self.calculate_experiment_threshold(image_group, 
                                                              workspace)
        if not workspace.frame is None:
            figure = workspace.create_or_find_figure(subplots=(1,1))
            figure.subplot_table(0,0,statistics)
    
    def run_on_image_group(self, image_group, workspace):
        '''Calculate statistics for a particular image'''
        statistics = []
        if image_group.check_blur.value:
            statistics += self.calculate_image_blur(image_group, workspace)
        if image_group.check_saturation.value:
            statistics += self.calculate_saturation(image_group, workspace)
        if image_group.calculate_threshold.value:
            statistics += self.calculate_threshold(image_group, workspace)
        return statistics
    
    def calculate_image_blur(self, image_group, workspace):
        '''Calculate a local blur measurement and a image-wide one
        
        '''
        window_size = image_group.window_size.value
        image = workspace.image_set.get_image(image_group.image_name.value,
                                              must_be_grayscale = True)
        pixel_data = image.pixel_data
        shape = image.pixel_data.shape
        if image.has_mask:
            pixel_data = pixel_data[image.mask]
        focus_score = 0
        if len(pixel_data):
            mean_image_value = np.mean(pixel_data)
            squared_normalized_image = (pixel_data - mean_image_value)**2
            if mean_image_value > 0:
                focus_score = (np.sum(squared_normalized_image) /
                               (np.product(pixel_data.shape) * mean_image_value))
        #
        # Create a labels matrix that grids the image to the dimensions
        # of the window size
        #
        i,j = np.mgrid[0:shape[0],0:shape[1]].astype(float)
        m,n = (np.array(shape) + window_size - 1)/window_size
        i = (i * float(m) / float(shape[0])).astype(int)
        j = (j * float(n) / float(shape[1])).astype(int)
        grid = i * m + j + 1
        if image.has_mask:
            grid[np.logical_not(image.mask)] = 0
        grid_range = np.arange(0, m*n+1)
        #
        # Do the math per label
        #
        local_means = fix(scind.mean(image.pixel_data, grid, grid_range))
        local_squared_normalized_image = (image.pixel_data -
                                          local_means[grid])**2
        #
        # Compute the sum of local_squared_normalized_image values for each
        # grid for means > 0. Exclude grid label = 0 because that's masked
        #
        nz_grid_range = grid_range[local_means != 0]
        if len(nz_grid_range) and nz_grid_range[0] == 0:
            nz_grid_range = nz_grid_range[1:]
            local_means = local_means[1:]
        local_focus_score = 0 # assume the worst - that we can't calculate it
        if len(nz_grid_range):
            sums = fix(scind.sum(local_squared_normalized_image, grid, 
                                 nz_grid_range)) 
            pixel_counts = fix(scind.sum(np.ones(shape), grid, nz_grid_range))
            local_norm_var = (sums / 
                              (pixel_counts * local_means[local_means != 0]))
            local_norm_median = np.median(local_norm_var)
            if np.isfinite(local_norm_median) and local_norm_median > 0:
                local_focus_score = np.var(local_norm_var) / local_norm_median
        #
        # Add the measurements
        #
        focus_score_name = "%s_%s_%s_%d"%(IMAGE_QUALITY,FOCUS_SCORE,
                                          image_group.image_name.value,
                                          window_size)
        workspace.add_measurement(cpmeas.IMAGE, focus_score_name,
                                  focus_score)
        
        local_focus_score_name = "%s_%s_%s_%d"%(IMAGE_QUALITY,
                                                LOCAL_FOCUS_SCORE,
                                                image_group.image_name.value,
                                                window_size)
        workspace.add_measurement(cpmeas.IMAGE, local_focus_score_name,
                                  local_focus_score)
        return [["%s focus score @%d"%(image_group.image_name.value,
                                       window_size), focus_score],
                ["%s local focus score @%d"%(image_group.image_name.value,
                                             window_size), local_focus_score]]
    
    def calculate_saturation(self, image_group, workspace):
        '''Count the # of pixels at saturation'''
        image_name = image_group.image_name.value
        image = workspace.image_set.get_image(image_name,
                                              must_be_grayscale = True)
        pixel_data = image.pixel_data
        if image.has_mask:
            pixel_data = pixel_data[image.mask]
        pixel_count = np.product(pixel_data.shape)
        if pixel_count == 0:
            percent_saturation = 0
            percent_maximal = 0
        else:
            number_pixels_saturated = np.sum(pixel_data == 1)
            number_pixels_maximal = np.sum(pixel_data == np.max(pixel_data))
            percent_saturation = (100.0 * float(number_pixels_saturated) /
                                  float(pixel_count))
            percent_maximal = (100.0 * float(number_pixels_maximal) /
                               float(pixel_count))
        percent_saturation_name = "%s_%s_%s"%(IMAGE_QUALITY, PERCENT_SATURATION,
                                              image_name)
        percent_maximal_name = "%s_%s_%s"%(IMAGE_QUALITY, PERCENT_MAXIMAL,
                                           image_name)
        workspace.add_measurement(cpmeas.IMAGE, percent_saturation_name,
                                  percent_saturation)
        workspace.add_measurement(cpmeas.IMAGE, percent_maximal_name,
                                  percent_maximal)
        return [["%s saturation"%image_name,"%.1f %%"%percent_saturation],
                ["%s maximal"%image_name, "%.1f %%"%percent_maximal]]
    
    def calculate_threshold(self, image_group, workspace):
        '''Calculate a threshold for this image'''
        image_name = image_group.image_name.value
        image = workspace.image_set.get_image(image_name,
                                              must_be_grayscale = True)
        threshold_method = image_group.threshold_algorithm
        object_fraction = image_group.object_fraction.value
        (local_threshold, global_threshold) = \
            (cpthresh.get_threshold(threshold_method,
                                    cpthresh.TM_GLOBAL,
                                    image.pixel_data,
                                    mask = image.mask,
                                    object_fraction = object_fraction)
             if image.has_mask
             else
             cpthresh.get_threshold(threshold_method,
                                    cpthresh.TM_GLOBAL,
                                    image.pixel_data,
                                    object_fraction = object_fraction))
        threshold_name = image_group.threshold_feature_name
        workspace.add_measurement(cpmeas.IMAGE, threshold_name,
                                  global_threshold)
        return [["%s %s threshold"%(image_name, threshold_method), 
                 str(global_threshold)]]
    
    def calculate_experiment_threshold(self, image_group, workspace):
        '''Calculate experiment-wide threshold mean, median and standard-deviation'''
        m = workspace.measurements
        statistics = []
        if image_group.calculate_threshold.value:
            image_name = image_group.image_name.value
            values = m.get_all_measurements(cpmeas.IMAGE, 
                                            image_group.threshold_feature_name)
            feature_name = "%s_Mean%s%s_%s"%(IMAGE_QUALITY,THRESHOLD,
                                             image_group.threshold_algorithm,
                                             image_name)
            mean_threshold = np.mean(values)
            m.add_experiment_measurement(feature_name, mean_threshold)
            statistics.append(["Mean %s %s threshold"%(image_name,
                                                       image_group.threshold_algorithm),
                               str(mean_threshold)])
            feature_name = "%s_Median%s%s_%s"%(IMAGE_QUALITY,THRESHOLD,
                                               image_group.threshold_algorithm,
                                               image_name)
            median_threshold = np.median(values)
            m.add_experiment_measurement(feature_name, median_threshold)
            statistics.append(["Median %s %s threshold"%(image_name,
                                                         image_group.threshold_algorithm),
                               str(median_threshold)])
            feature_name = "%s_Std%s%s_%s"%(IMAGE_QUALITY,THRESHOLD,
                                            image_group.threshold_algorithm,
                                            image_name)
            std_threshold = np.std(values)
            m.add_experiment_measurement(feature_name, std_threshold)
            statistics.append(["Std. dev. %s %s threshold"%(image_name,
                                                            image_group.threshold_algorithm),
                               str(std_threshold)])
        return statistics
