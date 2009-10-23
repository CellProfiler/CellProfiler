'''MeasureImageQuality: This module measures features that indicate image quality. This includes the
percentage of pixels in the image that are saturated. Measurements of blur 
(poor focus) are also calculated.
<hr>

Features measured:   
<ul><li>FocusScore: a measure of the intensity variance across image</li>
<li>LocalFocusScore: a measure of the intensity variance between image parts</li>
<li>PercentSaturation: percent of pixels with a value of 1</li>
<li>PercentMaximal: percent of pixels at the maximum intensity value</li>
<li>Threshold: calculated threshold for image</li></ul>

<h3>Saturation and PercentMaximal</h3> Saturation means that 
the pixel's intensity value is equal to the maximum possible intensity value 
for that image type. Sometimes images have undergone some kind of transformation
 such that no pixels ever reach the maximum possible intensity value of 
the image type. For this reason, the percentage of pixels at that <i>individual</i>
image's maximum intensity value is also calculated. Given noise in images, 
this should typically be a low percentage but if the images were saturated
during imaging, a higher than usual PercentMaximal will be observed.

<h3>Focus Score (Blur)</h3> The module can also measure blur by calculating a focus score
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

<h3>Local Focus Score</h3>
The LocalFocusScore is a local version of the original FocusScore. 
LocalFocusScore was just named after the original one to be consistent 
with naming. Note that these focus scores do not necessarily 
represent the qualities of focusing between different images. 
LocalFocusScore was added to differentiate good segmentation and bad 
segmentation images in the cases when bad segmentation images usually 
contain no cell objects with high background noise.

Example Output:
<table border="1">
<tr>
<td>Percent of pixels that are Saturated:</td>
<td>RescaledOrig: </td>
<td>0.002763</td>
</tr>
<tr>
<td>Percent of pixels that are in the Maximal Intensity:</td>
<td>RescaledOrig: </td>
<td>0.0002763</td>
</tr>
<tr>
<td>Focus Score:</td>
<td>RescaledOrig:</td>
<td> 0.016144</td>
</tr>
<tr>
<td>Suggested Threshold:</td>
<td>Orig: </td>
<td>0.0022854</td>
</tr>
</table>'''

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
import scipy.ndimage as scind

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
    module_name = "MeasureImageQuality"
    category = "Measurement"
    variable_revision_number = 1

    def create_settings(self):
        self.image_groups = []
        self.add_image_group()
        self.bottom_spacer = cps.Divider(line=False)
        self.add_button = cps.DoSomething("Add another image for measurement:",
                                          "Add", self.add_image_group)
    
    def add_image_group(self):
        group = MeasureImageQualitySettingsGroup() # helper class defined below
        group.append("image_name", cps.ImageNameSubscriber("Select an image to measure","None", 
                                                           doc = '''What did you call the grayscale images whose quality you want to measure?'''))
        group.append("check_blur", cps.Binary("Check for blur:",
                                              True, 
                                              doc = '''Would you like to check for blur? Blur is measured by calculating a focus score
                                                  (higher = better focus).'''))
        group.append("window_size", cps.Integer("Window Size:",
                                                20, minval =1,
                                                doc = '''The local focus score is measured within an NxN pixel window 
                                                  applied to the image. What value of N would you like to use? A suggested 
                                                  value is twice the typical object diameter. You
                                                  can measure the local focus score over multiple windows by adding an image
                                                  to the list more than once and by setting different window sizes for
                                                  each image.'''))
        group.append("check_saturation", cps.Binary("Check for saturation:",
                                                    True, doc = '''Would you like to check for saturation?'''))
        group.append("calculate_threshold", cps.Binary("Calculate threshold:",
                                                       True, doc = '''Would you like to calculate a suggested threshold?'''))
        group.append("threshold_method", cps.Choice("Select a thresholding method:",
                                                    cpthresh.TM_GLOBAL_METHODS,
                                                    cpthresh.TM_OTSU_GLOBAL, 
                                                    doc = '''This setting allows you to access the same automatic thresholding 
                                                       methods used in the <b>Identify</b> modules.  You may select any of these automatic thresholding 
                                                       methods, or choose "Manual" to enter a threshold manually.  To choose a binary image, select "Binary image". 
                                                       The output of <b>MeasureImageQuality</b> will be a numerical threshold, rather than objects.  
                                                       For more help on thresholding, see the Identify modules.'''))
        group.append("object_fraction", cps.Float("What fraction of the image is composed of objects?", 0.1,0,1))
        group.append("remove_button", cps.RemoveSettingButton("Remove the image above", "Remove", self.image_groups, group))
        group.append("divider", cps.Divider())
        self.image_groups.append(group)

    def prepare_settings(self, setting_values):
        '''Adjust self.image_groups to account for the expected # of images'''
        assert len(setting_values) % SETTINGS_PER_GROUP == 0
        group_count = len(setting_values) / SETTINGS_PER_GROUP
        del self.image_groups[group_count:]
        while len(self.image_groups) < group_count:
            self.add_image_group()

    def settings(self):
        '''The settings in the save / load order'''
        result = []
        for image_group in self.image_groups:
            result += [image_group.image_name, image_group.check_blur, image_group.window_size,
                       image_group.check_saturation, image_group.calculate_threshold,
                       image_group.threshold_method, image_group.object_fraction]
        return result

    def visible_settings(self):
        '''The settings as displayed to the user'''
        result = []
        for image_group in self.image_groups:
            result += [image_group.image_name, image_group.check_blur]
            if image_group.check_blur.value:
                result += [image_group.window_size]
            result += [image_group.check_saturation, image_group.calculate_threshold]
            if image_group.calculate_threshold.value:
                result += [image_group.threshold_method]
                if image_group.threshold_method == cpthresh.TM_MOG_GLOBAL:
                    result += [image_group.object_fraction]
            result += [image_group.remove_button, image_group.divider]
            
        # remove the last divider
        del result[-1]

        result += [self.bottom_spacer, self.add_button]
        return result

    def validate_module(self, pipeline):
        '''Make sure settings are compatible
        
        In particular, we make sure that no measurements are duplicated
        '''
        # check for duplicated measurements
        measurements, sources = self.get_measurement_columns(pipeline, return_sources=True)
        d = {}
        for m, s in zip(measurements, sources):
            if m in d:
                raise cps.ValidationError("%s measurement made twice."%(m[1]), s)
            d[m] = True

    def any_threshold(self):
        '''True if some image has its threshold calculated'''
        return any([ig.calculate_threshold.value 
                    for ig in self.image_groups])
    
    def any_saturation(self):
        '''True if some image has its saturation calculated'''
        return any([ig.check_saturation.value
                     for ig in self.image_groups])
    
    def any_blur(self):
        '''True if some image has its blur calculated'''
        return any([ig.check_blur.value
                    for ig in self.image_groups])
    
    def get_measurement_columns(self, pipeline, return_sources=False):
        '''Return column definitions for all measurements'''
        columns = []
        sources = []
        for ig in self.image_groups:
            if ig.check_blur.value:
                for feature in (FOCUS_SCORE,LOCAL_FOCUS_SCORE):
                    columns.append((cpmeas.IMAGE,
                                    '%s_%s_%s_%d'%(IMAGE_QUALITY, feature,
                                                   ig.image_name.value,
                                                   ig.window_size.value),
                                    cpmeas.COLTYPE_FLOAT))
                    sources.append(ig.image_name)
            if ig.check_saturation.value:
                for feature in (PERCENT_SATURATION, PERCENT_MAXIMAL):
                    columns.append((cpmeas.IMAGE,
                                    '%s_%s_%s'%(IMAGE_QUALITY, feature,
                                                ig.image_name.value),
                                    cpmeas.COLTYPE_FLOAT))
                    sources.append(ig.image_name)
            if ig.calculate_threshold.value:
                feature = ig.threshold_feature_name
                columns.append((cpmeas.IMAGE, feature, cpmeas.COLTYPE_FLOAT))
                sources.append(ig.image_name)
        if return_sources:
            return columns, sources
        else:
            return columns
            
    def get_categories(self, pipeline, object_name):
        if object_name == cpmeas.IMAGE:
            return [IMAGE_QUALITY]
        elif (object_name == cpmeas.EXPERIMENT and self.any_threshold()):
            return [IMAGE_QUALITY]
        return [] 

    def get_measurements(self, pipeline, object_name, category):
        if object_name == cpmeas.IMAGE and category == IMAGE_QUALITY:
            result = []
            if self.any_blur():
                result += [FOCUS_SCORE, LOCAL_FOCUS_SCORE]
            if self.any_saturation():
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
        image_name = image_group.image_name.value
        window_size = image_group.window_size.value
        image = workspace.image_set.get_image(image_name,
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
                                          image_name,
                                          window_size)
        workspace.add_measurement(cpmeas.IMAGE, focus_score_name,
                                  focus_score)
        
        local_focus_score_name = "%s_%s_%s_%d"%(IMAGE_QUALITY,
                                                LOCAL_FOCUS_SCORE,
                                                image_name,
                                                window_size)
        workspace.add_measurement(cpmeas.IMAGE, local_focus_score_name,
                                  local_focus_score)
        return [["%s focus score @%d"%(image_name,
                                       window_size), focus_score],
                ["%s local focus score @%d"%(image_name,
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
    
    def upgrade_settings(self, setting_values, variable_revision_number, 
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


class MeasureImageQualitySettingsGroup(cps.SettingsGroup):
    @property
    def threshold_algorithm(self):
        '''The thresholding algorithm to run'''
        return self.threshold_method.value.split(' ')[0]

    @property
    def  threshold_feature_name(self):
        '''The feature name of the threshold measurement generated'''
        return "%s_%s%s_%s"%(IMAGE_QUALITY, THRESHOLD, 
                             self.threshold_algorithm,
                             self.image_name.value)
                    
