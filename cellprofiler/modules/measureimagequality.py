'''<b>Measure Image Quality</b> measures features that indicate image quality, 
including measurements of blur (poor focus) and the percentage
of pixels in the image that are minimal and maximal (i.e., saturated)
<hr>

<h4>Available measurements</h4>  
<ul>
<li><i>PercentMaximal:</i> Percent of pixels at the maximum intensity value of the image</li>
<li><i>PercentMinimal:</i> Percent of pixels at the minimum intensity value of the image</li>
<li><i>FocusScore:</i> A measure of the intensity variance across image</li>
<li><i>LocalFocusScore:</i> A measure of the intensity variance between image parts</li>
<li><i>Threshold:</i> The automatically calculated threshold for each image for the 
thresholding method of choice.</li>
<li><i>MagnitudeLogLogSlope, PowerLogLogSlope:</i> The slope of the log-log magnitude and power spectra.</li>
</ul>

Example Output:
<table border="1">
<tr>
<td>Percent of pixels that are at the Maximal Intensity:</td>
<td>RescaledOrig: </td>
<td>0.0002763</td>
</tr>
<tr>
<td>Percent of pixels that are at the Minimal Intensity:</td>
<td>RescaledOrig: </td>
<td>0.0000352</td>
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
<tr>
<td>Magnitude Spectrum Slope:</td>
<td>RescaledOrig: </td>
<td>-2.2</td>
</tr>
<tr>
<td>Power Spectrum Slope:</td>
<td>RescaledOrig: </td>
<td>-2.9</td>
</tr>
</table>'''

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
import scipy.ndimage as scind
from scipy.linalg.basic import lstsq

from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
import cellprofiler.cpmath.threshold as cpthresh
import cellprofiler.cpmath.radial_power_spectrum as rps
from identify import O_TWO_CLASS, O_THREE_CLASS, O_WEIGHTED_VARIANCE, O_ENTROPY
from identify import O_FOREGROUND, O_BACKGROUND

IMAGE_QUALITY = 'ImageQuality'
FOCUS_SCORE = 'FocusScore'
LOCAL_FOCUS_SCORE = 'LocalFocusScore'
PERCENT_MAXIMAL = 'PercentMaximal'
PERCENT_MINIMAL = 'PercentMinimal'
THRESHOLD = 'Threshold'
MEAN_THRESH_ALL_IMAGES = 'MeanThresh_AllImages'
MEDIAN_THRESH_ALL_IMAGES = 'MedianThresh_AllImages'
STD_THRESH_ALL_IMAGES = 'StdThresh_AllImages'
POWER_SPECTRUM_FEATURES = ['MagnitudeLogLogSlope', 'PowerLogLogSlope']
SETTINGS_PER_GROUP = 11

class MeasureImageQuality(cpm.CPModule):
    module_name = "MeasureImageQuality"
    category = "Measurement"
    variable_revision_number = 3

    def create_settings(self):
        self.image_groups = []
        self.add_image_group(can_remove = False)
        self.bottom_spacer = cps.Divider(line=False)
        self.add_button = cps.DoSomething("", "Add another image", self.add_image_group)
    
    def add_image_group(self, can_remove = True):
        group = MeasureImageQualitySettingsGroup() # helper class defined below
        if can_remove:
            group.append("divider", cps.Divider(line=True))
        group.append("image_name", cps.ImageNameSubscriber("Select an image to measure","None", 
                                                           doc = '''What did you call the grayscale images whose quality you want to measure?'''))
        group.append("check_blur", cps.Binary("Check for blur?",
                                              True, 
                                              doc = '''Would you like to check for blur? If so, the module will calculate a focus score for each image, indicating how blurry an image is
                                            (higher Focus Score = better focus = less blurry). This calculation is slow, so it is optional. The score 
                                            is calculated using the normalized variance. We used this algorithm because it
                                            was ranked best in this paper:
                                            Sun, Y., Duthaler, S., Nelson, B. "Autofocusing in Computer Microscopy:
                                               Selecting the optimal focus algorithm," <i>Microscopy Research and
                                               Technique</i> 65:139-149 (2004).<br>
                                            <br>
                                            The calculation of the focus score is as follows:<br>
                                            [m,n] = size(Image);<br>
                                            MeanImageValue = mean(Image(:));<br>
                                            SquaredNormalizedImage = (Image-MeanImageValue).^2;<br>
                                            FocusScore{ImageNumber} = ...<br>
                                               sum(SquaredNormalizedImage(:))/(m*n*MeanImageValue);<br>
                                            <br>
                                            The above score is designed to determine which image of a particular field of view shows the 
                                            best focus (assuming the overall intensity and the number of objects in the image is 
                                            constant); it is not necessarily intended to compare images of different 
                                            fields of view, although it may be useful for this to some degree. 
                                            The Local Focus Score is a local version of the Focus Score, which is 
                                            potentially more useful for comparing focus between images of different fields of view. However, 
                                            like the Focus Score, the measurement should be used with caution 
                                            because it may fail to correspond well to the blurriness of images of 
                                            different fields of view, depending on the conditions.
                                            '''))
        group.append("window_size", cps.Integer("Window size for blur measurements",
                                                20, minval =1,
                                                doc = '''<i>(Used only if blur measurements are to be calculated)</i> <br> 
                                                  The Local Focus Score is measured within an NxN pixel window 
                                                  applied to the image. What value of N would you like to use? We suggest twice the typical object diameter. You
                                                  can measure the local focus score over multiple window sizes by adding an image
                                                  to the list more than once and by setting different window sizes for
                                                  each image. '''))
        group.append("check_saturation", cps.Binary("Check for saturation?",
                                                    True, 
                                                    doc = '''Would you like to check for saturation 
                                                    (maximal and minimal percentages)? The percentage of pixels at
                                                    the upper or lower limit of each individual image is 
                                                    calculated.  The hard limits of 0 and 1 are not used because images often
                                                    have undergone some kind of transformation such that no pixels
                                                    ever reach the absolute maximum or minimum of the image format.  Given
                                                    noise in images, this should typically be a low percentage but if the
                                                    images were saturated during imaging, a higher than usual
                                                    PercentMaximal will be observed, and if there are no objects, the
                                                    PercentMinimal will increase.'''))
        group.append("calculate_threshold", cps.Binary("Calculate threshold?",
                                                       True, doc = '''Would you like to automatically calculate a suggested 
                                                       threshold for each image? One indicator of image quality is that the 
                                                       automatically calculated suggested threshold is within a typical range. 
                                                       Outlier images with high or low thresholds often contain artifacts.'''))
        group.append("threshold_method", cps.Choice("Select a thresholding method",
                                                    cpthresh.TM_GLOBAL_METHODS,
                                                    cpthresh.TM_OTSU_GLOBAL, 
                                                    doc = '''<i>(Used only if thresholds are to be calculated)</i> <br> This setting allows you to access automatic thresholding 
                                                       methods used in the <b>Identify</b> modules.
                                                       For more help on thresholding, see the <b>Identify</b> modules.'''))
        group.append("object_fraction", cps.Float("Typical fraction of the image covered by objects", 0.1,0,1, doc = 
                                                  """<i>(Used only if threshold are calculated and MoG thresholding is chosen)</i> <br> 
                                                      Enter the approximate fraction of the typical image in the set
                                                      that is covered by objects."""))
        group.append("two_class_otsu", cps.Choice(
            'Two-class or three-class thresholding?',
            [O_TWO_CLASS, O_THREE_CLASS],doc="""
            <i>(Used only for the Otsu thresholding method)</i> <br>
            Select <i>Two</i> if the grayscale levels are readily distinguishable into foregound 
            (i.e., objects) and background. Select <i>Three</i> if there is a  
            middle set of grayscale levels that belongs to neither the
            foreground nor background. 
            <p>For example, three-class thresholding may
            be useful for images in which you have nuclear staining along with a
            low-intensity non-specific cell staining. Where two-class thresholding
            might incorrectly assign this intemediate staining to the nuclei 
            objects, three-class thresholding allows you to assign it to the 
            foreground or background as desired. However, in extreme cases where either 
            there are almost no objects or the entire field of view is covered with 
            objects, three-class thresholding may perform worse than two-class."""))
        
        group.append("use_weighted_variance", cps.Choice(
            'Minimize the weighted variance or the entropy?',
            [O_WEIGHTED_VARIANCE, O_ENTROPY]))
        
        group.append("assign_middle_to_foreground", cps.Choice(
            'Assign pixels in the middle intensity class to the foreground '
            'or the background?', [O_FOREGROUND, O_BACKGROUND],doc="""
            <i>(Used only for the Otsu thresholding method with three-class thresholding)</i><br>
            Choose whether you want the middle grayscale intensities to be assigned 
            to the foreground pixels or the background pixels."""))
        group.append("compute_power_spectrum", cps.Binary("Calculate quartiles and sum of radial power spectrum?", True,
                                                      doc = "Would you like to calculate the quartiles and sum of the radial power spectrum? The Power Spectrum is computed via FFT and the radii of the first three quartiles and the total power are measured."))
        if can_remove:
            group.append("remove_button", cps.RemoveSettingButton("", "Remove this image", self.image_groups, group))
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
                       image_group.threshold_method, image_group.object_fraction,
                       image_group.compute_power_spectrum,
                       image_group.two_class_otsu, 
                       image_group.use_weighted_variance, 
                       image_group.assign_middle_to_foreground]
        return result

    def visible_settings(self):
        '''The settings as displayed to the user'''
        result = []
        for i, image_group in enumerate(self.image_groups):
            if i != 0:
                result += [ image_group.divider ]
            result += [image_group.image_name, image_group.check_blur,
                       image_group.window_size, image_group.check_saturation,
                       image_group.calculate_threshold]
            if image_group.calculate_threshold:
                result += [image_group.threshold_method]
                if image_group.threshold_method == cpthresh.TM_MOG_GLOBAL:
                    result += [image_group.object_fraction]
                elif image_group.threshold_method == cpthresh.TM_OTSU_GLOBAL:
                    result += [image_group.use_weighted_variance, 
                               image_group.two_class_otsu]
                    if image_group.two_class_otsu == O_THREE_CLASS:
                        result += [image_group.assign_middle_to_foreground]
            result += [image_group.compute_power_spectrum]
            if i != 0:
                result += [image_group.remove_button]
        result += [self.add_button]
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
    
    def any_power_spectrum(self):
        '''True if some image has its radial power spectrum calculated'''
        return any([ig.compute_power_spectrum.value
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
                for feature in (PERCENT_MAXIMAL, PERCENT_MINIMAL):
                    columns.append((cpmeas.IMAGE,
                                    '%s_%s_%s'%(IMAGE_QUALITY, feature,
                                                ig.image_name.value),
                                    cpmeas.COLTYPE_FLOAT))
                    sources.append(ig.image_name)
            if ig.calculate_threshold.value:
                feature = ig.threshold_feature_name
                columns.append((cpmeas.IMAGE, feature, cpmeas.COLTYPE_FLOAT))
                sources.append(ig.image_name)
            if ig.compute_power_spectrum.value:
                for feature in POWER_SPECTRUM_FEATURES:
                    columns.append((cpmeas.IMAGE,
                                    '%s_%s_%s'%(IMAGE_QUALITY, feature,
                                                ig.image_name.value),
                                    cpmeas.COLTYPE_FLOAT))
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
                result += [PERCENT_MAXIMAL, PERCENT_MINIMAL]
            if self.any_power_spectrum():
                result += POWER_SPECTRUM_FEATURES
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
        if measurement in (PERCENT_MAXIMAL, PERCENT_MINIMAL):
            return [ig.image_name.value for ig in self.image_groups
                    if ig.check_saturation.value]
        if measurement.startswith(THRESHOLD):
            return [ig.image_name.value for ig in self.image_groups
                    if (ig.calculate_threshold.value and
                        measurement == THRESHOLD+ig.threshold_algorithm)]
        if measurement in POWER_SPECTRUM_FEATURES:
                return [ig.image_name.value for ig in self.image_groups
                        if ig.compute_power_spectrum.value]
    
    def get_measurement_scales(self, pipeline, object_name, category, 
                               measurement, image_name):
        '''Get the scales (window_sizes) for the given measurement'''
        if (object_name == cpmeas.IMAGE and
            category == IMAGE_QUALITY):
            if measurement in (FOCUS_SCORE, LOCAL_FOCUS_SCORE):
                return [ig.window_size for ig in self.image_groups
                        if ig.image_name == image_name]
            result = []
            for group in self.image_groups:
                if measurement == THRESHOLD+group.threshold_algorithm:
                    scale = group.threshold_scale
                    if scale is not None:
                        result += [scale]
            return result
        return []

    def run(self, workspace):
        '''Calculate statistics over all image groups'''
        statistics = []
        for image_group in self.image_groups:
            statistics += self.run_on_image_group(image_group, workspace)
        workspace.display_data.statistics = statistics

    def is_interactive(self):
        return False

    def display(self, workspace):
        if workspace.frame is not None:
            statistics = workspace.display_data.statistics
            figure = workspace.create_or_find_figure(subplots=(1,1))
            figure.subplot_table(0,0,statistics)
    
    def post_run(self, workspace):
        '''Calculate the experiment statistics at the end of a run'''
        statistics = []
        for image_group in self.image_groups:
            statistics += self.calculate_experiment_threshold(image_group, 
                                                              workspace)
    def run_on_image_group(self, image_group, workspace):
        '''Calculate statistics for a particular image'''
        statistics = []
        if image_group.check_blur.value:
            statistics += self.calculate_image_blur(image_group, workspace)
        if image_group.check_saturation.value:
            statistics += self.calculate_saturation(image_group, workspace)
        if image_group.calculate_threshold.value:
            statistics += self.calculate_threshold(image_group, workspace)
        if image_group.compute_power_spectrum.value:
            statistics += self.calculate_power_spectrum(image_group, workspace)
        
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
        grid_range = np.arange(0, m*n+1,dtype=int)
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
            percent_minimal = 0
        else:
            number_pixels_maximal = np.sum(pixel_data == np.max(pixel_data))
            number_pixels_minimal = np.sum(pixel_data == np.min(pixel_data))
            percent_maximal = (100.0 * float(number_pixels_maximal) /
                               float(pixel_count))
            percent_minimal = (100.0 * float(number_pixels_minimal) /
                               float(pixel_count))
        percent_maximal_name = "%s_%s_%s"%(IMAGE_QUALITY, PERCENT_MAXIMAL,
                                           image_name)
        percent_minimal_name = "%s_%s_%s"%(IMAGE_QUALITY, PERCENT_MINIMAL,
                                           image_name)
        workspace.add_measurement(cpmeas.IMAGE, percent_maximal_name,
                                  percent_maximal)
        workspace.add_measurement(cpmeas.IMAGE, percent_minimal_name,
                                  percent_minimal)
        return [["%s maximal"%image_name,"%.1f %%"%percent_maximal],
                ["%s minimal"%image_name, "%.1f %%"%percent_minimal]]

    

    def calculate_power_spectrum(self, image_group, workspace):
        image_name = image_group.image_name.value
        image = workspace.image_set.get_image(image_name,
                                              must_be_grayscale = True)

        pixel_data = image.pixel_data

        if image.has_mask:
            pixel_data = np.array(pixel_data) # make a copy
            masked_pixels = pixel_data[image.mask]
            pixel_count = np.product(masked_pixels.shape)
            if pixel_count > 0:
                pixel_data[~ image.mask] = np.mean(masked_pixels)
            else:
                pixel_data[~ image.mask] = 0
        
        radii, magnitude, power = rps.rps(pixel_data)
        if sum(magnitude) > 0:
            valid = (magnitude > 0)
            radii = radii[valid].reshape((-1, 1))
            magnitude = magnitude[valid].reshape((-1, 1))
            power = power[valid].reshape((-1, 1))
            if radii.shape[0] > 1:
                magslope = lstsq(np.hstack((np.log(radii), np.ones(radii.shape))), np.log(magnitude))[0][0]
                powerslope = lstsq(np.hstack((np.log(radii), np.ones(radii.shape))), np.log(power))[0][0]
            else:
                magslope = powerslope = 0
        else:
            magslope = powerslope = 0

        result = []
        for fname, val in zip(POWER_SPECTRUM_FEATURES, [magslope, powerslope]):
            workspace.add_measurement(cpmeas.IMAGE, 
                                      "%s_%s_%s"%(IMAGE_QUALITY, fname, image_name),
                                      val)
            result += [["%s %s"%(image_name, fname), "%.1f"%(val)]]
        return result

    def calculate_threshold(self, image_group, workspace):
        '''Calculate a threshold for this image'''
        image_name = image_group.image_name.value
        image = workspace.image_set.get_image(image_name,
                                              must_be_grayscale = True)
        threshold_method = image_group.threshold_algorithm
        object_fraction = image_group.object_fraction.value
        two_class_otsu = (image_group.two_class_otsu == O_TWO_CLASS)
        use_weighted_variance = (image_group.use_weighted_variance == O_WEIGHTED_VARIANCE)
        assign_middle_to_foreground = (image_group.assign_middle_to_foreground == O_FOREGROUND)
        (local_threshold, global_threshold) = \
            (cpthresh.get_threshold(threshold_method,
                                    cpthresh.TM_GLOBAL,
                                    image.pixel_data,
                                    mask = image.mask,
                                    object_fraction = object_fraction,
                                    two_class_otsu = two_class_otsu,
                                    use_weighted_variance = use_weighted_variance,
                                    assign_middle_to_foreground = assign_middle_to_foreground)
             if image.has_mask
             else
             cpthresh.get_threshold(threshold_method,
                                    cpthresh.TM_GLOBAL,
                                    image.pixel_data,
                                    object_fraction = object_fraction,
                                    two_class_otsu = two_class_otsu,
                                    use_weighted_variance = use_weighted_variance,
                                    assign_middle_to_foreground = assign_middle_to_foreground))
        threshold_name = image_group.threshold_feature_name
        scale = image_group.threshold_scale
        if scale is None:
            threshold_description = threshold_method
        else:
            threshold_description = threshold_method + " " + scale
        workspace.add_measurement(cpmeas.IMAGE, threshold_name,
                                  global_threshold)
        return [["%s %s threshold"%(image_name, threshold_description), 
                 str(global_threshold)]]

    def calculate_experiment_threshold(self, image_group, workspace):
        '''Calculate experiment-wide threshold mean, median and standard-deviation'''
        m = workspace.measurements
        statistics = []
        if image_group.calculate_threshold.value:
            image_name = image_group.image_name.value
            values = m.get_all_measurements(cpmeas.IMAGE, 
                                            image_group.threshold_feature_name)
            # Ignore missing values
            values = [v for v in values if v is not None]
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

        if (from_matlab and variable_revision_number == 4 and
            module_name == 'MeasureImageSaturationBlur'):
            image_names = []
            for image_name in setting_values[:6]:
                if image_name != cps.DO_NOT_USE:
                    image_names.append(image_name)
            wants_blur = setting_values[-2]
            local_focus_score = setting_values[-1]
            setting_values = []
            for image_name in image_names:
                setting_values += [image_name, 
                                   wants_blur, 
                                   local_focus_score,
                                   cps.YES, # check saturation
                                   cps.NO, # calculate threshold
                                   cpthresh.TM_OTSU_GLOBAL,
                                   .1, # object fraction
                                   cps.NO] # compute power spectrum
            variable_revision_number = 2
            from_matlab = False
            module_name = 'MeasureImageQuality'
            
        if (from_matlab and variable_revision_number == 1 and 
            module_name == 'MeasureImageQuality'):
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
        
        if (not from_matlab) and variable_revision_number == 1:
            # add power spectrum calculations
            assert (not from_matlab)
            assert len(setting_values) % 7 == 0
            num_images = len(setting_values) / 7
            new_settings = []
            for idx in range(num_images):
                new_settings += setting_values[(idx * 7):(idx * 7 + 7)]
                new_settings += [cps.YES]
            setting_values = new_settings
            variable_revision_number = 2
            
        if (not from_matlab) and variable_revision_number == 2:
            # add otsu threshold settings
            assert len(setting_values) % 8 == 0
            num_images = len(setting_values) / 8
            new_settings = []
            for idx in range(num_images):
                new_settings += setting_values[(idx * 8):(idx * 8 + 8)]
                new_settings += [O_TWO_CLASS, O_WEIGHTED_VARIANCE,
                                 O_FOREGROUND]
            setting_values = new_settings
            variable_revision_number = 3
        return setting_values, variable_revision_number, from_matlab


class MeasureImageQualitySettingsGroup(cps.SettingsGroup):
    @property
    def threshold_algorithm(self):
        '''The thresholding algorithm to run'''
        return self.threshold_method.value.split(' ')[0]

    @property
    def  threshold_feature_name(self):
        '''The feature name of the threshold measurement generated'''
        scale = self.threshold_scale
        if scale is None:
            return "%s_%s%s_%s"%(IMAGE_QUALITY, THRESHOLD, 
                                 self.threshold_algorithm,
                                 self.image_name.value)
        else:
            return "%s_%s%s_%s_%s" % (IMAGE_QUALITY, THRESHOLD,
                                      self.threshold_algorithm,
                                      self.image_name.value,
                                      scale)
    @property
    def threshold_scale(self):
        '''The "scale" for the threshold = minor parameterizations'''
        #
        # Distinguish Otsu choices from each other
        #
        threshold_algorithm = self.threshold_algorithm
        if threshold_algorithm == cpthresh.TM_OTSU:
            if self.two_class_otsu == O_TWO_CLASS:
                scale = "2"
            else:
                scale = "3"
                if self.assign_middle_to_foreground == O_FOREGROUND:
                    scale += "F"
                else:
                    scale += "B"
            if self.use_weighted_variance == O_WEIGHTED_VARIANCE:
                scale += "W"
            else:
                scale += "S"
            return scale
        elif threshold_algorithm == cpthresh.TM_MOG:
            return str(int(self.object_fraction.value * 100))
        
                    
