'''<b>Measure Image Quality</b> measures features that indicate image quality.
<hr>
This module can collect measurements indicating possible image abberations,
e.g. blur (poor focus), intensity, saturation (i.e., the percentage
of pixels in the image that are minimal and maximal). Details and guidance for
each of these measures is provided in the settings help.

<p>Please note that for best results, this module should be applied to the
original raw images, as opposed to images that have already been corrected for
illumination.</p>

<h4>Available measurements</h4>
<ul>
<li><b>Blur metrics</b>
<ul>
<li><i>FocusScore:</i> A measure of the intensity variance across the image.</li>
<li><i>LocalFocusScore:</i> A measure of the intensity variance between image sub-regions.</li>
<li><i>Correlation:</i> A measure of the correlation of the image for a given spatial scale.</li>
<li><i>PowerLogLogSlope:</i> The slope of the image log-log power spectrum.</li>
</ul>
</li>

<li><b>Saturation metrics</b>
<ul>
<li><i>PercentMaximal:</i> Percent of pixels at the maximum intensity value of the image.</li>
<li><i>PercentMinimal:</i> Percent of pixels at the minimum intensity value of the image.</li>
</ul>
</li>

<li><b>Intensity metrics</b>
<ul>
<li><i>TotalIntensity:</i> Sum of all pixel intensity values.</li>
<li><i>MeanIntensity, MedianIntensity:</i> Mean and median of pixel intensity values.</li>
<li><i>StdIntensity, MADIntensity:</i> Standard deviation and median absolute deviation (MAD) of pixel intensity values.</li>
<li><i>MinIntensity, MaxIntensity:</i> Minimum and maximum of pixel intensity values.</li>
<li><i>TotalArea:</i> Number of pixels measured.</li>
</ul>
</li>

<li><b>Threshold metrics:</b>
<ul>
<li><i>Threshold:</i> The automatically calculated threshold for each image for the
thresholding method of choice.
<p>Please note that these thresholds are recorded individually for each image and as an aggregate
statistic for all images. The mean, median and standard deviation of the threshold values are
computed for each of the threshold methods selected and recorded as a measurement in the
per-experiment table.</p></li>
</ul>
</li>
</ul>

<h4>References</h4>
<ul>
<li>Bray MA, Fraser AN, Hasaka TP, Carpenter AE (2012) "Workflow and metrics for image quality
control in large-scale high-content screens." <i>J Biomol Screen</i> 17(2):266-74.
<a href="http://dx.doi.org/10.1177/1087057111420292">(link)</a></li>
</ul>
'''

import logging

import numpy as np

logger = logging.getLogger(__name__)
import scipy.ndimage as scind
from scipy.linalg.basic import lstsq
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
import centrosome.haralick
import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.setting as cps
from cellprofiler.setting import YES, NO
import centrosome.threshold as cpthresh
import itertools
import centrosome.radial_power_spectrum as rps
from identify import O_TWO_CLASS, O_THREE_CLASS, O_WEIGHTED_VARIANCE, O_ENTROPY
from identify import O_FOREGROUND, O_BACKGROUND
from centrosome.threshold import TM_MOG, TM_OTSU
from loadimages import C_FILE_NAME, C_SCALING
import cellprofiler.preferences as cpprefs
from cellprofiler.preferences import \
    DEFAULT_OUTPUT_FOLDER_NAME, DEFAULT_INPUT_FOLDER_NAME, ABSOLUTE_FOLDER_NAME, \
    DEFAULT_INPUT_SUBFOLDER_NAME, DEFAULT_OUTPUT_SUBFOLDER_NAME, IO_FOLDER_CHOICE_HELP_TEXT

##############################################
#
# Choices for which images to include
#
##############################################

# Setting variables
'''Image selection'''
O_ALL_LOADED = "All loaded images"  # Use all loaded images
O_SELECT = "Select..."  # Select the images you want from a list, all treated the same

# Measurement names
'''Root module measurement name'''
C_IMAGE_QUALITY = 'ImageQuality'
F_FOCUS_SCORE = 'FocusScore'
F_LOCAL_FOCUS_SCORE = 'LocalFocusScore'
F_CORRELATION = 'Correlation'
F_POWER_SPECTRUM_SLOPE = 'PowerLogLogSlope'
F_TOTAL_AREA = 'TotalArea'
F_TOTAL_INTENSITY = 'TotalIntensity'
F_MEAN_INTENSITY = 'MeanIntensity'
F_MEDIAN_INTENSITY = 'MedianIntensity'
F_STD_INTENSITY = 'StdIntensity'
F_MAD_INTENSITY = 'MADIntensity'
F_MAX_INTENSITY = 'MaxIntensity'
F_MIN_INTENSITY = 'MinIntensity'
INTENSITY_FEATURES = [F_TOTAL_AREA, F_TOTAL_INTENSITY, F_MEAN_INTENSITY, F_MEDIAN_INTENSITY, F_STD_INTENSITY,
                      F_MAD_INTENSITY, F_MAX_INTENSITY, F_MIN_INTENSITY]
F_PERCENT_MAXIMAL = 'PercentMaximal'
F_PERCENT_MINIMAL = 'PercentMinimal'
SATURATION_FEATURES = [F_PERCENT_MAXIMAL, F_PERCENT_MINIMAL]
F_THRESHOLD = 'Threshold'
MEAN_THRESH_ALL_IMAGES = 'MeanThresh_AllImages'
MEDIAN_THRESH_ALL_IMAGES = 'MedianThresh_AllImages'
STD_THRESH_ALL_IMAGES = 'StdThresh_AllImages'

AGG_MEAN = "Mean"
AGG_MEDIAN = "Median"
AGG_STD = "Std"

SETTINGS_PER_GROUP_V3 = 11
IMAGE_GROUP_SETTING_OFFSET = 2


class MeasureImageQuality(cpm.Module):
    module_name = "MeasureImageQuality"
    category = "Measurement"
    variable_revision_number = 5

    def create_settings(self):
        self.images_choice = cps.Choice(
                "Calculate metrics for which images?",
                [O_ALL_LOADED, O_SELECT], doc="""
            This option lets you choose which images will have quality metrics calculated.
            <ul>
            <li><i>%(O_ALL_LOADED)s:</i> Use all images loaded with the <b>Input</b> modules.
            The quality metrics selected below will be applied to all loaded images.</li>
            <li><i>%(O_SELECT)s:</i> Select the desired images from a list. The quality
            metric settings selected will be applied to all these images. Additional lists
            can be added with separate settings.</li>
            </ul>""" % globals())

        self.divider = cps.Divider(line=True)

        self.image_groups = []
        self.image_count = cps.HiddenCount(self.image_groups, "Image count")
        self.add_image_group(can_remove=False)
        self.add_image_button = cps.DoSomething("", "Add another image list", self.add_image_group)

    def add_image_group(self, can_remove=True):
        group = cps.SettingsGroup()

        group.can_remove = can_remove
        if can_remove:
            group.append("divider", cps.Divider(line=True))

        group.append("image_names", cps.ImageNameSubscriberMultiChoice(
                "Select the images to measure", doc="""
            <i>(Used only if "%(O_SELECT)s" is chosen for selecting images)</i><br>
            Choose one or more images from this list. You can select multiple images by clicking
            using the shift or command keys. In addition to loaded images, the list includes
            the images that were created by prior modules.""" % globals()))

        group.append("include_image_scalings", cps.Binary(
                "Include the image rescaling value?",
                True, doc="""
            Select <i>%(YES)s</i> to add the image's rescaling
            value as a quality control metric. This value is set only for images
            that loaded using the <b>Input</b> modules. This is useful in confirming
            that all images are rescaled by the same value, since some acquisition
            device vendors may output this value differently.
            See <b>NamesAndTypes</b> for more information.""" % globals()))

        group.append("check_blur", cps.Binary(
                "Calculate blur metrics?",
                True, doc="""
            Select <i>%(YES)s</i> to compute a series of blur metrics. The blur metrics are the
            following, along with recomendations on their use:
            <ul>
            <li><i>%(F_POWER_SPECTRUM_SLOPE)s:</i> The power spectrum contains the frequency information
            of the image, and the slope gives a measure of image blur. A higher slope indicates
            more lower frequency components, and hence more blur (<i>Field, 1997</i>). This metric is
            recommended for blur detection in most cases.</li>
            <li><i>%(F_CORRELATION)s:</i> This is a measure of the image spatial intensity distribution
            computed across sub-regions of an image for a given spatial scale (<i>Haralick, 1973</i>).
            If an image is blurred, the correlation between neighboring pixels becomes high,
            producing a high correlation value. A similar approach was found to give optimal
            performance for fluorescence microscopy applications (<i>Vollath, 1987</i>). <br>
            Some care is required in selecting an appropriate spatial scale because differences
            in the spatial scale capture various features: moderate scales capture the
            blurring of intracellular features better than small scales and larger scales
            are more likely to reflect intercellular confluence than focal blur. A spatial scale
            no bigger than the feature of interest is recommended, although you can select as
            many scales as desired.</li>
            <li><i>%(F_FOCUS_SCORE)s:</i> This score is calculated using a normalized variance,
            which was the best-ranking algorithm for brightfield, phase contrast, and DIC images
            (<i>Sun, 2004</i>). Higher focus scores correspond to lower bluriness. <br>
            More specifically, the focus score computes the intensity variance of the entire
            image divided by mean image intensity. Since it is tailored for autofocusing
            applications (difference focus for the same field of view), it assumes that the
            overall intensity and the number of objects in the image is constant, making it less
            useful for comparison images of different fields of view. For distinguishing
            extremely blurry images, however, it performs well.</li>
            <li><i>%(F_LOCAL_FOCUS_SCORE)s:</i> A local version of the Focus Score, it subdivides the
            image into non-overlapping tiles, computes the normalized variance for each, and
            takes the mean of these values as the final metric. It is potentially more useful
            for comparing focus between images of different fields of view, but is subject
            to the same caveats as the Focus Score. It can be useful in differentiating good versus
            badly segmented images in the cases when badly segmented images usually contain no cell
            objects with high background noise.</li>
            </ul>
            <p><b>References</b><br>
            <ul>
            <li>Field DJ (1997) "Relations between the statistics of natural
            images and the response properties of cortical cells" <i>Journal of the Optical
            Society of America. A, Optics, image science, and vision</i>, 4(12):2379-94.
            <a href="http://redwood.psych.cornell.edu/papers/field_87.pdf"><(pdf)</a></li>
            <li>Haralick RM (1979) "Statistical and structural approaches to texture"
            Proc. IEEE, 67(5):786-804.
            <a href="http://dx.doi.org/10.1109/PROC.1979.11328">(link)</a></li>
            <li>Vollath D (1987) "Automatic focusing by correlative methods" <i>Journal of Microscopy</i>
            147(3):279-288.
            <a href="http://dx.doi.org/10.1111/j.1365-2818.1987.tb02839.x">(link)</a></li>
            <li>Sun Y, Duthaler S, Nelson B (2004) "Autofocusing in computer microscopy:
            Selecting the optimal focus algorithm" <i>Microscopy Research and
            Technique</i>, 65:139-149
            <a href="http://dx.doi.org/10.1002/jemt.20118">(link)</a></li>
            </ul>""" % globals()))

        group.append("include_local_blur", cps.Binary(
                "Include local blur metrics?",
                True, doc="""
            """))

        group.scale_groups = []

        group.scale_count = cps.HiddenCount(group.scale_groups, "Scale count")

        def add_scale_group(can_remove=True):
            self.add_scale_group(group, can_remove)

        add_scale_group(False)

        group.append("add_scale_button", cps.DoSomething("",
                                                         "Add another scale",
                                                         add_scale_group, doc="""
            Press this button to add another scale setting."""))

        group.append("check_saturation", cps.Binary(
                "Calculate saturation metrics?",
                True, doc="""
            Select <i>%(YES)s</i> to calculate the saturation metrics <i>%(F_PERCENT_MAXIMAL)s</i>
            and <i>%(F_PERCENT_MINIMAL)s</i>, i.e., the percentage of pixels at
            the upper or lower limit of each individual image.
            <p>For this calculation, the hard limits of 0 and 1 are not used because images often
            have undergone some kind of transformation such that no pixels
            ever reach the absolute maximum or minimum of the image format.  Given
            the noise typical in images, both these measures should be a low percentage but if the
            images were saturated during imaging, a higher than usual
            <i>%(F_PERCENT_MAXIMAL)s</i> will be observed, and if there are no objects, the
            <i>%(F_PERCENT_MINIMAL)s</i> value will increase.</p>""" % globals()))

        group.append("check_intensity", cps.Binary(
                "Calculate intensity metrics?",
                True, doc="""
            Select <i>%(YES)s</i> to calculate image-based
            intensity measures, namely the mean, maximum, minimum, standard deviation
            and median absolute deviation of pixel intensities. These measures
            are identical to those calculated by <b>MeasureImageIntensity</b>.""" % globals()))

        group.append("calculate_threshold", cps.Binary(
                "Calculate thresholds?",
                True, doc="""
            Automatically calculate a suggested
            threshold for each image. One indicator of image quality is that these threshold
            values lie within a typical range.
            Outlier images with high or low thresholds often contain artifacts."""))

        group.append("use_all_threshold_methods", cps.Binary(
                "Use all thresholding methods?",
                False, doc="""
            <i>(Used only if image thresholds are calculcated)</i><br>
            Select <i>%(YES)s</i> to calculate thresholds using all the available methods. Only the global methods
            are used. <br>
            While most methods are straightfoward, some methods have additional
            parameters that require special handling:
            <ul>
            <li><i>%(TM_OTSU)s:</i> Thresholds for all combinations of class number, minimzation
            parameter and middle class assignment are computed.</li>
            <li><i>Mixture of Gaussians (%(TM_MOG)s):</i> Thresholds for image coverage fractions
            of 0.05, 0.25, 0.75 and 0.95 are computed.</li>
            </ul>
            See the <b>IdentifyPrimaryObjects</b> module for more information on thresholding
            methods.""" % globals()))

        group.threshold_groups = []

        group.threshold_count = cps.HiddenCount(group.threshold_groups, "Threshold count")

        def add_threshold_group(can_remove=True):
            self.add_threshold_group(group, can_remove)

        add_threshold_group(False)

        group.append("add_threshold_button", cps.DoSomething("",
                                                             "Add another threshold method",
                                                             add_threshold_group, doc="""
            Press this button to add another set of threshold settings."""))

        if can_remove:
            group.append("remove_button",
                         cps.RemoveSettingButton("", "Remove this image list", self.image_groups, group))
        self.image_groups.append(group)
        return group

    def add_scale_group(self, image_group, can_remove=True):
        group = cps.SettingsGroup()
        image_group.scale_groups.append(group)

        group.image_names = image_group.image_names

        group.append("divider", cps.Divider(line=False))

        group.append('scale', cps.Integer(
                "Spatial scale for blur measurements",
                len(image_group.scale_groups) * 10 + 10, doc="""
            <i>(Used only if blur measurements are to be calculated)</i> <br>
            The <i>%(F_LOCAL_FOCUS_SCORE)s</i> is measured within an <i>N &times; N</i> pixel
            window applied to the image, whereas the <i>%(F_CORRELATION)s</i> of a pixel is
            measured with repsect to its neighbors <i>N</i> pixels away.
            <p>A higher number for the window size measures larger patterns of
            image blur whereas smaller numbers measure more localized patterns of
            blur. We suggest selecting a window size that is on the order of the feature of interest
            (e.g., the object diameter). You can measure these metrics for multiple window sizes
            by selecting additional scales for each image.</p>""" % globals()))

        group.can_remove = can_remove
        if can_remove:
            group.append("remove_button",
                         cps.RemoveSettingButton("", "Remove this scale", image_group.scale_groups, group))

    def add_threshold_group(self, image_group=None, can_remove=True):
        group = ImageQualitySettingsGroup()

        if image_group is not None:
            image_group.threshold_groups.append(group)
            group.image_names = image_group.image_names

        group.append("divider", cps.Divider(line=False))

        group.append("threshold_method", cps.Choice("Select a thresholding method",
                                                    cpthresh.TM_METHODS,
                                                    cpthresh.TM_OTSU, doc="""
            <i>(Used only if particular thresholds are to be calculated)</i> <br>
            This setting allows you to apply automatic thresholding
            methods used in the <b>Identify</b> modules. Only the global methods are applied.
            For more help on thresholding, see the <b>Identify</b> modules."""))

        group.append("object_fraction", cps.Float(
                "Typical fraction of the image covered by objects", 0.1, 0, 1, doc="""
            <i>(Used only if threshold are calculated and %(TM_MOG)s thresholding is chosen)</i> <br>
            Enter the approximate fraction of the typical image in the set
            that is covered by objects.""" % globals()))

        group.append("two_class_otsu", cps.Choice(
                'Two-class or three-class thresholding?',
                [O_TWO_CLASS, O_THREE_CLASS], doc="""
            <i>(Used only if thresholds are calculcated and the %(TM_OTSU)s thresholding method is used)</i> <br>
            Select <i>%(O_TWO_CLASS)s</i> if the grayscale levels are readily distinguishable into foregound
            (i.e., objects) and background. Select <i>%(O_THREE_CLASS)s</i> if there is a
            middle set of grayscale levels that belongs to neither the
            foreground nor background.
            <p>For example, three-class thresholding may
            be useful for images in which you have nuclear staining along with a
            low-intensity non-specific cell staining. Where two-class thresholding
            might incorrectly assign this intemediate staining to the nuclei
            objects, three-class thresholding allows you to assign it to the
            foreground or background as desired. However, in extreme cases where either
            there are almost no objects or the entire field of view is covered with
            objects, three-class thresholding may perform worse than two-class.""" % globals()))

        group.append("use_weighted_variance", cps.Choice(
                'Minimize the weighted variance or the entropy?',
                [O_WEIGHTED_VARIANCE, O_ENTROPY]))

        group.append("assign_middle_to_foreground", cps.Choice(
                'Assign pixels in the middle intensity class to the foreground or the background?',
                [O_FOREGROUND, O_BACKGROUND], doc="""
            <i>(Used only if thresholds are calculcated and the %(TM_OTSU)s thresholding method with %(O_THREE_CLASS)s is used)</i><br>
            Choose whether you want the middle grayscale intensities to be assigned
            to the foreground pixels or the background pixels.""" % globals()))

        group.can_remove = can_remove
        if can_remove and image_group is not None:
            group.append("remove_button", cps.RemoveSettingButton(
                    "", "Remove this threshold method", image_group.threshold_groups, group))

        if image_group is None:
            return group

    def prepare_settings(self, setting_values):
        '''Adjust image_groups and threshold_groups to account for the expected # of
            images, scales, and threshold methods'''
        image_group_count = int(setting_values[1])
        del self.image_groups[:]
        for i in range(image_group_count):
            can_remove = len(self.image_groups) > 0
            self.add_image_group(can_remove)
        for index, image_group in enumerate(self.image_groups):
            for count, group, fn in \
                    ((int(setting_values[IMAGE_GROUP_SETTING_OFFSET + 2 * index]), image_group.scale_groups,
                      self.add_scale_group),
                     (int(setting_values[IMAGE_GROUP_SETTING_OFFSET + 2 * index + 1]), image_group.threshold_groups,
                      self.add_threshold_group)):
                del group[:]
                for i in range(count):
                    can_remove = len(group) > 0
                    fn(image_group, can_remove)

    def settings(self):
        '''The settings in the save / load order'''
        result = [self.images_choice]
        result += [self.image_count]
        for image_group in self.image_groups:
            result += [image_group.scale_count,
                       image_group.threshold_count]
        for image_group in self.image_groups:
            result += [image_group.image_names]
            result += [image_group.include_image_scalings,
                       image_group.check_blur]
            for scale_group in image_group.scale_groups:
                result += [scale_group.scale]
            result += [image_group.check_saturation,
                       image_group.check_intensity]
            result += [image_group.calculate_threshold,
                       image_group.use_all_threshold_methods]
            for threshold_group in image_group.threshold_groups:
                result += [threshold_group.threshold_method,
                           threshold_group.object_fraction,
                           threshold_group.two_class_otsu,
                           threshold_group.use_weighted_variance,
                           threshold_group.assign_middle_to_foreground]
        return result

    def visible_settings(self):
        '''The settings as displayed to the user'''
        result = [self.images_choice]
        if self.images_choice.value == O_ALL_LOADED:
            del self.image_groups[1:]
        for image_group in self.image_groups:
            if image_group.can_remove:
                result += [image_group.divider]
            if self.images_choice.value == O_SELECT:
                result += [image_group.image_names]
            result += self.image_visible_settings(image_group)
            if image_group.can_remove:
                result += [image_group.remove_button]
        if self.images_choice.value == O_SELECT:
            result += [self.add_image_button]
        return result

    def image_visible_settings(self, image_group):
        result = [image_group.include_image_scalings, image_group.check_blur]
        if image_group.check_blur:
            result += self.scale_visible_settings(image_group)
        result += [image_group.check_intensity]
        result += [image_group.check_saturation, image_group.calculate_threshold]
        if image_group.calculate_threshold:
            result += [image_group.use_all_threshold_methods]
            if not image_group.use_all_threshold_methods.value:
                if image_group.threshold_count.value == 0:
                    self.add_threshold_group(image_group, False)
                result += self.threshold_visible_settings(image_group)
        return result

    def scale_visible_settings(self, image_group):
        result = []
        for scale_group in image_group.scale_groups:
            if scale_group.can_remove:
                result += [scale_group.divider]
            result += [scale_group.scale]
            if scale_group.can_remove:
                result += [scale_group.remove_button]
        result += [image_group.add_scale_button]
        return result

    def threshold_visible_settings(self, image_group):
        result = []
        for threshold_group in image_group.threshold_groups:
            if threshold_group.can_remove:
                result += [threshold_group.divider]
            result += [threshold_group.threshold_method]
            if threshold_group.threshold_method.value == cpthresh.TM_MOG:
                result += [threshold_group.object_fraction]
            elif threshold_group.threshold_method.value == cpthresh.TM_OTSU:
                result += [threshold_group.use_weighted_variance,
                           threshold_group.two_class_otsu]
                if threshold_group.two_class_otsu.value == O_THREE_CLASS:
                    result += [threshold_group.assign_middle_to_foreground]
            if threshold_group.can_remove:
                result += [threshold_group.remove_button]
        result += [image_group.add_threshold_button]
        return result

    def validate_module(self, pipeline):
        '''Make sure a mesurement is selected in image_names'''
        if self.images_choice.value == O_SELECT:
            for image_group in self.image_groups:
                if not image_group.image_names.get_selections():
                    raise cps.ValidationError("Please choose at least one image", image_group.image_names)

        '''Make sure settings are compatible. In particular, we make sure that no measurements are duplicated'''
        measurements, sources = self.get_measurement_columns(pipeline, return_sources=True)
        d = {}
        for m, s in zip(measurements, sources):
            m = (m[0], m[1])
            if m in d:
                raise cps.ValidationError("Measurement %s for image %s made twice." % (m[1], s[1]), s[0])
            d[m] = True

    def prepare_run(self, workspace):
        if cpprefs.get_headless():
            logger.warning(
                    "Experiment-wide values for mean threshold, etc calculated by MeasureImageQuality may be incorrect if the run is split into subsets of images.")
        return True

    def any_scaling(self):
        '''True if some image has its rescaling value calculated'''
        return any([image_group.include_image_scalings.value
                    for image_group in self.image_groups])

    def any_threshold(self):
        '''True if some image has its threshold calculated'''
        return any([image_group.calculate_threshold.value
                    for image_group in self.image_groups])

    def any_saturation(self):
        '''True if some image has its saturation calculated'''
        return any([image_group.check_saturation.value
                    for image_group in self.image_groups])

    def any_blur(self):
        '''True if some image has its blur calculated'''
        return any([image_group.check_blur.value
                    for image_group in self.image_groups])

    def any_intensity(self):
        '''True if some image has its intesnity calculated'''
        return any([image_group.check_intensity.value
                    for image_group in self.image_groups])

    def get_measurement_columns(self, pipeline, return_sources=False):
        '''Return column definitions for all measurements'''
        columns = []
        sources = []
        for image_group in self.image_groups:
            selected_images = self.images_to_process(image_group, None, pipeline)
            # Image scalings
            if image_group.include_image_scalings.value:
                for image_name in selected_images:
                    columns.append((cpmeas.IMAGE,
                                    '%s_%s_%s' % (C_IMAGE_QUALITY, C_SCALING,
                                                  image_name),
                                    cpmeas.COLTYPE_FLOAT))
                    sources.append([image_group.include_image_scalings, image_name])

            # Blur measurements
            if image_group.check_blur.value:
                for image_name in selected_images:
                    columns.append((cpmeas.IMAGE,
                                    '%s_%s_%s' % (C_IMAGE_QUALITY, F_FOCUS_SCORE,
                                                  image_name),
                                    cpmeas.COLTYPE_FLOAT))
                    sources.append([image_group.check_blur, image_name])

                    columns.append((cpmeas.IMAGE,
                                    '%s_%s_%s' % (C_IMAGE_QUALITY, F_POWER_SPECTRUM_SLOPE,
                                                  image_name),
                                    cpmeas.COLTYPE_FLOAT))
                    sources.append([image_group.check_blur, image_name])

                    for scale_group in image_group.scale_groups:
                        columns.append((cpmeas.IMAGE,
                                        '%s_%s_%s_%d' % (C_IMAGE_QUALITY, F_LOCAL_FOCUS_SCORE,
                                                         image_name,
                                                         scale_group.scale.value),
                                        cpmeas.COLTYPE_FLOAT))
                        sources.append([scale_group.scale, image_name])

                        columns.append((cpmeas.IMAGE,
                                        '%s_%s_%s_%d' % (C_IMAGE_QUALITY, F_CORRELATION,
                                                         image_name,
                                                         scale_group.scale.value),
                                        cpmeas.COLTYPE_FLOAT))
                        sources.append([scale_group.scale, image_name])

            # Intensity measurements
            if image_group.check_intensity.value:
                for image_name in selected_images:
                    for feature in INTENSITY_FEATURES:
                        measurement_name = image_name
                        columns.append((cpmeas.IMAGE,
                                        '%s_%s_%s' % (C_IMAGE_QUALITY, feature,
                                                      measurement_name),
                                        cpmeas.COLTYPE_FLOAT))
                        sources.append([image_group.check_intensity, image_name])

            # Saturation measurements
            if image_group.check_saturation.value:
                for image_name in selected_images:
                    for feature in SATURATION_FEATURES:
                        columns.append((cpmeas.IMAGE,
                                        '%s_%s_%s' % (C_IMAGE_QUALITY, feature,
                                                      image_name),
                                        cpmeas.COLTYPE_FLOAT))
                        sources.append([image_group.check_saturation, image_name])

            # Threshold measurements
            if image_group.calculate_threshold.value:
                all_threshold_groups = self.get_all_threshold_groups(image_group)
                for image_name in selected_images:
                    for threshold_group in all_threshold_groups:
                        feature = threshold_group.threshold_feature_name(image_name)
                        columns.append((cpmeas.IMAGE, feature, cpmeas.COLTYPE_FLOAT))
                        for agg in ("Mean", "Median", "Std"):
                            feature = threshold_group.threshold_feature_name(
                                    image_name, agg)
                            columns.append(
                                    (cpmeas.EXPERIMENT, feature, cpmeas.COLTYPE_FLOAT,
                                     {cpmeas.MCA_AVAILABLE_POST_RUN: True}))

                        if image_group.use_all_threshold_methods:
                            sources.append([image_group.use_all_threshold_methods, image_name])
                        else:
                            sources.append([threshold_group.threshold_method, image_name])

        if return_sources:
            return columns, sources
        else:
            return columns

    def get_categories(self, pipeline, object_name):
        if object_name == cpmeas.IMAGE:
            return [C_IMAGE_QUALITY]
        elif object_name == cpmeas.EXPERIMENT and self.any_threshold():
            return [C_IMAGE_QUALITY]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == cpmeas.IMAGE and category == C_IMAGE_QUALITY:
            result = []
            if self.any_scaling():
                result += [C_SCALING]
            if self.any_blur():
                result += [F_FOCUS_SCORE, F_LOCAL_FOCUS_SCORE, F_POWER_SPECTRUM_SLOPE, F_CORRELATION]
            if self.any_intensity():
                result += INTENSITY_FEATURES
            if self.any_saturation():
                result += SATURATION_FEATURES
            if self.any_threshold():
                thresholds = []
                for image_group in self.image_groups:
                    all_threshold_groups = self.build_threshold_parameter_list() \
                        if image_group.use_all_threshold_methods.value \
                        else image_group.threshold_groups
                    thresholds += [F_THRESHOLD + threshold_group.threshold_algorithm
                                   for threshold_group in all_threshold_groups
                                   if image_group.calculate_threshold.value]
                result += sorted(list(set(thresholds)))

            return result
        elif object_name == cpmeas.EXPERIMENT and category == C_IMAGE_QUALITY:
            return [MEAN_THRESH_ALL_IMAGES, MEDIAN_THRESH_ALL_IMAGES,
                    STD_THRESH_ALL_IMAGES]
        return []

    def get_measurement_images(self, pipeline, object_name, category,
                               measurement):

        if object_name != cpmeas.IMAGE or category != C_IMAGE_QUALITY:
            return []
        if measurement in (F_FOCUS_SCORE, F_LOCAL_FOCUS_SCORE, F_POWER_SPECTRUM_SLOPE, F_CORRELATION):
            result = []
            for image_group in self.image_groups:
                if image_group.check_blur.value:
                    result += self.images_to_process(image_group, None, pipeline)
            return result

        if measurement in SATURATION_FEATURES:
            result = []
            for image_group in self.image_groups:
                if image_group.check_saturation.value:
                    result += self.images_to_process(image_group, None, pipeline)
            return result

        if measurement in INTENSITY_FEATURES:
            result = []
            for image_group in self.image_groups:
                if image_group.check_intensity.value:
                    result += self.images_to_process(image_group, None, pipeline)
            return result

        if measurement.startswith(F_THRESHOLD):
            result = []
            for image_group in self.image_groups:
                all_threshold_groups = self.build_threshold_parameter_list() \
                    if image_group.use_all_threshold_methods.value \
                    else image_group.threshold_groups
                for threshold_group in all_threshold_groups:
                    if (image_group.calculate_threshold.value and
                                measurement == F_THRESHOLD + threshold_group.threshold_algorithm):
                        result += self.images_to_process(image_group, None, pipeline)
            return result

    def get_measurement_scales(self, pipeline, object_name, category,
                               measurement, image_names):
        '''Get the scales (window_sizes) for the given measurement'''
        if object_name == cpmeas.IMAGE and category == C_IMAGE_QUALITY:
            if measurement in (F_LOCAL_FOCUS_SCORE, F_CORRELATION):
                result = []
                for image_group in self.image_groups:
                    for scale_group in image_group.scale_groups:
                        if image_names in self.images_to_process(image_group, None, pipeline):
                            result += [scale_group.scale]
                return result
            if measurement.startswith(F_THRESHOLD):
                result = []
                for image_group in self.image_groups:
                    all_threshold_groups = self.build_threshold_parameter_list() \
                        if image_group.use_all_threshold_methods.value \
                        else image_group.threshold_groups
                    result += [threshold_group.threshold_scale for threshold_group in all_threshold_groups
                               if ((measurement == F_THRESHOLD + threshold_group.threshold_algorithm) and
                                   threshold_group.threshold_scale is not None)]
                return result
        return []

    def run(self, workspace):
        '''Calculate statistics over all image groups'''
        statistics = []
        for image_group in self.image_groups:
            statistics += self.run_on_image_group(image_group, workspace)
        workspace.display_data.statistics = statistics

    def display(self, workspace, figure):
        if self.show_window:
            statistics = workspace.display_data.statistics
            figure.set_subplots((1, 1))
            figure.subplot_table(0, 0, statistics)

    def post_run(self, workspace):
        '''Calculate the experiment statistics at the end of a run'''
        statistics = []
        for image_group in self.image_groups:
            statistics += self.calculate_experiment_threshold(image_group, workspace)

    def run_on_image_group(self, image_group, workspace):
        '''Calculate statistics for a particular image'''
        statistics = []
        if image_group.include_image_scalings.value:
            statistics += self.retrieve_image_scalings(image_group, workspace)
        if image_group.check_blur.value:
            statistics += self.calculate_focus_scores(image_group, workspace)
            statistics += self.calculate_correlation(image_group, workspace)
            statistics += self.calculate_power_spectrum(image_group, workspace)
        if image_group.check_saturation.value:
            statistics += self.calculate_saturation(image_group, workspace)
        if image_group.check_intensity.value:
            statistics += self.calculate_image_intensity(image_group, workspace)
        if image_group.calculate_threshold.value:
            statistics += self.calculate_thresholds(image_group, workspace)

        return statistics

    def retrieve_image_scalings(self, image_group, workspace):
        '''Grab the scalings from the image '''

        result = []
        for image_name in self.images_to_process(image_group, workspace):
            feature = "%s_%s_%s" % (C_IMAGE_QUALITY, C_SCALING, image_name)
            value = workspace.image_set.get_image(image_name).scale
            if not value:  # Set to NaN if not defined, such as for derived images
                value = np.NaN
            workspace.add_measurement(cpmeas.IMAGE, feature, value)
            result += [["%s scaling" % image_name, value]]
        return result

    def calculate_focus_scores(self, image_group, workspace):
        '''Calculate a local blur measurement and a image-wide one'''

        result = []
        for image_name in self.images_to_process(image_group, workspace):

            image = workspace.image_set.get_image(image_name,
                                                  must_be_grayscale=True)
            pixel_data = image.pixel_data
            shape = image.pixel_data.shape
            if image.has_mask:
                pixel_data = pixel_data[image.mask]

            local_focus_score = []
            for scale_group in image_group.scale_groups:
                scale = scale_group.scale.value

                focus_score = 0
                if len(pixel_data):
                    mean_image_value = np.mean(pixel_data)
                    squared_normalized_image = (pixel_data - mean_image_value) ** 2
                    if mean_image_value > 0:
                        focus_score = (np.sum(squared_normalized_image) /
                                       (np.product(pixel_data.shape) * mean_image_value))
                #
                # Create a labels matrix that grids the image to the dimensions
                # of the window size
                #
                if image.dimensions is 2:
                    i, j = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
                    m, n = (np.array(shape) + scale - 1) / scale
                    i = (i * float(m) / float(shape[0])).astype(int)
                    j = (j * float(n) / float(shape[1])).astype(int)
                    grid = i * n + j + 1
                else:
                    k, i, j = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(float)
                    o, m, n = (np.array(shape) + scale - 1) / scale
                    k = (k * float(o) / float(shape[0])).astype(int)
                    i = (i * float(m) / float(shape[1])).astype(int)
                    j = (j * float(n) / float(shape[2])).astype(int)
                    grid = k * o + i * n + j + 1  # hmm

                if image.has_mask:
                    grid[np.logical_not(image.mask)] = 0
                grid_range = np.arange(0, m * n + 1, dtype=np.int32)
                #
                # Do the math per label
                #
                local_means = fix(scind.mean(image.pixel_data, grid, grid_range))
                local_squared_normalized_image = (image.pixel_data -
                                                  local_means[grid]) ** 2
                #
                # Compute the sum of local_squared_normalized_image values for each
                # grid for means > 0. Exclude grid label = 0 because that's masked
                #
                grid_mask = (local_means != 0) & ~ np.isnan(local_means)
                nz_grid_range = grid_range[grid_mask]
                if len(nz_grid_range) and nz_grid_range[0] == 0:
                    nz_grid_range = nz_grid_range[1:]
                    local_means = local_means[1:]
                    grid_mask = grid_mask[1:]
                local_focus_score += [0]  # assume the worst - that we can't calculate it
                if len(nz_grid_range):
                    sums = fix(scind.sum(local_squared_normalized_image, grid,
                                         nz_grid_range))
                    pixel_counts = fix(scind.sum(np.ones(shape), grid, nz_grid_range))
                    local_norm_var = (sums /
                                      (pixel_counts * local_means[grid_mask]))
                    local_norm_median = np.median(local_norm_var)
                    if np.isfinite(local_norm_median) and local_norm_median > 0:
                        local_focus_score[-1] = np.var(local_norm_var) / local_norm_median

            #
            # Add the measurements
            #
            focus_score_name = "%s_%s_%s" % (C_IMAGE_QUALITY, F_FOCUS_SCORE,
                                             image_name)
            workspace.add_measurement(cpmeas.IMAGE, focus_score_name,
                                      focus_score)
            result += [["%s focus score @%d" % (image_name,
                                                scale), focus_score]]

            for idx, scale_group in enumerate(image_group.scale_groups):
                scale = scale_group.scale.value
                local_focus_score_name = "%s_%s_%s_%d" % (C_IMAGE_QUALITY,
                                                          F_LOCAL_FOCUS_SCORE,
                                                          image_name,
                                                          scale)
                workspace.add_measurement(cpmeas.IMAGE, local_focus_score_name,
                                          local_focus_score[idx])
                result += [["%s local focus score @%d" % (image_name,
                                                          scale), local_focus_score[idx]]]

        return result

    def calculate_correlation(self, image_group, workspace):
        '''Calculate a correlation measure from the Harlick feature set'''
        result = []
        for image_name in self.images_to_process(image_group, workspace):
            image = workspace.image_set.get_image(image_name,
                                                  must_be_grayscale=True)
            pixel_data = image.pixel_data

            # Compute Haralick's correlation texture for the given scales
            image_labels = np.ones(pixel_data.shape, int)
            if image.has_mask:
                image_labels[~image.mask] = 0
            for scale_group in image_group.scale_groups:
                scale = scale_group.scale.value

                value = centrosome.haralick.Haralick(pixel_data, image_labels, 0, scale).H3()
                if not np.isfinite(value):
                    value = 0.0
                workspace.add_measurement(cpmeas.IMAGE, "%s_%s_%s_%d" %
                                          (C_IMAGE_QUALITY, F_CORRELATION,
                                           image_name, scale),
                                          float(value))
                result += [["%s %s @%d" % (image_name, F_CORRELATION, scale), "%.2f" % (float(value))]]
        return result

    def calculate_saturation(self, image_group, workspace):
        '''Count the # of pixels at saturation'''

        result = []
        for image_name in self.images_to_process(image_group, workspace):
            image = workspace.image_set.get_image(image_name,
                                                  must_be_grayscale=True)
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
            percent_maximal_name = "%s_%s_%s" % (C_IMAGE_QUALITY, F_PERCENT_MAXIMAL,
                                                 image_name)
            percent_minimal_name = "%s_%s_%s" % (C_IMAGE_QUALITY, F_PERCENT_MINIMAL,
                                                 image_name)
            workspace.add_measurement(cpmeas.IMAGE, percent_maximal_name,
                                      percent_maximal)
            workspace.add_measurement(cpmeas.IMAGE, percent_minimal_name,
                                      percent_minimal)
            result += [["%s maximal" % image_name, "%.1f %%" % percent_maximal],
                       ["%s minimal" % image_name, "%.1f %%" % percent_minimal]]
        return result

    def calculate_image_intensity(self, image_group, workspace):
        '''Calculate intensity-based metrics, mostly from MeasureImageIntensity'''

        result = []
        for image_name in self.images_to_process(image_group, workspace):
            result += self.run_intensity_measurement(image_name, workspace)
        return result

    def run_intensity_measurement(self, image_name, workspace):
        image = workspace.image_set.get_image(image_name,
                                              must_be_grayscale=True)
        pixels = image.pixel_data
        if image.has_mask:
            pixels = pixels[image.mask]

        result = []
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
            pixel_mean = pixel_sum / float(pixel_count)
            pixel_std = np.std(pixels)
            pixel_median = np.median(pixels)
            pixel_mad = np.median(np.abs(pixels - pixel_median))
            pixel_min = np.min(pixels)
            pixel_max = np.max(pixels)

        m = workspace.measurements
        m.add_image_measurement("_".join((C_IMAGE_QUALITY, F_TOTAL_AREA, image_name)), pixel_count)
        m.add_image_measurement("_".join((C_IMAGE_QUALITY, F_TOTAL_INTENSITY, image_name)), pixel_sum)
        m.add_image_measurement("_".join((C_IMAGE_QUALITY, F_MEAN_INTENSITY, image_name)), pixel_mean)
        m.add_image_measurement("_".join((C_IMAGE_QUALITY, F_MEDIAN_INTENSITY, image_name)), pixel_median)
        m.add_image_measurement("_".join((C_IMAGE_QUALITY, F_STD_INTENSITY, image_name)), pixel_std)
        m.add_image_measurement("_".join((C_IMAGE_QUALITY, F_MAD_INTENSITY, image_name)), pixel_mad)
        m.add_image_measurement("_".join((C_IMAGE_QUALITY, F_MAX_INTENSITY, image_name)), pixel_max)
        m.add_image_measurement("_".join((C_IMAGE_QUALITY, F_MIN_INTENSITY, image_name)), pixel_min)

        result = [["%s %s" % (image_name,
                              feature_name),
                   "%.2f" % value]
                  for feature_name, value in (('Total intensity', pixel_sum),
                                              ('Mean intensity', pixel_mean),
                                              ('Median intensity', pixel_median),
                                              ('Std intensity', pixel_std),
                                              ('MAD intensity', pixel_mad),
                                              ('Min intensity', pixel_min),
                                              ('Max intensity', pixel_max),
                                              ('Total area', pixel_count))]
        return result

    def calculate_power_spectrum(self, image_group, workspace):
        result = []
        for image_name in self.images_to_process(image_group, workspace):
            image = workspace.image_set.get_image(image_name,
                                                  must_be_grayscale=True)

            if image.dimensions is 3:
                # TODO: calculate "radial power spectrum" for volumes.
                continue

            pixel_data = image.pixel_data

            if image.has_mask:
                pixel_data = np.array(pixel_data)  # make a copy
                masked_pixels = pixel_data[image.mask]
                pixel_count = np.product(masked_pixels.shape)
                if pixel_count > 0:
                    pixel_data[~ image.mask] = np.mean(masked_pixels)
                else:
                    pixel_data[~ image.mask] = 0

            radii, magnitude, power = rps.rps(pixel_data)
            if sum(magnitude) > 0 and len(np.unique(pixel_data)) > 1:
                valid = (magnitude > 0)
                radii = radii[valid].reshape((-1, 1))
                magnitude = magnitude[valid].reshape((-1, 1))
                power = power[valid].reshape((-1, 1))
                if radii.shape[0] > 1:
                    idx = np.isfinite(np.log(power))
                    powerslope = \
                        lstsq(np.hstack((np.log(radii)[idx][:, np.newaxis], np.ones(radii.shape)[idx][:, np.newaxis])),
                              np.log(power)[idx][:, np.newaxis])[0][0]
                else:
                    powerslope = 0
            else:
                powerslope = 0

            workspace.add_measurement(cpmeas.IMAGE,
                                      "%s_%s_%s" % (C_IMAGE_QUALITY, F_POWER_SPECTRUM_SLOPE, image_name),
                                      powerslope)
            result += [["%s %s" % (image_name, F_POWER_SPECTRUM_SLOPE), "%.1f" % powerslope]]
        return result

    def calculate_thresholds(self, image_group, workspace):
        '''Calculate a threshold for this image'''
        result = []
        all_threshold_groups = self.get_all_threshold_groups(image_group)

        for image_name in self.images_to_process(image_group, workspace):
            image = workspace.image_set.get_image(image_name,
                                                  must_be_grayscale=True)

            # TODO: works on 2D slice of image, i suspect the thresholding methods in centrosome aren't working in 3D
            pixel_data = image.pixel_data.astype(np.float32)

            for threshold_group in all_threshold_groups:
                threshold_method = threshold_group.threshold_algorithm
                object_fraction = threshold_group.object_fraction.value
                two_class_otsu = (threshold_group.two_class_otsu.value == O_TWO_CLASS)
                use_weighted_variance = (threshold_group.use_weighted_variance.value == O_WEIGHTED_VARIANCE)
                assign_middle_to_foreground = (threshold_group.assign_middle_to_foreground.value == O_FOREGROUND)
                (local_threshold, global_threshold) = \
                    (cpthresh.get_threshold(threshold_method,
                                            cpthresh.TM_GLOBAL,
                                            pixel_data,
                                            mask=image.mask,
                                            object_fraction=object_fraction,
                                            two_class_otsu=two_class_otsu,
                                            use_weighted_variance=use_weighted_variance,
                                            assign_middle_to_foreground=assign_middle_to_foreground)
                     if image.has_mask
                     else
                     cpthresh.get_threshold(threshold_method,
                                            cpthresh.TM_GLOBAL,
                                            pixel_data,
                                            object_fraction=object_fraction,
                                            two_class_otsu=two_class_otsu,
                                            use_weighted_variance=use_weighted_variance,
                                            assign_middle_to_foreground=assign_middle_to_foreground))

                scale = threshold_group.threshold_scale
                if scale is None:
                    threshold_description = threshold_method
                else:
                    threshold_description = threshold_method + " " + scale
                workspace.add_measurement(cpmeas.IMAGE, threshold_group.threshold_feature_name(image_name),
                                          global_threshold)
                result += [["%s %s threshold" % (image_name, threshold_description), str(global_threshold)]]

        return result

    def get_all_threshold_groups(self, image_group):
        '''Get all threshold groups to apply to an image group

        image_group - the image group to try thresholding on
        '''
        if image_group.use_all_threshold_methods.value:
            return self.build_threshold_parameter_list()
        return image_group.threshold_groups

    def calculate_experiment_threshold(self, image_group, workspace):
        '''Calculate experiment-wide threshold mean, median and standard-deviation'''
        m = workspace.measurements
        statistics = []
        all_threshold_groups = self.get_all_threshold_groups(image_group)
        if image_group.calculate_threshold.value:
            for image_name in self.images_to_process(image_group, workspace):
                for threshold_group in all_threshold_groups:
                    values = m.get_all_measurements(cpmeas.IMAGE,
                                                    threshold_group.threshold_feature_name(image_name))

                    values = values[np.isfinite(values)]

                    for feature in (F_THRESHOLD,):
                        for fn, agg in ((np.mean, AGG_MEAN),
                                        (np.median, AGG_MEDIAN),
                                        (np.std, AGG_STD)):
                            feature_name = threshold_group.threshold_feature_name(
                                    image_name, agg=agg)
                            feature_description = threshold_group.threshold_description(
                                    image_name, agg=agg)
                            val = fn(values)
                            m.add_experiment_measurement(feature_name, val)
                        statistics.append([feature_description, str(val)])
        return statistics

    def build_threshold_parameter_list(self):
        '''Build a set of temporary threshold groups containing all the threshold methods to be tested'''

        # Produce a list of meaningful combinations of threshold settings.'''
        threshold_args = []
        object_fraction = [0.05, 0.25, 0.75, 0.95]
        # Produce list of combinations of the special thresholding method parameters: Otsu, MoG
        z = itertools.product([cpthresh.TM_OTSU], [0], [O_WEIGHTED_VARIANCE, O_ENTROPY], [O_THREE_CLASS],
                    [O_FOREGROUND, O_BACKGROUND])
        threshold_args += [i for i in z]
        z = itertools.product([cpthresh.TM_OTSU], [0], [O_WEIGHTED_VARIANCE, O_ENTROPY], [O_TWO_CLASS], [O_FOREGROUND])
        threshold_args += [i for i in z]
        z = itertools.product([cpthresh.TM_MOG], object_fraction, [O_WEIGHTED_VARIANCE], [O_TWO_CLASS], [O_FOREGROUND])
        threshold_args += [i for i in z]
        # Tack on the remaining simpler methods
        leftover_methods = [i for i in cpthresh.TM_METHODS if i not in [cpthresh.TM_OTSU, cpthresh.TM_MOG]]
        z = itertools.product(leftover_methods, [0], [O_WEIGHTED_VARIANCE], [O_TWO_CLASS], [O_FOREGROUND])
        threshold_args += [i for i in z]

        # Assign the threshold values to a temporary threshold group
        threshold_groups = []
        for threshold_method, object_fraction, use_weighted_variance, two_class_otsu, assign_middle_to_foreground in threshold_args:
            threshold_groups.append(self.add_threshold_group(None, False))
            threshold_groups[-1].threshold_method.value = threshold_method
            threshold_groups[-1].object_fraction.value = object_fraction
            threshold_groups[-1].two_class_otsu.value = two_class_otsu
            threshold_groups[-1].use_weighted_variance.value = use_weighted_variance
            threshold_groups[-1].assign_middle_to_foreground.value = assign_middle_to_foreground

        return threshold_groups

    def images_to_process(self, image_group, workspace, pipeline=None):
        '''Return a list of input image names appropriate to the setting choice '''
        if self.images_choice.value == O_SELECT:
            return image_group.image_names.get_selections()
        elif self.images_choice.value == O_ALL_LOADED:
            # Grab all loaded images
            accepted_image_list = []
            if pipeline is None:
                pipeline = workspace.pipeline
            #
            # Get a dictionary of image name to (module, setting)
            #
            image_providers = pipeline.get_provider_dictionary(
                    cps.IMAGE_GROUP, self)
            for image_name in image_providers:
                for module, setting in image_providers[image_name]:
                    if (module.is_load_module() and
                            ((not isinstance(setting, cps.ImageNameProvider)) or
                                     cps.FILE_IMAGE_ATTRIBUTE in setting.provided_attributes)):
                        accepted_image_list.append(image_name)
            return accepted_image_list

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
                                   cps.YES,  # check saturation
                                   cps.NO,  # calculate threshold
                                   cpthresh.TM_OTSU_GLOBAL,
                                   .1,  # object fraction
                                   cps.NO]  # compute power spectrum
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
            for i in range(2, 14, 3):
                saturation_image = setting_values[i]
                threshold_image = setting_values[i + 1]
                threshold_method = setting_values[i + 2]
                if saturation_image != cps.DO_NOT_USE:
                    if not d.has_key(saturation_image):
                        d[saturation_image] = {"check_blur": check_blur,
                                               "check_saturation": cps.YES,
                                               "check_threshold": cps.NO,
                                               "threshold_method": threshold_method}
                    else:
                        d[saturation_image]["check_blur"] = check_blur
                        d[saturation_image]["check_saturation"] = cps.YES
                if threshold_image != cps.DO_NOT_USE:
                    if not d.has_key(threshold_image):
                        d[threshold_image] = {"check_blur": cps.NO,
                                              "check_saturation": cps.NO,
                                              "check_threshold": cps.YES,
                                              "threshold_method": threshold_method}
                    else:
                        d[threshold_image]["check_threshold"] = cps.YES
                        d[threshold_image]["threshold_method"] = threshold_method
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

        if (not from_matlab) and variable_revision_number == 3:
            # Rearrangement/consolidation of settings
            assert len(setting_values) % SETTINGS_PER_GROUP_V3 == 0
            num_images = len(setting_values) / SETTINGS_PER_GROUP_V3

            '''Since some settings are new/consolidated and can be repeated, handle
            the old settings by using a dict'''
            # Initialize the dictionary by image name
            d = {}
            unique_image_names = []
            for idx in range(num_images):
                # Get the settings belonging to each image
                im_settings = setting_values[
                              (idx * SETTINGS_PER_GROUP_V3):(idx * SETTINGS_PER_GROUP_V3 + SETTINGS_PER_GROUP_V3)]
                unique_image_names += [im_settings[0]]
            unique_image_names = sorted(set(unique_image_names), key=unique_image_names.index)
            # Assume that the user doesn't want blur and thresholds
            for image_name in unique_image_names:
                d[image_name] = {}
                d[image_name]["wants_scaling"] = True
                d[image_name]["wants_saturation"] = False
                d[image_name]["wants_blur"] = False
                d[image_name]["blur_scales"] = []
                d[image_name]["wants_intensity"] = True
                d[image_name]["wants_threshold"] = False
                d[image_name]["threshold_methods"] = []

            for idx in range(num_images):
                im_settings = setting_values[
                              (idx * SETTINGS_PER_GROUP_V3):(idx * SETTINGS_PER_GROUP_V3 + SETTINGS_PER_GROUP_V3)]
                image_name = im_settings[0]
                # Set blur and thresholds if the user sets any of the setting groups.
                d[image_name]["wants_saturation"] = d[image_name]["wants_saturation"] or (im_settings[3] == cps.YES)
                d[image_name]["wants_blur"] = d[image_name]["wants_blur"] or (
                    im_settings[1] == cps.YES or im_settings[7] == cps.YES)
                d[image_name]["wants_threshold"] = d[image_name]["wants_threshold"] or (im_settings[4] == cps.YES)
                #  Collect blur scales and threshold methods
                d[image_name]["blur_scales"] += [im_settings[2]]
                d[image_name]["threshold_methods"] += [im_settings[5:7] + im_settings[8:]]

            # Uniquify the scales and threshold methods
            import itertools
            for image_name in d.keys():
                d[image_name]["blur_scales"] = list(set(d[image_name]["blur_scales"]))
                d[image_name]["threshold_methods"] = [k for k, v in
                                                      itertools.groupby(sorted(d[image_name]["threshold_methods"]))]

            # Create the new settings
            new_settings = [O_SELECT, str(len(unique_image_names))]  # images_choice, image_count
            new_settings += [str(len(d[image_name]["blur_scales"])) for image_name in unique_image_names]  # scale_count
            new_settings += [str(len(d[image_name]["threshold_methods"])) for image_name in
                             unique_image_names]  # threshold_count
            for image_name in unique_image_names:
                new_settings += [image_name,  # image_name
                                 cps.YES if d[image_name]["wants_scaling"] else cps.NO,  # include_image_scalings
                                 cps.YES if d[image_name]["wants_blur"]    else cps.NO]  # check_blur
                new_settings += [k for k in d[image_name]["blur_scales"]]  # scale
                new_settings += [cps.YES if d[image_name]["wants_saturation"] else cps.NO,  # check_saturation
                                 cps.YES if d[image_name]["wants_intensity"]  else cps.NO,  # check_intensity
                                 cps.YES if d[image_name]["wants_threshold"]  else cps.NO,  # calculate_threshold,
                                 cps.NO]  # use_all_threshold_methods
                for k in d[image_name]["threshold_methods"]:
                    new_settings += k  # threshold_method, object_fraction, two_class_otsu, use_weighted_variance, assign_middle_to_foreground

            setting_values = new_settings
            variable_revision_number = 4

        if (not from_matlab) and variable_revision_number == 4:
            # Thresholding method name change: Strip off "Global"
            thresh_dict = dict(zip(cpthresh.TM_GLOBAL_METHODS, cpthresh.TM_METHODS))
            # Naturally, this method assumes that the user didn't name their images "Otsu Global" or something similar
            setting_values = [thresh_dict[x] if x in cpthresh.TM_GLOBAL_METHODS else x for x in setting_values]
            variable_revision_number = 5

        return setting_values, variable_revision_number, from_matlab

    def volumetric(self):
        return True


class ImageQualitySettingsGroup(cps.SettingsGroup):
    @property
    def threshold_algorithm(self):
        '''The thresholding algorithm to run'''
        return self.threshold_method.value.split(' ')[0]

    def threshold_feature_name(self, image_name, agg=None):
        '''The feature name of the threshold measurement generated'''
        scale = self.threshold_scale
        if agg is None:
            hdr = F_THRESHOLD
        else:
            hdr = F_THRESHOLD + agg
        if scale is None:
            return "%s_%s%s_%s" % (C_IMAGE_QUALITY, hdr,
                                   self.threshold_algorithm,
                                   image_name)
        else:
            return "%s_%s%s_%s_%s" % (C_IMAGE_QUALITY, hdr,
                                      self.threshold_algorithm,
                                      image_name,
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

    def threshold_description(self, image_name, agg=None):
        '''Return a description of the threshold meant to be seen by the user

        image_name - name of thresholded image

        agg - if present, the aggregating method, e.g. "Mean"
        '''
        if self.threshold_algorithm == cpthresh.TM_OTSU:
            if self.use_weighted_variance == O_WEIGHTED_VARIANCE:
                wvorentropy = "WV"
            else:
                wvorentropy = "S"
            if self.two_class_otsu == O_TWO_CLASS:
                result = "Otsu %s 2 cls" % wvorentropy
            else:
                result = "Otsu %s 3 cls" % wvorentropy
                if self.assign_middle_to_foreground == O_FOREGROUND:
                    result += " Fg"
                else:
                    result += " Bg"
        elif self.threshold_scale is not None:
            result = self.threshold_algorithm.lower() + " " + self.threshold_scale
        else:
            result = self.threshold_algorithm.lower()
        if agg is not None:
            result = agg + " " + image_name + result
        else:
            result = image_name + result
        return result
