"""
<b>Apply Threshold</b> sets pixel intensities below or above a certain threshold to zero
<hr>
<b>ApplyThreshold</b> produces a grayscale image based on a threshold which can be
pre-selected or calculated automatically using one of many methods.
"""

import centrosome.cpmorphology
import centrosome.threshold
import numpy
import scipy.ndimage.morphology

import cellprofiler.gui.help
import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.identify
import cellprofiler.setting
import identify

O_TWO_CLASS = "Two classes"
O_THREE_CLASS = "Three classes"

O_FOREGROUND = "Foreground"
O_BACKGROUND = "Background"

RB_MEAN = "Mean"
RB_MEDIAN = "Median"
RB_MODE = "Mode"
RB_SD = "Standard deviation"
RB_MAD = "Median absolute deviation"

TS_GLOBAL = "Global"
TS_ADAPTIVE = "Adaptive"
TS_MANUAL = "Manual"
TS_MEASUREMENT = "Measurement"

TS_ALL = [TS_GLOBAL, TS_ADAPTIVE, TS_MANUAL, TS_MEASUREMENT]

C_LOCATION = "Location"
C_NUMBER = "Number"
C_COUNT = "Count"
C_THRESHOLD = "Threshold"
C_PARENT = "Parent"
R_PARENT = "Parent"
C_CHILDREN = "Children"
R_CHILD = "Child"

FTR_CENTER_X = "Center_X"
M_LOCATION_CENTER_X = '%s_%s' % (C_LOCATION, FTR_CENTER_X)

FTR_CENTER_Y = "Center_Y"
M_LOCATION_CENTER_Y = '%s_%s' % (C_LOCATION, FTR_CENTER_Y)

FTR_OBJECT_NUMBER = "Object_Number"

M_NUMBER_OBJECT_NUMBER = '%s_%s' % (C_NUMBER, FTR_OBJECT_NUMBER)

FF_COUNT = '%s_%%s' % C_COUNT

FTR_FINAL_THRESHOLD = "FinalThreshold"
FF_FINAL_THRESHOLD = '%s_%s_%%s' % (C_THRESHOLD, FTR_FINAL_THRESHOLD)

FTR_ORIG_THRESHOLD = "OrigThreshold"
FF_ORIG_THRESHOLD = '%s_%s_%%s' % (C_THRESHOLD, FTR_ORIG_THRESHOLD)

FTR_WEIGHTED_VARIANCE = "WeightedVariance"
FF_WEIGHTED_VARIANCE = '%s_%s_%%s' % (C_THRESHOLD, FTR_WEIGHTED_VARIANCE)

FTR_SUM_OF_ENTROPIES = "SumOfEntropies"
FF_SUM_OF_ENTROPIES = '%s_%s_%%s' % (C_THRESHOLD, FTR_SUM_OF_ENTROPIES)

FF_CHILDREN_COUNT = "%s_%%s_Count" % C_CHILDREN

FF_PARENT = "%s_%%s" % C_PARENT

PROTIP_RECOMEND_ICON = "thumb-up.png"
PROTIP_AVOID_ICON = "thumb-down.png"
TECH_NOTE_ICON = "gear.png"


class ApplyThreshold(cellprofiler.module.ImageProcessing):
    module_name = "ApplyThreshold"

    variable_revision_number = 9

    def create_settings(self):
        super(ApplyThreshold, self).create_settings()

        self.threshold_scope = cellprofiler.setting.Choice(
            "Threshold strategy",
            TS_ALL,
            value=TS_GLOBAL,
            doc="""
            The thresholding strategy determines the type of input that is used to calculate the threshold. The
            image thresholds can be based on:
            <ul>
                <li>The pixel intensities of the input image (this is the most common).</li>
                <li>A single value manually provided by the user.</li>
                <li>A single value produced by a prior module measurement.</li>
            </ul>These options allow you to calculate a threshold based on the whole image or based on image
            sub-regions.<br>
            The choices for the threshold strategy are:<br>
            <ul>
                <li>
                    <i>{TS_GLOBAL}:</i> Calculate a single threshold value based on the unmasked pixels of the
                    input image and use that value to classify pixels above the threshold as foreground and
                    below as background.
                    <dl>
                        <dd><img src="memory:{PROTIP_RECOMEND_ICON}">&nbsp; This strategy is fast and robust,
                        especially if the background is uniformly illuminated.</dd>
                    </dl>
                </li>
                <li>
                    <i>{TS_ADAPTIVE}:</i> Partition the input image into tiles and calculate thresholds for
                    each tile. For each tile, the calculated threshold is applied only to the pixels within
                    that tile.<br>
                    <dl>
                        <dd><img src="memory:{PROTIP_RECOMEND_ICON}">&nbsp; This method is slower but can
                        produce better results for non-uniform backgrounds. However, for signifcant
                        illumination variation, using the <b>CorrectIllumination</b> modules is
                        preferable.</dd>
                    </dl>
                </li>
                <li>
                    <i>{TS_MANUAL}:</i> Enter a single value between zero and one that applies to all cycles
                    and is independent of the input image.
                    <dl>
                        <dd><img src="memory:{PROTIP_RECOMEND_ICON}">&nbsp; This approach is useful if the
                        input image has a stable or negligible background, or if the input image is the
                        probability map output of the <b>ClassifyPixels</b> module (in which case, a value of
                        0.5 should be chosen). If the input image is already binary (i.e., where the foreground
                        is 1 and the background is 0), a manual value of 0.5 will identify the objects.</dd>
                    </dl>
                </li>
                <li>
                    <i>{TS_MEASUREMENT}:</i> Use a prior image measurement as the threshold. The measurement
                    should have values between zero and one. This strategy can be used to apply a
                    pre-calculated threshold imported as per-image metadata via the <b>Metadata</b> module.
                    <dl>
                        <dd><img src="memory:{PROTIP_RECOMEND_ICON}">&nbsp; Like manual thresholding, this
                        approach can be useful when you are certain what the cutoff should be. The difference
                        in this case is that the desired threshold does vary from image to image in the
                        experiment but can be measured using another module, such as one of the <b>Measure</b>
                        modules, <b>ApplyThreshold</b> or an <b>Identify</b> module.</dd>
                    </dl>
                </li>
            </ul>
            """.format(**{
                "PROTIP_RECOMEND_ICON": PROTIP_RECOMEND_ICON,
                "TM_MCT": centrosome.threshold.TM_MCT,
                "TM_OTSU": centrosome.threshold.TM_OTSU,
                "TS_ADAPTIVE": TS_ADAPTIVE,
                "TS_GLOBAL": TS_GLOBAL,
                "TS_MANUAL": TS_MANUAL,
                "TS_MEASUREMENT": TS_MEASUREMENT
            })
        )

        self.threshold_method = cellprofiler.setting.Choice(
            "Thresholding method",
            [
                centrosome.threshold.TM_MCT,
                centrosome.threshold.TM_OTSU,
                centrosome.threshold.TM_ROBUST_BACKGROUND
            ],
            value=centrosome.threshold.TM_MCT,
            doc="""
            The intensity threshold affects the decision of whether each pixel will be considered foreground
            (region(s) of interest) or background. A higher threshold value will result in only the brightest
            regions being identified, whereas a lower threshold value will include dim regions. You can have
            the threshold automatically calculated from a choice of several methods, or you can enter a number
            manually between 0 and 1 for the threshold.
            <p>Both the automatic and manual options have advantages and disadvantages.</p>
            <dl>
                <dd><img src="memory:{PROTIP_RECOMEND_ICON}">&nbsp; An automatically-calculated threshold
                adapts to changes in lighting/staining conditions between images and is usually more
                robust/accurate. In the vast majority of cases, an automatic method is sufficient to achieve
                the desired thresholding, once the proper method is selected.</dd>
                <dd>In contrast, an advantage of a manually-entered number is that it treats every image
                identically, so use this option when you have a good sense for what the threshold should be
                across all images. To help determine the choice of threshold manually, you can inspect the
                pixel intensities in an image of your choice. {HELP_ON_PIXEL_INTENSITIES}.</dd>
                <dd><img src="memory:{PROTIP_AVOID_ICON}">&nbsp; The manual method is not robust with regard to
                slight changes in lighting/staining conditions between images.</dd>
                <dd>The automatic methods may ocasionally produce a poor threshold for unusual or artifactual
                images. It also takes a small amount of time to calculate, which can add to processing time for
                analysis runs on a large number of images.</dd>
            </dl>
            <p></p>
            <p>The threshold that is used for each image is recorded as a per-image measurement, so if you are
            surprised by unusual measurements from one of your images, you might check whether the
            automatically calculated threshold was unusually high or low compared to the other images. See the
            <b>FlagImage</b> module if you would like to flag an image based on the threshold value.</p>
            <p>There are a number of methods for finding thresholds automatically:</p>
            <ul>
                <li>
                    <i>{TM_OTSU}:</i> This approach calculates the threshold separating the two classes of
                    pixels (foreground and background) by minimizing the variance within the each class.
                    <dl>
                        <dd><img src="memory:{PROTIP_RECOMEND_ICON}">&nbsp; This method is a good initial
                        approach if you do not know much about the image characteristics of all the images in
                        your experiment, especially if the percentage of the image covered by foreground varies
                        substantially from image to image.</dd>
                    </dl>
                    <dl>
                        <dd><img src="memory:{TECH_NOTE_ICON}">&nbsp; Our implementation of Otsu's method
                        allows for assigning the threshold value based on splitting the image into either two
                        classes (foreground and background) or three classes (foreground, mid-level, and
                        background). See the help below for more details.</dd>
                    </dl>
                </li>
                <li>
                    <i>{TM_ROBUST_BACKGROUND}:</i> This method assumes that the background distribution approximates a
                    Gaussian by trimming the brightest and dimmest 5% of pixel intensities. It then calculates the mean
                    and standard deviation of the remaining pixels and calculates the threshold as the mean + 2 times
                    the standard deviation.
                    <dl>
                        <dd><img src="memory:{PROTIP_RECOMEND_ICON}">&nbsp; This thresholding method can be
                        helpful if the majority of the image is background. It can also be helpful
                        if your images vary in overall brightness, but the objects of interest are consistently
                        <i>N</i> times brighter than the background level of the image.</dd>
                    </dl>
                </li>
                <li>
                    <i>Maximum correlation thresholding ({TM_MCT}):</i> This method computes the maximum
                    correlation between the binary mask created by thresholding and the thresholded image and
                    is somewhat similar mathematically to <i>{TM_OTSU}</i>.
                    <dl>
                        <dd><img src="memory:{PROTIP_RECOMEND_ICON}">&nbsp; The authors of this method claim
                        superior results when thresholding images of neurites and other images that have sparse
                        foreground densities.</dd>
                    </dl>
                    <dl>
                        <dd><img src="memory:{TECH_NOTE_ICON}">&nbsp; This is an implementation of the method
                        described in Padmanabhan <i>et al</i>, 2010.</dd>
                    </dl>
                </li>
            </ul>
            <p><b>References</b></p>
            <ul>
                <li>Sezgin M, Sankur B (2004) "Survey over image thresholding techniques and quantitative
                performance evaluation." <i>Journal of Electronic Imaging</i>, 13(1), 146-165. (<a href=
                "http://dx.doi.org/10.1117/1.1631315">link</a>)
                </li>
                <li>Padmanabhan K, Eddy WF, Crowley JC (2010) "A novel algorithm for optimal image thresholding
                of biological data" <i>Journal of Neuroscience Methods</i> 193, 380-384. (<a href=
                "http://dx.doi.org/10.1016/j.jneumeth.2010.08.031">link</a>)
                </li>
            </ul>
            <p></p>
            """.format(**{
                "HELP_ON_PIXEL_INTENSITIES": cellprofiler.gui.help.HELP_ON_PIXEL_INTENSITIES,
                "PROTIP_AVOID_ICON": PROTIP_AVOID_ICON,
                "PROTIP_RECOMEND_ICON": PROTIP_RECOMEND_ICON,
                "TECH_NOTE_ICON": TECH_NOTE_ICON,
                "TM_MCT": centrosome.threshold.TM_MCT,
                "TM_OTSU": centrosome.threshold.TM_OTSU,
                "TM_ROBUST_BACKGROUND": centrosome.threshold.TM_ROBUST_BACKGROUND
            })
        )

        self.threshold_smoothing_scale = cellprofiler.setting.Float(
            "Threshold smoothing scale",
            0,
            minval=0,
            doc="""
            This setting controls the scale used to smooth the input image before the threshold is applied.<br>
            The input image can be optionally smoothed before being thresholded. Smoothing can improve the
            uniformity of the resulting objects, by removing holes and jagged edges caused by noise in the
            acquired image. Smoothing is most likely <i>not</i> appropriate if the input image is binary, if it
            has already been smoothed or if it is an output of the <i>ClassifyPixels</i> module.<br>
            The scale should be approximately the size of the artifacts to be eliminated by smoothing. A Gaussian
            is used with a sigma adjusted so that 1/2 of the Gaussian's distribution falls within the diameter
            given by the scale (sigma = scale / 0.674)<br>
            Use a value of 0 for no smoothing. Use a value of 1.3488 for smoothing with a sigma of 1.
            """
        )

        self.threshold_correction_factor = cellprofiler.setting.Float(
            "Threshold correction factor",
            1,
            doc="""
            This setting allows you to adjust the threshold as calculated by the above method. The value
            entered here adjusts the threshold either upwards or downwards, by multiplying it by this value. A
            value of 1 means no adjustment, 0 to 1 makes the threshold more lenient and &gt; 1 makes the
            threshold more stringent.
            <dl>
                <dd><img src="memory:{PROTIP_RECOMEND_ICON}">&nbsp; When the threshold is calculated
                automatically, you may find that the value is consistently too stringent or too lenient across
                all images. This setting is helpful for adjusting the threshold to a value that you empirically
                determine is more suitable. For example, the {TM_OTSU} automatic thresholding inherently
                assumes that 50% of the image is covered by objects. If a larger percentage of the image is
                covered, the Otsu method will give a slightly biased threshold that may have to be corrected
                using this setting.</dd>
            </dl>
            """.format(**{
                "PROTIP_RECOMEND_ICON": PROTIP_RECOMEND_ICON,
                "TM_OTSU": centrosome.threshold.TM_OTSU
            })
        )

        self.threshold_range = cellprofiler.setting.FloatRange(
            "Lower and upper bounds on threshold",
            (0, 1),
            minval=0,
            maxval=1,
            doc="""
            Enter the minimum and maximum allowable threshold, a value from 0 to 1. This is helpful as a safety
            precaution when the threshold is calculated automatically, by overriding the automatic threshold.
            <dl>
                <dd><img src="memory:{PROTIP_RECOMEND_ICON}">&nbsp; For example, if there are no objects in the
                field of view, the automatic threshold might be calculated as unreasonably low; the algorithm
                will still attempt to divide the foreground from background (even though there is no
                foreground), and you may end up with spurious false positive foreground regions. In such cases,
                you can estimate the background pixel intensity and set the lower bound according to this
                empirically-determined value.</dd>
                <dd>{HELP_ON_PIXEL_INTENSITIES}</dd>
            </dl>
            """.format(**{
                "HELP_ON_PIXEL_INTENSITIES": cellprofiler.gui.help.HELP_ON_PIXEL_INTENSITIES,
                "PROTIP_RECOMEND_ICON": PROTIP_RECOMEND_ICON
            })
        )

        self.manual_threshold = cellprofiler.setting.Float(
            "Manual threshold",
            value=0.0,
            minval=0.0,
            maxval=1.0,
            doc="""
            <i>(Used only if Manual selected for thresholding method)</i><br>
            Enter the value that will act as an absolute threshold for the images, a value from 0 to 1.
            """
        )

        self.thresholding_measurement = cellprofiler.setting.Measurement(
            "Select the measurement to threshold with",
            lambda: cellprofiler.measurement.IMAGE,
            doc="""
            <i>(Used only if Measurement is selected for thresholding method)</i><br>
            Choose the image measurement that will act as an absolute threshold for the images.
            """
        )

        self.two_class_otsu = cellprofiler.setting.Choice(
            "Two-class or three-class thresholding?",
            [O_TWO_CLASS, O_THREE_CLASS],
            doc="""
            <i>(Used only for the Otsu thresholding method)</i><br>
            <ul>
                <li><i>{O_TWO_CLASS}:</i> Select this option if the grayscale levels are readily
                distinguishable into only two classes: foreground (i.e., regions of interest) and
                background.</li>
                <li><i>{O_THREE_CLASS}</i>: Choose this option if the grayscale levels fall instead into three
                classes: foreground, background and a middle intensity between the two. You will then be asked
                whether the middle intensity class should be added to the foreground or background class in
                order to generate the final two-class output.</li>
            </ul>Note that whether two- or three-class thresholding is chosen, the image pixels are always
            finally assigned two classes: foreground and background.
            <dl>
                <dd><img src="memory:{PROTIP_RECOMEND_ICON}">&nbsp; Three-class thresholding may be useful for
                images in which you have nuclear staining along with low-intensity non-specific cell staining.
                Where two-class thresholding might incorrectly assign this intermediate staining to the nuclei
                objects for some cells, three-class thresholding allows you to assign it to the foreground or
                background as desired.</dd>
            </dl>
            <dl>
                <dd><img src="memory:{PROTIP_AVOID_ICON}">&nbsp; However, in extreme cases where either there
                are almost no objects or the entire field of view is covered with objects, three-class
                thresholding may perform worse than two-class.</dd>
            </dl>
            """.format(**{
                "O_THREE_CLASS": O_THREE_CLASS,
                "O_TWO_CLASS": O_TWO_CLASS,
                "PROTIP_AVOID_ICON": PROTIP_AVOID_ICON,
                "PROTIP_RECOMEND_ICON": PROTIP_RECOMEND_ICON
            })
        )

        self.assign_middle_to_foreground = cellprofiler.setting.Choice(
            "Assign pixels in the middle intensity class to the foreground or the background?",
            [O_FOREGROUND, O_BACKGROUND],
            doc="""
            <i>(Used only for three-class thresholding)</i><br>
            Choose whether you want the pixels with middle grayscale intensities to be assigned to the
            foreground class or the background class.
            """
        )

        self.lower_outlier_fraction = cellprofiler.setting.Float(
            "Lower outlier fraction",
            0.05,
            minval=0,
            maxval=1,
            doc="""
            <i>(Used only when customizing the {TM_ROBUST_BACKGROUND} method)</i><br>
            Discard this fraction of the pixels in the image starting with those of the lowest intensity.
            """.format(**{
                "TM_ROBUST_BACKGROUND": centrosome.threshold.TM_ROBUST_BACKGROUND
            })
        )

        self.upper_outlier_fraction = cellprofiler.setting.Float(
            "Upper outlier fraction",
            0.05,
            minval=0,
            maxval=1,
            doc="""
            <i>(Used only when customizing the {TM_ROBUST_BACKGROUND} method)</i><br>
            Discard this fraction of the pixels in the image starting with those of the highest intensity.
            """.format(**{
                "TM_ROBUST_BACKGROUND": centrosome.threshold.TM_ROBUST_BACKGROUND
            })
        )

        self.averaging_method = cellprofiler.setting.Choice(
            "Averaging method",
            [RB_MEAN, RB_MEDIAN, RB_MODE],
            doc="""
            <i>(Used only when customizing the {TM_ROBUST_BACKGROUND} method)</i><br>
            This setting determines how the intensity midpoint is determined.<br>
            <ul>
                <li><i>{RB_MEAN}</i>: Use the mean of the pixels remaining after discarding the outliers. This
                is a good choice if the cell density is variable or high.</li>
                <li><i>{RB_MEDIAN}</i>: Use the median of the pixels. This is a good choice if, for all images,
                more than half of the pixels are in the background after removing outliers.</li>
                <li><i>{RB_MODE}</i>: Use the most frequently occurring value from among the pixel values. The
                {TM_ROBUST_BACKGROUND} method groups the intensities into bins (the number of bins is the
                square root of the number of pixels in the unmasked portion of the image) and chooses the
                intensity associated with the bin with the most pixels.</li>
            </ul>
            """.format(**{
                "RB_MEAN": RB_MEAN,
                "RB_MEDIAN": RB_MEDIAN,
                "RB_MODE": RB_MODE,
                "TM_ROBUST_BACKGROUND": centrosome.threshold.TM_ROBUST_BACKGROUND
            })
        )

        self.variance_method = cellprofiler.setting.Choice(
            "Variance method",
            [RB_SD, RB_MAD],
            doc="""
            <i>(Used only when customizing the {TM_ROBUST_BACKGROUND} method)</i><br>
            Robust background adds a number of deviations (standard or MAD) to the average to get the final
            background. This setting chooses the method used to assess the variance in the pixels, after
            removing outliers.<br>
            Choose one of <i>{RB_SD}</i> or <i>{RB_MAD}</i> (the median of the absolute difference of the pixel
            intensities from their median).
            """.format(**{
                "RB_MAD": RB_MAD,
                "RB_SD": RB_SD,
                "TM_ROBUST_BACKGROUND": centrosome.threshold.TM_ROBUST_BACKGROUND
            })
        )

        self.number_of_deviations = cellprofiler.setting.Float(
            "# of deviations",
            2,
            doc="""
            <i>(Used only when customizing the {TM_ROBUST_BACKGROUND} method)</i><br>
            Robust background calculates the variance, multiplies it by the value given by this setting and
            adds it to the average. Adding several deviations raises the background value above the average,
            which should be close to the average background, excluding most background pixels. Use a larger
            number to exclude more background pixels. Use a smaller number to include more low-intensity
            foreground pixels. It's possible to use a negative number to lower the threshold if the averaging
            method picks a threshold that is within the range of foreground pixel intensities.
            """.format(**{
                "TM_ROBUST_BACKGROUND": centrosome.threshold.TM_ROBUST_BACKGROUND
            })
        )

        self.adaptive_window_size = cellprofiler.setting.Integer(
            "Size of adaptive window",
            50,
            doc="""
            Enter the window for the adaptive method. For example, you may want to use a multiple of the
            largest expected object size.
            """
        )

    def visible_settings(self):
        visible_settings = super(ApplyThreshold, self).visible_settings()

        visible_settings += [self.threshold_scope]

        if self.threshold_scope == TS_MANUAL:
            visible_settings += [self.manual_threshold]
        elif self.threshold_scope == TS_MEASUREMENT:
            visible_settings += [self.thresholding_measurement]
        elif self.threshold_scope in [TS_GLOBAL, TS_ADAPTIVE]:
            visible_settings += [self.threshold_method]

            if self.threshold_method == centrosome.threshold.TM_OTSU:
                visible_settings += [self.two_class_otsu]

                if self.two_class_otsu == O_THREE_CLASS:
                    visible_settings += [self.assign_middle_to_foreground]
            elif self.threshold_method == centrosome.threshold.TM_ROBUST_BACKGROUND:
                visible_settings += [
                    self.lower_outlier_fraction,
                    self.upper_outlier_fraction,
                    self.averaging_method,
                    self.variance_method,
                    self.number_of_deviations
                ]

        if self.threshold_scope not in [TS_MEASUREMENT, TS_MANUAL]:
            visible_settings += [self.threshold_smoothing_scale]

        if self.threshold_scope != centrosome.threshold.TM_MANUAL:
            visible_settings += [self.threshold_correction_factor, self.threshold_range]

        if self.threshold_scope == centrosome.threshold.TM_ADAPTIVE:
            visible_settings += [self.adaptive_window_size]

        return visible_settings

    def settings(self):
        settings = super(ApplyThreshold, self).settings()

        return settings + [
            self.threshold_scope,
            self.threshold_method,
            self.threshold_smoothing_scale,
            self.threshold_correction_factor,
            self.threshold_range,
            self.manual_threshold,
            self.thresholding_measurement,
            self.two_class_otsu,
            self.assign_middle_to_foreground,
            self.adaptive_window_size,
            self.lower_outlier_fraction,
            self.upper_outlier_fraction,
            self.averaging_method,
            self.variance_method,
            self.number_of_deviations
        ]

    def help_settings(self):
        return [
            self.x_name,
            self.y_name,
            self.threshold_scope,
            self.threshold_method,
            self.manual_threshold,
            self.thresholding_measurement,
            self.two_class_otsu,
            self.assign_middle_to_foreground,
            self.lower_outlier_fraction,
            self.upper_outlier_fraction,
            self.averaging_method,
            self.variance_method,
            self.number_of_deviations,
            self.adaptive_window_size,
            self.threshold_correction_factor,
            self.threshold_range,
            self.threshold_smoothing_scale
        ]

    def run(self, workspace):
        input = workspace.image_set.get_image(self.x_name.value, must_be_grayscale=True)

        local_threshold, global_threshold = self.get_threshold(input.pixel_data, input.mask, workspace)

        self.add_threshold_measurements(
            self.get_measurement_objects_name(),
            workspace.measurements,
            local_threshold,
            global_threshold
        )

        binary_image, _ = self.apply_threshold(input.pixel_data, input.mask, local_threshold)

        self.add_fg_bg_measurements(
            self.get_measurement_objects_name(),
            workspace.measurements,
            input.pixel_data,
            input.mask,
            binary_image
        )

        output = cellprofiler.image.Image(binary_image, parent_image=input)

        workspace.image_set.add(self.y_name.value, output)

        if self.show_window:
            workspace.display_data.input_pixel_data = input.pixel_data
            workspace.display_data.output_pixel_data = output.pixel_data
            statistics = workspace.display_data.statistics = []
            workspace.display_data.col_labels = ("Feature", "Value")

            for column in self.get_measurement_columns(workspace.pipeline):
                value = workspace.measurements.get_current_image_measurement(column[1])
                statistics += [(column[1].split('_')[1], str(value))]

    @property
    def threshold_modifier(self):
        """The threshold algorithm modifier"""
        if self.threshold_scope.value in (TS_GLOBAL, TS_MANUAL, TS_MEASUREMENT):
            return centrosome.threshold.TM_GLOBAL

        return centrosome.threshold.TM_ADAPTIVE

    def apply_threshold(self, image, mask, threshold, automatic=False):
        if not automatic and self.threshold_scope in [TS_MEASUREMENT, TS_MANUAL]:
            return (image >= threshold) & mask, 0

        if automatic:
            sigma = 1
        else:
            # Convert from a scale into a sigma. What I've done here
            # is to structure the Gaussian so that 1/2 of the smoothed
            # intensity is contributed from within the smoothing diameter
            # and 1/2 is contributed from outside.
            sigma = self.threshold_smoothing_scale.value / 0.6744 / 2.0

        blurred_image = centrosome.smooth.smooth_with_function_and_mask(
            image,
            lambda x: scipy.ndimage.gaussian_filter(x, sigma, mode="constant", cval=0),
            mask
        )

        return (blurred_image >= threshold) & mask, sigma

    def get_threshold(self, img, mask, workspace, automatic=False):
        if automatic:
            local_threshold, global_threshold = centrosome.threshold.get_threshold(
                centrosome.threshold.TM_MCT,
                TS_GLOBAL,
                img,
                mask=mask,
                labels=None,
                adaptive_window_size=None,
                threshold_range_min=0.0,
                threshold_range_max=1.0,
                threshold_correction_factor=1.0
            )
        elif self.threshold_scope == centrosome.threshold.TM_MANUAL:
            local_threshold = global_threshold = self.manual_threshold.value
        elif self.threshold_scope == centrosome.threshold.TM_MEASUREMENT:
            m = workspace.measurements

            # Thresholds are stored as single element arrays.  Cast to float to extract the value.
            value = float(m.get_current_image_measurement(self.thresholding_measurement.value))

            value *= self.threshold_correction_factor.value

            if self.threshold_range.min is not None:
                value = max(value, self.threshold_range.min)

            if self.threshold_range.max is not None:
                value = min(value, self.threshold_range.max)

            local_threshold = global_threshold = value
        else:
            labels = None

            block_size = None

            if self.threshold_scope == TS_ADAPTIVE:
                block_size = self.adaptive_window_size.value * numpy.array([1, 1])

            kwparams = {
                "threshold_range_min": self.threshold_range.min,
                "threshold_range_max": self.threshold_range.max,
                "threshold_correction_factor": self.threshold_correction_factor.value,
                "two_class_otsu": self.two_class_otsu.value == O_TWO_CLASS,
                "assign_middle_to_foreground": self.assign_middle_to_foreground.value == O_FOREGROUND,
                "lower_outlier_fraction": self.lower_outlier_fraction.value,
                "upper_outlier_fraction": self.upper_outlier_fraction.value,
                "deviations_above_average": self.number_of_deviations.value
            }

            kwparams["average_fn"] = {
                RB_MEAN: numpy.mean,
                RB_MEDIAN: numpy.median,
                RB_MODE: centrosome.threshold.binned_mode
            }.get(self.averaging_method.value, numpy.mean)

            kwparams["variance_fn"] = {
                RB_SD: numpy.std,
                RB_MAD: centrosome.threshold.mad
            }.get(self.variance_method.value, numpy.std)

            local_threshold, global_threshold = centrosome.threshold.get_threshold(
                self.threshold_method.value,
                self.threshold_modifier,
                img,
                mask=mask,
                labels=labels,
                adaptive_window_size=block_size,
                **kwparams
            )

        return local_threshold, global_threshold

    def display(self, workspace, figure):
        figure.set_subplots((3, 1))

        figure.subplot_imshow_grayscale(0, 0, workspace.display_data.input_pixel_data,
                                        title="Original image: %s" %
                                              self.x_name.value)

        figure.subplot_imshow_grayscale(1, 0, workspace.display_data.output_pixel_data,
                                        title="Thresholded image: %s" %
                                              self.y_name.value,
                                        sharexy=figure.subplot(0, 0))
        figure.subplot_table(
                2, 0, workspace.display_data.statistics,
                workspace.display_data.col_labels)

    def get_measurement_objects_name(self):
        return self.y_name.value

    def add_threshold_measurements(self, objname, measurements, local_threshold, global_threshold):
        ave_threshold = numpy.mean(numpy.atleast_1d(local_threshold))

        measurements.add_measurement(cellprofiler.measurement.IMAGE, FF_FINAL_THRESHOLD % objname, ave_threshold)

        measurements.add_measurement(cellprofiler.measurement.IMAGE, FF_ORIG_THRESHOLD % objname, global_threshold)

    def add_fg_bg_measurements(self, objname, measurements, image, mask, binary_image):
        wv = centrosome.threshold.weighted_variance(image, mask, binary_image)

        measurements.add_measurement(
            cellprofiler.measurement.IMAGE,
            FF_WEIGHTED_VARIANCE % objname,
            numpy.array([wv], dtype=float)
        )

        entropies = centrosome.threshold.sum_of_entropies(image, mask, binary_image)

        measurements.add_measurement(
            cellprofiler.measurement.IMAGE,
            FF_SUM_OF_ENTROPIES % objname,
            numpy.array([entropies], dtype=float)
        )

    def get_measurement_columns(self, pipeline):
        return image_measurement_columns(self.get_measurement_objects_name())

    def get_categories(self, pipeline, object_name):
        if object_name == cellprofiler.measurement.IMAGE:
            return [C_THRESHOLD]

        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == cellprofiler.measurement.IMAGE and category == C_THRESHOLD:
            return [
                FTR_ORIG_THRESHOLD,
                FTR_FINAL_THRESHOLD,
                FTR_SUM_OF_ENTROPIES,
                FTR_WEIGHTED_VARIANCE
            ]

        return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        if measurement in self.get_measurements(pipeline, object_name, category):
            return [self.get_measurement_objects_name()]

        return []

    def upgrade_settings(self, setting_values, variable_revision_number, module_name, from_matlab):
        if from_matlab:
            raise NotImplementedError("There is no automatic upgrade path for this module from MatLab pipelines.")

        if variable_revision_number < 7:
            raise NotImplementedError("Automatic upgrade for this module is not supported in CellProfiler 3.0.")

        if variable_revision_number == 7:
            setting_values = setting_values[:2] + setting_values[6:]

            setting_values = setting_values[:2] + self.upgrade_threshold_settings(setting_values[2:])

            variable_revision_number = 8

        if variable_revision_number == 8:
            setting_values = setting_values[:2] + setting_values[3:]

            variable_revision_number = 9

        return setting_values, variable_revision_number, False

    def upgrade_threshold_settings(self, setting_values):
        '''Upgrade the threshold settings to the current version

        use the first setting which is the version to determine the
        threshold settings version and upgrade as appropriate
        '''
        version = int(setting_values[0])

        if version == 1:
            # Added robust background settings
            #
            setting_values = setting_values + [
                "Default",  # Robust background custom choice
                .05, .05,  # lower and upper outlier fractions
                RB_MEAN,  # averaging method
                RB_SD,  # variance method
                2]  # of standard deviations
            version = 2

        if version == 2:
            if setting_values[1] in ["Binary image", "Per object"]:
                setting_values[1] = "None"

            if setting_values[1] == "Automatic":
                setting_values[1] = TS_GLOBAL
                setting_values[2] = centrosome.threshold.TM_MCT
                setting_values[3] = "Manual"
                setting_values[4] = "1.3488"
                setting_values[5] = "1"
                setting_values[6] = "(0.0, 1.0)"

            removed_threshold_methods = [
                centrosome.threshold.TM_KAPUR,
                centrosome.threshold.TM_MOG,
                centrosome.threshold.TM_RIDLER_CALVARD
            ]

            if setting_values[2] in removed_threshold_methods:
                setting_values[2] = "None"

            if setting_values[2] == centrosome.threshold.TM_BACKGROUND:
                setting_values[2] = centrosome.threshold.TM_ROBUST_BACKGROUND
                setting_values[17] = "Custom"
                setting_values[18] = "0.02"
                setting_values[19] = "0.02"
                setting_values[20] = RB_MODE
                setting_values[21] = RB_SD
                setting_values[22] = "0"

                correction_factor = float(setting_values[5])

                if correction_factor == 0:
                    correction_factor = 2
                else:
                    correction_factor *= 2

                setting_values[5] = str(correction_factor)

            if setting_values[3] == "No smoothing":
                setting_values[4] = "0"

            if setting_values[3] == "Automatic":
                setting_values[4] = "1.3488"

            if setting_values[17] == "Default":
                setting_values[18] = "0.05"
                setting_values[19] = "0.05"
                setting_values[20] = RB_MEAN
                setting_values[21] = RB_SD
                setting_values[22] = "2"

            new_setting_values = setting_values[:3]
            new_setting_values += setting_values[4:7]
            new_setting_values += setting_values[8:10]
            new_setting_values += setting_values[12:13]
            new_setting_values += setting_values[14:15]
            new_setting_values += setting_values[16:17]
            new_setting_values += setting_values[18:]

            setting_values = new_setting_values

        return setting_values

    def validate_module(self, pipeline):
        if self.threshold_scope in [TS_ADAPTIVE, TS_GLOBAL] and \
                self.threshold_method.value == centrosome.threshold.TM_ROBUST_BACKGROUND and \
                self.lower_outlier_fraction.value + self.upper_outlier_fraction.value >= 1:
            raise cellprofiler.setting.ValidationError(
                """
                The sum of the lower robust background outlier fraction ({0:f}) and the upper fraction ({1:f}) must be
                less than one.
                """.format(
                    self.lower_outlier_fraction.value,
                    self.upper_outlier_fraction.value
                ),
                self.upper_outlier_fraction
            )


def image_measurement_columns(image_name):
    return [(cellprofiler.measurement.IMAGE, FF_FINAL_THRESHOLD % image_name, cellprofiler.measurement.COLTYPE_FLOAT),
            (cellprofiler.measurement.IMAGE, FF_ORIG_THRESHOLD % image_name, cellprofiler.measurement.COLTYPE_FLOAT),
            (cellprofiler.measurement.IMAGE, FF_WEIGHTED_VARIANCE % image_name, cellprofiler.measurement.COLTYPE_FLOAT),
            (cellprofiler.measurement.IMAGE, FF_SUM_OF_ENTROPIES % image_name, cellprofiler.measurement.COLTYPE_FLOAT)]
