"""
<b>Apply Threshold</b> sets pixel intensities below or above a certain threshold to zero
<hr>
<b>ApplyThreshold</b> produces either a grayscale or binary image based on a threshold which can be
pre-selected or calculated automatically using one of many methods.
"""

import centrosome.cpmorphology
import centrosome.threshold
import scipy.ndimage.morphology

import cellprofiler.image
import cellprofiler.module
import cellprofiler.modules.identify
import cellprofiler.setting
import identify

RETAIN = "Retain"
SHIFT = "Shift"
GRAYSCALE = "Grayscale"
BINARY = "Binary (black and white)"

TH_BELOW_THRESHOLD = "Below threshold"
TH_ABOVE_THRESHOLD = "Above threshold"

'''# of non-threshold settings in current revision'''
N_SETTINGS = 6


class ApplyThreshold(identify.Identify):
    module_name = "ApplyThreshold"
    variable_revision_number = 7
    category = "Image Processing"

    def create_settings(self):
        threshold_methods = [
            method for method in centrosome.threshold.TM_METHODS if method != centrosome.threshold.TM_BINARY_IMAGE
        ]

        self.image_name = cellprofiler.setting.ImageNameSubscriber(
            "Select the input image",
            doc="Choose the image to be thresholded."
        )

        self.thresholded_image_name = cellprofiler.setting.ImageNameProvider(
            "Name the output image",
            "ThreshBlue",
            doc="Enter a name for the thresholded image."
        )

        self.binary = cellprofiler.setting.Choice(
            "Select the output image type",
            [GRAYSCALE, BINARY],
            doc="""
            Two types of output images can be produced:<br>
            <ul>
                <li><i>{GRAYSCALE}:</i> The pixels that are retained after some pixels are set to zero or
                shifted (based on your selections for thresholding options) will have their original intensity
                values.</li>
                <li><i>{BINARY}:</i> The pixels that are retained after some pixels are set to zero (based on
                your selections for thresholding options) will be white and all other pixels will be black
                (zeroes).</li>
            </ul>
            """.format(**{
                "GRAYSCALE": GRAYSCALE,
                "BINARY": BINARY
            })
        )

        # if not binary:
        self.low_or_high = cellprofiler.setting.Choice(
            "Set pixels below or above the threshold to zero?",
            [TH_BELOW_THRESHOLD, TH_ABOVE_THRESHOLD],
            doc="""
            <i>(Used only when "{GRAYSCALE}" thresholding is selected)</i><br>
            This option adjusts how pixels above or below the threshold are handled:
            <ul>
                <li><i>{TH_BELOW_THRESHOLD}:</i> Set the dim pixels below the threshold to zero.</li>
                <li><i>{TH_ABOVE_THRESHOLD}:</i> Set the bright pixels above the threshold to zero.</li>
            </ul>
            """.format(**{
                "GRAYSCALE": GRAYSCALE,
                "TH_BELOW_THRESHOLD": TH_BELOW_THRESHOLD,
                "TH_ABOVE_THRESHOLD": TH_ABOVE_THRESHOLD
            })
        )

        # if not binary and below threshold
        self.shift = cellprofiler.setting.Binary(
            "Subtract the threshold value from the remaining pixel intensities?",
            False,
            doc='''
            <i>(Used only if the output image is {GRAYSCALE} and pixels below a given intensity are to be set
            to zero)</i><br>
            Select <i>{YES}</i> to shift the value of the dim pixels by the threshold value.
            '''.format(**{
                "GRAYSCALE": GRAYSCALE,
                "YES": cellprofiler.setting.YES
            })
        )

        # if not binary and above threshold
        self.dilation = cellprofiler.setting.Float(
            "Number of pixels by which to expand the thresholding around those excluded bright pixels",
            0.0,
            doc="""
            <i>(Used only if the output image is grayscale and pixels above a given intensity are to be set to
            zero)</i><br>
            This setting is useful when attempting to exclude bright artifactual objects: first, set the
            threshold to exclude these bright objects; it may also be desirable to expand the thresholded
            region around those bright objects by a certain distance so as to avoid a "halo" effect.
            """
        )

        self.create_threshold_settings(threshold_methods)

        self.threshold_smoothing_choice.value = identify.TSM_NONE

    def visible_settings(self):
        vv = [self.image_name, self.thresholded_image_name, self.binary]
        if self.binary.value == GRAYSCALE:
            vv.append(self.low_or_high)
            if self.low_or_high.value == TH_BELOW_THRESHOLD:
                vv.extend([self.shift])
            else:
                vv.extend([self.dilation])
        vv += self.get_threshold_visible_settings()
        return vv

    def settings(self):
        return [self.image_name, self.thresholded_image_name,
                self.binary, self.low_or_high,
                self.shift, self.dilation] + self.get_threshold_settings()

    def help_settings(self):
        """Return all settings in a consistent order"""
        return [self.image_name, self.thresholded_image_name,
                self.binary, self.low_or_high,
                self.shift, self.dilation] + \
               self.get_threshold_help_settings()

    def run(self, workspace):
        """Run the module

        workspace    - the workspace contains:
            pipeline     - instance of CellProfiler.Pipeline for this run
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - display within this frame (or None to not display)
        """
        input = workspace.image_set.get_image(self.image_name.value,
                                              must_be_grayscale=True)
        pixels = input.pixel_data.copy()
        binary_image, local_thresh = self.threshold_image(
                self.image_name.value, workspace, wants_local_threshold=True)
        if self.binary != 'Grayscale':
            pixels = binary_image & input.mask
        else:
            if self.low_or_high == TH_BELOW_THRESHOLD:
                pixels[input.mask & ~ binary_image] = 0
                if self.shift.value:
                    pixels[input.mask & binary_image] -= \
                        local_thresh if self.threshold_modifier == centrosome.threshold.TM_GLOBAL \
                            else local_thresh[input.mask & binary_image]
            elif self.low_or_high == TH_ABOVE_THRESHOLD:
                undilated = input.mask & binary_image
                dilated = scipy.ndimage.morphology.binary_dilation(undilated,
                                                                   centrosome.cpmorphology.strel_disk(self.dilation.value),
                                                                   mask=input.mask)
                pixels[dilated] = 0
            else:
                raise NotImplementedError(
                        """Threshold setting, "%s" is not "%s" or "%s".""" %
                        (self.low_or_high.value, TH_BELOW_THRESHOLD,
                         TH_ABOVE_THRESHOLD))
        output = cellprofiler.image.Image(pixels, parent_image=input)
        workspace.image_set.add(self.thresholded_image_name.value, output)
        if self.show_window:
            workspace.display_data.input_pixel_data = input.pixel_data
            workspace.display_data.output_pixel_data = output.pixel_data
            statistics = workspace.display_data.statistics = []
            workspace.display_data.col_labels = ("Feature", "Value")

            for column in self.get_measurement_columns(workspace.pipeline):
                value = workspace.measurements.get_current_image_measurement(column[1])
                statistics += [(column[1].split('_')[1], str(value))]

    def display(self, workspace, figure):
        figure.set_subplots((3, 1))

        figure.subplot_imshow_grayscale(0, 0, workspace.display_data.input_pixel_data,
                                        title="Original image: %s" %
                                              self.image_name.value)

        figure.subplot_imshow_grayscale(1, 0, workspace.display_data.output_pixel_data,
                                        title="Thresholded image: %s" %
                                              self.thresholded_image_name.value,
                                        sharexy=figure.subplot(0, 0))
        figure.subplot_table(
                2, 0, workspace.display_data.statistics,
                workspace.display_data.col_labels)

    def get_measurement_objects_name(self):
        '''Return the name of the "objects" used to name thresholding measurements

        In the case of ApplyThreshold, we use the image name to name the
        measurements, so the code here works, but is misnamed.
        '''
        return self.thresholded_image_name.value

    def get_measurement_columns(self, pipeline):
        return cellprofiler.modules.identify.get_threshold_measurement_columns(self.thresholded_image_name.value)

    def get_categories(self, pipeline, object_name):
        return self.get_threshold_categories(pipeline, object_name)

    def get_measurements(self, pipeline, object_name, category):
        return self.get_threshold_measurements(pipeline, object_name, category)

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        return self.get_threshold_measurement_objects(
                pipeline, object_name, category, measurement)

    def upgrade_settings(self, setting_values,
                         variable_revision_number, module_name,
                         from_matlab):
        if from_matlab and variable_revision_number < 4:
            raise NotImplementedError, ("TODO: Handle Matlab CP pipelines for "
                                        "ApplyThreshold with revision < 4")
        if from_matlab and variable_revision_number == 4:
            setting_values = [setting_values[0],  # ImageName
                              setting_values[1],  # ThresholdedImageName
                              None,
                              None,
                              None,
                              setting_values[2],  # LowThreshold
                              setting_values[3],  # Shift
                              setting_values[4],  # HighThreshold
                              setting_values[5],  # DilationValue
                              centrosome.threshold.TM_MANUAL,  # Manual thresholding
                              setting_values[6],  # BinaryChoice
                              "0,1",  # Threshold range
                              "1",  # Threshold correction factor
                              ".2",  # Object fraction
                              cellprofiler.setting.NONE  # Enclosing objects name
                              ]
            setting_values[2] = (BINARY if float(setting_values[10]) > 0
                                 else GRAYSCALE)  # binary flag
            setting_values[3] = (cellprofiler.setting.YES if float(setting_values[5]) > 0
                                 else cellprofiler.setting.NO)  # low threshold set
            setting_values[4] = (cellprofiler.setting.YES if float(setting_values[7]) > 0
                                 else cellprofiler.setting.NO)  # high threshold set
            variable_revision_number = 2
            from_matlab = False
        if (not from_matlab) and variable_revision_number == 1:
            setting_values = (setting_values[:9] +
                              [centrosome.threshold.TM_MANUAL, setting_values[9], "O,1", "1",
                               ".2", cellprofiler.setting.NONE])
            variable_revision_number = 2
        if (not from_matlab) and variable_revision_number == 2:
            # Added Otsu options
            setting_values = list(setting_values)
            setting_values += [identify.O_TWO_CLASS, identify.O_WEIGHTED_VARIANCE,
                               identify.O_FOREGROUND]
            variable_revision_number = 3

        if (not from_matlab) and variable_revision_number == 3:
            #
            # Only low or high, not both + removed manual threshold settings
            #
            if setting_values[3] == cellprofiler.setting.YES:
                th = TH_BELOW_THRESHOLD
            else:
                th = TH_ABOVE_THRESHOLD
            if setting_values[2] == GRAYSCALE:
                # Grayscale used to have just manual thresholding
                setting_values = list(setting_values)
                setting_values[9] = centrosome.threshold.TM_MANUAL
                if th == TH_BELOW_THRESHOLD:
                    # Set to old low threshold
                    setting_values[10] = setting_values[5]
                else:
                    setting_values[10] = setting_values[7]
            setting_values = [setting_values[0],  # Image name
                              setting_values[1],  # Thresholded image
                              setting_values[2],  # binary or gray
                              th,
                              setting_values[6],  # shift
                              ] + setting_values[8:]
            variable_revision_number = 4

        if (not from_matlab) and variable_revision_number == 4:
            # Added measurements to threshold methods
            setting_values = setting_values + [cellprofiler.setting.NONE]
            variable_revision_number = 5

        if (not from_matlab) and variable_revision_number == 5:
            # Added adaptive thresholding settings
            setting_values += [identify.FI_IMAGE_SIZE, "10"]
            variable_revision_number = 6

        if (not from_matlab) and variable_revision_number == 6:
            image_name, thresholded_image_name, binary, low_or_high, \
            shift, dilation, threshold_method, manual_threshold, \
            threshold_range, threshold_correction_factor, \
            object_fraction, enclosing_objects_name, \
            two_class_otsu, use_weighted_variance, \
            assign_middle_to_foreground, thresholding_measurement = \
                setting_values[:16]
            setting_values = [
                                 image_name, thresholded_image_name, binary, low_or_high,
                                 shift, dilation] + self.upgrade_legacy_threshold_settings(
                    threshold_method, identify.TSM_NONE, threshold_correction_factor,
                    threshold_range, object_fraction, manual_threshold,
                    thresholding_measurement, cellprofiler.setting.NONE, two_class_otsu,
                    use_weighted_variance, assign_middle_to_foreground,
                    identify.FI_IMAGE_SIZE, "10", masking_objects=enclosing_objects_name)
            variable_revision_number = 7
        #
        # Upgrade the threshold settings
        #
        setting_values = setting_values[:N_SETTINGS] + \
                         self.upgrade_threshold_settings(setting_values[N_SETTINGS:])
        return setting_values, variable_revision_number, from_matlab
