__doc__ = """\
ClassifyObjects
===============

**ClassifyObjects** classifies objects into different classes according
to the value of measurements you choose.

This module classifies objects into a number of different bins according
to the value of a measurement (e.g., by size, intensity, shape). It
reports how many objects fall into each class as well as the percentage
of objects that fall into each class. The module asks you to select the
measurement feature to be used to classify your objects and specify the
bins to use. It also requires you to have run a measurement or
**CalculateMath** previous to this module in the pipeline so that the
measurement values can be used to classify the objects.

There are two flavors of classification:

-  The first classifies each object according to the measurements you
   choose and assigns each object to one class per measurement. You may
   specify more than two classification bins per measurement.
-  The second classifies each object according to two measurements and
   two threshold values. The module classifies each object once per
   measurement resulting in four possible object classes. The module
   then stores one measurement per object, based on the object’s class.

Note that objects without a measurement are not counted as belonging in
a classification bin and will not show up in the output image (shown in
the module display window); in the object classification they will have
a value of False for all bins. However, they are still counted in the
total number of objects and hence are reflected in the classification
percentages.
 
|		
 		
============ ============ ===============		
Supports 2D? Supports 3D? Respects masks?		
============ ============ ===============		
YES          NO           NO		
============ ============ ===============		
 		
See also		
^^^^^^^^		
 		
See also **CalculateMath** and any of the modules in the **Measure**	category.		
 
Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **Image measurements:**

   -  *NumObjectsPerBin:* The number of objects that are classified into
      each bin.
   -  *PctObjectsPerBin:* The percentage of total objects that are
      classified into each bin.

-  **Object measurements:**

   -  Single measurement: Classification (true/false) of the
      N\ :sup:`th` bin for the M\ :sup:`th` measurement.
   -  Two measurement: Classification (true/false) of the 1\ :sup:`st`
      measurement versus the 2\ :sup:`nd` measurement binned into bins
      above (“high”) and below (“low”) the cutoff.
"""

import functools

import numpy
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.constants.measurement import COLTYPE_INTEGER
from cellprofiler_core.constants.measurement import IMAGE
from cellprofiler_core.image import Image
from cellprofiler_core.module import Module
from cellprofiler_core.preferences import get_default_colormap
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting import Divider
from cellprofiler_core.setting import HiddenCount
from cellprofiler_core.setting import Measurement
from cellprofiler_core.setting import SettingsGroup
from cellprofiler_core.setting import ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething
from cellprofiler_core.setting.do_something import RemoveSettingButton
from cellprofiler_core.setting.subscriber import LabelSubscriber
from cellprofiler_core.setting.text import Alphanumeric
from cellprofiler_core.setting.text import Float
from cellprofiler_core.setting.text import ImageName
from cellprofiler_core.setting.text import Integer
from cellprofiler_core.setting.text import Text

BY_SINGLE_MEASUREMENT = "Single measurement"
BY_TWO_MEASUREMENTS = "Pair of measurements"
TM_MEAN = "Mean"
TM_MEDIAN = "Median"
TM_CUSTOM = "Custom"

BC_EVEN = "Evenly spaced bins"
BC_CUSTOM = "Custom-defined bins"

M_CATEGORY = "Classify"
F_PCT_PER_BIN = "PctObjectsPerBin"
F_NUM_PER_BIN = "NumObjectsPerBin"


class ClassifyObjects(Module):
    category = "Object Processing"
    module_name = "ClassifyObjects"
    variable_revision_number = 2

    def create_settings(self):
        """Create the settings for the module

        Create the settings for the module during initialization.
        """
        self.contrast_choice = Choice(
            "Make each classification decision on how many measurements?",
            [BY_SINGLE_MEASUREMENT, BY_TWO_MEASUREMENTS],
            doc="""\
This setting controls how many measurements are used to make a
classifications decision for each object:

-  *%(BY_SINGLE_MEASUREMENT)s:* Classifies each object based on a
   single measurement.
-  *%(BY_TWO_MEASUREMENTS)s:* Classifies each object based on a pair
   of measurements taken together (that is, an object must meet two
   criteria to belong to a class).
"""
            % globals(),
        )

        ############### Single measurement settings ##################
        #
        # A list holding groupings for each of the single measurements
        # to be done
        #
        self.single_measurements = []
        #
        # A count of # of measurements
        #
        self.single_measurement_count = HiddenCount(self.single_measurements)
        #
        # Add one single measurement to start off
        #
        self.add_single_measurement(False)
        #
        # A button to press to get another measurement
        #
        self.add_measurement_button = DoSomething(
            "", "Add another classification", self.add_single_measurement
        )
        #
        ############### Two-measurement settings #####################
        #
        # The object for the contrasting method
        #
        self.object_name = LabelSubscriber(
            "Select the object name",
            "None",
            doc="""\
Choose the object that you want to measure from the list. This should be
an object created by a previous module such as
**IdentifyPrimaryObjects**, **IdentifySecondaryObjects**, **IdentifyTertiaryObjects**, or **Watershed**
""",
        )

        #
        # The two measurements for the contrasting method
        #
        def object_fn():
            return self.object_name.value

        self.first_measurement = Measurement(
            "Select the first measurement",
            object_fn,
            doc="""\
*(Used only if using a pair of measurements)*

Choose a measurement made on the above object. This is the first of two
measurements that will be contrasted together. The measurement should be
one made on the object in a prior module.
""",
        )

        self.first_threshold_method = Choice(
            "Method to select the cutoff",
            [TM_MEAN, TM_MEDIAN, TM_CUSTOM],
            doc="""\
*(Used only if using a pair of measurements)*

Objects are classified as being above or below a cutoff value for a
measurement. You can set this cutoff threshold in one of three ways:

-  *%(TM_MEAN)s*: At the mean of the measurement’s value for all
   objects in the image cycle.
-  *%(TM_MEDIAN)s*: At the median of the measurement’s value for all
   objects in the image set.
-  *%(TM_CUSTOM)s*: You specify a custom threshold value.
"""
            % globals(),
        )

        self.first_threshold = Float(
            "Enter the cutoff value",
            0.5,
            doc="""\
*(Used only if using a pair of measurements)*

This is the cutoff value separating objects in the two classes.""",
        )

        self.second_measurement = Measurement(
            "Select the second measurement",
            object_fn,
            doc="""\
*(Used only if using a pair of measurements)*

Select a measurement made on the above object. This is
the second of two measurements that will be contrasted together.
The measurement should be one made on the object in a prior
module.""",
        )

        self.second_threshold_method = Choice(
            "Method to select the cutoff",
            [TM_MEAN, TM_MEDIAN, TM_CUSTOM],
            doc="""\
*(Used only if using a pair of measurements)*

Objects are classified as being above or below a cutoff value for a
measurement. You can set this cutoff threshold in one of three ways:

-  *%(TM_MEAN)s:* At the mean of the measurement’s value for all
   objects in the image cycle.
-  *%(TM_MEDIAN)s:* At the median of the measurement’s value for all
   objects in the image set.
-  *%(TM_CUSTOM)s:* You specify a custom threshold value.
"""
            % globals(),
        )

        self.second_threshold = Float(
            "Enter the cutoff value",
            0.5,
            doc="""\
*(Used only if using a pair of measurements)*

This is the cutoff value separating objects in the two classes.""",
        )

        self.wants_custom_names = Binary(
            "Use custom names for the bins?",
            False,
            doc="""\
*(Used only if using a pair of measurements)*

Select "*Yes*" if you want to specify the names of each bin
measurement.

Select "*No*" to create names based on the measurements. For instance,
for “Intensity_MeanIntensity_Green” and
“Intensity_TotalIntensity_Blue”, the module generates measurements
such as
“Classify_Intensity_MeanIntensity_Green_High_Intensity_TotalIntensity_Low”.
"""
            % globals(),
        )

        self.low_low_custom_name = Alphanumeric(
            "Enter the low-low bin name",
            "low_low",
            doc="""\
*(Used only if using a pair of measurements)*

Name of the measurement for objects that fall below the threshold for
both measurements.
""",
        )

        self.low_high_custom_name = Alphanumeric(
            "Enter the low-high bin name",
            "low_high",
            doc="""\
*(Used only if using a pair of measurements)*

Name of the measurement for objects whose
first measurement is below threshold and whose second measurement
is above threshold.
""",
        )

        self.high_low_custom_name = Alphanumeric(
            "Enter the high-low bin name",
            "high_low",
            doc="""\
*(Used only if using a pair of measurements)*

Name of the measurement for objects whose
first measurement is above threshold and whose second measurement
is below threshold.""",
        )

        self.high_high_custom_name = Alphanumeric(
            "Enter the high-high bin name",
            "high_high",
            doc="""\
*(Used only if using a pair of measurements)*

Name of the measurement for objects that
are above the threshold for both measurements.""",
        )

        self.wants_image = Binary(
            "Retain an image of the classified objects?",
            False,
            doc="""\
Select "*Yes*" to retain the image of the objects color-coded
according to their classification, for use later in the pipeline (for
example, to be saved by a **SaveImages** module).
"""
            % globals(),
        )

        self.image_name = ImageName(
            "Enter the image name",
            "None",
            doc="""\
*(Used only if the classified object image is to be retained for later use in the pipeline)*

Enter the name to be given to the classified object image.""",
        )

    def add_single_measurement(self, can_delete=True):
        """Add a single measurement to the group of single measurements

        can_delete - True to include a "remove" button, False if you're not
                     allowed to remove it.
        """
        group = SettingsGroup()
        if can_delete:
            group.append("divider", Divider(line=True))

        group.append(
            "object_name",
            LabelSubscriber(
                "Select the object to be classified",
                "None",
                doc="""\
The name of the objects to be classified. You can choose from objects
created by any previous module. See **IdentifyPrimaryObjects**,
**IdentifySecondaryObjects**, **IdentifyTertiaryObjects**, or **Watershed**
""",
            ),
        )

        def object_fn():
            return group.object_name.value

        group.append(
            "measurement",
            Measurement(
                "Select the measurement to classify by",
                object_fn,
                doc="""\
*(Used only if using a single measurement)*

Select a measurement made by a previous module. The objects will be
classified according to their values for this measurement.
""",
            ),
        )

        group.append(
            "bin_choice",
            Choice(
                "Select bin spacing",
                [BC_EVEN, BC_CUSTOM],
                doc="""\
*(Used only if using a single measurement)*

Select how you want to define the spacing of the bins. You have the
following options:

-  *%(BC_EVEN)s:* Choose this if you want to specify bins of equal
   size, bounded by upper and lower limits. If you want two bins, choose
   this option and then provide a single threshold when asked.
-  *%(BC_CUSTOM)s:* Choose this option to create the indicated number
   of bins at evenly spaced intervals between the low and high
   threshold. You also have the option to create bins for objects that
   fall below or above the low and high threshold.
"""
                % globals(),
            ),
        )

        group.append(
            "bin_count",
            Integer(
                "Number of bins",
                3,
                minval=1,
                doc="""\
*(Used only if using a single measurement)*

This is the number of bins that will be created between
the low and high threshold""",
            ),
        )

        group.append(
            "low_threshold",
            Float(
                "Lower threshold",
                0,
                doc="""\
*(Used only if using a single measurement and "%(BC_EVEN)s" selected)*

This is the threshold that separates the lowest bin from the others. The
lower threshold, upper threshold, and number of bins define the
thresholds of bins between the lowest and highest.
"""
                % globals(),
            ),
        )

        group.append(
            "wants_low_bin",
            Binary(
                "Use a bin for objects below the threshold?",
                False,
                doc="""\
*(Used only if using a single measurement)*

Select "*Yes*" if you want to create a bin for objects whose values
fall below the low threshold. Select "*No*" if you do not want a bin
for these objects.
"""
                % globals(),
            ),
        )

        group.append(
            "high_threshold",
            Float(
                "Upper threshold",
                1,
                doc="""\
*(Used only if using a single measurement and "%(BC_EVEN)s" selected)*

This is the threshold that separates the last bin from the others. Note
that if you would like two bins, you should select "*%(BC_CUSTOM)s*".
"""
                % globals(),
            ),
        )

        group.append(
            "wants_high_bin",
            Binary(
                "Use a bin for objects above the threshold?",
                False,
                doc="""\
*(Used only if using a single measurement)*

Select "*Yes*" if you want to create a bin for objects whose values
are above the high threshold.

Select "*No*" if you do not want a bin for these objects.
"""
                % globals(),
            ),
        )

        group.append(
            "custom_thresholds",
            Text(
                "Enter the custom thresholds separating the values between bins",
                "0,1",
                doc="""\
*(Used only if using a single measurement and "%(BC_CUSTOM)s" selected)*

This setting establishes the threshold values for the bins. You should
enter one threshold between each bin, separating thresholds with commas
(for example, *0.3, 1.5, 2.1* for four bins). The module will create one
more bin than there are thresholds.
"""
                % globals(),
            ),
        )

        group.append(
            "wants_custom_names",
            Binary(
                "Give each bin a name?",
                False,
                doc="""\
*(Used only if using a single measurement)*

Select "*Yes*" to assign custom names to bins you have specified.

Select "*No*" for the module to automatically assign names based on
the measurements and the bin number.
"""
                % globals(),
            ),
        )

        group.append(
            "bin_names",
            Text(
                "Enter the bin names separated by commas",
                "None",
                doc="""\
*(Used only if "Give each bin a name?" is checked)*

Enter names for each of the bins, separated by commas.
An example including three bins might be *First,Second,Third*.""",
            ),
        )

        group.append(
            "wants_images",
            Binary(
                "Retain an image of the classified objects?",
                False,
                doc="""\
Select "*Yes*" to keep an image of the objects which is color-coded
according to their classification, for use later in the pipeline (for
example, to be saved by a **SaveImages** module).
"""
                % globals(),
            ),
        )

        group.append(
            "image_name",
            ImageName(
                "Name the output image",
                "ClassifiedNuclei",
                doc="""Enter the name to be given to the classified object image.""",
            ),
        )

        group.can_delete = can_delete

        def number_of_bins():
            """Return the # of bins in this classification"""
            if group.bin_choice == BC_EVEN:
                value = group.bin_count.value
            else:
                value = len(group.custom_thresholds.value.split(",")) - 1
            if group.wants_low_bin:
                value += 1
            if group.wants_high_bin:
                value += 1
            return value

        group.number_of_bins = number_of_bins

        def measurement_name():
            """Get the measurement name to use inside the bin name

            Account for conflicts with previous measurements
            """
            measurement_name = group.measurement.value
            other_same = 0
            for other in self.single_measurements:
                if id(other) == id(group):
                    break
                if other.measurement.value == measurement_name:
                    other_same += 1
            if other_same > 0:
                measurement_name += str(other_same)
            return measurement_name

        def bin_feature_names():
            """Return the feature names for each bin"""
            if group.wants_custom_names:
                return [name.strip() for name in group.bin_names.value.split(",")]
            return [
                "_".join((measurement_name(), "Bin_%d" % (i + 1)))
                for i in range(number_of_bins())
            ]

        group.bin_feature_names = bin_feature_names

        def validate_group():
            bin_name_count = len(bin_feature_names())
            bin_count = number_of_bins()
            if bin_count < 1:
                bad_setting = (
                    group.bin_count
                    if group.bin_choice == BC_EVEN
                    else group.custom_thresholds
                )
                raise ValidationError(
                    "You must have at least one bin in order to take measurements. "
                    "Either add more bins or ask for bins for objects above or below threshold",
                    bad_setting,
                )
            if bin_name_count != number_of_bins():
                raise ValidationError(
                    "The number of bin names (%d) does not match the number of bins (%d)."
                    % (bin_name_count, bin_count),
                    group.bin_names,
                )
            for bin_feature_name in bin_feature_names():
                Alphanumeric.validate_alphanumeric_text(
                    bin_feature_name, group.bin_names, True
                )
            if group.bin_choice == BC_CUSTOM:
                try:
                    [float(x.strip()) for x in group.custom_thresholds.value.split(",")]
                except ValueError:
                    raise ValidationError(
                        "Custom thresholds must be a comma-separated list "
                        'of numbers (example: "1.0, 2.3, 4.5")',
                        group.custom_thresholds,
                    )
            elif group.bin_choice == BC_EVEN:
                if group.low_threshold.value >= group.high_threshold.value:
                    raise ValidationError(
                        "Lower Threshold must be less than Upper Threshold",
                        group.low_threshold,
                    )

        group.validate_group = validate_group

        if can_delete:
            group.remove_settings_button = RemoveSettingButton(
                "", "Remove this classification", self.single_measurements, group
            )
        self.single_measurements.append(group)

    def settings(self):
        result = [self.contrast_choice, self.single_measurement_count]
        result += functools.reduce(
            lambda x, y: x + y,
            [group.pipeline_settings() for group in self.single_measurements],
        )
        result += [
            self.object_name,
            self.first_measurement,
            self.first_threshold_method,
            self.first_threshold,
            self.second_measurement,
            self.second_threshold_method,
            self.second_threshold,
            self.wants_custom_names,
            self.low_low_custom_name,
            self.low_high_custom_name,
            self.high_low_custom_name,
            self.high_high_custom_name,
            self.wants_image,
            self.image_name,
        ]
        return result

    def visible_settings(self):
        result = [self.contrast_choice]
        if self.contrast_choice == BY_TWO_MEASUREMENTS:
            #
            # Visible settings if there are two measurements
            #
            result += [self.object_name]
            for measurement_setting, threshold_method_setting, threshold_setting in (
                (
                    self.first_measurement,
                    self.first_threshold_method,
                    self.first_threshold,
                ),
                (
                    self.second_measurement,
                    self.second_threshold_method,
                    self.second_threshold,
                ),
            ):
                result += [measurement_setting, threshold_method_setting]
                if threshold_method_setting == TM_CUSTOM:
                    result += [threshold_setting]
            result += [self.wants_custom_names]
            if self.wants_custom_names:
                result += [
                    self.low_low_custom_name,
                    self.low_high_custom_name,
                    self.high_low_custom_name,
                    self.high_high_custom_name,
                ]
            result += [self.wants_image]
            if self.wants_image:
                result += [self.image_name]
        else:
            #
            # Visible results per single measurement
            #
            for group in self.single_measurements:
                if group.can_delete:
                    result += [group.divider]
                result += [group.object_name, group.measurement, group.bin_choice]
                if group.bin_choice == BC_EVEN:
                    result += [
                        group.bin_count,
                        group.low_threshold,
                        group.wants_low_bin,
                        group.high_threshold,
                        group.wants_high_bin,
                    ]
                else:
                    result += [
                        group.custom_thresholds,
                        group.wants_low_bin,
                        group.wants_high_bin,
                    ]
                result += [group.wants_custom_names]
                if group.wants_custom_names:
                    result += [group.bin_names]
                result += [group.wants_images]
                if group.wants_images:
                    result += [group.image_name]
                if group.can_delete:
                    result += [group.remove_settings_button]
            result += [self.add_measurement_button]
        return result

    def run(self, workspace):
        """Classify the objects in the image cycle"""
        if self.contrast_choice == BY_SINGLE_MEASUREMENT:
            if self.show_window:
                workspace.display_data.labels = []
                workspace.display_data.bins = []
                workspace.display_data.values = []
            for group in self.single_measurements:
                self.run_single_measurement(group, workspace)
        elif self.contrast_choice == BY_TWO_MEASUREMENTS:
            self.run_two_measurements(workspace)
        else:
            raise ValueError(
                "Invalid classification method: %s" % self.contrast_choice.value
            )

    def display(self, workspace, figure):
        if self.contrast_choice == BY_TWO_MEASUREMENTS:
            self.display_two_measurements(workspace, figure)
        else:
            self.display_single_measurement(workspace, figure)

    def get_feature_name_matrix(self):
        """Get a 2x2 matrix of feature names for two measurements"""
        if self.wants_custom_names:
            return numpy.array(
                [
                    [self.low_low_custom_name.value, self.low_high_custom_name.value],
                    [self.high_low_custom_name.value, self.high_high_custom_name.value],
                ]
            )
        else:
            m1 = self.first_measurement.value
            m2 = self.second_measurement.value
            return numpy.array(
                [
                    ["_".join((m1, a1, m2, a2)) for a2 in ("low", "high")]
                    for a1 in ("low", "high")
                ]
            )

    def run_two_measurements(self, workspace):
        measurements = workspace.measurements
        in_high_class = []
        saved_values = []
        objects = workspace.object_set.get_objects(self.object_name.value)
        has_nan_measurement = numpy.zeros(objects.count, bool)
        for feature, threshold_method, threshold in (
            (self.first_measurement, self.first_threshold_method, self.first_threshold),
            (
                self.second_measurement,
                self.second_threshold_method,
                self.second_threshold,
            ),
        ):
            values = measurements.get_current_measurement(
                self.object_name.value, feature.value
            )
            if len(values) < objects.count:
                values = numpy.hstack(
                    (values, [numpy.NaN] * (objects.count - len(values)))
                )
            saved_values.append(values)
            has_nan_measurement = has_nan_measurement | numpy.isnan(values)
            if threshold_method == TM_CUSTOM:
                t = threshold.value
            elif len(values) == 0:
                t = 0
            elif threshold_method == TM_MEAN:
                t = numpy.mean(values[~numpy.isnan(values)])
            elif threshold_method == TM_MEDIAN:
                t = numpy.median(values[~numpy.isnan(values)])
            else:
                raise ValueError(
                    "Unknown threshold method: %s" % threshold_method.value
                )
            in_high_class.append(values >= t)
        feature_names = self.get_feature_name_matrix()
        num_values = len(values)
        for i in range(2):
            for j in range(2):
                in_class = (
                    (in_high_class[0].astype(int) == i)
                    & (in_high_class[1].astype(int) == j)
                    & (~has_nan_measurement)
                )
                measurements.add_measurement(
                    self.object_name.value,
                    "_".join((M_CATEGORY, feature_names[i, j])),
                    in_class.astype(int),
                )
                num_hits = in_class.sum()
                measurement_name = "_".join(
                    (M_CATEGORY, feature_names[i, j], F_NUM_PER_BIN)
                )
                measurements.add_measurement(IMAGE, measurement_name, num_hits)
                measurement_name = "_".join(
                    (M_CATEGORY, feature_names[i, j], F_PCT_PER_BIN)
                )
                measurements.add_measurement(
                    IMAGE,
                    measurement_name,
                    100.0 * float(num_hits) / num_values if num_values > 0 else 0,
                )

        if self.wants_image:
            class_1, class_2 = in_high_class
            object_codes = class_1.astype(int) + class_2.astype(int) * 2 + 1
            object_codes = numpy.hstack(([0], object_codes))
            object_codes[numpy.hstack((False, numpy.isnan(values)))] = 0
            nobjects = len(class_1)
            mapping = numpy.zeros(nobjects + 1, int)
            mapping[1:] = numpy.arange(1, nobjects + 1)
            labels = object_codes[mapping[objects.segmented]]
            colors = self.get_colors(4)
            image = colors[labels, :3]
            image = Image(image, parent_image=objects.parent_image)
            workspace.image_set.add(self.image_name.value, image)

        if self.show_window:
            workspace.display_data.in_high_class = in_high_class
            workspace.display_data.labels = (objects.segmented,)
            workspace.display_data.saved_values = saved_values

    def display_two_measurements(self, workspace, figure):
        figure.set_subplots((2, 2))
        object_name = self.object_name.value
        for i, feature_name in (
            (0, self.first_measurement.value),
            (1, self.second_measurement.value),
        ):
            saved_values = workspace.display_data.saved_values[i]
            good_saved_values = saved_values[~numpy.isnan(saved_values)]
            if len(good_saved_values) == 0:
                figure.subplot_table(i, 0, [["No %s objects found" % object_name]])
            else:
                axes = figure.subplot(i, 0)
                axes.hist(good_saved_values)
                axes.set_xlabel(feature_name)
                axes.set_ylabel("# of %s" % object_name)
        class_1, class_2 = workspace.display_data.in_high_class
        object_codes = class_1.astype(int) + class_2.astype(int) * 2 + 1
        object_codes = numpy.hstack(([0], object_codes))
        nobjects = len(class_1)
        mapping = numpy.zeros(nobjects + 1, int)
        mapping[1:] = numpy.arange(1, nobjects + 1)
        for i in range(2):
            saved_values = workspace.display_data.saved_values[i]
            mapping[1:][numpy.isnan(saved_values)] = 0
        labels = object_codes[mapping[workspace.display_data.labels]]
        figure.subplot_imshow_labels(0, 1, labels, title=object_name)
        #
        # Draw a 4-bar histogram
        #
        axes = figure.subplot(1, 1)
        values = object_codes[1:]
        axes.hist(values[~numpy.isnan(values)], bins=4, range=(0.5, 4.5))
        axes.set_xticks((1, 2, 3, 4))
        if self.wants_custom_names:
            axes.set_xticklabels(
                (
                    self.low_low_custom_name.value,
                    self.high_low_custom_name.value,
                    self.low_high_custom_name.value,
                    self.high_high_custom_name.value,
                )
            )
        else:
            axes.set_xticklabels(("low\nlow", "high\nlow", "low\nhigh", "high\nhigh"))
        axes.set_ylabel("# of %s" % object_name)
        colors = self.get_colors(len(axes.patches))
        #
        # The patches are the rectangles in the histogram
        #
        for i, patch in enumerate(axes.patches):
            patch.set_facecolor(colors[i + 1, :])

    def run_single_measurement(self, group, workspace):
        """Classify objects based on one measurement"""
        object_name = group.object_name.value
        feature = group.measurement.value
        objects = workspace.object_set.get_objects(object_name)
        measurements = workspace.measurements
        values = measurements.get_current_measurement(object_name, feature)
        #
        # Pad values if too few (defensive programming).
        #
        if len(values) < objects.count:
            values = numpy.hstack((values, [numpy.NaN] * (objects.count - len(values))))
        if group.bin_choice == BC_EVEN:
            low_threshold = group.low_threshold.value
            high_threshold = group.high_threshold.value
            if low_threshold >= high_threshold:
                raise ValueError("Lower Threshold must be less than Upper Threshold")
            bin_count = group.bin_count.value
            thresholds = (
                numpy.arange(bin_count + 1)
                * (high_threshold - low_threshold)
                / float(bin_count)
                + low_threshold
            )
        else:
            thresholds = [
                float(x.strip()) for x in group.custom_thresholds.value.split(",")
            ]
        #
        # Put infinities at either end of the thresholds so we can bin the
        # low and high bins
        #
        thresholds = numpy.hstack(
            (
                [-numpy.inf] if group.wants_low_bin else [],
                thresholds,
                [numpy.inf] if group.wants_high_bin else [],
            )
        )
        #
        # Do a cross-product of objects and threshold comparisons
        #
        ob_idx, th_idx = numpy.mgrid[0 : len(values), 0 : len(thresholds) - 1]
        bin_hits = (values[ob_idx] > thresholds[th_idx]) & (
            values[ob_idx] <= thresholds[th_idx + 1]
        )
        num_values = len(values)
        for bin_idx, feature_name in enumerate(group.bin_feature_names()):
            measurement_name = "_".join((M_CATEGORY, feature_name))
            measurements.add_measurement(
                object_name, measurement_name, bin_hits[:, bin_idx].astype(int)
            )
            measurement_name = "_".join((M_CATEGORY, feature_name, F_NUM_PER_BIN))
            num_hits = bin_hits[:, bin_idx].sum()
            measurements.add_measurement(IMAGE, measurement_name, num_hits)
            measurement_name = "_".join((M_CATEGORY, feature_name, F_PCT_PER_BIN))
            measurements.add_measurement(
                IMAGE,
                measurement_name,
                100.0 * float(num_hits) / num_values if num_values > 0 else 0,
            )
        if group.wants_images or self.show_window:
            colors = self.get_colors(bin_hits.shape[1])
            object_bins = numpy.sum(bin_hits * th_idx, 1) + 1
            object_color = numpy.hstack(([0], object_bins))
            object_color[numpy.hstack((False, numpy.isnan(values)))] = 0
            labels = object_color[objects.segmented]
            if group.wants_images:
                image = colors[labels, :3]
                workspace.image_set.add(
                    group.image_name.value,
                    Image(image, parent_image=objects.parent_image),
                )

            if self.show_window:
                workspace.display_data.bins.append(object_bins[~numpy.isnan(values)])
                workspace.display_data.labels.append(labels)
                workspace.display_data.values.append(values[~numpy.isnan(values)])

    def display_single_measurement(self, workspace, figure):
        """Display an array of single measurements"""
        figure.set_subplots((3, len(self.single_measurements)))
        for i, group in enumerate(self.single_measurements):
            bin_hits = workspace.display_data.bins[i]
            labels = workspace.display_data.labels[i]
            values = workspace.display_data.values[i]
            if len(values) == 0:
                continue
            #
            # A histogram of the values
            #
            axes = figure.subplot(0, i)
            axes.hist(values[~numpy.isnan(values)])
            axes.set_xlabel(group.measurement.value)
            axes.set_ylabel("# of %s" % group.object_name.value)
            #
            # A histogram of the labels yielding the bins
            #
            axes = figure.subplot(1, i)
            axes.hist(
                bin_hits,
                bins=group.number_of_bins(),
                range=(0.5, group.number_of_bins() + 0.5),
            )
            axes.set_xticks(numpy.arange(1, group.number_of_bins() + 1))
            if group.wants_custom_names:
                axes.set_xticklabels(group.bin_names.value.split(","))
            axes.set_xlabel(group.measurement.value)
            axes.set_ylabel("# of %s" % group.object_name.value)
            colors = self.get_colors(len(axes.patches))
            for j, patch in enumerate(axes.patches):
                patch.set_facecolor(colors[j + 1, :])
            #
            # The labels matrix
            #
            figure.subplot_imshow_labels(
                2,
                i,
                labels,
                title=group.object_name.value,
                sharexy=figure.subplot(2, 0),
            )

    def get_colors(self, count):
        """Get colors used for two-measurement labels image"""
        import matplotlib.cm as cm

        cmap = cm.get_cmap(get_default_colormap())
        #
        # Trick the colormap into divulging the values used.
        #
        sm = cm.ScalarMappable(cmap=cmap)
        colors = sm.to_rgba(numpy.arange(count) + 1)
        return numpy.vstack((numpy.zeros(colors.shape[1]), colors))

    def prepare_settings(self, setting_values):
        """Do any sort of adjustment to the settings required for the given values

        setting_values - the values for the settings

        This method allows a module to specialize itself according to
        the number of settings and their value. For instance, a module that
        takes a variable number of images or objects can increase or decrease
        the number of relevant settings so they map correctly to the values."""

        single_measurement_count = int(setting_values[1])
        if single_measurement_count < len(self.single_measurements):
            del self.single_measurements[single_measurement_count:]
        while single_measurement_count > len(self.single_measurements):
            self.add_single_measurement(True)

    def validate_module(self, pipeline):
        if self.contrast_choice == BY_SINGLE_MEASUREMENT:
            for group in self.single_measurements:
                group.validate_group()

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        """Adjust setting values if they came from a previous revision

        setting_values - a sequence of strings representing the settings
                         for the module as stored in the pipeline
        variable_revision_number - the variable revision number of the
                         module at the time the pipeline was saved. Use this
                         to determine how the incoming setting values map
                         to those of the current module version.
        module_name - the name of the module that did the saving. This can be
                      used to import the settings from another module if
                      that module was merged into the current module
        """
        if variable_revision_number == 1:
            # we modified this in the code but didn't want to bump the variable revision number.
            if BY_SINGLE_MEASUREMENT in setting_values[0]:
                contrast_choice = BY_SINGLE_MEASUREMENT
            else:
                contrast_choice = BY_TWO_MEASUREMENTS
            #
            # We inserted wants_low_bin and wants_high_bin in each group
            #
            new_setting_values = [contrast_choice, setting_values[1]]
            setting_values = setting_values[2:]
            for i in range(int(new_setting_values[1])):
                new_setting_values += setting_values[:3]
                #
                # Bin count changed: don't count the outer 2 bins
                #
                new_setting_values += [str(int(setting_values[3]) - 2)]
                new_setting_values += [setting_values[4]] + ["Yes"]
                new_setting_values += [setting_values[5]] + ["Yes"]
                new_setting_values += setting_values[6:11]
                setting_values = setting_values[11:]
            new_setting_values += setting_values
            setting_values = new_setting_values
            variable_revision_number = 2

        return setting_values, variable_revision_number

    def get_measurement_columns(self, pipeline):
        columns = []
        if self.contrast_choice == BY_SINGLE_MEASUREMENT:
            for group in self.single_measurements:
                columns += [
                    (
                        IMAGE,
                        "_".join((M_CATEGORY, feature_name, F_NUM_PER_BIN)),
                        COLTYPE_INTEGER,
                    )
                    for feature_name in group.bin_feature_names()
                ]
                columns += [
                    (
                        IMAGE,
                        "_".join((M_CATEGORY, feature_name, F_PCT_PER_BIN)),
                        COLTYPE_FLOAT,
                    )
                    for feature_name in group.bin_feature_names()
                ]
                columns += [
                    (
                        group.object_name.value,
                        "_".join((M_CATEGORY, feature_name)),
                        COLTYPE_INTEGER,
                    )
                    for feature_name in group.bin_feature_names()
                ]
        else:
            names = self.get_feature_name_matrix()
            columns += [
                (IMAGE, "_".join((M_CATEGORY, name, F_NUM_PER_BIN)), COLTYPE_INTEGER,)
                for name in names.flatten()
            ]
            columns += [
                (IMAGE, "_".join((M_CATEGORY, name, F_PCT_PER_BIN)), COLTYPE_FLOAT,)
                for name in names.flatten()
            ]
            columns += [
                (self.object_name.value, "_".join((M_CATEGORY, name)), COLTYPE_INTEGER,)
                for name in names.flatten()
            ]
        return columns

    def get_categories(self, pipeline, object_name):
        """Return the categories of measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        if (
            (object_name == IMAGE)
            or (
                self.contrast_choice == BY_SINGLE_MEASUREMENT
                and object_name
                in [group.object_name.value for group in self.single_measurements]
            )
            or (
                self.contrast_choice == BY_TWO_MEASUREMENTS
                and object_name == self.object_name
            )
        ):
            return [M_CATEGORY]

        return []

    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        if category != M_CATEGORY:
            return []
        if self.contrast_choice == BY_SINGLE_MEASUREMENT:
            result = []
            for group in self.single_measurements:
                if group.object_name == object_name:
                    return group.bin_feature_names()
                elif object_name == IMAGE:
                    for image_features in (F_NUM_PER_BIN, F_PCT_PER_BIN):
                        for bin_feature_names in group.bin_feature_names():
                            result += ["_".join((bin_feature_names, image_features))]
                    return result
        elif self.contrast_choice == BY_TWO_MEASUREMENTS:
            if self.object_name == object_name:
                return self.get_feature_name_matrix().flatten().tolist()
            elif object_name == IMAGE:
                result = []
                for image_features in (F_NUM_PER_BIN, F_PCT_PER_BIN):
                    for bin_feature_names in (
                        self.get_feature_name_matrix().flatten().tolist()
                    ):
                        result += ["_".join((bin_feature_names, image_features))]
                return result
        return []
