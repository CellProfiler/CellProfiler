"""
CalculateStatistics
===================

**CalculateStatistics** calculates measures of assay quality (V and Z’
factors) and dose-response data (EC50) for all measured features made
from images.

The V and Z’ factors are statistical measures of assay quality and are
calculated for each per-image measurement and for each average
per-object measurement that you have made in the pipeline. Placing this
module at the end of a pipeline in order to calculate these values
allows you to identify which measured features are most powerful for
distinguishing positive and negative control samples (Z' factor), or for accurately
quantifying the assay’s response to dose (V factor). These measurements will be
calculated for all measured values (Intensity, AreaShape, Texture,
etc.) upstream in the pipeline. The statistics calculated by this module
can be exported as the “Experiment” set of data.

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

See also
^^^^^^^^

See also **CalculateMath**.

What do I need as input?
^^^^^^^^^^^^^^^^^^^^^^^^
Example format for a file to be loaded by **LoadData** for this module:

**LoadData** loads information from a CSV file. The first line of this
file is a header that names the items. Each subsequent line represents
data for one image cycle, so your file should have the header line
plus one line per image to be processed. You can also make a file for
**LoadData** to load that contains the positive/negative control and
dose designations *plus* the image file names to be processed, which
is a good way to guarantee that images are matched with the correct
data. The control and dose information can be designated in one of two
ways:

.. _(link): https://doi.org/10.1177/108705719900400206
.. _Ilya Ravkin: http://www.ravkin.net

-  As metadata (so that the column header is prefixed with the
   “Metadata\_” tag). “Metadata” is the category and the name after the
   underscore is the measurement.
-  As some other type of data, in which case the header needs to be of
   the form *<prefix>\_<measurement>*. Select *<prefix>* as the category
   and *<measurement>* as the measurement.

Here is an example file:

+-------------------------+-------------------------+------------------+--------------+
| Image\_FileName\_CY3,   | Image\_PathName\_CY3,   | Data\_Control,   | Data\_Dose   |
+-------------------------+-------------------------+------------------+--------------+
| “Plate1\_A01.tif”,      | “/images”,              | -1,              | 0            |
+-------------------------+-------------------------+------------------+--------------+
| “Plate1\_A02.tif”,      | “/images”,              | 1,               | 1E10         |
+-------------------------+-------------------------+------------------+--------------+
| “Plate1\_A03.tif”,      | “/images”,              | 0,               | 3E4          |
+-------------------------+-------------------------+------------------+--------------+
| “Plate1\_A04.tif”,      | “/images”,              | 0,               | 5E5          |
+-------------------------+-------------------------+------------------+--------------+

|

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  **Experiment features:** Whereas most CellProfiler measurements are
   calculated for each object (per-object) or for each image
   (per-image), this module produces *per-experiment* values; for
   example, one Z’ factor is calculated for each measurement, across the
   entire analysis run.

   -  *Zfactor:* The Z’-factor indicates how well separated the positive
      and negative controls are. A Z’-factor > 0 is potentially
      screenable; a Z’-factor > 0.5 is considered an excellent assay.
      The formula is 1 - 3 × (σ\ :sub:`p` +
      σ\ :sub:`n`)/\|μ\ :sub:`p` - μ\ :sub:`n`\ \| where σ\ :sub:`p` and
      σ\ :sub:`n` are the standard deviations of the positive and
      negative controls, and μ\ :sub:`p` and μ\ :sub:`n` are the means
      of the positive and negative controls.
   -  *Vfactor:* The V-factor is a generalization of the Z’-factor, and
      is calculated as 1 - 6 × mean(σ)/\|μ\ :sub:`p` -
      μ\ :sub:`n`\ \| where σ are the standard deviations of the data,
      and μ\ :sub:`p` and μ\ :sub:`n` are defined as above.
   -  *EC50:* The half maximal effective concentration (EC50) is the
      concentration of a treatment required to induce a response that
      is 50% of the maximal response.
   -  *OneTailedZfactor:* This measure is an attempt to overcome a
      limitation of the original Z’-factor formulation (it assumes a
      Gaussian distribution) and is informative for populations with
      moderate or high amounts of skewness. In these cases, long tails
      opposite to the mid-range point lead to a high standard deviation
      for either population, which results in a low Z’ factor even
      though the population means and samples between the means may be
      well-separated. Therefore, the one-tailed Z’ factor is calculated
      with the same formula but using only those samples that lie
      between the positive/negative population means. **This is not yet
      a well established measure of assay robustness, and should be
      considered experimental.**

For both Z’ and V factors, the highest possible value (best assay
quality) is 1, and they can range into negative values (for assays where
distinguishing between positive and negative controls is difficult or
impossible). The Z’ factor is based only on positive and negative
controls. The V factor is based on an entire dose-response curve rather
than on the minimum and maximum responses. When there are only two doses
in the assay (positive and negative controls only), the V factor will
equal the Z’ factor.

Note that if the standard deviation of a measured feature is zero for a
particular set of samples (e.g., all the positive controls), the Z’ and
V factors will equal 1 despite the fact that the assay quality is poor.
This can occur when there is only one sample at each dose. This also
occurs for some non-informative measured features, like the number of
cytoplasm compartments per cell, which is always equal to 1.

This module can create MATLAB scripts that display the EC50 curves for
each measurement. These scripts will require MATLAB and the statistics
toolbox in order to run. See *Create dose-response plots?* below.

References
^^^^^^^^^^

-  *Z’ factor:* Zhang JH, Chung TD, et al. (1999) “A simple statistical
   parameter for use in evaluation and validation of high throughput
   screening assays” *J Biomolecular Screening* 4(2): 67-73. `(link)`_
-  *V factor:* Ravkin I (2004): Poster #P12024 - Quality Measures for
   Imaging-based Cellular Assays. *Society for Biomolecular Screening
   Annual Meeting Abstracts*.
-  Code for the calculation of Z’ and V factors was kindly donated by
   `Ilya Ravkin`_. Carlos Evangelista donated his copyrighted
   dose-response-related code.
"""

import functools
import os

import numpy
import scipy.optimize
from cellprofiler_core.constants.measurement import EXPERIMENT
from cellprofiler_core.constants.measurement import IMAGE
from cellprofiler_core.constants.measurement import NEIGHBORS
from cellprofiler_core.constants.module import (
    IO_FOLDER_CHOICE_HELP_TEXT,
    IO_WITH_METADATA_HELP_TEXT,
)
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.module import Module
from cellprofiler_core.preferences import ABSOLUTE_FOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_INPUT_FOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_INPUT_SUBFOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_OUTPUT_FOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_OUTPUT_SUBFOLDER_NAME
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting import Divider
from cellprofiler_core.setting import Measurement
from cellprofiler_core.setting import SettingsGroup
from cellprofiler_core.setting import ValidationError
from cellprofiler_core.setting.do_something import DoSomething
from cellprofiler_core.setting.do_something import RemoveSettingButton
from cellprofiler_core.setting.text import Directory
from cellprofiler_core.setting.text import Text

"""# of settings aside from the dose measurements"""
FIXED_SETTING_COUNT = 1
VARIABLE_SETTING_COUNT = 5

PC_CUSTOM = "Custom"


class CalculateStatistics(Module):
    module_name = "CalculateStatistics"
    category = "Data Tools"
    variable_revision_number = 2

    def create_settings(self):
        """Create your settings by subclassing this function

        create_settings is called at the end of initialization.

        You should create the setting variables for your module here:
            # Ask the user for the input image
            self.image_name = .ImageSubscriber(...)
            # Ask the user for the name of the output image
            self.output_image = .ImageName(...)
            # Ask the user for a parameter
            self.smoothing_size = .Float(...)"""

        self.grouping_values = Measurement(
            "Select the image measurement describing the positive and negative control status",
            lambda: IMAGE,
            doc="""\
The Z’ factor, a measure of assay quality, is calculated by this module
based on measurements from images that are specified as positive
controls and images that are specified as negative controls. Images
that are neither are ignored. The module assumes that all of the
negative controls are specified by a minimum value, all of the positive
controls are specified by a maximum value, and all other images have an
intermediate value; this might allow you to use your dosing information
to also specify the positive and negative controls. If you don’t use
actual dose data to designate your controls, a common practice is to
designate -1 as a negative control, 0 as an experimental sample, and 1
as a positive control. In other words, positive controls should all be
specified by a single high value (for instance, 1) and negative controls
should all be specified by a single low value (for instance, -1). Other
samples should have an intermediate value to exclude them from the Z’
factor analysis.

The typical way to provide this information in the pipeline is to create
a text comma-delimited (CSV) file outside of CellProfiler and then load
that file into the pipeline using the **Metadata** module or the legacy
**LoadData** module. In that case, choose the measurement that matches
the column header of the measurement in the input file. See the main
module help for this module or for the **Metadata** module for an
example text file.
""",
        )
        self.dose_values = []
        self.add_dose_value(can_remove=False)
        self.add_dose_button = DoSomething(
            "", "Add another dose specification", self.add_dose_value
        )

    def add_dose_value(self, can_remove=True):
        """Add a dose value measurement to the list

        can_delete - set this to False to keep from showing the "remove"
                     button for images that must be present."""
        group = SettingsGroup()
        group.append(
            "measurement",
            Measurement(
                "Select the image measurement describing the treatment dose",
                lambda: IMAGE,
                doc="""\
The V and Z’ factors, metrics of assay quality, and the EC50,
indicating dose-response, are calculated by this module based on each
image being specified as a particular treatment dose. Choose a
measurement that gives the dose of some treatment for each of your
images. See the help for the previous setting for details.""",
            ),
        )

        group.append(
            "log_transform",
            Binary(
                "Log-transform the dose values?",
                False,
                doc="""\
Select *Yes* if you have dose-response data and you want to
log-transform the dose values before fitting a sigmoid curve.

Select *No* if your data values indicate only positive vs. negative
controls.
"""
                % globals(),
            ),
        )

        group.append(
            "wants_save_figure",
            Binary(
                """Create dose-response plots?""",
                False,
                doc="""Select *Yes* if you want to create and save dose-response plots.
You will be asked for information on how to save the plots."""
                % globals(),
            ),
        )

        group.append(
            "figure_name",
            Text(
                "Figure prefix",
                "",
                doc="""\
*(Used only when creating dose-response plots)*

CellProfiler will create a file name by appending the measurement name
to the prefix you enter here. For instance, if you specify a prefix
of “Dose\_”, when saving a file related to objects you have chosen (for
example, *Cells*) and a particular measurement (for example, *AreaShape_Area*),
CellProfiler will save the figure as *Dose_Cells_AreaShape_Area.m*.
Leave this setting blank if you do not want a prefix.
""",
            ),
        )
        group.append(
            "pathname",
            Directory(
                "Output file location",
                dir_choices=[
                    DEFAULT_OUTPUT_FOLDER_NAME,
                    DEFAULT_INPUT_FOLDER_NAME,
                    ABSOLUTE_FOLDER_NAME,
                    DEFAULT_OUTPUT_SUBFOLDER_NAME,
                    DEFAULT_INPUT_SUBFOLDER_NAME,
                ],
                doc="""\
*(Used only when creating dose-response plots)*

This setting lets you choose the folder for the output files. {fcht}

{mht}
""".format(
                    fcht=IO_FOLDER_CHOICE_HELP_TEXT, mht=IO_WITH_METADATA_HELP_TEXT
                ),
            ),
        )

        group.append("divider", Divider())

        group.append(
            "remover",
            RemoveSettingButton(
                "", "Remove this dose measurement", self.dose_values, group
            ),
        )
        self.dose_values.append(group)

    def settings(self):
        """Return the settings to be loaded or saved to/from the pipeline

        These are the settings (from cellprofiler_core.settings) that are
        either read from the strings in the pipeline or written out
        to the pipeline. The settings should appear in a consistent
        order so they can be matched to the strings in the pipeline.
        """
        return [self.grouping_values] + functools.reduce(
            lambda x, y: x + y,
            [
                [
                    value.measurement,
                    value.log_transform,
                    value.wants_save_figure,
                    value.figure_name,
                    value.pathname,
                ]
                for value in self.dose_values
            ],
        )

    def visible_settings(self):
        """The settings that are visible in the UI
        """
        result = [self.grouping_values]
        for index, dose_value in enumerate(self.dose_values):
            if index > 0:
                result.append(dose_value.divider)
            result += [
                dose_value.measurement,
                dose_value.log_transform,
                dose_value.wants_save_figure,
            ]
            if dose_value.wants_save_figure:
                result += [dose_value.figure_name, dose_value.pathname]
            if index > 0:
                result += [dose_value.remover]
        result.append(self.add_dose_button)
        return result

    def prepare_settings(self, setting_values):
        """Do any sort of adjustment to the settings required for the given values

        setting_values - the values for the settings

        This method allows a module to specialize itself according to
        the number of settings and their value. For instance, a module that
        takes a variable number of images or objects can increase or decrease
        the number of relevant settings so they map correctly to the values.

        See cellprofiler.modules.measureobjectsizeshape for an example.
        """
        value_count = len(setting_values)
        if (value_count - FIXED_SETTING_COUNT) % VARIABLE_SETTING_COUNT != 0:
            raise ValueError(
                "Invalid # of settings (%d) for the CalculateStatistics module"
                % value_count
            )
        dose_count = (value_count - FIXED_SETTING_COUNT) / VARIABLE_SETTING_COUNT
        if len(self.dose_values) > dose_count:
            del self.dose_values[dose_count:]
        while len(self.dose_values) < dose_count:
            self.add_dose_value()

    def run(self, workspace):
        """Run the module

        workspace    - The workspace contains
            pipeline     - instance of cpp for this run
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - the parent frame to whatever frame is created. None means don't draw.

        CalculateStatistics does all of its work after running. Do nothing here.
        """

    def run_as_data_tool(self, workspace):
        self.post_run(workspace)
        workspace.post_run_display(self)

    def get_image_measurements(self, measurements, feature_name):
        assert isinstance(measurements, Measurements)
        image_numbers = measurements.get_image_numbers()
        result = numpy.zeros(len(image_numbers))
        for i, image_number in enumerate(image_numbers):
            value = measurements.get_measurement(IMAGE, feature_name, image_number)
            result[i] = (
                None if value is None else value if numpy.isscalar(value) else value[0]
            )
        return result

    def aggregate_measurement(self, measurements, object_name, feature_name):
        assert isinstance(measurements, Measurements)
        image_numbers = measurements.get_image_numbers()
        result = numpy.zeros(len(image_numbers))
        for i, image_number in enumerate(image_numbers):
            values = measurements.get_measurement(
                object_name, feature_name, image_number
            )
            if values is None:
                result[i] = numpy.nan
            elif numpy.isscalar(values):
                result[i] = values
            elif numpy.any(numpy.isfinite(values)):
                values = numpy.array(values)
                result[i] = numpy.mean(values[numpy.isfinite(values)])
            else:
                result[i] = numpy.nan
        return result

    def post_run(self, workspace):
        """Do post-processing after the run completes

        workspace - the workspace at the end of the run
        """
        measurements = workspace.measurements
        assert isinstance(measurements, Measurements)
        all_objects = [
            x
            for x in measurements.get_object_names()
            if x not in [EXPERIMENT, NEIGHBORS]
        ]
        feature_set = []
        image_numbers = measurements.get_image_numbers()
        for object_name in all_objects:
            all_features = [
                x
                for x in measurements.get_feature_names(object_name)
                if self.include_feature(measurements, object_name, x, image_numbers)
            ]
            feature_set += [
                (object_name, feature_name) for feature_name in all_features
            ]
        grouping_data = self.get_image_measurements(
            measurements, self.grouping_values.value
        )
        grouping_data = grouping_data.flatten()
        data = numpy.zeros((len(grouping_data), len(feature_set)))
        for i, (object_name, feature_name) in enumerate(feature_set):
            data[:, i] = self.aggregate_measurement(
                measurements, object_name, feature_name
            )

        z, z_one_tailed, OrderedUniqueDoses, OrderedAverageValues = z_factors(
            grouping_data, data
        )
        #
        # For now, use first dose value only
        #
        dose_data = self.get_image_measurements(
            measurements, self.dose_values[0].measurement.value
        )
        dose_data = numpy.array(dose_data).flatten()
        v = v_factors(dose_data, data)
        expt_measurements = {
            "Zfactor": z,
            "Vfactor": v,
            "OneTailedZfactor": z_one_tailed,
        }
        for dose_group in self.dose_values:
            dose_feature = dose_group.measurement.value
            dose_data = self.get_image_measurements(measurements, dose_feature)
            ec50_coeffs = calculate_ec50(
                dose_data, data, dose_group.log_transform.value
            )
            if len(self.dose_values) == 1:
                name = "EC50"
            else:
                name = "EC50_" + dose_feature
            expt_measurements[name] = ec50_coeffs[:, 2]
            if dose_group.wants_save_figure:
                pathname = dose_group.pathname.get_absolute_path(measurements)
                if not os.path.exists(pathname):
                    os.makedirs(pathname)
                write_figures(
                    dose_group.figure_name,
                    pathname,
                    dose_feature,
                    dose_data,
                    data,
                    ec50_coeffs,
                    feature_set,
                    dose_group.log_transform.value,
                )

        for i, (object_name, feature_name) in enumerate(feature_set):
            for statistic, value in list(expt_measurements.items()):
                sfeature_name = "_".join((statistic, object_name, feature_name))
                measurements.add_experiment_measurement(sfeature_name, value[i])
        if self.show_window:
            workspace.display_data.expt_measurements = expt_measurements
            workspace.display_data.feature_set = feature_set

    def display_post_run(self, workspace, figure):
        expt_measurements = workspace.display_data.expt_measurements
        feature_set = workspace.display_data.feature_set
        figure.set_subplots((2, 1))
        for ii, key in enumerate(("Zfactor", "Vfactor")):
            a = expt_measurements[key]
            indexes = numpy.lexsort((-a,))
            col_labels = ["Object", "Feature", key]
            stats = [[feature_set[i][0], feature_set[i][1], a[i]] for i in indexes[:10]]
            figure.subplot_table(ii, 0, stats, col_labels=col_labels)

    def include_feature(self, measurements, object_name, feature_name, image_numbers):
        """Return true if we should analyze a feature"""
        if feature_name.find("Location") != -1:
            return False
        if feature_name.find("ModuleError") != -1:
            return False
        if feature_name.find("ExecutionTime") != -1:
            return False
        if object_name == IMAGE and feature_name == self.grouping_values:
            # Don't measure the pos/neg controls
            return False
        if object_name == IMAGE and feature_name in [
            g.measurement.value for g in self.dose_values
        ]:
            return False
        if len(image_numbers) == 0:
            return False
        for image_number in image_numbers:
            v = measurements.get_measurement(object_name, feature_name, image_number)
            if v is not None:
                break
        else:
            return False
        if numpy.isscalar(v):
            return not (isinstance(v, str))
        #
        # Make sure the measurement isn't a string or other oddity
        #
        return numpy.asanyarray(v).dtype.kind not in "OSU"

    def validate_module_warnings(self, pipeline):
        """Warn user re: Test mode """
        if pipeline.test_mode:
            raise ValidationError(
                "CalculateStatistics will not produce any output in test mode",
                self.grouping_values,
            )

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):

        PC_DEFAULT = "Default output folder"
        PC_WITH_IMAGE = "Same folder as image"

        if variable_revision_number == 1:
            #
            # Minor change: Default output directory -> folder
            #
            new_setting_values = [setting_values[0]]
            for offset in range(1, len(setting_values), 6):
                dir_choice = setting_values[offset + 4]
                custom_path = setting_values[offset + 5]
                if dir_choice == PC_CUSTOM:
                    if custom_path[0] == ".":
                        dir_choice = DEFAULT_OUTPUT_SUBFOLDER_NAME
                    elif custom_path[0] == "&":
                        dir_choice = DEFAULT_OUTPUT_SUBFOLDER_NAME
                        custom_path = "." + custom_path[1:]
                    else:
                        dir_choice = ABSOLUTE_FOLDER_NAME
                directory = Directory.static_join_string(dir_choice, custom_path)
                new_setting_values += setting_values[offset : (offset + 4)]
                new_setting_values += [directory]
            setting_values = new_setting_values
            variable_revision_number = 2

        # Standardize input/output directory name references
        setting_values = list(setting_values)
        for offset in range(5, len(setting_values), VARIABLE_SETTING_COUNT):
            setting_values[offset] = Directory.upgrade_setting(setting_values[offset])

        return setting_values, variable_revision_number


########################################################
#
# The following code is adapted from Matlab code donated by Ilya Ravkin
#
# http://www.ravkin.net
########################################################
def z_factors(xcol, ymatr):
    """xcol is (Nobservations,1) column vector of grouping values
           (in terms of dose curve it may be Dose).
       ymatr is (Nobservations, Nmeasures) matrix, where rows correspond to
           observations and columns corresponds to different measures.

       returns v, z, z_one_tailed, OrderedUniqueDoses, OrderedAverageValues
       z and z_bwtn_mean are (1, Nmeasures) row vectors containing Z'- and
       between-mean Z'-factors for the corresponding measures.

       When ranges are zero, we set the Z' factors to a very negative
       value."""

    xs, avers, stds = loc_shrink_mean_std(xcol, ymatr)
    # Z' factor is defined by the positive and negative controls, so we take the
    # extremes BY DOSE of the averages and stdevs.
    zrange = numpy.abs(avers[0, :] - avers[-1, :])
    zstd = stds[0, :] + stds[-1, :]
    zstd[zrange == 0] = 1
    zrange[zrange == 0] = 0.000001
    z = 1 - 3 * (zstd / zrange)

    # The one-tailed Z' factor is defined by using only the samples between the
    # means, again defined by DOSE extremes
    zrange = numpy.abs(avers[0, :] - avers[-1, :])
    exp1_vals = ymatr[xcol == xs[0], :]
    exp2_vals = ymatr[xcol == xs[-1], :]
    #
    # Sort the average positive control values and negative control values
    # so that the lowest is in index 0 and the highest is in index 1 independent
    # of whether the control is negative or positive
    #
    sort_avers = numpy.sort(numpy.array((avers[0, :], avers[-1, :])), 0)

    for i in range(sort_avers.shape[1]):
        # Here the std must be calculated using the full formula
        exp1_cvals = exp1_vals[:, i]
        exp2_cvals = exp2_vals[:, i]
        vals1 = exp1_cvals[
            (exp1_cvals >= sort_avers[0, i]) & (exp1_cvals <= sort_avers[1, i])
        ]
        vals2 = exp2_cvals[
            (exp2_cvals >= sort_avers[0, i]) & (exp2_cvals <= sort_avers[1, i])
        ]
        stds[0, i] = numpy.sqrt(numpy.sum((vals1 - sort_avers[0, i]) ** 2) / len(vals1))
        stds[1, i] = numpy.sqrt(numpy.sum((vals2 - sort_avers[1, i]) ** 2) / len(vals2))

    zstd = stds[0, :] + stds[1, :]

    # If means aren't the same and stdev aren't NaN, calculate the value
    z_one_tailed = 1 - 3 * (zstd / zrange)
    # Otherwise, set it to a really negative value
    z_one_tailed[(~numpy.isfinite(zstd)) | (zrange == 0)] = -1e5
    return z, z_one_tailed, xs, avers


def v_factors(xcol, ymatr):
    """xcol is (Nobservations,1) column vector of grouping values
           (in terms of dose curve it may be Dose).
       ymatr is (Nobservations, Nmeasures) matrix, where rows correspond to
           observations and columns corresponds to different measures.

        Calculate the V factor = 1-6 * mean standard deviation / range
    """
    xs, avers, stds = loc_shrink_mean_std(xcol, ymatr)
    #
    # Range of averages per label
    #
    vrange = numpy.max(avers, 0) - numpy.min(avers, 0)
    #
    # Special handling for labels that have no ranges
    #
    vstd = numpy.zeros(len(vrange))
    vstd[vrange == 0] = 1
    vstd[vrange != 0] = numpy.mean(stds[:, vrange != 0], 0)
    vrange[vrange == 0] = 0.000001
    v = 1 - 6 * (vstd / vrange)
    return v


def loc_shrink_mean_std(xcol, ymatr):
    """Compute mean and standard deviation per label

    xcol - column of image labels or doses
    ymatr - a matrix with rows of values per image and columns
            representing different measurements

    returns xs - a vector of unique doses
            avers - the average value per label
            stds - the standard deviation per label
    """
    ncols = ymatr.shape[1]
    labels, labnum, xs = loc_vector_labels(xcol)
    avers = numpy.zeros((labnum, ncols))
    stds = avers.copy()
    for ilab in range(labnum):
        labinds = labels == ilab
        labmatr = ymatr[labinds, :]
        if labmatr.shape[0] == 1:
            avers[ilab, :] = labmatr[0, :]
        else:
            avers[ilab, :] = numpy.mean(labmatr, 0)
            stds[ilab, :] = numpy.std(labmatr, 0)
    return xs, avers, stds


def loc_vector_labels(x):
    """Identify unique labels from the vector of image labels

    x - a vector of one label or dose per image

    returns labels, labnum, uniqsortvals
    labels - a vector giving an ordinal per image where that ordinal
             is an index into the vector of unique labels (uniqsortvals)
    labnum - # of unique labels in x
    uniqsortvals - a vector containing the unique labels in x
    """
    #
    # Get the index of each image's label in the sorted array
    #
    order = numpy.lexsort((x,))
    reverse_order = numpy.lexsort((order,))
    #
    # Get a sorted view of the labels
    #
    sorted_x = x[order]
    #
    # Find the elements that start a new run of labels in the sorted array
    # ex: 0,0,0,3,3,3,5,5,5
    #     1,0,0,1,0,0,1,0,0
    #
    # Then cumsum - 1 turns into:
    #     0,0,0,1,1,1,2,2,2
    #
    # and sorted_x[first_occurrence] gives the unique labels in order
    first_occurrence = numpy.ones(len(x), bool)
    first_occurrence[1:] = sorted_x[:-1] != sorted_x[1:]
    sorted_labels = numpy.cumsum(first_occurrence) - 1
    labels = sorted_labels[reverse_order]
    uniqsortvals = sorted_x[first_occurrence]
    return labels, len(uniqsortvals), uniqsortvals


#######################################################
#
# The following code computes the EC50 dose response
#
#######################################################
def calculate_ec50(conc, responses, Logarithmic):
    """EC50 Function to fit a dose-response data to a 4 parameter dose-response
       curve.

       Inputs: 1. a 1 dimensional array of drug concentrations
               2. the corresponding m x n array of responses
       Algorithm: generate a set of initial coefficients including the Hill
                  coefficient
                  fit the data to the 4 parameter dose-response curve using
                  nonlinear least squares
       Output: a matrix of the 4 parameters
               results[m,1]=min
               results[m,2]=max
               results[m,3]=ec50
               results[m,4]=Hill coefficient

       Original Matlab code Copyright 2004 Carlos Evangelista
       send comments to CCEvangelista@aol.com
       """
    # If we are using a log-domain set of doses, we have a better chance of
    # fitting a sigmoid to the curve if the concentrations are
    # log-transformed.
    if Logarithmic:
        conc = numpy.log(conc)

    n = responses.shape[1]
    results = numpy.zeros((n, 4))

    def error_fn(v, x, y):
        """Least-squares error function

        This measures the least-squares error of fitting the sigmoid
        with parameters in v to the x and y data.
        """
        return numpy.sum((sigmoid(v, x) - y) ** 2)

    for i in range(n):
        response = responses[:, i]
        v0 = calc_init_params(conc, response)
        v = scipy.optimize.fmin(
            error_fn, v0, args=(conc, response), maxiter=1000, maxfun=1000, disp=False
        )
        results[i, :] = v
    return results


def sigmoid(v, x):
    """This is the EC50 sigmoid function

    v is a vector of parameters:
        v[0] = minimum allowed value
        v[1] = maximum allowed value
        v[2] = ec50
        v[3] = Hill coefficient
    """
    p_min, p_max, ec50, hill = v
    return p_min + ((p_max - p_min) / (1 + (x / ec50) ** hill))


def calc_init_params(x, y):
    """This generates the min, max, x value at the mid-y value, and Hill
      coefficient. These values are starting points for the sigmoid fitting.

      x & y are the points to be fit
      returns minimum, maximum, ec50 and hill coefficient starting points
      """
    min_0 = min(y)
    max_0 = max(y)

    # Parameter 3
    # OLD:  parms(3)=(min(x)+max(x))/2;
    # This is an estimate of the EC50, i.e., the half maximal effective
    # concentration (here denoted as x-value)
    #
    # Note that this was originally simply mean([max(x); min(x)]).  This does not
    # take into account the y-values though, so it was changed to be the
    # x-value that corresponded to the y-value closest to the mean([max(y); min(y)]).
    # Unfortunately, for x-values with only two categories e.g., [0 1], this results in
    # an initial EC50 of either 0 or 1 (min(x) or max(x)), which seems a bad estimate.
    # 5 We will take a two-pronged approach: Use the estimate from this latter approach,
    # unless the parameter will equal either the max(x) or min(x).  In this case, we will use the
    # former approach, namely (mean([max(x); min(x)]).  DL 2007.09.24
    YvalueAt50thPercentile = (min(y) + max(y)) / 2
    DistanceToCentralYValue = numpy.abs(y - YvalueAt50thPercentile)
    LocationOfNearest = numpy.argmin(DistanceToCentralYValue)
    XvalueAt50thPercentile = x[LocationOfNearest]
    if XvalueAt50thPercentile == min(x) or XvalueAt50thPercentile == max(x):
        ec50 = (min(x) + max(x)) / 2
    else:
        ec50 = XvalueAt50thPercentile

    # Parameter 4
    # The OLD way used 'size' oddly - perhaps meant 'length'?  It would cause
    # divide-by-zero warnings since 'x(2)-x(sizex)' would necessarily have
    # zeros.
    # The NEW way just checks to see whether the depenmdent var is increasing (note
    # negative hillc) or decreasing (positive hillc) and sets them initially
    # to +/-1.  This could be smarter about how to initialize hillc, but +/-1 seems ok for now
    # DL 2007.09.25

    # OLD
    # sizey=size(y);
    # sizex=size(x);
    # if (y(1)-y(sizey))./(x(2)-x(sizex))>0
    #     init_params(4)=(y(1)-y(sizey))./(x(2)-x(sizex));
    # else
    #     init_params(4)=1;
    # end

    # I've made this look at the Y response at the minimum and maximum dosage
    # whereas before, it was looking at the Y response at the first and last
    # point which could just happen to be the same.
    min_idx = numpy.argmin(x)
    max_idx = numpy.argmax(x)
    x0 = x[min_idx]
    x1 = x[max_idx]
    y0 = y[min_idx]
    y1 = y[max_idx]

    if x0 == x1:
        # If all of the doses are the same, why are we doing this?
        # There's not much point in fitting.
        raise ValueError(
            "All doses or labels for all image sets are %s. Can't calculate dose-response curves."
            % x0
        )
    elif y1 > y0:
        hillc = -1
    else:
        hillc = 1
    return min_0, max_0, ec50, hillc


def write_figures(
    prefix,
    directory,
    dose_name,
    dose_data,
    data,
    ec50_coeffs,
    feature_set,
    log_transform,
):
    """Write out figure scripts for each measurement

    prefix - prefix for file names
    directory - write files into this directory
    dose_name - name of the dose measurement
    dose_data - doses per image
    data - data per image
    ec50_coeffs - coefficients calculated by calculate_ec50
    feature_set - tuples of object name and feature name in same order as data
    log_transform - true to log-transform the dose data
    """
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_pdf import FigureCanvasPdf

    if log_transform:
        dose_data = numpy.log(dose_data)
    for i, (object_name, feature_name) in enumerate(feature_set):
        fdata = data[:, i]
        fcoeffs = ec50_coeffs[i, :]
        filename = "%s%s_%s.pdf" % (prefix, object_name, feature_name)
        pathname = os.path.join(directory, filename)
        f = Figure()
        canvas = FigureCanvasPdf(f)
        ax = f.add_subplot(1, 1, 1)
        x = numpy.linspace(0, numpy.max(dose_data), num=100)
        y = sigmoid(fcoeffs, x)
        ax.plot(x, y)
        dose_y = sigmoid(fcoeffs, dose_data)
        ax.plot(dose_data, dose_y, "o")
        ax.set_xlabel("Dose")
        ax.set_ylabel("Response")
        ax.set_title("%s_%s" % (object_name, feature_name))
        f.savefig(pathname)
