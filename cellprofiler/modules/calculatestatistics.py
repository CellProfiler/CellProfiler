'''<b>Calculate Statistics</b> calculates measures of assay quality 
(V and Z' factors) and dose response data (EC50) for all measured features
made from images.
<hr>
The V and Z' factors are statistical measures of assay quality and are
calculated for each per-image measurement and for each average per-object
measurement that you have made in the pipeline. Placing
this module at the end of a pipeline in order to calculate these values allows
you to identify which measured features are most powerful for distinguishing
positive and negative control samples, or for accurately quantifying the assay's
response to dose. These measurements will be calculated for all
measured values (Intensity, AreaShape, Texture, etc.). These measurements
can be exported as the "Experiment" set of data.

<h4>Available measurements</h4>

<ul>
<li><b>Experiment features:</b> Whereas most CellProfiler measurements are calculated for each object (per-object)
or for each image (per-image), this module produces <i>per-experiment</i> values;
for example, one Z' factor is calculated for each measurement, across the entire analysis run.
<ul>
<li><i>Zfactor:</i> The Z'-factor indicates how well separated the positive and negative controls are.
A Z'-factor &gt; 0 is potentially screenable; a Z'-factor &gt; 0.5 is considered an excellent assay.
The formula is 1 &minus 3 &times; (&sigma;<sub>p</sub> + &sigma;<sub>n</sub>)/|&mu;<sub>p</sub> - &mu;<sub>n</sub>|
where &sigma;<sub>p</sub> and &sigma;<sub>n</sub> are the standard deviations of the positive and negative controls,
and &mu;<sub>p</sub> and &mu;<sub>n</sub> are the means of the positive and negative controls.</li>
<li><i>Vfactor:</i> The V-factor is a generalization of the Z'-factor, and is
calculated as 1 &minus 6 &times; mean(&sigma;)/|&mu;<sub>p</sub> - &mu;<sub>n</sub>| where
&sigma; are the standard deviations of the data, and &mu;<sub>p</sub> and &mu;<sub>n</sub>
are defined as above.</li>
<li><i>EC50:</i> The half maximal effective concentration (EC50) is the concentration of a
treatment required to induce a response which is 50%% of the maximal response.</li>
<li><i>OneTailedZfactor:</i> This measure is an attempt to overcome a limitation of
the original Z'-factor formulation (it assumes a Gaussian distribution) and is
informative for populations with moderate or high amounts of skewness. In these
cases, long tails opposite to the mid-range point lead to a high standard deviation
for either population, which results in a low Z' factor even though the population
means and samples between the means may be well-separated. Therefore, the
one-tailed Z' factor is calculated with the same formula but using only those samples that lie
between the positive/negative population means. <b>This is not yet a well established
measure of assay robustness, and should be considered experimental.</b></li>
</ul>
</li>
</ul>

For both Z' and V factors, the highest possible value (best assay
quality) is 1, and they can range into negative values (for assays where
distinguishing between positive and negative controls is difficult or
impossible). The Z' factor is based only on positive and negative controls. The V
factor is based on an entire dose-response curve rather than on the
minimum and maximum responses. When there are only two doses in the assay
(positive and negative controls only), the V factor will equal the Z'
factor.

<p><i>Note:</i> If the standard deviation of a measured feature is zero for a
particular set of samples (e.g., all the positive controls), the Z' and V
factors will equal 1 despite the fact that the assay quality is poor.
This can occur when there is only one sample at each dose.
This also occurs for some non-informative measured features, like the
number of cytoplasm compartments per cell, which is always equal to 1.</p>

<p>This module can create MATLAB scripts that display the EC50 curves for
each measurement. These scripts will require MATLAB and the statistics
toolbox in order to run. See <a href='#wants_save_figure'>
<i>Create dose/response plots?</i></a> below.</p>

<h4>References</h4>
<ul>
<li><i>Z' factor:</i> Zhang JH, Chung TD, et al. (1999) "A
simple statistical parameter for use in evaluation and validation of high
throughput screening assays" <i>J Biomolecular Screening</i> 4(2): 67-73.
<a href="http://dx.doi.org/10.1177/108705719900400206">(link)</a></li>
<li><i>V factor:</i> Ravkin I (2004): Poster #P12024 - Quality
Measures for Imaging-based Cellular Assays. <i>Society for Biomolecular
Screening Annual Meeting Abstracts</i>. </li>
<li>Code for the calculation of Z' and V factors was kindly donated by
<a href="http://www.ravkin.net">Ilya Ravkin</a>. Carlos
Evangelista donated his copyrighted dose-response-related code.</li>
</ul>

<p><i>Example format for a file to be loaded by <b>LoadData</b> for this module:</i><br>

<b>LoadData</b> loads information from a CSV file. The first line of this file is a
header that names the items.
Each subsequent line represents data for one image cycle, so your file should have
the header line plus one line per image to be processed. You can also make a
file for <b>LoadData</b> to load that contains the positive/negative control and
dose designations <i>plus</i> the image file names to be processed, which is a good way
to guarantee that images are matched with the correct data. The control and dose
information can be designated in one of two ways:
<ul>
<li>As metadata (so that the column header is prefixed with
the "Metadata_" tag). "Metadata" is the category and the name after the underscore
is the measurement.</li>
<li>As some other type of data, in which case the header needs
to be of the form <i>&lt;prefix&gt;_&lt;measurement&gt;</i>. Select <i>&lt;prefix&gt;</i> as
the category and <i>&lt;measurement&gt;</i> as the measurement.</li>
</ul>
Here is an example file:<br><br>
<code>
<tt><table>
<tr><td>Image_FileName_CY3,</td><td>Image_PathName_CY3,</td><td>Data_Control,</td><td>Data_Dose</td></tr>
<tr><td>"Plate1_A01.tif",</td><td>"/images",</td><td>-1,</td><td>0</td></tr>
<tr><td>"Plate1_A02.tif",</td><td>"/images",</td><td>1,</td><td>1E10</td></tr>
<tr><td>"Plate1_A03.tif",</td><td>"/images",</td><td>0,</td><td>3E4</td></tr>
<tr><td>"Plate1_A04.tif",</td><td>"/images",</td><td>0,</td><td>5E5</td></tr>
</table></tt>
</code>
<br>

See also the <b>Metadata</b> and legacy <b>LoadData</b> modules.
'''

import os

import numpy as np
import scipy.optimize

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.preferences as cpprefs
import cellprofiler.settings as cps
from cellprofiler.gui.help import USING_METADATA_TAGS_REF, USING_METADATA_HELP_REF
from cellprofiler.preferences import standardize_default_folder_names, \
     DEFAULT_INPUT_FOLDER_NAME, DEFAULT_OUTPUT_FOLDER_NAME, \
     IO_FOLDER_CHOICE_HELP_TEXT, IO_WITH_METADATA_HELP_TEXT
from cellprofiler.settings import YES, NO

'''# of settings aside from the dose measurements'''
FIXED_SETTING_COUNT = 1
VARIABLE_SETTING_COUNT = 5

PC_CUSTOM      = "Custom"

class CalculateStatistics(cpm.CPModule):
    module_name = "CalculateStatistics"
    category = "Data Tools"
    variable_revision_number = 2
    def create_settings(self):
        """Create your settings by subclassing this function

        create_settings is called at the end of initialization.

        You should create the setting variables for your module here:
            # Ask the user for the input image
            self.image_name = cellprofiler.settings.ImageNameSubscriber(...)
            # Ask the user for the name of the output image
            self.output_image = cellprofiler.settings.ImageNameProvider(...)
            # Ask the user for a parameter
            self.smoothing_size = cellprofiler.settings.Float(...)"""

        self.grouping_values = cps.Measurement(
            "Select the image measurement describing the positive and negative control status",
            lambda : cpmeas.IMAGE, doc = '''
            The Z' factor, a measure of assay quality, is calculated by this
            module based on measurements from images that are specified as positive controls
            and images that are specified as negative controls. (Images that are neither are
            ignored.) The module assumes that
            all of the negative controls are specified by a minimum value, all of the
            positive controls are specified by a maximum value, and all other images have an
            intermediate value; this might allow you to use your dosing information to also
            specify the positive and negative controls. If you don't use actual dose
            data to designate your controls, a common practice is to designate -1 as a
            negative control, 0 as an experimental sample, and 1 as a positive control.
            In other words, positive controls should all be specified by a single high
            value (for instance, 1) and negative controls should all be specified by a
            single low value (for instance, 0). Other samples should have an intermediate value
            to exclude them from the Z' factor analysis.<p>
            The typical way to provide this information in the pipeline is to create
            a text comma-delimited (CSV) file outside of CellProfiler and then load that file into the pipeline
            using the <b>Metadata</b> module or the legacy <b>LoadData</b> module. In that case, choose the
            measurement that matches the column header of the measurement
            in the input file. See the <b>Metadata</b> module help for an example text file.''')
        self.dose_values = []
        self.add_dose_value(can_remove = False)
        self.add_dose_button = cps.DoSomething("","Add another dose specification",
                                               self.add_dose_value)

    def add_dose_value(self,can_remove = True):
        '''Add a dose value measurement to the list

        can_delete - set this to False to keep from showing the "remove"
                     button for images that must be present.'''
        group = cps.SettingsGroup()
        group.append("measurement",
                     cps.Measurement("Select the image measurement describing the treatment dose",
                                     lambda : cpmeas.IMAGE,
                                     doc = """
            The V and Z' factor, a measure of assay quality, and the EC50, indicating
            dose/response, are calculated by this module based on each image being
            specified as a particular treatment dose. Choose a measurement that gives
            the dose of some treatment for each of your images. <p>
            The typical way to provide this information in the pipeline is to create
            a comma-delimited text file (CSV) outside of CellProfiler and then load that file into the pipeline
            using <b>Metadata</b> or the <b>LoadData</b>. In that case, choose the
            measurement that matches the column header of the measurement
            in the CSV input file. See <b>LoadData</b> help for an example text file.
            """))

        group.append("log_transform",cps.Binary(
            "Log-transform the dose values?",False,doc = '''
            Select <i>%(YES)s</i> if you have dose-response data and you want to log-transform
            the dose values before fitting a sigmoid curve.
            <p>Select <i>%(NO)s</i> if your data values indicate only positive vs. negative
            controls.</p>'''%globals()))

        group.append('wants_save_figure', cps.Binary(
            '''Create dose/response plots?''',False,doc = '''<a name='wants_save_figure'></a>
            Select <i>%(YES)s</i> if you want to create and save
            dose response plots. You will be asked for information on how to save the plots.'''%globals()))

        group.append('figure_name', cps.Text(
            "Figure prefix","", doc = '''
            <i>(Used only when creating dose/response plots)</i><br>
            CellProfiler will create a file name by appending the measurement name
            to the prefix you enter here. For instance, if you have objects
            named, "Cells", the "AreaShape_Area measurement", and a prefix of "Dose_",
            CellProfiler will save the figure as <i>Dose_Cells_AreaShape_Area.m</i>.
            Leave this setting blank if you do not want a prefix.'''
        ))
        group.append('pathname', cps.DirectoryPath(
            "Output file location",
            dir_choices = [
                cps.DEFAULT_OUTPUT_FOLDER_NAME, cps.DEFAULT_INPUT_FOLDER_NAME,
                cps.ABSOLUTE_FOLDER_NAME, cps.DEFAULT_OUTPUT_SUBFOLDER_NAME,
                cps.DEFAULT_INPUT_SUBFOLDER_NAME], doc="""
            <i>(Used only when creating dose/response plots)</i><br>
            This setting lets you choose the folder for the output
            files. %(IO_FOLDER_CHOICE_HELP_TEXT)s

            <p>%(IO_WITH_METADATA_HELP_TEXT)s %(USING_METADATA_TAGS_REF)s
            For instance, if you have a metadata tag named
            "Plate", you can create a per-plate folder by selecting one of the subfolder options
            and then specifying the subfolder name as "\g&lt;Plate&gt;". The module will
            substitute the metadata values for the current image set for any metadata tags in the
            folder name. %(USING_METADATA_HELP_REF)s.</p>"""%globals()))

        group.append("divider", cps.Divider())

        group.append("remover", cps.RemoveSettingButton("", "Remove this dose measurement",
                                                        self.dose_values,
                                                        group))
        self.dose_values.append(group)

    def settings(self):
        """Return the settings to be loaded or saved to/from the pipeline

        These are the settings (from cellprofiler.settings) that are
        either read from the strings in the pipeline or written out
        to the pipeline. The settings should appear in a consistent
        order so they can be matched to the strings in the pipeline.
        """
        return ([self.grouping_values] +
                reduce(lambda x,y:x+y,
                       [[value.measurement, value.log_transform,
                         value.wants_save_figure, value.figure_name,
                         value.pathname]
                        for value in self.dose_values]))

    def visible_settings(self):
        """The settings that are visible in the UI
        """
        result = [self.grouping_values]
        for index,dose_value in enumerate(self.dose_values):
            if index > 0:
                result.append(dose_value.divider)
            result += [dose_value.measurement, dose_value.log_transform,
                       dose_value.wants_save_figure]
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
            raise ValueError("Invalid # of settings (%d) for the CalculateStatistics module" %
                             value_count)
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
        assert isinstance(measurements, cpmeas.Measurements)
        image_numbers = measurements.get_image_numbers()
        result = np.zeros(len(image_numbers))
        for i, image_number in enumerate(image_numbers):
            value = measurements.get_measurement(
                cpmeas.IMAGE, feature_name, image_number)
            result[i] = (None if value is None
                         else value if np.isscalar(value) else value[0])
        return result

    def aggregate_measurement(self, measurements, object_name, feature_name):
        assert isinstance(measurements, cpmeas.Measurements)
        image_numbers = measurements.get_image_numbers()
        result = np.zeros(len(image_numbers))
        for i, image_number in enumerate(image_numbers):
            values = measurements.get_measurement(
                object_name, feature_name, image_number)
            if values is None:
                result[i] = np.nan
            elif np.isscalar(values):
                result[i] = values
            elif np.any(np.isfinite(values)):
                values = np.array(values)
                result[i] = np.mean(values[np.isfinite(values)])
            else:
                result[i] = np.nan
        return result

    def post_run(self, workspace):
        """Do post-processing after the run completes

        workspace - the workspace at the end of the run
        """
        measurements = workspace.measurements
        assert isinstance(measurements, cpmeas.Measurements)
        all_objects = [x for x in measurements.get_object_names()
                       if x not in [cpmeas.EXPERIMENT, cpmeas.NEIGHBORS]]
        feature_set = []
        image_numbers = measurements.get_image_numbers()
        for object_name in all_objects:
            all_features = [
                x for x in measurements.get_feature_names(object_name)
                if self.include_feature(
                    measurements, object_name, x, image_numbers)]
            feature_set += [(object_name, feature_name)
                            for feature_name in all_features]
        grouping_data = self.get_image_measurements(
            measurements, self.grouping_values.value)
        grouping_data = grouping_data.flatten()
        data = np.zeros((len(grouping_data), len(feature_set)))
        for i, (object_name, feature_name) in enumerate(feature_set):
            data[:,i] = self.aggregate_measurement(
                measurements, object_name, feature_name)

        z, z_one_tailed, OrderedUniqueDoses, OrderedAverageValues = \
             z_factors(grouping_data, data)
        #
        # For now, use first dose value only
        #
        dose_data = self.get_image_measurements(
            measurements, self.dose_values[0].measurement.value)
        dose_data = np.array(dose_data).flatten()
        v = v_factors(dose_data, data)
        expt_measurements = {
            "Zfactor": z,
            "Vfactor": v,
            "OneTailedZfactor":z_one_tailed
            }
        for dose_group in self.dose_values:
            dose_feature = dose_group.measurement.value
            dose_data = self.get_image_measurements(
                measurements, dose_feature)
            ec50_coeffs = calculate_ec50(dose_data, data,
                                         dose_group.log_transform.value)
            if len(self.dose_values) == 1:
                name = "EC50"
            else:
                name = "EC50_"+dose_feature
            expt_measurements[name] = ec50_coeffs[:,2]
            if dose_group.wants_save_figure:
                pathname = dose_group.pathname.get_absolute_path(measurements)
                if not os.path.exists(pathname):
                    os.makedirs(pathname)
                write_figures(dose_group.figure_name, pathname, dose_feature,
                              dose_data, data, ec50_coeffs, feature_set,
                              dose_group.log_transform.value)

        for i, (object_name, feature_name) in enumerate(feature_set):
            for statistic, value in expt_measurements.iteritems():
                sfeature_name = '_'.join((statistic, object_name, feature_name))
                measurements.add_experiment_measurement(sfeature_name, value[i])
        if self.show_window:
            workspace.display_data.expt_measurements = expt_measurements
            workspace.display_data.feature_set = feature_set

    def display_post_run(self, workspace, figure):
        expt_measurements = workspace.display_data.expt_measurements
        feature_set = workspace.display_data.feature_set
        figure.set_subplots((2, 1))
        for ii, key in enumerate(("Zfactor","Vfactor")):
            a = expt_measurements[key]
            indexes = np.lexsort((-a,))
            col_labels = ["Object","Feature",key]
            stats = [[feature_set[i][0], feature_set[i][1], a[i]]
                       for i in indexes[:10]]
            figure.subplot_table(ii,0, stats, col_labels=col_labels)

    def include_feature(self, measurements, object_name, feature_name,
                        image_numbers):
        '''Return true if we should analyze a feature'''
        if feature_name.find("Location") != -1:
            return False
        if feature_name.find("ModuleError") != -1:
            return False
        if feature_name.find("ExecutionTime") != -1:
            return False
        if (object_name == cpmeas.IMAGE and
            feature_name == self.grouping_values):
            # Don't measure the pos/neg controls
            return False
        if (object_name == cpmeas.IMAGE and
            feature_name in [g.measurement.value for g in self.dose_values]):
            return False
        if len(image_numbers) == 0:
            return False
        for image_number in image_numbers:
            v = measurements.get_measurement(object_name,
                                             feature_name,
                                             image_number)
            if v is not None:
                break
        else:
            return False
        if np.isscalar(v):
            return not (isinstance(v, (str, unicode)))
        #
        # Make sure the measurement isn't a string or other oddity
        #
        return np.asanyarray(v).dtype.kind not in "OSU"

    def validate_module_warnings(self, pipeline):
        '''Warn user re: Test mode '''
        if pipeline.test_mode:
            raise cps.ValidationError(
                "CalculateStatistics will not produce any output in test mode",
                self.grouping_values)

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):

        PC_DEFAULT     = "Default output folder"
        PC_WITH_IMAGE  = "Same folder as image"

        if from_matlab and variable_revision_number == 3:
            data_name = setting_values[0]
            logarithmic = setting_values[1]
            figure_name = setting_values[2]
            wants_save_figure = (cps.NO if figure_name == cps.DO_NOT_USE
                                 else cps.YES)
            setting_values = [data_name,
                              data_name,
                              logarithmic,
                              wants_save_figure,
                              figure_name,
                              PC_DEFAULT,
                              cps.DO_NOT_USE]
            variable_revision_number = 1
            from_matlab = False
        if variable_revision_number == 1 and not from_matlab:
            #
            # Minor change: Default output directory -> folder
            #
            new_setting_values = [setting_values[0]]
            for offset in range(1, len(setting_values), 6):
                dir_choice = setting_values[offset+4]
                custom_path = setting_values[offset+5]
                if dir_choice == PC_CUSTOM:
                    if custom_path[0] == '.':
                        dir_choice = cps.DEFAULT_OUTPUT_SUBFOLDER_NAME
                    elif custom_path[0] == '&':
                        dir_choice = cps.DEFAULT_OUTPUT_SUBFOLDER_NAME
                        custom_path = "."+custom_path[1:]
                    else:
                        dir_choice = cps.ABSOLUTE_FOLDER_NAME
                directory = cps.DirectoryPath.static_join_string(
                    dir_choice, custom_path)
                new_setting_values += setting_values[offset:(offset+4)]
                new_setting_values += [directory]
            setting_values = new_setting_values
            variable_revision_number = 2

        # Standardize input/output directory name references
        setting_values = list(setting_values)
        for offset in range(5, len(setting_values), VARIABLE_SETTING_COUNT):
            setting_values[offset] = cps.DirectoryPath.upgrade_setting(
                setting_values[offset])

        return setting_values, variable_revision_number, from_matlab

########################################################
#
# The following code is adapted from Matlab code donated by Ilya Ravkin
#
# http://www.ravkin.net
########################################################
def z_factors(xcol, ymatr):
    '''xcol is (Nobservations,1) column vector of grouping values
           (in terms of dose curve it may be Dose).
       ymatr is (Nobservations, Nmeasures) matrix, where rows correspond to
           observations and columns corresponds to different measures.

       returns v, z, z_one_tailed, OrderedUniqueDoses, OrderedAverageValues
       z and z_bwtn_mean are (1, Nmeasures) row vectors containing Z'- and
       between-mean Z'-factors for the corresponding measures.

       When ranges are zero, we set the Z' factors to a very negative
       value.'''

    xs, avers, stds = loc_shrink_mean_std(xcol, ymatr)
    # Z' factor is defined by the positive and negative controls, so we take the
    # extremes BY DOSE of the averages and stdevs.
    zrange = np.abs(avers[0, :] - avers[-1, :])
    zstd = stds[0, :] + stds[-1, :]
    zstd[zrange == 0] = 1;
    zrange[zrange == 0] = 0.000001;
    z = 1 - 3 * (zstd / zrange)

    # The one-tailed Z' factor is defined by using only the samples between the
    # means, again defined by DOSE extremes
    zrange = np.abs(avers[0, :] - avers[-1, :]);
    exp1_vals = ymatr[xcol == xs[0],:]
    exp2_vals = ymatr[xcol == xs[-1],:]
    #
    # Sort the average positive control values and negative control values
    # so that the lowest is in index 0 and the highest is in index 1 independent
    # of whether the control is negative or positive
    #
    sort_avers = np.sort(np.array((avers[0,:],avers[-1,:])),0)

    for i in range(sort_avers.shape[1]):
        # Here the std must be calculated using the full formula
        exp1_cvals = exp1_vals[:,i]
        exp2_cvals = exp2_vals[:,i]
        vals1 = exp1_cvals[(exp1_cvals >= sort_avers[0,i]) &
                           (exp1_cvals <= sort_avers[1,i])]
        vals2 = exp2_cvals[(exp2_cvals >= sort_avers[0,i]) &
                           (exp2_cvals <= sort_avers[1,i])]
        stds[0,i] = np.sqrt(np.sum((vals1 - sort_avers[0,i])**2) / len(vals1))
        stds[1,i] = np.sqrt(np.sum((vals2 - sort_avers[1,i])**2) / len(vals2))

    zstd = stds[0, :] + stds[1, :]

    # If means aren't the same and stdev aren't NaN, calculate the value
    z_one_tailed = 1 - 3 * (zstd / zrange)
    # Otherwise, set it to a really negative value
    z_one_tailed[(~ np.isfinite(zstd)) | (zrange == 0)] = -1e5
    return (z, z_one_tailed, xs, avers)

def v_factors(xcol, ymatr):
    '''xcol is (Nobservations,1) column vector of grouping values
           (in terms of dose curve it may be Dose).
       ymatr is (Nobservations, Nmeasures) matrix, where rows correspond to
           observations and columns corresponds to different measures.

        Calculate the V factor = 1-6 * mean standard deviation / range
    '''
    xs, avers, stds = loc_shrink_mean_std(xcol, ymatr)
    #
    # Range of averages per label
    #
    vrange = np.max(avers,0) - np.min(avers,0)
    #
    # Special handling for labels that have no ranges
    #
    vstd = np.zeros(len(vrange))
    vstd[vrange == 0] = 1;
    vstd[vrange != 0] = np.mean(stds[:,vrange !=0], 0)
    vrange[vrange == 0] = 0.000001;
    v = 1 - 6 * (vstd / vrange)
    return v

def loc_shrink_mean_std(xcol, ymatr):
    '''Compute mean and standard deviation per label

    xcol - column of image labels or doses
    ymatr - a matrix with rows of values per image and columns
            representing different measurements

    returns xs - a vector of unique doses
            avers - the average value per label
            stds - the standard deviation per label
    '''
    ncols = ymatr.shape[1]
    labels, labnum, xs = loc_vector_labels(xcol)
    avers = np.zeros((labnum, ncols))
    stds  = avers.copy()
    for ilab in range(labnum):
        labinds = (labels == ilab)
        labmatr = ymatr[labinds,:]
        if labmatr.shape[0] == 1:
            avers[ilab,:] = labmatr[0,:]
        else:
            avers[ilab, :] = np.mean(labmatr,0)
            stds[ilab, :] = np.std(labmatr, 0)
    return xs, avers, stds

def loc_vector_labels(x):
    '''Identify unique labels from the vector of image labels

    x - a vector of one label or dose per image

    returns labels, labnum, uniqsortvals
    labels - a vector giving an ordinal per image where that ordinal
             is an index into the vector of unique labels (uniqsortvals)
    labnum - # of unique labels in x
    uniqsortvals - a vector containing the unique labels in x
    '''
    #
    # Get the index of each image's label in the sorted array
    #
    order = np.lexsort((x,))
    reverse_order = np.lexsort((order,))
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
    first_occurrence = np.ones(len(x), bool)
    first_occurrence[1:] = sorted_x[:-1] != sorted_x[1:]
    sorted_labels = np.cumsum(first_occurrence)-1
    labels = sorted_labels[reverse_order]
    uniqsortvals = sorted_x[first_occurrence]
    return (labels, len(uniqsortvals), uniqsortvals)

#######################################################
#
# The following code computes the EC50 dose response
#
#######################################################
def calculate_ec50(conc, responses, Logarithmic):
    '''EC50 Function to fit a dose-response data to a 4 parameter dose-response
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
       '''
    # If we are using a log-domain set of doses, we have a better chance of
    # fitting a sigmoid to the curve if the concentrations are
    # log-transformed.
    if Logarithmic:
        conc = np.log(conc)

    n=responses.shape[1]
    results=np.zeros((n,4))

    def error_fn(v, x, y):
        '''Least-squares error function

        This measures the least-squares error of fitting the sigmoid
        with parameters in v to the x and y data.
        '''
        return np.sum((sigmoid(v,x) - y)**2)

    for i in range(n):
        response=responses[:,i]
        v0 = calc_init_params(conc,response)
        v = scipy.optimize.fmin(error_fn, v0, args=(conc, response),
                                maxiter=1000, maxfun = 1000,
                                disp=False)
        results[i,:] = v
    return results

def sigmoid(v, x):
        '''This is the EC50 sigmoid function

        v is a vector of parameters:
            v[0] = minimum allowed value
            v[1] = maximum allowed value
            v[2] = ec50
            v[3] = Hill coefficient
        '''
        p_min, p_max, ec50, hill = v
        return p_min + ((p_max - p_min) /
                        (1+(x/ec50)**hill))

def calc_init_params(x,y):
    '''This generates the min, max, x value at the mid-y value, and Hill
      coefficient. These values are starting points for the sigmoid fitting.

      x & y are the points to be fit
      returns minimum, maximum, ec50 and hill coefficient starting points
      '''
    min_0 = min(y)
    max_0 = max(y)

    # Parameter 3
    # OLD:  parms(3)=(min(x)+max(x))/2;
    # This is an estimate of the EC50, i.e. the half maximal effective
    # concentration (here denoted as x-value)
    #
    # Note: this was originally simply mean([max(x); min(x)]).  This does not
    # take into account the y-values though, so it was changed to be the
    # x-value that corresponded to the y-value closest to the mean([max(y); min(y)]).
    # Unfortunately, for x-values with only two categories e.g. [0 1], this results in
    # an initial EC50 of either 0 or 1 (min(x) or max(x)), which seems a bad estimate.
    # 5 We will take a two-pronged approach: Use the estimate from this latter approach,
    # unless the parameter will equal either the max(x) or min(x).  In this case, we will use the
    # former approach, namely (mean([max(x); min(x)]).  DL 2007.09.24
    YvalueAt50thPercentile = (min(y)+max(y))/2
    DistanceToCentralYValue = np.abs(y - YvalueAt50thPercentile)
    LocationOfNearest = np.argmin(DistanceToCentralYValue)
    XvalueAt50thPercentile = x[LocationOfNearest]
    if XvalueAt50thPercentile == min(x) or XvalueAt50thPercentile == max(x):
        ec50 = (min(x)+max(x))/2
    else:
        ec50 = XvalueAt50thPercentile

    # Parameter 4
    # The OLD way used 'size' oddly - perhaps meant 'length'?  It would cause
    # divide-by-zero warnings since 'x(2)-x(sizex)' would necessarily have
    # zeros.
    # The NEW way just checks to see whether the depenmdent var is increasing (note
    # negative hillc) or descreasing (positive hillc) and sets them initally
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
    min_idx = np.argmin(x)
    max_idx = np.argmax(x)
    x0 = x[min_idx]
    x1 = x[max_idx]
    y0 = y[min_idx]
    y1 = y[max_idx]

    if x0 == x1:
        # If all of the doses are the same, why are we doing this?
        # There's not much point in fitting.
        raise ValueError("All doses or labels for all image sets are %s. Can't calculate dose/response curves."%x0)
    elif y1 > y0:
        hillc = -1
    else:
        hillc = 1
    return (min_0, max_0, ec50, hillc)

def write_figures(prefix, directory, dose_name,
                  dose_data, data, ec50_coeffs,
                  feature_set, log_transform):
    '''Write out figure scripts for each measurement

    prefix - prefix for file names
    directory - write files into this directory
    dose_name - name of the dose measurement
    dose_data - doses per image
    data - data per image
    ec50_coeffs - coefficients calculated by calculate_ec50
    feature_set - tuples of object name and feature name in same order as data
    log_transform - true to log-transform the dose data
    '''
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_pdf import FigureCanvasPdf

    if log_transform:
        dose_data = np.log(dose_data)
    for i, (object_name, feature_name) in enumerate(feature_set):
        fdata = data[:,i]
        fcoeffs = ec50_coeffs[i,:]
        filename = "%s%s_%s.pdf"%(prefix, object_name, feature_name)
        pathname = os.path.join(directory, filename)
        f = Figure()
        canvas = FigureCanvasPdf(f)
        ax = f.add_subplot(1,1,1)
        x = np.linspace(0, np.max(dose_data), num=100)
        y = sigmoid(fcoeffs, x)
        ax.plot(x, y)
        dose_y = sigmoid(fcoeffs, dose_data)
        ax.plot(dose_data, dose_y, "o")
        ax.set_xlabel('Dose')
        ax.set_ylabel('Response')
        ax.set_title('%s_%s'%(object_name, feature_name))
        f.savefig(pathname)
