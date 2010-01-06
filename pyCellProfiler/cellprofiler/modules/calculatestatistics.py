'''<b>CalculateStatistics</b> calculates measures of assay quality 
(V and Z' factors) and dose response data (EC50) for all measured features
made from images.
<hr>
The V and Z' factors are statistical measures of assay quality and are
calculated for each per-image measurement and for each average per-object 
measurement that you have made
in the pipeline. For example, the Z' factor indicates how well-separated
the positive and negative controls are. Calculating these values by
placing this module at the end of a pipeline allows you to identify which
measured features are most powerful for distinguishing positive and
negative control samples, or for accurately quantifying the assay's
response to dose. Both Z' and V factors will be calculated for all
measured values (Intensity, AreaShape, Texture, etc.). These measurements
can be exported as the "Experiment" set of data.
<p>
For both Z' and V factors, the highest possible value (best assay
quality) is 1 and they can range into negative values (for assays where
distinguishing between positive and negative controls is difficult or
impossible). A Z' factor > 0 is potentially screenable; a Z' factor > 0.5
is considered an excellent assay.
<p>
The Z' factor is based only on positive and negative controls. The V
factor is based on an entire dose-response curve rather than on the
minimum and maximum responses. When there are only two doses in the assay
(positive and negative controls only), the V factor will equal the Z'
factor.
<p>
The one-tailed Z' factor is an attempt to overcome a limitation of the original 
Z'-factor formulation (it assumes a gaussian distribution) and is 
informative for populations with moderate or high amounts
of skewness. In these cases, long tails opposite to the mid-range point
lead to a high standard deviation for either population, which results 
in a low Z' factor even though the population means and samples between
the means may be well-separated. Therefore, the one-tailed Z' factor is 
calculated with the same formula but using only those samples that lie 
between the positive/negative population means. This is not yet a well-
established measure of assay robustness.
<p>
NOTE: If the standard deviation of a measured feature is zero for a
particular set of samples (e.g. all the positive controls), the Z' and V
factors will equal 1 despite the fact that the assay quality is poor. 
This can occur when there is only one sample at each dose.
This also occurs for some non-informative measured features, like the
number of cytoplasm compartments per cell, which is always equal to 1.
<p>
This module can create Matlab scripts that display the EC50 curves for
each measurement. These scripts will require Matlab and the statistics
toolbox in order to run. See <a href='#wants_save_figure'>
Do you want to create dose/response plots?</a>
<p>
The reference for Z' factor is: JH Zhang, TD Chung, et al. (1999) "A
simple statistical parameter for use in evaluation and validation of high
throughput screening assays." J Biomolecular Screening 4(2): 67-73.
<p>
The reference for V factor is: I Ravkin (2004): Poster #P12024 - Quality
Measures for Imaging-based Cellular Assays. Society for Biomolecular
Screening Annual Meeting Abstracts. This is likely to be published.
<p>
Code for the calculation of Z' and V factors was kindly donated by Ilya
Ravkin: http://www.ravkin.net. Carlos Evangelista donated his copyrighted 
dose-response-related code.
<p>
Features measured:
Note: whereas most CellProfiler measurements are calculated for each object (per-object) or for each image (per-image), the Calculate Statistics module produces per-experiment values; for example, one Z' factor is calculated for each measurement, across the entire analysis run.
<ul>
<li>Zfactor</li>
<li>Vfactor</li>
<li>EC50</li>
<li>One-tailed Zfactor</li>
</ul>
<p>
Example format for a file to be loaded by <b>LoadText</b> for this module:
LoadText loads information from a CSV file. The first line of this file is a 
header that names the items.
Each subsequent line represents data for one image set, so your file should have the header line plus one line per image to be processed. You can also make a file for LoadText to load that contains the positive/negative control and dose designations *plus* the image file names to be processed, which is a good way to guarantee that images are matched
with the correct data. Here is an example file:<br>
<code>
<table>
<tr><td>Image_FileName_CY3,</td><td>Image_PathName_CY3,</td><td>Control,</td><td>Dose</td></tr>
<tr><td>"Plate1_A01.tif",</td><td>"/images",</td><td>-1,</td><td>0</td></tr>
<tr><td>"Plate1_A02.tif",</td><td>"/images",</td><td>1,</td><td>1E10</td></tr>
<tr><td>"Plate1_A03.tif",</td><td>"/images",</td><td>0,</td><td>3E4</td></tr>
<tr><td>"Plate1_A04.tif",</td><td>"/images",</td><td>0,</td><td>5E5</td></tr>
</table>
</code>
<br>
'''
__version__="$Revision$"

import numpy as np
import scipy.optimize
import os

import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.settings as cps
import cellprofiler.preferences as cpprefs

'''# of settings aside from the dose measurements'''
FIXED_SETTING_COUNT = 1
VARIABLE_SETTING_COUNT = 6

PC_DEFAULT     = "Default output folder"
PC_WITH_IMAGE  = "Same folder as image"
PC_CUSTOM      = "Custom"

class CalculateStatistics(cpm.CPModule):
    module_name = "CalculateStatistics"
    category = "Measurement"
    variable_revision_number = 1
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
            "Where is information about the positive and negative control status of each image?",
            lambda : cpmeas.IMAGE,
            doc = '''The Z' factor, a measure of assay quality, is calculated by this module based on measurements from images that are specified as positive controls and images that are specified as negative controls (images that are neither are ignored when calculating this statistic). The module uses the convention that all of the negative controls are specified by a minimum value, all of the
positive controls are specified by a maximum value, and all other images have an intermediate value - this might allow you to use your dosing information to also specify the positive and negative controls. If you are not using actual dose data to designate your controls, a common way to designate them is: -1 is a negative control, 0 is an experimental sample, and 1 is a positive control.  In other words, positive
            controls should all be specified by a single high value (for instance, 1)
            and negative controls should all be specified by a single low value (for
            instance, 0). Other samples should have an intermediate value
            to exclude them from the Z' factor analysis.<p>
            The typical way to provide this information in the pipeline is to create 
            a text file outside of CellProfiler and then load that file in the pipeline
            using <b>LoadText</b>. In that case, choose the
            measurement that matches the column header of the measurement
            in LoadText's input file. See the help for this module for an example text file.''')
        self.dose_values = []
        self.add_dose_value()
        self.add_dose_button = cps.DoSomething("","Add another dose specification",
                                               self.add_dose_value)
        
    def add_dose_value(self):
        '''Add a dose value measurement to the list'''
        group = cps.SettingsGroup()
        group.append("measurement",
                     cps.Measurement("Where is information about the treatment dose for each image?",
                                     lambda : cpmeas.IMAGE,
                                     doc = 
            """The V factor, a measure of assay quality, and the EC50, indicating dose/response, are calculated by this module based on each image being specified as a particular treatment dose. Choose a measurement that gives the dose of some treatment
            for each of your images. <p>
            The typical way to provide this information in the pipeline is to create 
            a text file outside of CellProfiler and then load that file in the pipeline
            using <b>LoadText</b>. In that case, choose the
            measurement that matches the column header of the measurement
            in LoadText's input file. See the help for this module for an example text file.
            """))
        group.append("log_transform",cps.Binary(
            "Log-transform dose values?",
            False,
            doc = '''This option allows you to log-transform the dose values 
            before fitting a sigmoid curve. Check
            this box if you have dose-response data. Leave the box unchecked
            if your data values only indicate positive vs negative controls.'''))
        group.append('wants_save_figure', cps.Binary(
            '''Create dose/response plots?''',
            False,
            doc = '''<a name='wants_save_figure'/>Check this box if you want to create and save dose response plots. 
            If you check the box, you will be asked for information on how to save the plots.</a>'''))
        group.append('figure_name', cps.Text(
            "Figure prefix?","",
            doc = '''CellProfiler will create a file name by appending the measurement name
            to the prefix you enter here. For instance, if you have objects
            named, "Cells", the AreaShape_Area measurement, and a prefix of "Dose_",
            CellProfiler will save the figure as "Dose_Cells_AreaShape_Area.m".
            Leave this setting blank if you do not want a prefix.'''
        ))
        group.append('pathname_choice', cps.Choice(
            "File output location",
            [PC_DEFAULT, PC_CUSTOM],
            doc="""
            This setting lets you control the folder used to store the file. The
            choices are:
            <ul>
            <li><i>Default output folder</i></li>
            <li><i>Custom:</i> The file will be stored in a customizable folder. You can
            prefix the folder name with "." (a period) to make the root folder the default
            output folder or "&" (an ampersand) to make the root folder the default image
            folder.</li></ul>"""))
        group.append('pathname', cps.Text(
            "Folder pathname:",
            ".",doc="""
                Enter the pathname to save the images here. The pathname can be referenced with respect 
                to the default output folder specified in the main CellProfiler window with a period (".") or the default input 
                folder with an ampersand ("&") as the root folder."""))
            
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
                         value.pathname_choice, value.pathname]
                        for value in self.dose_values]))
    
    def visible_settings(self):
        """The settings that are visible in the UI
        """
        result = [self.grouping_values]
        for dose_value in self.dose_values:
            result += [dose_value.measurement, dose_value.log_transform,
                       dose_value.wants_save_figure]
            if dose_value.wants_save_figure:
                result += [dose_value.figure_name, dose_value.pathname_choice]
                if dose_value.pathname_choice == PC_CUSTOM:
                    result += [dose_value.pathname]
            result += [dose_value.remover]
        result += [self.add_dose_button]
        return result
    
    def prepare_settings(self, setting_values):
        """Do any sort of adjustment to the settings required for the given values
        
        setting_values - the values for the settings

        This method allows a module to specialize itself according to
        the number of settings and their value. For instance, a module that
        takes a variable number of images or objects can increase or decrease
        the number of relevant settings so they map correctly to the values.
        
        See cellprofiler.modules.measureobjectareashape for an example.
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
        
    def post_run(self, workspace):
        """Do post-processing after the run completes
        
        workspace - the workspace at the end of the run
        """
        measurements = workspace.measurements
        assert isinstance(measurements, cpmeas.Measurements)
        all_objects = [x for x in measurements.get_object_names()
                       if x not in [cpmeas.EXPERIMENT, cpmeas.NEIGHBORS]]
        feature_set = []
        for object_name in all_objects:
            all_features = [x for x in measurements.get_feature_names(object_name)
                            if self.include_feature(measurements, object_name, x)]
            feature_set += [(object_name, feature_name) 
                            for feature_name in all_features]
        grouping_data = np.array(measurements.get_all_measurements(
            cpmeas.IMAGE,self.grouping_values.value))
        data = np.zeros((len(grouping_data), len(feature_set)))
        for i, (object_name, feature_name) in enumerate(feature_set):
            fdata = measurements.get_all_measurements(object_name, feature_name)
            fdata = np.array([np.mean(e) for e in fdata])
            data[:,i] = fdata
    
        v, z, z_one_tailed, OrderedUniqueDoses, OrderedAverageValues = \
             vz_factors(grouping_data, data)
        expt_measurements = {
            "Zfactor": z,
            "Vfactor": v,
            "OneTailedZfactor":z_one_tailed
            }
        for dose_group in self.dose_values:
            dose_feature = dose_group.measurement.value
            dose_data = measurements.get_all_measurements(cpmeas.IMAGE,
                                                          dose_feature)
            dose_data = np.array(dose_data)
            ec50_coeffs = calculate_ec50(dose_data, data, 
                                         dose_group.log_transform.value)
            if len(self.dose_values) == 1:
                name = "EC50"
            else:
                name = "EC50_"+dose_feature
            expt_measurements[name] = ec50_coeffs[:,2]
            if dose_group.wants_save_figure:
                if dose_group.pathname_choice == PC_DEFAULT:
                    pathname = cpprefs.get_default_output_directory()
                elif dose_group.pathname_choice == PC_CUSTOM:
                    pathname = cpprefs.get_absolute_path(dose_group.pathname.value,
                                                         cpprefs.ABSPATH_OUTPUT)
                write_figures(dose_group.figure_name, pathname, dose_feature,
                              dose_data, data, ec50_coeffs, feature_set,
                              dose_group.log_transform.value)
                
        for i, (object_name, feature_name) in enumerate(feature_set):
            for statistic, value in expt_measurements.iteritems():
                sfeature_name = '_'.join((statistic, object_name, feature_name))
                measurements.add_experiment_measurement(sfeature_name, value[i])
        if workspace.frame is not None:
            #
            # Create tables for the top 10 Z and V
            #
            figure = workspace.create_or_find_figure(subplots=(2,1))
            for ii, key in enumerate(("Zfactor","Vfactor")):
                a = expt_measurements[key]
                indexes = np.lexsort((-a,))
                stats = [["Object","Feature",key]]
                stats += [[feature_set[i][0], feature_set[i][1], a[i]]
                           for i in indexes[:10]]
                figure.subplot_table(ii,0, stats, (.3,.5,.2))
                figure.set_subplot_title("Top 10 by %s"%key, ii,0)
                
    def include_feature(self, measurements, object_name, feature_name):
        '''Return true if we should analyze a feature'''
        if feature_name.find("Location") != -1:
            return False
        if feature_name.find("ModuleError") != -1:
            return False
        all_measurements = measurements.get_all_measurements(object_name, 
                                                             feature_name)
        if len(all_measurements) == 0:
            return False
        if np.isscalar(all_measurements[0]):
            return not (isinstance(all_measurements[0], str),
                        isinstance(all_measurements[0], unicode))
        #
        # Make sure the measurement isn't a string or other oddity
        #
        return all_measurements[0].dtype.kind not in "OSU"
        
    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
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
            for choice in (PC_DEFAULT, PC_WITH_IMAGE):
                if setting_values[5].startswith(choice[:4]):
                    setting_values = (setting_values[:5] + [choice] +
                                      setting_values[6:])
        return setting_values, variable_revision_number, from_matlab                              

########################################################
#
# The following code is adapted from Matlab code donated by Ilya Ravkin
#
# http://www.ravkin.net
########################################################        
def vz_factors(xcol, ymatr):
    '''xcol is (Nobservations,1) column vector of grouping values
           (in terms of dose curve it may be Dose).
       ymatr is (Nobservations, Nmeasures) matrix, where rows correspond to 
           observations and columns corresponds to different measures.
       
       returns v, z, z_one_tailed, OrderedUniqueDoses, OrderedAverageValues
       v, z and z_bwtn_mean are (1, Nmeasures) row vectors containing V-, Z'- and 
       between-mean Z'-factors for the corresponding measures.

       When ranges are zero, we set the V and Z' factors to a very negative
       value.'''

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
    return (v, z, z_one_tailed, xs, avers)

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
    if log_transform:
        dose_data = np.log(dose_data)
    for i, (object_name, feature_name) in enumerate(feature_set):
        fdata = data[:,i]
        fcoeffs = ec50_coeffs[i,:]
        filename = "%s%s_%s.m"%(prefix, object_name, feature_name)
        pathname = os.path.join(directory, filename)
        fd = open(pathname,'w')
        fd.write('%%%%%% EC50 dose/response for %s vs %s %s\n'%
                 (dose_name, object_name, feature_name))
        fd.write('dose=[%s];\n'%
                 ','.join([str(x) for x in dose_data]))
        fd.write('response=[%s];\n'%
                 ','.join([str(x) for x in fdata]))
        fd.write('coeffs=[%s];\n'%
                 ','.join([str(x) for x in fcoeffs]))
        fd.write('%% A lambda function that computes the sigmoid\n')
        fd.write('sigmoid_fn=@(v,x) v(1)+(v(2)-v(1))./(1+(x(:,1)/v(3)).^v(4))\n')
        fd.write("nlintool(dose, response, sigmoid_fn, coeffs, .05,'%s','%s_%s');\n" %
                 (dose_name, object_name, feature_name))
        fd.close()
        
