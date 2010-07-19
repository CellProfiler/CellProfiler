'''<b>Measure Object Size Shape </b> measures several area and shape features of identified objects
<hr>

Given an image with identified objects (e.g. nuclei or cells), this
module extracts area and shape features of each one. Note that these
features are only reliable for objects that are completely inside the
image borders, so you may wish to exclude objects touching the edge of
the image using <b>IdentifyPrimaryObjects</b>.

<h4>Available measurements</h4>
<ul>
<li><i>Area:</i> The actual number of pixels in the region.</li>

<li><i>Perimeter:</i> The total number of pixels around the boundary of each
region in the image.</li>

<li><i>FormFactor:</i> Calculated as 4*&pi;*Area/Perimeter<sup>2</sup>. Equals 1 for a 
perfectly circular object.</li>

<li><i>Eccentricity:</i> The eccentricity of the ellipse that has the
same second-moments as the region. The eccentricity is the ratio of the
distance between the foci of the ellipse and its major axis length. The
value is between 0 and 1. (0 and 1 are degenerate cases; an ellipse whose
eccentricity is 0 is actually a circle, while an ellipse whose eccentricity
is 1 is a line segment.) This property is supported only for 2D input
label matrices.</li>

<li><i>Solidity:</i> The proportion of the pixels in the convex hull that
are also in the region. Also known as <i>convexity</i>. Computed as Area/ConvexArea.</li>

<li><i>Extent:</i> The proportion of the pixels in the bounding box that
are also in the region. Computed as the Area divided by the area of the
bounding box.</li>

<li><i>EulerNumber:</i> The number of objects in the region
minus the number of holes in those objects, assuming 8-connectivity.</li>

<li><i>MajorAxisLength:</li> The length (in pixels) of the major axis of
the ellipse that has the same normalized second central moments as the
region.</li>

<li><i>MinorAxisLength:</i> The length (in pixels) of the minor axis of
the ellipse that has the same normalized second central moments as the
region.</li>

<li><i>Orientation:</i> The angle (in degrees ranging from -90 to 90
degrees) between the x-axis and the major axis of the ellipse that has the
same second-moments as the region.</li>

<li><i>Zernike shape features:</i> Measure shape by describing a binary object (or
more precisely, a patch with background and an object in the center) in a
basis of Zernike polynomials, using the coefficients as features <i>(Boland
et al., 1998</i>. Currently, Zernike polynomials from order 0 to order 9 are
calculated, giving in total 30 measurements. While there is no limit to
the order which can be calculated (and indeed users could add more by
adjusting the code), the higher order polynomials carry less information.</li>
</ul>

See also <b>MeasureImageAreaOccupied</b>.
'''

# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

__version__="$Revision: 1 $"

import numpy as np
import scipy.ndimage as scind

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpmath.zernike as cpmz
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result
from cellprofiler.cpmath.cpmorphology import ellipse_from_second_moments
from cellprofiler.cpmath.cpmorphology import calculate_extents
from cellprofiler.cpmath.cpmorphology import calculate_perimeters
from cellprofiler.cpmath.cpmorphology import calculate_solidity
from cellprofiler.cpmath.cpmorphology import euler_number
from cellprofiler.measurements import COLTYPE_FLOAT

"""The category of the per-object measurements made by this module"""
AREA_SHAPE = 'AreaShape'

"""Calculate Zernike features for N,M where N=0 through ZERNIKE_N"""
ZERNIKE_N = 9

F_AREA = "Area"
F_ECCENTRICITY = 'Eccentricity'
F_SOLIDITY = 'Solidity'
F_EXTENT = 'Extent'
F_EULER_NUMBER = 'EulerNumber'
F_PERIMETER = 'Perimeter'
F_FORM_FACTOR = 'FormFactor'
F_MAJOR_AXIS_LENGTH = 'MajorAxisLength'
F_MINOR_AXIS_LENGTH = 'MinorAxisLength'
F_ORIENTATION = 'Orientation'
"""The non-Zernike features"""
F_STANDARD = [ F_AREA, F_ECCENTRICITY, F_SOLIDITY, F_EXTENT,
               F_EULER_NUMBER, F_PERIMETER, F_FORM_FACTOR,
               F_MAJOR_AXIS_LENGTH, F_MINOR_AXIS_LENGTH,
               F_ORIENTATION ]
class MeasureObjectSizeShape(cpm.CPModule):

    module_name = "MeasureObjectSizeShape"
    variable_revision_number = 1
    category = 'Measurement'
    
    def create_settings(self):
        """Create the settings for the module at startup and set the module name
        
        The module allows for an unlimited number of measured objects, each
        of which has an entry in self.object_groups.
        """ 
        self.object_groups = []
        self.add_object(can_remove = False)
        self.spacer = cps.Divider(line = True)
        self.add_objects = cps.DoSomething("", "Add another object",self.add_object)
        self.calculate_zernikes = cps.Binary('Calculate the Zernike features?',True, doc="""
                                            Check this box to calculate the Zernike shape features. Since the
                                            first 10 Zernike polynomials (from order 0 to order 9) are
                                            calculated, this operation can be time consuming if the image
                                            contains a lot of objects.""")
    
    def add_object(self, can_remove = True):
        """Add a slot for another object"""
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=False))
        
        group.append("name", cps.ObjectNameSubscriber("Select objects to measure","None",doc="""
                                                What did you call the objects you want to measure?"""))
        if can_remove:
            group.append("remove", cps.RemoveSettingButton("", "Remove this object", self.object_groups, group))
        
        self.object_groups.append(group)
        
    def settings(self):
        """The settings as they appear in the save file"""
        result = [og.name for og in self.object_groups]
        result.append(self.calculate_zernikes)
        return result
    
    def prepare_settings(self,setting_values):
        """Adjust the number of object groups based on the number of setting_values"""
        object_group_count = len(setting_values)-1
        while len(self.object_groups) > object_group_count:
            self.remove_object(object_group_count)
        
        while len(self.object_groups) < object_group_count:
            self.add_object()
        
    def visible_settings(self):
        """The settings as they appear in the module viewer"""
        result = []
        for og in self.object_groups:
            result += og.visible_settings()
        result.extend([self.add_objects, self.spacer, self.calculate_zernikes])
        return result
    
    def get_categories(self,pipeline, object_name):
        """Get the categories of measurements supplied for the given object name
        
        pipeline - pipeline being run
        object_name - name of labels in question (or 'Images')
        returns a list of category names
        """
        if object_name in [og.name for og in self.object_groups]:
            return [AREA_SHAPE]
        else:
            return []

    def get_zernike_numbers(self):
        """The Zernike numbers measured by this module"""
        if self.calculate_zernikes.value:
            return cpmz.get_zernike_indexes(ZERNIKE_N+1)
        else:
            return []
    
    def get_zernike_name(self,zernike_index):
        """Return the name of a Zernike feature, given a (N,M) 2-tuple
        
        zernike_index - a 2 element sequence organized as N,M
        """
        return "Zernike_%d_%d"%(zernike_index[0],zernike_index[1])
    
    def get_feature_names(self):
        """Return the names of the features measured"""
        result = list(F_STANDARD)
        result.extend([self.get_zernike_name(index) 
                       for index in self.get_zernike_numbers()])
        return result

    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces
        
        object_name - return measurements made on this object 
                      (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        if (category == AREA_SHAPE and
            self.get_categories(pipeline,object_name)):
            return self.get_feature_names()
        return []
    
    def is_interactive(self):
        return False

    def run(self, workspace):
        """Run, computing the area measurements for the objects"""
        
        if workspace.frame is not None:
            workspace.display_data.statistics = \
                     [("Object","Feature","Mean","Median","STD")]
        for object_group in self.object_groups:
            self.run_on_objects(object_group.name.value, workspace)
    
    def run_on_objects(self,object_name, workspace):
        """Run, computing the area measurements for a single map of objects"""
        objects = workspace.get_objects(object_name)
        #
        # Compute the area as the sum of 1s over a label matrix
        #
        self.perform_ndmeasurement(workspace, scind.sum,
                                   object_name, F_AREA)
        #
        # Do the ellipse-related measurements
        #
        centers,eccentricity,major_axis_length,minor_axis_length,theta =\
            objects.fn_of_ones_label_and_index(ellipse_from_second_moments)
        self.record_measurement(workspace, object_name,
                                F_ECCENTRICITY, eccentricity)
        self.record_measurement(workspace, object_name,
                                F_MAJOR_AXIS_LENGTH, major_axis_length)
        self.record_measurement(workspace, object_name, 
                                F_MINOR_AXIS_LENGTH, minor_axis_length)
        self.record_measurement(workspace, object_name, F_ORIENTATION, 
                                theta * 180 / np.pi)
        #
        # The extent (area / bounding box area)
        #
        self.perform_measurement(workspace, calculate_extents,
                                 object_name, F_EXTENT)
        #
        # The perimeter distance
        #
        self.perform_measurement(workspace, calculate_perimeters,
                                 object_name, F_PERIMETER)
        #
        # Solidity
        #
        self.perform_measurement(workspace, calculate_solidity,
                                 object_name, F_SOLIDITY)
        #
        # Form factor
        #
        ff = form_factor(objects)
        self.record_measurement(workspace, object_name, 
                                F_FORM_FACTOR, ff)
        #
        # Euler number
        self.perform_measurement(workspace, euler_number,
                                 object_name, F_EULER_NUMBER)
        #
        # Zernike features
        #
        if self.calculate_zernikes.value:
            zernike_numbers = self.get_zernike_numbers()
            if len(objects.indices) > 0:
                zernike_features = cpmz.zernike(zernike_numbers, 
                                                objects.segmented,
                                                objects.indices)
            else:
                zernike_features = np.zeros((0,zernike_numbers.shape[0]))
            for i in range(zernike_numbers.shape[0]):
                zernike_number = zernike_numbers[i]
                zernike_feature = zernike_features[:,i]
                feature_name = self.get_zernike_name(zernike_number)
                self.record_measurement(workspace, object_name, feature_name, 
                                        zernike_feature)
            
    def display(self, workspace):
        figure = workspace.create_or_find_figure(title="MeasureObjectSizeShape, image cycle #%d"%(
                workspace.measurements.image_set_number),subplots=(1,1))
        figure.subplot_table(0,0,workspace.display_data.statistics,
                             ratio=(.25,.45,.1,.1,.1))
        
    def perform_measurement(self, workspace, function,
                            object_name, feature_name):
        """Perform a measurement on a label matrix
        
        workspace   - the workspace for the run
        function    - a function with the following sort of signature:
                      image - an image to be fed into the function which for
                              our case is all ones
                      labels - the label matrix from the objects
                      index  - a sequence of label indexes to pay attention to
        object_name - name of object to retrieve from workspace and deposit
                      in measurements
        feature_name- name of feature to deposit in measurements
        """
        objects = workspace.get_objects(object_name)
        if len(objects.indices) > 0:
            data = objects.fn_of_label_and_index(function) 
        else:
            data = np.zeros((0,))
        self.record_measurement(workspace, object_name, feature_name, data)
        
    def perform_ndmeasurement(self, workspace, function, 
                              object_name, feature_name ):
        """Perform a scipy.ndimage-style measurement on a label matrix
        
        workspace   - the workspace for the run
        function    - a function with the following sort of signature:
                      image - an image to be fed into the function which for
                              our case is all ones
                      labels - the label matrix from the objects
                      index  - a sequence of label indexes to pay attention to
        object_name - name of object to retrieve from workspace and deposit
                      in measurements
        feature_name- name of feature to deposit in measurements
        """
        objects = workspace.get_objects(object_name)
        if len(objects.indices) > 0:
            data = objects.fn_of_ones_label_and_index(function)
        else:
            data = np.zeros((0,))
        self.record_measurement(workspace, object_name, feature_name, data)

    def record_measurement(self,workspace,  
                           object_name, feature_name, result):
        """Record the result of a measurement in the workspace's measurements"""
        data = fixup_scipy_ndimage_result(result)
        workspace.add_measurement(object_name, 
                                  "%s_%s"%(AREA_SHAPE,feature_name), 
                                  data)
        if workspace.frame is not None and len(data) > 0:
            workspace.display_data.statistics.append(
                (object_name, feature_name, 
                 "%.2f"%np.mean(data),
                 "%.2f"%np.median(data),
                 "%.2f"%np.std(data)))
        
    def get_measurement_columns(self, pipeline):
        '''Return measurement column definitions. 
        All cols returned as float even though "Area" will only ever be int'''
        object_names = [s.value for s in self.settings()][:-1]
        measurement_names = self.get_feature_names()
        cols = []
        for oname in object_names:
            for mname in measurement_names:
                cols += [(oname, AREA_SHAPE+'_'+mname, COLTYPE_FLOAT)]
        return cols
        
            
        
        
    def upgrade_settings(self,setting_values,variable_revision_number,
                         module_name,from_matlab):
        """Adjust the setting_values for older save file versions
        
        setting_values - a list of strings representing the settings for
                         this module.
        variable_revision_number - the variable revision number of the module
                                   that saved the settings
        module_name - the name of the module that saved the settings
        from_matlab - true if it was a Matlab module that saved the settings
        
        returns the modified settings, revision number and "from_matlab" flag
        """
        if from_matlab and variable_revision_number == 2:
            # Added Zernike question at revision # 2
            setting_values = list(setting_values)
            setting_values.append(cps.NO)
            variable_revision_number = 3
        
        if from_matlab and variable_revision_number == 3:
            # Remove the "Do not use" objects from the list
            setting_values = np.array(setting_values)
            setting_values = list(setting_values[setting_values !=
                                                 cps.DO_NOT_USE])
            variable_revision_number = 1
            from_matlab = False
        return setting_values, variable_revision_number, from_matlab

def form_factor(objects):
    """FormFactor = 4/pi*Area/Perimeter^2, equals 1 for a perfectly circular"""
    if len(objects.indices) > 0:
        areas = fixup_scipy_ndimage_result(
                    objects.fn_of_ones_label_and_index(scind.sum))
        perimeter = objects.fn_of_label_and_index(calculate_perimeters)
        return 4.0*np.pi*areas / perimeter**2
    else:
        return np.zeros((0,))

MeasureObjectAreaShape = MeasureObjectSizeShape
