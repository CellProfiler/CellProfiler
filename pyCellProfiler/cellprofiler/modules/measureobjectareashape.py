"""measureobjectareashape.py - Measure area features for an object

"""
__version__='$Revision: 1 $'

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

"""The category of the per-object measurements made by this module"""
AREA_SHAPE = 'AreaShape'

"""The "name" slot in the object group dictionary entry"""
OG_NAME = 'name'

"""The "remove"slot in the object group dictionary entry"""
OG_REMOVE = 'remove'

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
class MeasureObjectAreaShape(cpm.CPModule):
    """SHORT DESCRIPTION:
Measures several area and shape features of identified objects.
*************************************************************************

Given an image with objects identified (e.g. nuclei or cells), this
module extracts area and shape features of each object. Note that these
features are only reliable for objects that are completely inside the
image borders, so you may wish to exclude objects touching the edge of
the image in Identify modules.

Basic shape features:     Feature Number:

Zernike shape features measure shape by describing a binary object (or
more precisely, a patch with background and an object in the center) in a
basis of Zernike polynomials, using the coefficients as features (Boland
et al., 1998). Currently, Zernike polynomials from order 0 to order 9 are
calculated, giving in total 30 measurements. While there is no limit to
the order which can be calculated (and indeed users could add more by
adjusting the code), the higher order polynomials carry less information.

Details about how measurements are calculated:
This module retrieves objects in label matrix format and measures them.
The label matrix image should be "compacted": that is, each number should
correspond to an object, with no numbers skipped. So, if some objects
were discarded from the label matrix image, the image should be converted
to binary and re-made into a label matrix image before feeding into this
module.

*Area - Computed from the the actual number of pixels in the region.
*Eccentricity - Also known as elongation or elongatedness. For an ellipse
that has the same second-moments as the object, the eccentricity is the
ratio of the between-foci distance and the major axis length. The value
is between 0 (a circle) and 1 (a line segment).
*Solidity - Also known as convexity. The proportion of the pixels in the
convex hull that are also in the object. Computed as Area/ConvexArea.
*Extent - The proportion of the pixels in the bounding box that are also
in the region. Computed as the Area divided by the area of the bounding box.
*EulerNumber - Equal to the number of objects in the image minus the
number of holes in those objects. For modules built to date, the number
of objects in the image is always 1.
*MajorAxisLength - The length (in pixels) of the major axis of the
ellipse that has the same normalized second central moments as the
region.
*MinorAxisLength - The length (in pixels) of the minor axis of the
ellipse that has the same normalized second central moments as the
region.
*Perimeter - the total number of pixels around the boundary of each
region in the image.

In addition, the following feature is calculated:

FormFactor = 4*pi*Area/Perimeter^2, equals 1 for a perfectly circular
object

HERE IS MORE DETAILED INFORMATION ABOUT THE MEASUREMENTS FOR YOUR
REFERENCE

'Area' ? Scalar; the actual number of pixels in the region. (This value
might differ slightly from the value returned by bwarea, which weights
different patterns of pixels differently.)

'Eccentricity' ? Scalar; the eccentricity of the ellipse that has the
same second-moments as the region. The eccentricity is the ratio of the
distance between the foci of the ellipse and its major axis length. The
value is between 0 and 1. (0 and 1 are degenerate cases; an ellipse whose
eccentricity is 0 is actually a circle, while an ellipse whose eccentricity
is 1 is a line segment.) This property is supported only for 2-D input
label matrices.

'Solidity' -? Scalar; the proportion of the pixels in the convex hull that
are also in the region. Computed as Area/ConvexArea. This property is
supported only for 2-D input label matrices.

'Extent' ? Scalar; the proportion of the pixels in the bounding box that
are also in the region. Computed as the Area divided by the area of the
bounding box. This property is supported only for 2-D input label matrices.

'EulerNumber' ? Scalar; equal to the number of objects in the region
minus the number of holes in those objects. This property is supported
only for 2-D input label matrices. regionprops uses 8-connectivity to
compute the EulerNumber measurement. To learn more about connectivity,
see Pixel Connectivity.

'perimeter' ? p-element vector containing the distance around the boundary
of each contiguous region in the image, where p is the number of regions.
regionprops computes the perimeter by calculating the distance between
each adjoining pair of pixels around the border of the region. If the
image contains discontiguous regions, regionprops returns unexpected
results. The following figure shows the pixels included in the perimeter
calculation for this object

'MajorAxisLength' ? Scalar; the length (in pixels) of the major axis of
the ellipse that has the same normalized second central moments as the
region. This property is supported only for 2-D input label matrices.

'MinorAxisLength' ? Scalar; the length (in pixels) of the minor axis of
the ellipse that has the same normalized second central moments as the
region. This property is supported only for 2-D input label matrices.

'Orientation' ? Scalar; the angle (in degrees ranging from -90 to 90
degrees) between the x-axis and the major axis of the ellipse that has the
same second-moments as the region. This property is supported only for
2-D input label matrices.

See also MeasureImageAreaOccupied.
"""

    variable_revision_number = 1
    category = 'Measurement'
    
    def create_settings(self):
        """Create the settings for the module at startup and set the module name
        
        The module allows for an unlimited number of measured objects, each
        of which has an entry in self.object_groups.
        """ 
        self.module_name = "MeasureObjectAreaShape"
        self.object_groups = []
        self.add_object_cb()
        self.add_objects = cps.DoSomething("Add another object","Add",self.add_object_cb)
        self.calculate_zernikes = cps.Binary("Would you like to calculate the Zernike features for each object  (with lots of objects, this can be very slow)?",True)
    
    def settings(self):
        """The settings as they appear in the save file"""
        result = [og[OG_NAME] for og in self.object_groups]
        result.append(self.calculate_zernikes)
        return result
    
    def backwards_compatibilize(self,setting_values,variable_revision_number,
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

    def prepare_to_set_values(self,setting_values):
        """Adjust the number of object groups based on the number of setting_values"""
        object_group_count = len(setting_values)-1
        while len(self.object_groups) > object_group_count:
            self.remove_object_cb(object_group_count)
        
        while len(self.object_groups) < object_group_count:
            self.add_object_cb()
        
    def visible_settings(self):
        """The settings as they appear in the module viewer"""
        result = []
        for og in self.object_groups:
            result.extend([og[OG_NAME],og[OG_REMOVE]])
        result.extend([self.add_objects, self.calculate_zernikes])
        return result
    
    def add_object_cb(self):
        """Add a slot for another object"""
        index = len(self.object_groups)
        self.object_groups.append({OG_NAME:cps.ObjectNameSubscriber("What did you call the objects you want to measure?","None"),
                                   OG_REMOVE:cps.DoSomething("Remove the above objects","Remove",self.remove_object_cb,index)})
        
    def remove_object_cb(self, index):
        """Remove the indexed object from the to-do list"""
        del self.object_groups[index]
        
    def get_categories(self,pipeline, object_name):
        """Get the categories of measurements supplied for the given object name
        
        pipeline - pipeline being run
        object_name - name of labels in question (or 'Images')
        returns a list of category names
        """
        if any([object_name == og[OG_NAME] for og in self.object_groups]):
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

    def run(self, workspace):
        """Run, computing the area measurements for the objects"""
        
        for object_group in self.object_groups:
            self.run_on_objects(object_group[OG_NAME].value, workspace)
    
    def run_on_objects(self,object_name, workspace):
        """Run, computing the area measurements for a single map of objects"""
        objects = workspace.get_objects(object_name)
        #
        # Compute the area as the sum of 1's over a label matrix
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
            zernike_features,z_images = cpmz.zernike(zernike_numbers, 
                                                     objects.segmented,
                                                     objects.indices)
            for i in range(zernike_numbers.shape[0]):
                zernike_number = zernike_numbers[i]
                zernike_feature = zernike_features[:,i]
                feature_name = self.get_zernike_name(zernike_number)
                self.record_measurement(workspace, object_name, feature_name, 
                                        zernike_feature)
            
    
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
        data    = objects.fn_of_label_and_index(function) 
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
        data    = objects.fn_of_ones_label_and_index(function) 
        self.record_measurement(workspace, object_name, feature_name, data)

    def record_measurement(self,workspace,  
                           object_name, feature_name, result):
        """Record the result of a measurement in the workspace's measurements"""
        data = fixup_scipy_ndimage_result(result)
        workspace.add_measurement(object_name, 
                                  "%s_%s"%(AREA_SHAPE,feature_name), 
                                  data)
        
def form_factor(objects):
    """FormFactor = 4/pi*Area/Perimeter^2, equals 1 for a perfectly circular"""
    areas = fixup_scipy_ndimage_result(
                objects.fn_of_ones_label_and_index(scind.sum))
    perimeter = objects.fn_of_label_and_index(calculate_perimeters)
    return 4.0*np.pi*areas / perimeter**2

