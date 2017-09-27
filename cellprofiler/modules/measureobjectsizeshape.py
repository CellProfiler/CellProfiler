# coding=utf-8

import cellprofiler.icons

import numpy as np
import scipy.ndimage as scind

import cellprofiler.module as cpm
import cellprofiler.object as cpo
import cellprofiler.setting as cps
from cellprofiler.setting import YES, NO
import centrosome.zernike as cpmz
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from centrosome.cpmorphology import ellipse_from_second_moments_ijv
from centrosome.cpmorphology import calculate_extents
from centrosome.cpmorphology import calculate_perimeters
from centrosome.cpmorphology import calculate_solidity
from centrosome.cpmorphology import euler_number
from centrosome.cpmorphology import distance_to_edge
from centrosome.cpmorphology import maximum_position_of_labels
from centrosome.cpmorphology import median_of_labels
from centrosome.cpmorphology import feret_diameter
from centrosome.cpmorphology import convex_hull_ijv
from cellprofiler.measurement import COLTYPE_FLOAT
import skimage.measure
import _help

__doc__ = """\
MeasureObjectSizeShape
======================

**MeasureObjectSizeShape** measures several area and shape features
of identified objects.

Given an image with identified objects (e.g., nuclei or cells), this
module extracts area and shape features of each one. Note that these
features are only reliable for objects that are completely inside the
image borders, so you may wish to exclude objects touching the edge of
the image using **Identify** settings for 2D objects or **ClearBorder** for 3D objects.

The display window for this module shows per-image
aggregates for the per-object measurements. If you want to view the
per-object measurements themselves, you will need to use an
**Export** module to export them, or use **DisplayDataOnImage** to
display the object measurements of choice overlaid on an image of
choice.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

See also
^^^^^^^^

See also **MeasureImageAreaOccupied**.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Some measurements are available for 3D and 2D objects, while some are 2D
only.

See the *Technical Notes* below for an explanation of a key step
underlying many of the following metrics: creating an
ellipse with the same second-moments as each object.

-  *Area:* The number of pixels (2D) or voxels (3D) in the region.
-  *Perimeter:* The total number of pixels (2D) or voxels (3D) around the boundary of each
   region in the image. In 3D, this is more commonly described as the surface area.
-  *FormFactor:* *(2D only)* Calculated as 4\*π\*Area/Perimeter\ :sup:`2`. Equals 1
   for a perfectly circular object.
-  *Solidity:* *(2D only)* The proportion of the pixels in the convex hull that are
   also in the object, i.e., *ObjectArea/ConvexHullArea*.
-  *Extent:* The proportion of the pixels (2D) or voxels (3D) in the bounding box
   that are also in the region. Computed as the area/volume of the object divided
   by the area/volume of the bounding box.
-  *EulerNumber:* *(2D only)* The number of objects in the region minus the number
   of holes in those objects, assuming 8-connectivity.
-  *Center\_X, Center\_Y, Center\_Z:* The *x*-, *y*-, and (for 3D objects) *z-*
   coordinates of the point farthest away from any object edge (the *centroid*).
   Note that this is not the same as the *Location-X* and *-Y* measurements
   produced by the **Identify** or **Watershed**
   modules or the *Location-Z* measurement produced by the **Watershed** module.
-  *Eccentricity:* *(2D only)* The eccentricity of the ellipse that has the same
   second-moments as the region. The eccentricity is the ratio of the
   distance between the foci of the ellipse and its major axis length.
   The value is between 0 and 1. (0 and 1 are degenerate cases; an
   ellipse whose eccentricity is 0 is actually a circle, while an
   ellipse whose eccentricity is 1 is a line segment.)

    |MOSS_image0|


-  *MajorAxisLength:* *(2D only)* The length (in pixels) of the major axis of the
   ellipse that has the same normalized second central moments as the
   region.
-  *MinorAxisLength:* *(2D only)* The length (in pixels) of the minor axis of the
   ellipse that has the same normalized second central moments as the
   region.
-  *Orientation:* *(2D only)* The angle (in degrees ranging from -90 to 90 degrees)
   between the x-axis and the major axis of the ellipse that has the
   same second-moments as the region.
-  *Compactness:* *(2D only)* The mean squared distance of the object’s pixels from
   the centroid divided by the area. A filled circle will have a
   compactness of 1, with irregular objects or objects with holes having
   a value greater than 1.
-  *MaximumRadius:* *(2D only)* The maximum distance of any pixel in the object to
   the closest pixel outside of the object. For skinny objects, this is
   1/2 of the maximum width of the object.
-  *MedianRadius:* *(2D only)* The median distance of any pixel in the object to the
   closest pixel outside of the object.
-  *MeanRadius:* *(2D only)* The mean distance of any pixel in the object to the
   closest pixel outside of the object.
-  *MinFeretDiameter, MaxFeretDiameter:* *(2D only)* The Feret diameter is the
   distance between two parallel lines tangent on either side of the
   object (imagine taking a caliper and measuring the object at various
   angles). The minimum and maximum Feret diameters are the smallest and
   largest possible diameters, rotating the calipers along all possible
   angles.
-  *Zernike shape features:* *(2D only)* These metrics of shape describe a binary object
   (or more precisely, a patch with background and an object in the
   center) in a basis of Zernike polynomials, using the coefficients as
   features (*Boland et al., 1998*). Currently, Zernike polynomials from
   order 0 to order 9 are calculated, giving in total 30 measurements.
   While there is no limit to the order which can be calculated (and
   indeed you could add more by adjusting the code), the higher order
   polynomials carry less information.

Technical notes
^^^^^^^^^^^^^^^

A number of the object measurements are generated by creating an ellipse
with the same second-moments as the original object region. This is
essentially the best-fitting ellipse for a given object with the same
statistical properties. Furthermore, they are not affected by the
translation or uniform scaling of a region.

Following computer vision conventions, the origin of the X and Y axes is at the top
left of the image rather than the bottom left; the orientation of objects whose topmost point
is on their right (or are rotated counter-clockwise from the horizontal) will therefore
have a negative orientation, while objects whose topmost point is on their left
(or are rotated clockwise from the horizontal) will have a positive orientation.

The Zernike features are computed within the minimum enclosing circle of
the object, i.e., the circle of the smallest diameter that contains all
of the object’s pixels.

References
^^^^^^^^^^

-  Rocha L, Velho L, Carvalho PCP, “Image moments-based structuring and
   tracking of objects”, Proceedings from XV Brazilian Symposium on
   Computer Graphics and Image Processing, 2002. `(pdf)`_
-  Principles of Digital Image Processing: Core Algorithms
   (Undergraduate Topics in Computer Science): `Section 2.4.3 -
   Statistical shape properties`_
-  Chrystal P (1885), “On the problem to construct the minimum circle
   enclosing n given points in a plane”, *Proceedings of the Edinburgh
   Mathematical Society*, vol 3, p. 30

.. _(pdf): http://sibgrapi.sid.inpe.br/col/sid.inpe.br/banon/2002/10.23.11.34/doc/35.pdf
.. _Section 2.4.3 - Statistical shape properties: http://www.scribd.com/doc/58004056/Principles-of-Digital-Image-Processing#page=49
.. |MOSS_image0| image:: {ECCENTRICITY_ICON}
""".format(**{
                "ECCENTRICITY_ICON": _help.__image_resource("MeasureObjectSizeShape_Eccentricity.png")
            })

"""The category of the per-object measurements made by this module"""
AREA_SHAPE = 'AreaShape'

"""Calculate Zernike features for N,M where N=0 through ZERNIKE_N"""
ZERNIKE_N = 9

F_AREA = "Area"
F_ECCENTRICITY = 'Eccentricity'
F_SOLIDITY = 'Solidity'
F_EXTENT = 'Extent'
F_CENTER_X = 'Center_X'
F_CENTER_Y = 'Center_Y'
F_CENTER_Z = 'Center_Z'
F_EULER_NUMBER = 'EulerNumber'
F_PERIMETER = 'Perimeter'
F_FORM_FACTOR = 'FormFactor'
F_MAJOR_AXIS_LENGTH = 'MajorAxisLength'
F_MINOR_AXIS_LENGTH = 'MinorAxisLength'
F_ORIENTATION = 'Orientation'
F_COMPACTNESS = 'Compactness'
F_MAXIMUM_RADIUS = 'MaximumRadius'
F_MEDIAN_RADIUS = 'MedianRadius'
F_MEAN_RADIUS = 'MeanRadius'
F_MIN_FERET_DIAMETER = 'MinFeretDiameter'
F_MAX_FERET_DIAMETER = 'MaxFeretDiameter'

"""The non-Zernike features"""
F_STANDARD = [F_AREA, F_ECCENTRICITY, F_SOLIDITY, F_EXTENT,
              F_EULER_NUMBER, F_PERIMETER, F_FORM_FACTOR,
              F_MAJOR_AXIS_LENGTH, F_MINOR_AXIS_LENGTH,
              F_ORIENTATION, F_COMPACTNESS, F_CENTER_X, F_CENTER_Y, F_CENTER_Z,
              F_MAXIMUM_RADIUS, F_MEAN_RADIUS, F_MEDIAN_RADIUS,
              F_MIN_FERET_DIAMETER, F_MAX_FERET_DIAMETER]


class MeasureObjectSizeShape(cpm.Module):
    module_name = "MeasureObjectSizeShape"
    variable_revision_number = 1
    category = 'Measurement'

    def create_settings(self):
        """Create the settings for the module at startup and set the module name

        The module allows for an unlimited number of measured objects, each
        of which has an entry in self.object_groups.
        """
        self.object_groups = []
        self.add_object(can_remove=False)
        self.spacer = cps.Divider(line=True)
        self.add_objects = cps.DoSomething("", "Add another object", self.add_object)

        self.calculate_zernikes = cps.Binary(
                'Calculate the Zernike features?', True, doc="""\
Select *%(YES)s* to calculate the Zernike shape features. Because the
first 10 Zernike polynomials (from order 0 to order 9) are calculated,
this operation can be time consuming if the image contains a lot of
objects. Select *%(NO)s* if you are measuring 3D objects with this
module.""" % globals())

    def add_object(self, can_remove=True):
        """Add a slot for another object"""
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=False))

        group.append("name", cps.ObjectNameSubscriber(
                "Select objects to measure", cps.NONE, doc="""Select the objects that you want to measure."""))

        if can_remove:
            group.append("remove", cps.RemoveSettingButton("", "Remove this object", self.object_groups, group))

        self.object_groups.append(group)

    def settings(self):
        """The settings as they appear in the save file"""
        result = [og.name for og in self.object_groups]
        result.append(self.calculate_zernikes)
        return result

    def prepare_settings(self, setting_values):
        """Adjust the number of object groups based on the number of setting_values"""
        object_group_count = len(setting_values) - 1
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

    def validate_module(self, pipeline):
        """Make sure chosen objects are selected only once"""
        objects = set()
        for group in self.object_groups:
            if group.name.value in objects:
                raise cps.ValidationError(
                        "%s has already been selected" % group.name.value,
                        group.name)
            objects.add(group.name.value)

    def get_categories(self, pipeline, object_name):
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
            return cpmz.get_zernike_indexes(ZERNIKE_N + 1)
        else:
            return []

    def get_zernike_name(self, zernike_index):
        """Return the name of a Zernike feature, given a (N,M) 2-tuple

        zernike_index - a 2 element sequence organized as N,M
        """
        return "Zernike_%d_%d" % (zernike_index[0], zernike_index[1])

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
                self.get_categories(pipeline, object_name)):
            return self.get_feature_names()
        return []

    def run(self, workspace):
        """Run, computing the area measurements for the objects"""

        if self.show_window:
            workspace.display_data.col_labels = ("Object", "Feature", "Mean", "Median", "STD")

            workspace.display_data.statistics = []

        for object_group in self.object_groups:
            self.run_on_objects(object_group.name.value, workspace)

    def run_on_objects(self, object_name, workspace):
        """Run, computing the area measurements for a single map of objects"""
        objects = workspace.get_objects(object_name)

        if len(objects.shape) == 2:
            #
            # Do the ellipse-related measurements
            #
            i, j, l = objects.ijv.transpose()
            centers, eccentricity, major_axis_length, minor_axis_length, \
            theta, compactness = \
                ellipse_from_second_moments_ijv(i, j, 1, l, objects.indices, True)
            del i
            del j
            del l
            self.record_measurement(workspace, object_name,
                                    F_ECCENTRICITY, eccentricity)
            self.record_measurement(workspace, object_name,
                                    F_MAJOR_AXIS_LENGTH, major_axis_length)
            self.record_measurement(workspace, object_name,
                                    F_MINOR_AXIS_LENGTH, minor_axis_length)
            self.record_measurement(workspace, object_name, F_ORIENTATION,
                                    theta * 180 / np.pi)
            self.record_measurement(workspace, object_name, F_COMPACTNESS,
                                    compactness)
            is_first = False
            if len(objects.indices) == 0:
                nobjects = 0
            else:
                nobjects = np.max(objects.indices)
            mcenter_x = np.zeros(nobjects)
            mcenter_y = np.zeros(nobjects)
            mextent = np.zeros(nobjects)
            mperimeters = np.zeros(nobjects)
            msolidity = np.zeros(nobjects)
            euler = np.zeros(nobjects)
            max_radius = np.zeros(nobjects)
            median_radius = np.zeros(nobjects)
            mean_radius = np.zeros(nobjects)
            min_feret_diameter = np.zeros(nobjects)
            max_feret_diameter = np.zeros(nobjects)
            zernike_numbers = self.get_zernike_numbers()
            zf = {}
            for n, m in zernike_numbers:
                zf[(n, m)] = np.zeros(nobjects)
            if nobjects > 0:
                chulls, chull_counts = convex_hull_ijv(objects.ijv, objects.indices)
                for labels, indices in objects.get_labels():
                    to_indices = indices - 1
                    distances = distance_to_edge(labels)
                    mcenter_y[to_indices], mcenter_x[to_indices] = \
                        maximum_position_of_labels(distances, labels, indices)
                    max_radius[to_indices] = fix(scind.maximum(
                            distances, labels, indices))
                    mean_radius[to_indices] = fix(scind.mean(
                            distances, labels, indices))
                    median_radius[to_indices] = median_of_labels(
                            distances, labels, indices)
                    #
                    # The extent (area / bounding box area)
                    #
                    mextent[to_indices] = calculate_extents(labels, indices)
                    #
                    # The perimeter distance
                    #
                    mperimeters[to_indices] = calculate_perimeters(labels, indices)
                    #
                    # Solidity
                    #
                    msolidity[to_indices] = calculate_solidity(labels, indices)
                    #
                    # Euler number
                    #
                    euler[to_indices] = euler_number(labels, indices)
                    #
                    # Zernike features
                    #
                    if self.calculate_zernikes.value:
                        zf_l = cpmz.zernike(zernike_numbers, labels, indices)
                        for (n, m), z in zip(zernike_numbers, zf_l.transpose()):
                            zf[(n, m)][to_indices] = z
                #
                # Form factor
                #
                ff = 4.0 * np.pi * objects.areas / mperimeters ** 2
                #
                # Feret diameter
                #
                min_feret_diameter, max_feret_diameter = \
                    feret_diameter(chulls, chull_counts, objects.indices)

            else:
                ff = np.zeros(0)

            for f, m in ([(F_AREA, objects.areas),
                          (F_CENTER_X, mcenter_x),
                          (F_CENTER_Y, mcenter_y),
                          (F_CENTER_Z, np.ones_like(mcenter_x)),
                          (F_EXTENT, mextent),
                          (F_PERIMETER, mperimeters),
                          (F_SOLIDITY, msolidity),
                          (F_FORM_FACTOR, ff),
                          (F_EULER_NUMBER, euler),
                          (F_MAXIMUM_RADIUS, max_radius),
                          (F_MEAN_RADIUS, mean_radius),
                          (F_MEDIAN_RADIUS, median_radius),
                          (F_MIN_FERET_DIAMETER, min_feret_diameter),
                          (F_MAX_FERET_DIAMETER, max_feret_diameter)] +
                             [(self.get_zernike_name((n, m)), zf[(n, m)])
                              for n, m in zernike_numbers]):
                self.record_measurement(workspace, object_name, f, m)
        else:
            labels = objects.segmented

            props = skimage.measure.regionprops(labels)

            # Area
            areas = [prop.area for prop in props]

            self.record_measurement(workspace, object_name, F_AREA, areas)

            # Extent
            extents = [prop.extent for prop in props]

            self.record_measurement(workspace, object_name, F_EXTENT, extents)

            # Centers of mass
            centers = objects.center_of_mass()

            center_z, center_x, center_y = centers.transpose()

            self.record_measurement(workspace, object_name, F_CENTER_X, center_x)

            self.record_measurement(workspace, object_name, F_CENTER_Y, center_y)

            self.record_measurement(workspace, object_name, F_CENTER_Z, center_z)

            # Perimeters
            perimeters = []

            for label in np.unique(labels):
                if label == 0:
                    continue

                volume = np.zeros_like(labels, dtype='bool')

                volume[labels == label] = True

                verts, faces, _, _ = skimage.measure.marching_cubes(
                    volume,
                    spacing=objects.parent_image.spacing if objects.has_parent_image else (1.0,) * labels.ndim,
                    level=0
                )

                perimeters += [skimage.measure.mesh_surface_area(verts, faces)]

            if len(perimeters) == 0:
                self.record_measurement(workspace, object_name, F_PERIMETER, [0])
            else:
                self.record_measurement(workspace, object_name, F_PERIMETER, perimeters)

            for feature in self.get_feature_names():
                if feature in [F_AREA, F_EXTENT, F_CENTER_X, F_CENTER_Y, F_CENTER_Z, F_PERIMETER]:
                    continue

                self.record_measurement(workspace, object_name, feature, [np.nan])

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))
        figure.subplot_table(0, 0,
                             workspace.display_data.statistics,
                             col_labels=workspace.display_data.col_labels)

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
                              object_name, feature_name):
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

    def record_measurement(self, workspace,
                           object_name, feature_name, result):
        """Record the result of a measurement in the workspace's measurements"""
        data = fix(result)
        workspace.add_measurement(object_name,
                                  "%s_%s" % (AREA_SHAPE, feature_name),
                                  data)
        if self.show_window and np.any(np.isfinite(data)) > 0:
            data = data[np.isfinite(data)]
            workspace.display_data.statistics.append(
                    (object_name, feature_name,
                     "%.2f" % np.mean(data),
                     "%.2f" % np.median(data),
                     "%.2f" % np.std(data)))

    def get_measurement_columns(self, pipeline):
        '''Return measurement column definitions.
        All cols returned as float even though "Area" will only ever be int'''
        object_names = [s.value for s in self.settings()][:-1]
        measurement_names = self.get_feature_names()
        cols = []
        for oname in object_names:
            for mname in measurement_names:
                cols += [(oname, AREA_SHAPE + '_' + mname, COLTYPE_FLOAT)]
        return cols

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
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

    def volumetric(self):
        return True


def form_factor(objects):
    """FormFactor = 4/pi*Area/Perimeter^2, equals 1 for a perfectly circular"""
    if len(objects.indices) > 0:
        perimeter = objects.fn_of_label_and_index(calculate_perimeters)
        return 4.0 * np.pi * objects.areas / perimeter ** 2
    else:
        return np.zeros((0,))


MeasureObjectAreaShape = MeasureObjectSizeShape
