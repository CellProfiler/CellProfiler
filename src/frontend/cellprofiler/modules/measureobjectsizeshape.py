import centrosome.cpmorphology
import centrosome.zernike
import numpy
import scipy.ndimage
import skimage.measure
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.module import Module
from cellprofiler_core.object import Objects
from cellprofiler_core.setting import Divider, Binary, ValidationError
from cellprofiler_core.setting.subscriber import LabelListSubscriber

import cellprofiler.gui.help.content
import cellprofiler.icons

from cellprofiler_library.modules import measureobjectsizeshape
from cellprofiler_library.opts.objectsizeshapefeatures import ObjectSizeShapeFeatures

__doc__ = """\
MeasureObjectSizeShape
======================

**MeasureObjectSizeShape** measures several area and shape features
of identified objects.

Given an image with identified objects (e.g., nuclei or cells), this
module extracts area and shape features of each one. Note that these
features are only reliable for objects that are completely inside the
image borders, so you may wish to exclude objects touching the edge of
the image using **Identify** settings for 2D objects, or by applying
**FilterObjects** downstream.

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

-  *Area:* *(2D only)* The number of pixels in the region.
-  *Volume:* *(3D only)* The number of voxels in the region.
-  *Perimeter:* *(2D only)* The total number of pixels around the boundary of each
   region in the image.
-  *SurfaceArea:* *(3D only)* The total number of voxels around the boundary of
   each region in the image.
-  *FormFactor:* *(2D only)* Calculated as 4\*π\*Area/Perimeter\ :sup:`2`. Equals 1
   for a perfectly circular object.
-  *Convex Area:* The area of a convex polygon containing the whole object.
   Best imagined as a rubber band stretched around the object. 
-  *Solidity:* The proportion of the pixels in the convex hull that are
   also in the object, i.e., *ObjectArea/ConvexHullArea*.
-  *Extent:* The proportion of the pixels (2D) or voxels (3D) in the bounding box
   that are also in the region. Computed as the area/volume of the object divided
   by the area/volume of the bounding box.
-  *EulerNumber:* The number of objects in the region minus the number
   of holes in those objects, assuming 8-connectivity.
-  *Center\_X, Center\_Y, Center\_Z:* The *x*-, *y*-, and (for 3D objects) *z-*
   coordinates of the point farthest away from any object edge (the *centroid*).
   Note that this is not the same as the *Location-X* and *-Y* measurements
   produced by the **Identify** or **Watershed**
   modules or the *Location-Z* measurement produced by the **Watershed** module.
-  *BoundingBoxMinimum/Maximum\_X/Y/Z:* The minimum/maximum *x*-, *y*-, and (for 3D objects)
   *z-* coordinates of the object.
-  *BoundingBoxArea:* *(2D only)* The area of a box containing the object.
-  *BoundingBoxVolume:* *(3D only)* The volume of a box containing the object.
-  *Eccentricity:* *(2D only)* The eccentricity of the ellipse that has the same
   second-moments as the region. The eccentricity is the ratio of the
   distance between the foci of the ellipse and its major axis length.
   The value is between 0 and 1. (0 and 1 are degenerate cases; an
   ellipse whose eccentricity is 0 is actually a circle, while an
   ellipse whose eccentricity is 1 is a line segment.)

    |MOSS_image0|


-  *MajorAxisLength:* The length (in pixels) of the major axis of the
   ellipse that has the same normalized second central moments as the
   region.
-  *MinorAxisLength:* The length (in pixels) of the minor axis of the
   ellipse that has the same normalized second central moments as the
   region.
-  *EquivalentDiameter:* The diameter of a circle or sphere with the same area
   as the object.
-  *Orientation:* *(2D only)* The angle (in degrees ranging from -90 to 90 degrees)
   between the x-axis and the major axis of the ellipse that has the
   same second-moments as the region.
-  *Compactness:* *(2D only)* Calculated as Perimeter\ :sup:`2`/4\*π\*Area, related to 
   Form Factor. A filled circle will have a compactness of 1, with irregular objects or 
   objects with holes having a value greater than 1.
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
-  *Spatial Moment features:* *(2D only)* A series of weighted averages 
   representing the shape, size, rotation and location of the object.
-  *Central Moment features:* *(2D only)* Similar to spatial moments, but
   normalized to the object's centroid. These are therefore not influenced
   by an object's location within an image.
-  *Normalized Moment features:* *(2D only)* Similar to central moments,
   but further normalized to be scale invariant. These moments are therefore
   not impacted by an object's size (or location).
-  *Hu Moment features:* *(2D only)* Hu's set of image moment features. These
   are not altered by the object's location, size or rotation. This means that
   they primarily describe the shape of the object.
-  *Inertia Tensor features:* *(2D only)* A representation of rotational
   inertia of the object relative to it's center.
-  *Inertia Tensor Eigenvalues features:* *(2D only)* Values describing 
   the movement of the Inertia Tensor array.



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
-  Hu MK (1962), “Visual pattern recognition by moment invariants”, *IRE
   transactions on information theory*, 8(2), pp.179-187 `(link)`_

.. _(pdf): http://sibgrapi.sid.inpe.br/col/sid.inpe.br/banon/2002/10.23.11.34/doc/35.pdf
.. _Section 2.4.3 - Statistical shape properties: http://www.scribd.com/doc/58004056/Principles-of-Digital-Image-Processing#page=49
.. _(link): https://ieeexplore.ieee.org/abstract/document/1057692
.. |MOSS_image0| image:: {ECCENTRICITY_ICON}
""".format(
    **{
        "ECCENTRICITY_ICON": cellprofiler.gui.help.content.image_resource(
            "MeasureObjectSizeShape_Eccentricity.png"
        )
    }
)


class MeasureObjectSizeShape(Module):
    module_name = "MeasureObjectSizeShape"
    variable_revision_number = 3
    category = "Measurement"

    def create_settings(self):
        """Create the settings for the module at startup and set the module name

        The module allows for an unlimited number of measured objects, each
        of which has an entry in self.object_groups.
        """
        self.objects_list = LabelListSubscriber(
            "Select object sets to measure",
            [],
            doc="""Select the object sets whose size and shape you want to measure.""",
        )
        self.spacer = Divider(line=True)

        self.calculate_advanced = Binary(
            text="Calculate the advanced features?",
            value=False,
            doc="""\
Select *{YES}* to calculate additional statistics for object moments
and intertia tensors in **2D mode**. These features should not require much additional time
to calculate, but do add many additional columns to the resulting output 
files.

In **3D mode** this setting enables the Solidity measurement, which can be time-consuming
to calculate.""".format(
                **{"YES": "Yes"}
            ),
        )

        self.calculate_zernikes = Binary(
            text="Calculate the Zernike features?",
            value=True,
            doc="""\
Select *{YES}* to calculate the Zernike shape features. Because the
first 10 Zernike polynomials (from order 0 to order 9) are calculated,
this operation can be time consuming if the image contains a lot of
objects. Select *{NO}* if you are measuring 3D objects with this
module.""".format(
                **{"YES": "Yes", "NO": "No"}
            ),
        )

    def settings(self):
        """The settings as they appear in the save file"""
        result = [self.objects_list, self.calculate_zernikes, self.calculate_advanced]
        return result

    def visible_settings(self):
        """The settings as they appear in the module viewer"""
        result = [
            self.objects_list,
            self.spacer,
            self.calculate_zernikes,
            self.calculate_advanced,
        ]
        return result

    def validate_module(self, pipeline):
        """Make sure chosen objects are selected only once"""
        objects = set()
        if len(self.objects_list.value) == 0:
            raise ValidationError("No object sets selected", self.objects_list)

        for object_name in self.objects_list.value:
            if object_name in objects:
                raise ValidationError(
                    "%s has already been selected" % object_name, object_name
                )
            objects.add(object_name)

    def get_categories(self, pipeline, object_name):
        """Get the categories of measurements supplied for the given object name

        pipeline - pipeline being run
        object_name - name of labels in question (or 'Images')
        returns a list of category names
        """
        for object_set in self.objects_list.value:
            if object_set == object_name:
                return [ObjectSizeShapeFeatures.AREA_SHAPE.value]
        else:
            return []

    def get_zernike_numbers(self):
        """The Zernike numbers measured by this module"""
        if self.calculate_zernikes.value:
            return centrosome.zernike.get_zernike_indexes(
                ObjectSizeShapeFeatures.ZERNIKE_N.value + 1
            )
        else:
            return []

    def get_zernike_name(self, zernike_index):
        """Return the name of a Zernike feature, given a (N,M) 2-tuple

        zernike_index - a 2 element sequence organized as N,M
        """
        return "Zernike_%d_%d" % (zernike_index[0], zernike_index[1])

    def get_feature_names(self, pipeline):
        """Return the names of the features measured"""
        feature_names = list(ObjectSizeShapeFeatures.F_STANDARD.value)

        if pipeline.volumetric():
            feature_names += list(ObjectSizeShapeFeatures.F_STD_3D.value)
            if self.calculate_advanced.value:
                feature_names += list(ObjectSizeShapeFeatures.F_ADV_3D.value)
        else:
            feature_names += list(ObjectSizeShapeFeatures.F_STD_2D.value)
            if self.calculate_zernikes.value:
                feature_names += [
                    self.get_zernike_name(index) for index in self.get_zernike_numbers()
                ]
            if self.calculate_advanced.value:
                feature_names += list(get_feature_names.F_ADV_2D.values)

        return feature_names

    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces

        object_name - return measurements made on this object
                      (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        if (
            category == ObjectSizeShapeFeatures.AREA_SHAPE.value
            and self.get_categories(pipeline, object_name)
        ):
            return self.get_feature_names(pipeline)
        return []

    def run(self, workspace):
        """Run, computing the area measurements for the objects"""

        if self.show_window:
            workspace.display_data.col_labels = (
                "Object",
                "Feature",
                "Mean",
                "Median",
                "STD",
            )

            workspace.display_data.statistics = []
        for object_name in self.objects_list.value:

            objects = workspace.get_objects(object_name)

            features_to_record = measureobjectsizeshape(
                objects=objects.dense,
                calculate_advanced=self.calculate_advanced.value,
                calculate_zernikes=self.calculate_zernikes.value,
                volumetric=workspace.pipeline.volumetric(),
                spacing=objects.parent_image.spacing
                if objects.has_parent_image
                else (1.0,) * objects.dimensions,  # TODO: Check this change is OK
            )

            for f, m in features_to_record.items():
                self.record_measurement(workspace, object_name, f, m)

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))
        figure.subplot_table(
            0,
            0,
            workspace.display_data.statistics,
            col_labels=workspace.display_data.col_labels,
            title="default",
        )

    def record_measurement(self, workspace, object_name, feature_name, result):
        """Record the result of a measurement in the workspace's measurements"""
        data = centrosome.cpmorphology.fixup_scipy_ndimage_result(result)
        workspace.add_measurement(
            object_name,
            "%s_%s" % (ObjectSizeShapeFeatures.AREA_SHAPE.value, feature_name),
            data,
        )
        if self.show_window and numpy.any(numpy.isfinite(data)) > 0:
            data = data[numpy.isfinite(data)]
            workspace.display_data.statistics.append(
                (
                    object_name,
                    feature_name,
                    "%.2f" % numpy.mean(data),
                    "%.2f" % numpy.median(data),
                    "%.2f" % numpy.std(data),
                )
            )

    def get_measurement_columns(self, pipeline):
        """Return measurement column definitions.
        All cols returned as float even though "Area" will only ever be int"""
        measurement_names = self.get_feature_names(pipeline)
        cols = []
        for oname in self.objects_list.value:
            for mname in measurement_names:
                cols += [
                    (
                        oname,
                        ObjectSizeShapeFeatures.AREA_SHAPE.value + "_" + mname,
                        COLTYPE_FLOAT,
                    )
                ]
        return cols

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        """Adjust the setting_values for older save file versions"""
        if variable_revision_number == 1:
            objects_list = setting_values[:-1]
            setting_values = [", ".join(map(str, objects_list)), setting_values[-1]]
            variable_revision_number = 2
        if variable_revision_number == 2:
            # Add advanced features toggle
            setting_values.append("No")
            variable_revision_number = 3
        return setting_values, variable_revision_number

    def volumetric(self):
        return True


MeasureObjectAreaShape = MeasureObjectSizeShape
