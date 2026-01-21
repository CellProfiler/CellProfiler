"""
MeasureObjectNeighbors
======================

**MeasureObjectNeighbors** calculates how many neighbors each object
has and records various properties about the neighbors’ relationships,
including the percentage of an object’s edge pixels that touch a
neighbor. Please note that the distances reported for object 
measurements are center-to-center distances, not edge-to-edge distances.

Given an image with objects identified (e.g., nuclei or cells), this
module determines how many neighbors each object has. You can specify
the distance within which objects should be considered neighbors, or
that objects are only considered neighbors if they are directly
touching.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

See also
^^^^^^^^

See also the **Identify** modules.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Object measurements**

-  *NumberOfNeighbors:* Number of neighbor objects.
-  *PercentTouching:* Percent of the object’s boundary pixels that touch
   neighbors, after the objects have been expanded to the specified
   distance.
-  *FirstClosestObjectNumber:* The index of the closest object.
-  *FirstClosestDistance:* The distance to the closest object (in units
   of pixels), measured between object centers.
-  *SecondClosestObjectNumber:* The index of the second closest object.
-  *SecondClosestDistance:* The distance to the second closest object (in units
   of pixels), measured between object centers.
-  *AngleBetweenNeighbors:* The angle formed with the object center as
   the vertex and the first and second closest object centers along the
   vectors.

**Object relationships:** The identity of the neighboring objects, for
each object. Since per-object output is one-to-one and neighbors
relationships are often many-to-one, they may be saved as a separate
file in **ExportToSpreadsheet** by selecting *Object relationships* from
the list of objects to export.

Technical notes
^^^^^^^^^^^^^^^

Objects discarded via modules such as **IdentifyPrimaryObjects** or
**IdentifySecondaryObjects** will still register as neighbors for the
purposes of accurate measurement. For instance, if an object touches a
single object and that object had been discarded, *NumberOfNeighbors*
will be positive, but there may not be a corresponding
*ClosestObjectNumber*. This can be disabled in module settings.

"""

import matplotlib.cm
import numpy
import scipy.ndimage
import scipy.signal
import skimage.morphology
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.constants.measurement import COLTYPE_INTEGER
from cellprofiler_core.constants.measurement import MCA_AVAILABLE_EACH_CYCLE
from cellprofiler_core.constants.measurement import NEIGHBORS
from cellprofiler_core.image import Image
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.module import Module
from cellprofiler_core.object import Objects
from cellprofiler_core.preferences import get_default_colormap
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.choice import Choice, Colormap
from cellprofiler_core.setting.subscriber import LabelSubscriber
from cellprofiler_core.setting.text import ImageName
from cellprofiler_core.setting.text import Integer
from cellprofiler_core.workspace import Workspace
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from centrosome.cpmorphology import strel_disk, centers_of_labels
from centrosome.outline import outline
from cellprofiler_library.opts.measureobjectneighbors import DistanceMethod, Measurement, MeasurementScale, C_NEIGHBORS, M_ALL, D_ALL
from typing import List, Optional, Union, Tuple
from numpy.typing import NDArray
from cellprofiler_library.types import ObjectSegmentation, ObjectLabel
from cellprofiler_library.functions.segmentation import convert_label_set_to_ijv, areas_from_ijv, cast_labels_to_label_set, indices_from_ijv
from cellprofiler_library.functions.object_processing import relate_labels
from cellprofiler_library.modules._measureobjectneighbors import measure_object_neighbors
# DistanceMethod.ADJACENT = "Adjacent"
# DistanceMethod.EXPAND = "Expand until adjacent"
# DistanceMethod.WITHIN = "Within a specified distance"
# D_ALL = [DistanceMethod.ADJACENT, DistanceMethod.EXPAND, DistanceMethod.WITHIN]

# Measurement.NUMBER_OF_NEIGHBORS = "NumberOfNeighbors"
# Measurement.PERCENT_TOUCHING = "PercentTouching"
# Measurement.FIRST_CLOSEST_OBJECT_NUMBER = "FirstClosestObjectNumber"
# Measurement.FIRST_CLOSEST_DISTANCE = "FirstClosestDistance"
# Measurement.SECOND_CLOSEST_OBJECT_NUMBER = "SecondClosestObjectNumber"
# Measurement.SECOND_CLOSEST_DISTANCE = "SecondClosestDistance"
# Measurement.ANGLE_BETWEEN_NEIGHBORS = "AngleBetweenNeighbors"
# M_ALL = [
#     Measurement.NUMBER_OF_NEIGHBORS,
#     Measurement.PERCENT_TOUCHING,
#     Measurement.FIRST_CLOSEST_OBJECT_NUMBER,
#     Measurement.FIRST_CLOSEST_DISTANCE,
#     Measurement.SECOND_CLOSEST_OBJECT_NUMBER,
#     Measurement.SECOND_CLOSEST_DISTANCE,
#     Measurement.ANGLE_BETWEEN_NEIGHBORS,
# ]

# C_NEIGHBORS = "Neighbors"

# MeasurementScale.EXPANDED = "Expanded"
# MeasurementScale.ADJACENT = "Adjacent"


class MeasureObjectNeighbors(Module):
    module_name = "MeasureObjectNeighbors"
    category = "Measurement"
    variable_revision_number = 3

    def create_settings(self):
        self.object_name = LabelSubscriber(
            "Select objects to measure",
            "None",
            doc="""\
Select the objects whose neighbors you want to measure.""",
        )

        self.neighbors_name = LabelSubscriber(
            "Select neighboring objects to measure",
            "None",
            doc="""\
This is the name of the objects that are potential
neighbors of the above objects. You can find the neighbors
within the same set of objects by selecting the same objects
as above.""",
        )

        self.distance_method = Choice(
            "Method to determine neighbors",
            D_ALL,
            DistanceMethod.EXPAND,
            doc="""\
There are several methods by which to determine whether objects are
neighbors:

-  *{D_ADJACENT}:* In this mode, two objects must have adjacent
   boundary pixels to be neighbors.
-  *{D_EXPAND}:* The objects are expanded until all pixels on the
   object boundaries are touching another. Two objects are neighbors if
   any of their boundary pixels are adjacent after expansion.
-  *{D_WITHIN}:* Each object is expanded by the number of pixels you
   specify. Two objects are neighbors if they have adjacent pixels after
   expansion. Note that *all* objects are expanded by this amount (e.g., 
   if this distance is set to 10, a pair of objects will count as 
   neighbors if their edges are 20 pixels apart or closer).

For *{D_ADJACENT}* and *{D_EXPAND}*, the
*{M_PERCENT_TOUCHING}* measurement is the percentage of pixels on
the boundary of an object that touch adjacent objects. For
*{D_WITHIN}*, two objects are touching if any of their boundary
pixels are adjacent after expansion and *{M_PERCENT_TOUCHING}*
measures the percentage of boundary pixels of an *expanded* object that
touch adjacent objects.
""".format(
    **{
        "D_ADJACENT": DistanceMethod.ADJACENT.value,
        "D_EXPAND": DistanceMethod.EXPAND.value,
        "D_WITHIN": DistanceMethod.WITHIN.value,
        "M_PERCENT_TOUCHING": Measurement.PERCENT_TOUCHING.value,
    }
),
        )

        self.distance = Integer(
            "Neighbor distance",
            5,
            1,
            doc="""\
*(Used only when “%(D_WITHIN)s” is selected)*

The Neighbor distance is the number of pixels that each object is
expanded for the neighbor calculation. Expanded objects that touch are
considered neighbors.
""".format(
    **{
        "D_WITHIN": DistanceMethod.WITHIN.value,
    }
),
        )

        self.wants_count_image = Binary(
            "Retain the image of objects colored by numbers of neighbors?",
            False,
            doc="""\
An output image showing the input objects colored by numbers of
neighbors may be retained. A colormap of your choice shows how many
neighbors each object has. The background is set to -1. Objects are
colored with an increasing color value corresponding to the number of
neighbors, such that objects with no neighbors are given a color
corresponding to 0. Use the **SaveImages** module to save this image to
a file.""",
        )

        self.count_image_name = ImageName(
            "Name the output image",
            "ObjectNeighborCount",
            doc="""\
*(Used only if the image of objects colored by numbers of neighbors is
to be retained for later use in the pipeline)*

Specify a name that will allow the image of objects colored by numbers
of neighbors to be selected later in the pipeline.""",
        )

        self.count_colormap = Colormap(
            "Select colormap",
            value="Blues",
            doc="""\
*(Used only if the image of objects colored by numbers of neighbors is
to be retained for later use in the pipeline)*

Select the colormap to use to color the neighbor number image. All
available colormaps can be seen `here`_.

.. _here: http://matplotlib.org/examples/color/colormaps_reference.html""",
        )

        self.wants_percent_touching_image = Binary(
            "Retain the image of objects colored by percent of touching pixels?",
            False,
            doc="""\
Select *Yes* to keep an image of the input objects colored by the
percentage of the boundary touching their neighbors. A colormap of your
choice is used to show the touching percentage of each object. Use the
**SaveImages** module to save this image to a file.
"""
            % globals(),
        )

        self.touching_image_name = ImageName(
            "Name the output image",
            "PercentTouching",
            doc="""\
*(Used only if the image of objects colored by percent touching is to be
retained for later use in the pipeline)*

Specify a name that will allow the image of objects colored by percent
of touching pixels to be selected later in the pipeline.""",
        )

        self.touching_colormap = Colormap(
            "Select colormap",
            value="Oranges",
            doc="""\
*(Used only if the image of objects colored by percent touching is to be
retained for later use in the pipeline)*

Select the colormap to use to color the percent touching image. All
available colormaps can be seen `here`_.

.. _here: http://matplotlib.org/examples/color/colormaps_reference.html""",
        )

        self.wants_excluded_objects = Binary(
            "Consider objects discarded for touching image border?",
            True,
            doc="""\
When set to *{YES}*, objects which were previously discarded for touching
the image borders will be considered as potential object neighbours in this
analysis. You may want to disable this if using object sets which were
further filtered, since those filters won't have been applied to the
previously discarded objects.""".format(
                **{"YES": "Yes"}
            ),
        )

    def settings(self):
        return [
            self.object_name,
            self.neighbors_name,
            self.distance_method,
            self.distance,
            self.wants_excluded_objects,
            self.wants_count_image,
            self.count_image_name,
            self.count_colormap,
            self.wants_percent_touching_image,
            self.touching_image_name,
            self.touching_colormap,
        ]

    def visible_settings(self):
        result = [self.object_name, self.neighbors_name, self.distance_method]
        if self.distance_method == DistanceMethod.WITHIN:
            result += [self.distance]
        result += [self.wants_excluded_objects, self.wants_count_image]
        if self.wants_count_image.value:
            result += [self.count_image_name, self.count_colormap]
        result += [self.wants_percent_touching_image]
        if self.wants_percent_touching_image.value:
            result += [self.touching_image_name, self.touching_colormap]
        return result

    @property
    def neighbors_are_objects(self):
        """True if the neighbors are taken from the same object set as objects"""
        return self.object_name.value == self.neighbors_name.value



    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.object_name.value)
        # labels: ObjectSegmentation = objects.small_removed_segmented
        objects_small_removed_segmented: ObjectSegmentation = objects.small_removed_segmented
        kept_labels: ObjectSegmentation = objects.segmented
        assert isinstance(objects, Objects)
        has_pixels = objects.areas > 0

        neighbor_objects = workspace.object_set.get_objects(self.neighbors_name.value)
        # neighbor_labels: ObjectSegmentation = neighbor_objects.small_removed_segmented
        neighbor_small_removed_segmented: ObjectSegmentation = neighbor_objects.small_removed_segmented
        neighbor_kept_labels: ObjectSegmentation = neighbor_objects.segmented
        assert isinstance(neighbor_objects, Objects)
       
        dimensions = len(objects.shape)


        (
            measurements,
            (first_objects, second_objects),
            expanded_labels,
        ) = measure_object_neighbors( 
            objects_small_removed_segmented, 
            kept_labels,
            neighbor_small_removed_segmented, 
            neighbor_kept_labels,
            self.object_name.value,
            self.neighbors_name.value,
            self.neighbors_are_objects,
            dimensions, 
            self.distance.value,
            self.distance_method.value, 
            self.wants_excluded_objects.value,
        )


        #
        # Record the measurements
        #
        assert isinstance(workspace, Workspace)
        m = workspace.measurements
        assert isinstance(m, Measurements)
        image_set = workspace.image_set
        
        # Record Image Measurements
        for feature_name, value in measurements.image.items():
            m.add_image_measurement(feature_name, value)
        
        # Record Object Measurements
        for object_name, features in measurements.objects.items():
            for feature_name, data in features.items():
                m.add_measurement(object_name, feature_name, data)

        if len(first_objects) > 0:
            m.add_relate_measurement(
                self.module_num,
                NEIGHBORS,
                self.object_name.value,
                self.object_name.value if self.neighbors_are_objects else self.neighbors_name.value,
                m.image_set_number * numpy.ones(first_objects.shape, int),
                first_objects,
                m.image_set_number * numpy.ones(second_objects.shape, int),
                second_objects,
            )

        labels = kept_labels
        neighbor_labels = neighbor_kept_labels
        
        # Retrieve data for display
        neighbor_count_feature = self.get_measurement_name(Measurement.NUMBER_OF_NEIGHBORS.value)
        neighbor_count = measurements.objects[self.object_name.value][neighbor_count_feature]
        
        percent_touching_feature = self.get_measurement_name(Measurement.PERCENT_TOUCHING.value)
        percent_touching = measurements.objects[self.object_name.value][percent_touching_feature]

        neighbor_count_image = numpy.zeros(labels.shape, int)
        object_mask = objects.segmented != 0
        object_indexes = objects.segmented[object_mask] - 1
        neighbor_count_image[object_mask] = neighbor_count[object_indexes]
        workspace.display_data.neighbor_count_image = neighbor_count_image

        percent_touching_image = numpy.zeros(labels.shape)
        percent_touching_image[object_mask] = percent_touching[object_indexes]
        workspace.display_data.percent_touching_image = percent_touching_image

        image_set = workspace.image_set
        if self.wants_count_image.value:
            neighbor_cm_name = self.count_colormap.value
            neighbor_cm = get_colormap(neighbor_cm_name)
            sm = matplotlib.cm.ScalarMappable(cmap=neighbor_cm)
            img = sm.to_rgba(neighbor_count_image)[:, :, :3]
            img[:, :, 0][~object_mask] = 0
            img[:, :, 1][~object_mask] = 0
            img[:, :, 2][~object_mask] = 0
            count_image = Image(img, masking_objects=objects)
            image_set.add(self.count_image_name.value, count_image)
        else:
            neighbor_cm_name = "Blues"
            neighbor_cm = matplotlib.cm.get_cmap(neighbor_cm_name)
        if self.wants_percent_touching_image:
            percent_touching_cm_name = self.touching_colormap.value
            percent_touching_cm = get_colormap(percent_touching_cm_name)
            sm = matplotlib.cm.ScalarMappable(cmap=percent_touching_cm)
            img = sm.to_rgba(percent_touching_image)[:, :, :3]
            img[:, :, 0][~object_mask] = 0
            img[:, :, 1][~object_mask] = 0
            img[:, :, 2][~object_mask] = 0
            touching_image = Image(img, masking_objects=objects)
            image_set.add(self.touching_image_name.value, touching_image)
        else:
            percent_touching_cm_name = "Oranges"
            percent_touching_cm = matplotlib.cm.get_cmap(percent_touching_cm_name)

        if self.show_window:
            workspace.display_data.neighbor_cm_name = neighbor_cm_name
            workspace.display_data.percent_touching_cm_name = percent_touching_cm_name
            workspace.display_data.orig_labels = objects.segmented
            workspace.display_data.neighbor_labels = neighbor_labels
            workspace.display_data.expanded_labels = expanded_labels
            workspace.display_data.object_mask = object_mask
            workspace.display_data.dimensions = dimensions

    def display(self, workspace, figure):
        dimensions = workspace.display_data.dimensions
        figure.set_subplots((2, 2), dimensions=dimensions)
        figure.subplot_imshow_labels(
            0,
            0,
            workspace.display_data.orig_labels,
            "Original: %s" % self.object_name.value,
        )

        object_mask = workspace.display_data.object_mask
        expanded_labels = workspace.display_data.expanded_labels
        neighbor_count_image = workspace.display_data.neighbor_count_image
        neighbor_count_image[~object_mask] = -1
        neighbor_cm = get_colormap(workspace.display_data.neighbor_cm_name)
        neighbor_cm.set_under((0, 0, 0))
        neighbor_cm = matplotlib.cm.ScalarMappable(cmap=neighbor_cm)
        percent_touching_cm = get_colormap(
            workspace.display_data.percent_touching_cm_name
        )
        percent_touching_cm.set_under((0, 0, 0))
        percent_touching_image = workspace.display_data.percent_touching_image
        percent_touching_image[~object_mask] = -1
        percent_touching_cm = matplotlib.cm.ScalarMappable(cmap=percent_touching_cm)
        expandplot_position = 0
        if not self.neighbors_are_objects:
            # Display the neighbor object set, move expanded objects plot out of the way
            expandplot_position = 1
            figure.subplot_imshow_labels(
                1,
                0,
                workspace.display_data.neighbor_labels,
                "Neighbors: %s" % self.neighbors_name.value,
                sharexy=figure.subplot(0, 0),
            )
        if numpy.any(object_mask):
            figure.subplot_imshow(
                0,
                1,
                neighbor_count_image,
                "%s colored by # of neighbors" % self.object_name.value,
                colormap=neighbor_cm,
                colorbar=True,
                vmin=0,
                vmax=max(neighbor_count_image.max(), 1),
                normalize=False,
                sharexy=figure.subplot(0, 0),
            )
            if self.neighbors_are_objects:
                figure.subplot_imshow(
                    1,
                    1,
                    percent_touching_image,
                    "%s colored by pct touching" % self.object_name.value,
                    colormap=percent_touching_cm,
                    colorbar=True,
                    vmin=0,
                    vmax=max(percent_touching_image.max(), 1),
                    normalize=False,
                    sharexy=figure.subplot(0, 0),
                )
        else:
            # No objects - colorbar blows up.
            figure.subplot_imshow(
                0,
                1,
                neighbor_count_image,
                "%s colored by # of neighbors" % self.object_name.value,
                colormap=neighbor_cm,
                vmin=0,
                vmax=max(neighbor_count_image.max(), 1),
                sharexy=figure.subplot(0, 0),
            )
            if self.neighbors_are_objects:
                figure.subplot_imshow(
                    1,
                    1,
                    percent_touching_image,
                    "%s colored by pct touching" % self.object_name.value,
                    colormap=percent_touching_cm,
                    vmin=0,
                    vmax=max(neighbor_count_image.max(), 1),
                    sharexy=figure.subplot(0, 0),
                )

        if self.distance_method == DistanceMethod.EXPAND:
            figure.subplot_imshow_labels(
                1,
                expandplot_position,
                expanded_labels,
                "Expanded %s" % self.object_name.value,
                sharexy=figure.subplot(0, 0),
            )

    @property
    def all_features(self):
        return M_ALL

    def get_measurement_name(self, feature):
        if self.distance_method == DistanceMethod.EXPAND:
            scale = MeasurementScale.EXPANDED
        elif self.distance_method == DistanceMethod.WITHIN:
            scale = str(self.distance.value)
        elif self.distance_method == DistanceMethod.ADJACENT:
            scale = MeasurementScale.ADJACENT
        if self.neighbors_are_objects:
            return "_".join((C_NEIGHBORS, feature, scale))
        else:
            return "_".join((C_NEIGHBORS, feature, self.neighbors_name.value, scale))

    def get_measurement_columns(self, pipeline):
        """Return column definitions for measurements made by this module"""
        coltypes = dict(
            [
                (
                    feature,
                    COLTYPE_INTEGER
                    if feature
                    in (
                        Measurement.NUMBER_OF_NEIGHBORS,
                        Measurement.FIRST_CLOSEST_OBJECT_NUMBER,
                        Measurement.SECOND_CLOSEST_OBJECT_NUMBER,
                    )
                    else COLTYPE_FLOAT,
                )
                for feature in self.all_features
            ]
        )
        return [
            (
                self.object_name.value,
                self.get_measurement_name(feature_name),
                coltypes[feature_name],
            )
            for feature_name in self.all_features
        ]

    def get_object_relationships(self, pipeline):
        """Return column definitions for object relationships output by module"""
        objects_name = self.object_name.value
        if self.neighbors_are_objects:
            neighbors_name = objects_name
        else:
            neighbors_name = self.neighbors_name.value
        return [(NEIGHBORS, objects_name, neighbors_name, MCA_AVAILABLE_EACH_CYCLE,)]

    def get_categories(self, pipeline, object_name):
        if object_name == self.object_name:
            return [C_NEIGHBORS]
        return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.object_name and category == C_NEIGHBORS:
            return list(M_ALL)
        return []

    def get_measurement_objects(self, pipeline, object_name, category, measurement):
        if self.neighbors_are_objects or measurement not in self.get_measurements(
            pipeline, object_name, category
        ):
            return []
        return [self.neighbors_name.value]

    def get_measurement_scales(
        self, pipeline, object_name, category, measurement, image_name
    ):
        if measurement in self.get_measurements(pipeline, object_name, category):
            if self.distance_method == DistanceMethod.EXPAND:
                return [MeasurementScale.EXPANDED]
            elif self.distance_method == DistanceMethod.ADJACENT:
                return [MeasurementScale.ADJACENT]
            elif self.distance_method == DistanceMethod.WITHIN:
                return [str(self.distance.value)]
            else:
                raise ValueError(
                    "Unknown distance method: %s" % self.distance_method.value
                )
        return []

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            # Added neighbor objects
            # To upgrade, repeat object_name twice
            #
            setting_values = setting_values[:1] * 2 + setting_values[1:]
            variable_revision_number = 2
        if variable_revision_number == 2:
            # Added border object exclusion
            setting_values = setting_values[:4] + [True] + setting_values[4:]
            variable_revision_number = 3
        return setting_values, variable_revision_number

    def volumetric(self):
        return True


def get_colormap(name):
    """Get colormap, accounting for possible request for default"""
    if name == "Default":
        name = get_default_colormap()
    return matplotlib.cm.get_cmap(name)
