import centrosome.cpmorphology
import centrosome.propagate
import numpy
import scipy.ndimage
import skimage.segmentation
from cellprofiler_core.constants.measurement import (
    FF_CHILDREN_COUNT,
    FF_PARENT,
    FTR_CENTER_Z,
    FTR_CENTER_Y,
    FTR_CENTER_X,
    C_LOCATION,
    C_NUMBER,
    FTR_OBJECT_NUMBER,
    C_PARENT,
    C_CHILDREN,
    FF_COUNT,
    C_COUNT,
)
from cellprofiler_core.module.image_segmentation import ObjectProcessing
from cellprofiler_core.object import Objects
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import Integer, Float, LabelName
from cellprofiler_core.utilities.core.object import size_similarly

from cellprofiler.modules import _help, threshold

__doc__ = """\
IdentifySecondaryObjects
========================

**IdentifySecondaryObjects** identifies objects (e.g., cells)
using objects identified by another module (e.g., nuclei) as a starting
point.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

See also
^^^^^^^^

See also the other **Identify** modules.

What is a secondary object?
^^^^^^^^^^^^^^^^^^^^^^^^^^^

{DEFINITION_OBJECT}

We define an
object as *secondary* when it can be found in an image by using another
cellular feature as a reference for guiding detection.

For densely-packed cells (such as those in a confluent monolayer),
determining the cell borders using a cell body stain can be quite
difficult since they often have irregular intensity patterns and are
lower-contrast with more diffuse staining. In addition, cells often
touch their neighbors making it harder to delineate the cell borders. It
is often easier to identify an organelle which is well separated
spatially (such as the nucleus) as an object first and then use that
object to guide the detection of the cell borders. See the
**IdentifyPrimaryObjects** module for details on how to identify a
primary object.

In order to identify the edges of secondary objects, this module
performs two tasks:

#. Finds the dividing lines between secondary objects that touch each
   other.
#. Finds the dividing lines between the secondary objects and the
   background of the image. In most cases, this is done by thresholding
   the image stained for the secondary objects.

What do I need as input?
^^^^^^^^^^^^^^^^^^^^^^^^

This module identifies secondary objects based on two types of input:

#. An *object* (e.g., nuclei) identified from a prior module. These are
   typically produced by an **IdentifyPrimaryObjects** module, but any
   object produced by another module may be selected for this purpose.
#. (*optional*) An *image* highlighting the image features defining the edges of the
   secondary objects (e.g., cell edges).
   This is typically a fluorescent stain for the cell body, membrane or
   cytoskeleton (e.g., phalloidin staining for actin). However, any
   image that produces these features can be used for this purpose. For
   example, an image processing module might be used to transform a
   brightfield image into one that captures the characteristics of a
   cell body fluorescent stain. This input is optional because you can
   instead define secondary objects as a fixed distance around each
   primary object.

What do I get as output?
^^^^^^^^^^^^^^^^^^^^^^^^

A set of secondary objects are produced by this module, which can be
used in downstream modules for measurement purposes or other operations.
Because each primary object is used as the starting point for producing
a corresponding secondary object, keep in mind the following points:

-  The primary object will always be completely contained within a
   secondary object. For example, nuclei are completely enclosed within
   cells identified by actin staining.
-  There will always be at most one secondary object for each primary
   object.

Once the module has finished processing, the module display window will
show the following panels;
note that these are just for display: you must use the **SaveImages**
module if you would like to save any of these images to the hard drive
(as well, the **OverlayOutlines** module or **ConvertObjectsToImage**
modules might be needed):

-  *Upper left:* The raw, original image.
-  *Upper right:* The identified objects shown as a color image where
   connected pixels that belong to the same object are assigned the same
   color (*label image*). Note that assigned colors
   are arbitrary; they are used simply to help you distinguish the
   various objects.
-  *Lower left:* The raw image overlaid with the colored outlines of the
   identified secondary objects. The objects are shown with the
   following colors:

   -  Magenta: Secondary objects
   -  Green: Primary objects

   If you need to change the color defaults, you can make adjustments in
   *File > Preferences*.
-  *Lower right:* A table showing some of the settings you chose,
   as well as those calculated by the module in order to produce
   the objects shown.

{HELP_ON_SAVING_OBJECTS}

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Image measurements:**

-  *Count:* The number of secondary objects identified.
-  *OriginalThreshold:* The global threshold for the image.
-  *FinalThreshold:* For the global threshold methods, this value is the
   same as *OriginalThreshold*. For the adaptive or per-object methods,
   this value is the mean of the local thresholds.
-  *WeightedVariance:* The sum of the log-transformed variances of the
   foreground and background pixels, weighted by the number of pixels in
   each distribution.
-  *SumOfEntropies:* The sum of entropies computed from the foreground
   and background distributions.

**Object measurements:**

-  *Parent:* The identity of the primary object associated with each
   secondary object.
-  *Location\_X, Location\_Y:* The pixel (X,Y) coordinates of the center
   of mass of the identified secondary objects.

""".format(
    **{
        "DEFINITION_OBJECT": _help.DEFINITION_OBJECT,
        "HELP_ON_SAVING_OBJECTS": _help.HELP_ON_SAVING_OBJECTS,
    }
)

M_PROPAGATION = "Propagation"
M_WATERSHED_G = "Watershed - Gradient"
M_WATERSHED_I = "Watershed - Image"
M_DISTANCE_N = "Distance - N"
M_DISTANCE_B = "Distance - B"

"""# of setting values other than thresholding ones"""
N_SETTING_VALUES = 10

"""Parent (seed) relationship of input objects to output objects"""
R_PARENT = "Parent"


class IdentifySecondaryObjects(ObjectProcessing):
    module_name = "IdentifySecondaryObjects"

    variable_revision_number = 10

    category = "Object Processing"

    def __init__(self):
        self.threshold = threshold.Threshold()

        super(IdentifySecondaryObjects, self).__init__()

    def volumetric(self):
        return False

    def create_settings(self):
        super(IdentifySecondaryObjects, self).create_settings()

        self.x_name.text = "Select the input objects"

        self.x_name.doc = """\
What did you call the objects you want to use as primary objects ("seeds") to identify a secondary
object around each one? By definition, each primary object must be associated with exactly one
secondary object and completely contained within it."""

        self.y_name.text = "Name the objects to be identified"

        self.y_name.doc = "Enter the name that you want to call the objects identified by this module."

        self.method = Choice(
            "Select the method to identify the secondary objects",
            [M_PROPAGATION, M_WATERSHED_G, M_WATERSHED_I, M_DISTANCE_N, M_DISTANCE_B],
            M_PROPAGATION,
            doc="""\
There are several methods available to find the dividing lines between
secondary objects that touch each other:

-  *{M_PROPAGATION:s}:* This method will find dividing lines between
   clumped objects where the image stained for secondary objects shows a
   change in staining (i.e., either a dimmer or a brighter line).
   Smoother lines work better, but unlike the Watershed method, small
   gaps are tolerated. This method is considered an improvement on the
   traditional *Watershed* method. The dividing lines between objects
   are determined by a combination of the distance to the nearest
   primary object and intensity gradients. This algorithm uses local
   image similarity to guide the location of boundaries between cells.
   Boundaries are preferentially placed where the image’s local
   appearance changes perpendicularly to the boundary (*Jones et al,
   2005*).

   |image0| The {M_PROPAGATION:s} algorithm is the default approach for secondary object
   creation. Each primary object is a "seed" for its corresponding
   secondary object, guided by the input
   image and limited to the foreground region as determined by the chosen
   thresholding method. λ is a regularization parameter; see the help for
   the setting for more details. Propagation of secondary object labels is
   by the shortest path to an adjacent primary object from the starting
   (“seeding”) primary object. The seed-to-pixel distances are calculated
   as the sum of absolute differences in a 3x3 (8-connected) image
   neighborhood, combined with λ via sqrt(differences\ :sup:`2` +
   λ\ :sup:`2`).
-  *{M_WATERSHED_G:s}:* This method uses the watershed algorithm
   (*Vincent and Soille, 1991*) to assign pixels to the primary objects
   which act as seeds for the watershed. In this variant, the watershed
   algorithm operates on the Sobel transformed image which computes an
   intensity gradient. This method works best when the image intensity
   drops off or increases rapidly near the boundary between cells.
-  *{M_WATERSHED_I:s}:* This method is similar to the above, but it
   uses the inverted intensity of the image for the watershed. The areas
   of lowest intensity will be detected as the boundaries between cells. This
   method works best when there is a saddle of relatively low intensity
   at the cell-cell boundary.
-  *Distance:* In this method, the edges of the primary objects are
   expanded a specified distance to create the secondary objects. For
   example, if nuclei are labeled but there is no stain to help locate
   cell edges, the nuclei can simply be expanded in order to estimate
   the cell’s location. This is often called the “doughnut” or “annulus”
   or “ring” approach for identifying the cytoplasm. There are two
   methods that can be used:

   -  *{M_DISTANCE_N:s}*: In this method, the image of the secondary
      staining is not used at all; the expanded objects are the final
      secondary objects.
   -  *{M_DISTANCE_B:s}*: Thresholding of the secondary staining image
      is used to eliminate background regions from the secondary
      objects. This allows the extent of the secondary objects to be
      limited to a certain distance away from the edge of the primary
      objects without including regions of background.

References
^^^^^^^^^^

Jones TR, Carpenter AE, Golland P (2005) “Voronoi-Based Segmentation of
Cells on Image Manifolds”, *ICCV Workshop on Computer Vision for
Biomedical Image Applications*, 535-543. (`link1`_)

Vincent L, Soille P (1991) "Watersheds in Digital Spaces: An Efficient
Algorithm Based on Immersion Simulations", *IEEE Transactions on Pattern
Analysis and Machine Intelligence*, Vol. 13, No. 6, 583-598 (`link2`_)

.. _link1: http://people.csail.mit.edu/polina/papers/JonesCarpenterGolland_CVBIA2005.pdf
.. _link2: http://www.cse.msu.edu/~cse902/S03/watershed.pdf

.. |image0| image:: {TECH_NOTE_ICON}
""".format(
                **{
                    "M_PROPAGATION": M_PROPAGATION,
                    "M_WATERSHED_G": M_WATERSHED_G,
                    "M_WATERSHED_I": M_WATERSHED_I,
                    "M_DISTANCE_N": M_DISTANCE_N,
                    "M_DISTANCE_B": M_DISTANCE_B,
                    "TECH_NOTE_ICON": _help.TECH_NOTE_ICON,
                }
            ),
        )

        self.image_name = ImageSubscriber(
            "Select the input image",
            "None",
            doc="""\
The selected image will be used to find the edges of the secondary
objects. For *{M_DISTANCE_N:s}* this will not affect object
identification, only the module's display.
""".format(
                **{"M_DISTANCE_N": M_DISTANCE_N}
            ),
        )

        self.distance_to_dilate = Integer(
            "Number of pixels by which to expand the primary objects",
            10,
            minval=1,
            doc="""\
*(Used only if "{M_DISTANCE_B:s}" or "{M_DISTANCE_N:s}" method is selected)*

This option allows you to define the number of pixels by which the primary objects
will be expanded. This option becomes useful in situations when no staining was
used to define cell cytoplasm but the cell edges must be defined for further
measurements.
""".format(
                **{"M_DISTANCE_N": M_DISTANCE_N, "M_DISTANCE_B": M_DISTANCE_B}
            ),
        )

        self.regularization_factor = Float(
            "Regularization factor",
            0.05,
            minval=0,
            doc="""\
*(Used only if "{M_PROPAGATION:s}" method is selected)*

The regularization factor λ can be anywhere in the range 0 to
infinity. This method takes two factors into account when deciding
where to draw the dividing line between two touching secondary
objects: the distance to the nearest primary object, and the intensity
of the secondary object image. The regularization factor controls the
balance between these two considerations:

-  A λ value of 0 means that the distance to the nearest primary object
   is ignored and the decision is made entirely on the intensity
   gradient between the two competing primary objects.
-  Larger values of λ put more and more weight on the distance between
   the two objects. This relationship is such that small changes in λ
   will have fairly different results (e.g., 0.01 vs 0.001). However, the
   intensity image is almost completely ignored at λ much greater than
   1.
-  At infinity, the result will look like {M_DISTANCE_B:s}, masked to
   the secondary staining image.
""".format(
                **{"M_PROPAGATION": M_PROPAGATION, "M_DISTANCE_B": M_DISTANCE_B}
            ),
        )

        self.wants_discard_edge = Binary(
            "Discard secondary objects touching the border of the image?",
            False,
            doc="""\
Select *{YES:s}* to discard secondary objects that touch the image
border. Select *{NO:s}* to retain objects regardless of whether they
touch the image edge or not.

Note: the objects are discarded with respect to downstream measurement
modules, but they are retained in memory as “Unedited objects”; this
allows them to be considered in downstream modules that modify the
segmentation.
""".format(
                **{"YES": "Yes", "NO": "No"}
            ),
        )

        self.fill_holes = Binary(
            "Fill holes in identified objects?",
            True,
            doc="""\
Select *{YES:s}* to fill any holes inside objects.

Please note that if an object is located within a hole and this option is
enabled, the object will be lost when the hole is filled in.
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.wants_discard_primary = Binary(
            "Discard the associated primary objects?",
            False,
            doc="""\
*(Used only if discarding secondary objects touching the image
border)*

It might be appropriate to discard the primary object for any
secondary object that touches the edge of the image.

Select *{YES:s}* to create a new set of objects that are identical to
the original set of primary objects, minus the objects for which the
associated secondary object touches the image edge.
""".format(
                **{"YES": "Yes"}
            ),
        )

        self.new_primary_objects_name = LabelName(
            "Name the new primary objects",
            "FilteredNuclei",
            doc="""\
*(Used only if associated primary objects are discarded)*

You can name the primary objects that remain after the discarding step.
These objects will all have secondary objects that do not touch the edge
of the image. Note that any primary object whose secondary object
touches the edge will be retained in memory as an “unedited object”;
this allows them to be considered in downstream modules that modify the
segmentation.""",
        )

        self.threshold_setting_version = Integer(
            "Threshold setting version", value=self.threshold.variable_revision_number
        )

        self.threshold.create_settings()

        self.threshold.threshold_smoothing_scale.value = 0

    def settings(self):
        settings = super(IdentifySecondaryObjects, self).settings()

        return (
            settings
            + [
                self.method,
                self.image_name,
                self.distance_to_dilate,
                self.regularization_factor,
                self.wants_discard_edge,
                self.wants_discard_primary,
                self.new_primary_objects_name,
                self.fill_holes,
            ]
            + [self.threshold_setting_version]
            + self.threshold.settings()[2:]
        )

    def visible_settings(self):
        visible_settings = [self.image_name]

        visible_settings += super(IdentifySecondaryObjects, self).visible_settings()

        visible_settings += [self.method]

        if self.method != M_DISTANCE_N:
            visible_settings += self.threshold.visible_settings()[2:]

        if self.method in (M_DISTANCE_B, M_DISTANCE_N):
            visible_settings += [self.distance_to_dilate]
        elif self.method == M_PROPAGATION:
            visible_settings += [self.regularization_factor]

        visible_settings += [self.fill_holes, self.wants_discard_edge]

        if self.wants_discard_edge:
            visible_settings += [self.wants_discard_primary]

            if self.wants_discard_primary:
                visible_settings += [self.new_primary_objects_name]

        return visible_settings

    def help_settings(self):
        help_settings = [self.x_name, self.y_name, self.method, self.image_name]

        help_settings += self.threshold.help_settings()[2:]

        help_settings += [
            self.distance_to_dilate,
            self.regularization_factor,
            self.fill_holes,
            self.wants_discard_edge,
            self.wants_discard_primary,
            self.new_primary_objects_name,
        ]

        return help_settings

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number < 9:
            raise NotImplementedError(
                "Automatic upgrade for this module is not supported in CellProfiler 3."
            )

        if variable_revision_number == 9:
            setting_values = (
                setting_values[:6] + setting_values[8:11] + setting_values[13:]
            )

            variable_revision_number = 10

        threshold_setting_values = setting_values[N_SETTING_VALUES:]

        threshold_settings_version = int(threshold_setting_values[0])

        if threshold_settings_version < 4:
            threshold_setting_values = self.threshold.upgrade_threshold_settings(
                threshold_setting_values
            )

            threshold_settings_version = 9

        (
            threshold_upgrade_settings,
            threshold_settings_version,
        ) = self.threshold.upgrade_settings(
            ["None", "None"] + threshold_setting_values[1:],
            threshold_settings_version,
            "Threshold",
        )

        threshold_upgrade_settings = [
            str(threshold_settings_version)
        ] + threshold_upgrade_settings[2:]

        setting_values = setting_values[:N_SETTING_VALUES] + threshold_upgrade_settings

        return setting_values, variable_revision_number

    def run(self, workspace):
        image_name = self.image_name.value
        image = workspace.image_set.get_image(image_name, must_be_grayscale=True)
        workspace.display_data.statistics = []
        img = image.pixel_data
        mask = image.mask
        objects = workspace.object_set.get_objects(self.x_name.value)
        if img.shape != objects.shape:
            raise ValueError(
                "This module requires that the input image and object sets are the same size.\n"
                "The %s image and %s objects are not (%s vs %s).\n"
                "If they are paired correctly you may want to use the Resize, ResizeObjects or "
                "Crop module(s) to make them the same size."
                % (image_name, self.x_name.value, img.shape, objects.shape,)
            )
        global_threshold = None
        if self.method == M_DISTANCE_N:
            has_threshold = False
        else:
            thresholded_image, global_threshold, sigma = self._threshold_image(
                image_name, workspace
            )
            workspace.display_data.global_threshold = global_threshold
            workspace.display_data.threshold_sigma = sigma
            has_threshold = True

        #
        # Get the following labels:
        # * all edited labels
        # * labels touching the edge, including small removed
        #
        labels_in = objects.unedited_segmented.copy()
        labels_touching_edge = numpy.hstack(
            (labels_in[0, :], labels_in[-1, :], labels_in[:, 0], labels_in[:, -1])
        )
        labels_touching_edge = numpy.unique(labels_touching_edge)
        is_touching = numpy.zeros(numpy.max(labels_in) + 1, bool)
        is_touching[labels_touching_edge] = True
        is_touching = is_touching[labels_in]

        labels_in[(~is_touching) & (objects.segmented == 0)] = 0
        #
        # Stretch the input labels to match the image size. If there's no
        # label matrix, then there's no label in that area.
        #
        if tuple(labels_in.shape) != tuple(img.shape):
            tmp = numpy.zeros(img.shape, labels_in.dtype)
            i_max = min(img.shape[0], labels_in.shape[0])
            j_max = min(img.shape[1], labels_in.shape[1])
            tmp[:i_max, :j_max] = labels_in[:i_max, :j_max]
            labels_in = tmp

        if self.method in (M_DISTANCE_B, M_DISTANCE_N):
            if self.method == M_DISTANCE_N:
                distances, (i, j) = scipy.ndimage.distance_transform_edt(
                    labels_in == 0, return_indices=True
                )
                labels_out = numpy.zeros(labels_in.shape, int)
                dilate_mask = distances <= self.distance_to_dilate.value
                labels_out[dilate_mask] = labels_in[i[dilate_mask], j[dilate_mask]]
            else:
                labels_out, distances = centrosome.propagate.propagate(
                    img, labels_in, thresholded_image, 1.0
                )
                labels_out[distances > self.distance_to_dilate.value] = 0
                labels_out[labels_in > 0] = labels_in[labels_in > 0]
            if self.fill_holes:
                label_mask = labels_out == 0
                small_removed_segmented_out = centrosome.cpmorphology.fill_labeled_holes(
                    labels_out, mask=label_mask
                )
            else:
                small_removed_segmented_out = labels_out
            #
            # Create the final output labels by removing labels in the
            # output matrix that are missing from the segmented image
            #
            segmented_labels = objects.segmented
            segmented_out = self.filter_labels(
                small_removed_segmented_out, objects, workspace
            )
        elif self.method == M_PROPAGATION:
            labels_out, distance = centrosome.propagate.propagate(
                img, labels_in, thresholded_image, self.regularization_factor.value
            )
            if self.fill_holes:
                label_mask = labels_out == 0
                small_removed_segmented_out = centrosome.cpmorphology.fill_labeled_holes(
                    labels_out, mask=label_mask
                )
            else:
                small_removed_segmented_out = labels_out.copy()
            segmented_out = self.filter_labels(
                small_removed_segmented_out, objects, workspace
            )
        elif self.method == M_WATERSHED_G:
            #
            # First, apply the sobel filter to the image (both horizontal
            # and vertical). The filter measures gradient.
            #
            sobel_image = numpy.abs(scipy.ndimage.sobel(img))
            #
            # Combine the image mask and threshold to mask the watershed
            #
            watershed_mask = numpy.logical_or(thresholded_image, labels_in > 0)
            watershed_mask = numpy.logical_and(watershed_mask, mask)

            #
            # Perform the first watershed
            #

            labels_out = skimage.segmentation.watershed(
                connectivity=numpy.ones((3, 3), bool),
                image=sobel_image,
                markers=labels_in,
                mask=watershed_mask,
            )

            if self.fill_holes:
                label_mask = labels_out == 0
                small_removed_segmented_out = centrosome.cpmorphology.fill_labeled_holes(
                    labels_out, mask=label_mask
                )
            else:
                small_removed_segmented_out = labels_out.copy()
            segmented_out = self.filter_labels(
                small_removed_segmented_out, objects, workspace
            )
        elif self.method == M_WATERSHED_I:
            #
            # invert the image so that the maxima are filled first
            # and the cells compete over what's close to the threshold
            #
            inverted_img = 1 - img
            #
            # Same as above, but perform the watershed on the original image
            #
            watershed_mask = numpy.logical_or(thresholded_image, labels_in > 0)
            watershed_mask = numpy.logical_and(watershed_mask, mask)
            #
            # Perform the watershed
            #

            labels_out = skimage.segmentation.watershed(
                connectivity=numpy.ones((3, 3), bool),
                image=inverted_img,
                markers=labels_in,
                mask=watershed_mask,
            )

            if self.fill_holes:
                label_mask = labels_out == 0
                small_removed_segmented_out = centrosome.cpmorphology.fill_labeled_holes(
                    labels_out, mask=label_mask
                )
            else:
                small_removed_segmented_out = labels_out
            segmented_out = self.filter_labels(
                small_removed_segmented_out, objects, workspace
            )

        if self.wants_discard_edge:
            lookup = scipy.ndimage.maximum(
                segmented_out,
                objects.segmented,
                list(range(numpy.max(objects.segmented) + 1)),
            )
            lookup = centrosome.cpmorphology.fixup_scipy_ndimage_result(lookup)
            lookup[0] = 0
            lookup[lookup != 0] = numpy.arange(numpy.sum(lookup != 0)) + 1
            segmented_labels = lookup[objects.segmented]
            segmented_out = lookup[segmented_out]

        
            if self.wants_discard_primary:
                #
                # Make a new primary object
                #
                new_objects = Objects()
                new_objects.segmented = segmented_labels
                if objects.has_unedited_segmented:
                    new_objects.unedited_segmented = objects.unedited_segmented
                if objects.has_small_removed_segmented:
                    new_objects.small_removed_segmented = objects.small_removed_segmented
                new_objects.parent_image = objects.parent_image

        #
        # Add the objects to the object set
        #
        objects_out = Objects()
        objects_out.unedited_segmented = small_removed_segmented_out
        objects_out.small_removed_segmented = small_removed_segmented_out
        objects_out.segmented = segmented_out
        objects_out.parent_image = image
        objname = self.y_name.value
        workspace.object_set.add_objects(objects_out, objname)
        object_count = numpy.max(segmented_out)
        #
        # Add measurements
        #
        measurements = workspace.measurements
        super(IdentifySecondaryObjects, self).add_measurements(workspace)
        #
        # Relate the secondary objects to the primary ones and record
        # the relationship.
        #
        children_per_parent, parents_of_children = objects.relate_children(objects_out)
        measurements.add_measurement(
            self.x_name.value, FF_CHILDREN_COUNT % objname, children_per_parent,
        )
        measurements.add_measurement(
            objname, FF_PARENT % self.x_name.value, parents_of_children,
        )
        image_numbers = (
            numpy.ones(len(parents_of_children), int) * measurements.image_set_number
        )
        mask = parents_of_children > 0
        measurements.add_relate_measurement(
            self.module_num,
            R_PARENT,
            self.x_name.value,
            self.y_name.value,
            image_numbers[mask],
            parents_of_children[mask],
            image_numbers[mask],
            numpy.arange(1, len(parents_of_children) + 1)[mask],
        )
        #
        # If primary objects were created, add them
        #
        if self.wants_discard_edge and self.wants_discard_primary:
            workspace.object_set.add_objects(
                new_objects, self.new_primary_objects_name.value
            )
            super(IdentifySecondaryObjects, self).add_measurements(
                workspace,
                input_object_name=self.x_name.value,
                output_object_name=self.new_primary_objects_name.value,
            )

            children_per_parent, parents_of_children = new_objects.relate_children(
                objects_out
            )

            measurements.add_measurement(
                self.new_primary_objects_name.value,
                FF_CHILDREN_COUNT % objname,
                children_per_parent,
            )

            measurements.add_measurement(
                objname,
                FF_PARENT % self.new_primary_objects_name.value,
                parents_of_children,
            )

        if self.show_window:
            object_area = numpy.sum(segmented_out > 0)
            workspace.display_data.object_pct = (
                100 * object_area / numpy.product(segmented_out.shape)
            )
            workspace.display_data.img = img
            workspace.display_data.segmented_out = segmented_out
            workspace.display_data.primary_labels = objects.segmented
            workspace.display_data.global_threshold = global_threshold
            workspace.display_data.object_count = object_count

    def _threshold_image(self, image_name, workspace, automatic=False):
        image = workspace.image_set.get_image(image_name, must_be_grayscale=True)

        final_threshold, orig_threshold, guide_threshold, binary_image, sigma = self.threshold.get_threshold(
            image, workspace, automatic
        )

        self.threshold.add_threshold_measurements(
            self.y_name.value,
            workspace.measurements,
            final_threshold,
            orig_threshold,
            guide_threshold,
        )

        self.threshold.add_fg_bg_measurements(
            self.y_name.value, workspace.measurements, image, binary_image
        )

        return binary_image, numpy.mean(numpy.atleast_1d(final_threshold)), sigma

    def display(self, workspace, figure):
        object_pct = workspace.display_data.object_pct
        img = workspace.display_data.img
        primary_labels = workspace.display_data.primary_labels
        segmented_out = workspace.display_data.segmented_out
        global_threshold = workspace.display_data.global_threshold
        object_count = workspace.display_data.object_count
        statistics = workspace.display_data.statistics

        if global_threshold is not None:
            statistics.append(["Threshold", "%0.3g" % global_threshold])

        if object_count > 0:
            areas = scipy.ndimage.sum(
                numpy.ones(segmented_out.shape),
                segmented_out,
                numpy.arange(1, object_count + 1),
            )
            areas.sort()
            low_diameter = numpy.sqrt(float(areas[object_count // 10]) / numpy.pi) * 2
            median_diameter = numpy.sqrt(float(areas[object_count // 2]) / numpy.pi) * 2
            high_diameter = (
                numpy.sqrt(float(areas[object_count * 9 // 10]) / numpy.pi) * 2
            )
            statistics.append(["10th pctile diameter", "%.1f pixels" % low_diameter])
            statistics.append(["Median diameter", "%.1f pixels" % median_diameter])
            statistics.append(["90th pctile diameter", "%.1f pixels" % high_diameter])
            if self.method != M_DISTANCE_N:
                statistics.append(
                    [
                        "Thresholding filter size",
                        "%.1f" % workspace.display_data.threshold_sigma,
                    ]
                )
            statistics.append(["Area covered by objects", "%.1f %%" % object_pct])
        workspace.display_data.statistics = statistics

        figure.set_subplots((2, 2))
        title = "Input image, cycle #%d" % workspace.measurements.image_number
        figure.subplot_imshow_grayscale(0, 0, img, title)
        figure.subplot_imshow_labels(
            1,
            0,
            segmented_out,
            "%s objects" % self.y_name.value,
            sharexy=figure.subplot(0, 0),
        )

        cplabels = [
            dict(name=self.x_name.value, labels=[primary_labels]),
            dict(name=self.y_name.value, labels=[segmented_out]),
        ]
        title = "%s and %s outlines" % (self.x_name.value, self.y_name.value)
        figure.subplot_imshow_grayscale(
            0, 1, img, title=title, cplabels=cplabels, sharexy=figure.subplot(0, 0)
        )
        figure.subplot_table(
            1,
            1,
            [[x[1]] for x in workspace.display_data.statistics],
            row_labels=[x[0] for x in workspace.display_data.statistics],
        )

    def filter_labels(self, labels_out, objects, workspace):
        """Filter labels out of the output

        Filter labels that are not in the segmented input labels. Optionally
        filter labels that are touching the edge.

        labels_out - the unfiltered output labels
        objects    - the objects thing, containing both segmented and
                     small_removed labels
        """
        segmented_labels = objects.segmented
        max_out = numpy.max(labels_out)
        if max_out > 0:
            segmented_labels, m1 = size_similarly(labels_out, segmented_labels)
            segmented_labels[~m1] = 0
            lookup = scipy.ndimage.maximum(
                segmented_labels, labels_out, list(range(max_out + 1))
            )
            lookup = numpy.array(lookup, int)
            lookup[0] = 0
            segmented_labels_out = lookup[labels_out]
        else:
            segmented_labels_out = labels_out.copy()
        if self.wants_discard_edge:
            image = workspace.image_set.get_image(self.image_name.value)
            if image.has_mask:
                mask_border = image.mask & ~scipy.ndimage.binary_erosion(image.mask)
                edge_labels = segmented_labels_out[mask_border]
            else:
                edge_labels = numpy.hstack(
                    (
                        segmented_labels_out[0, :],
                        segmented_labels_out[-1, :],
                        segmented_labels_out[:, 0],
                        segmented_labels_out[:, -1],
                    )
                )
            edge_labels = numpy.unique(edge_labels)
            #
            # Make a lookup table that translates edge labels to zero
            # but translates everything else to itself
            #
            lookup = numpy.arange(max(max_out, numpy.max(segmented_labels)) + 1)
            lookup[edge_labels] = 0
            #
            # Run the segmented labels through this to filter out edge
            # labels
            segmented_labels_out = lookup[segmented_labels_out]

        return segmented_labels_out

    def is_object_identification_module(self):
        return True

    def get_measurement_columns(self, pipeline):
        if self.wants_discard_edge and self.wants_discard_primary:
            columns = super(IdentifySecondaryObjects, self).get_measurement_columns(
                pipeline,
                additional_objects=[
                    (self.x_name.value, self.new_primary_objects_name.value)
                ],
            )

            columns += [
                (
                    self.new_primary_objects_name.value,
                    FF_CHILDREN_COUNT % self.y_name.value,
                    "integer",
                ),
                (
                    self.y_name.value,
                    FF_PARENT % self.new_primary_objects_name.value,
                    "integer",
                ),
            ]
        else:
            columns = super(IdentifySecondaryObjects, self).get_measurement_columns(
                pipeline
            )

        if self.method != M_DISTANCE_N:
            columns += self.threshold.get_measurement_columns(
                pipeline, object_name=self.y_name.value
            )

        return columns

    def get_categories(self, pipeline, object_name):
        categories = super(IdentifySecondaryObjects, self).get_categories(
            pipeline, object_name
        )

        if self.method != M_DISTANCE_N:
            categories += self.threshold.get_categories(pipeline, object_name)

        if self.wants_discard_edge and self.wants_discard_primary:
            if object_name == self.new_primary_objects_name.value:
                # new_primary_objects_name objects has the same categories as y_name objects
                categories += super(IdentifySecondaryObjects, self).get_categories(
                    pipeline, self.y_name.value
                )

                categories += [C_CHILDREN]

        return categories

    def get_measurements(self, pipeline, object_name, category):
        measurements = super(IdentifySecondaryObjects, self).get_measurements(
            pipeline, object_name, category
        )

        if self.method.value != M_DISTANCE_N:
            measurements += self.threshold.get_measurements(
                pipeline, object_name, category
            )

        if self.wants_discard_edge and self.wants_discard_primary:
            if object_name == "Image" and category == C_COUNT:
                measurements += [self.new_primary_objects_name.value]

            if object_name == self.y_name.value and category == C_PARENT:
                measurements += [self.new_primary_objects_name.value]

            if object_name == self.new_primary_objects_name.value:
                if category == C_LOCATION:
                    measurements += [
                        FTR_CENTER_X,
                        FTR_CENTER_Y,
                        FTR_CENTER_Z,
                    ]

                if category == C_NUMBER:
                    measurements += [FTR_OBJECT_NUMBER]

                if category == C_PARENT:
                    measurements += [self.x_name.value]

            if category == C_CHILDREN:
                if object_name == self.x_name.value:
                    measurements += ["%s_Count" % self.new_primary_objects_name.value]

                if object_name == self.new_primary_objects_name.value:
                    measurements += ["%s_Count" % self.y_name.value]

        return measurements

    def get_measurement_objects(self, pipeline, object_name, category, measurement):
        threshold_measurements = self.threshold.get_measurements(
            pipeline, object_name, category
        )

        if self.method != M_DISTANCE_N and measurement in threshold_measurements:
            return [self.y_name.value]

        return []
