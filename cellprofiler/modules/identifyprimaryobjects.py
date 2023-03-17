import math

import cellprofiler_core.module.image_segmentation
import cellprofiler_core.object
import centrosome.cpmorphology
import centrosome.outline
import centrosome.propagate
import centrosome.threshold
import numpy
import scipy.ndimage
import scipy.sparse
import skimage.morphology
import skimage.segmentation
from cellprofiler_core.setting import Binary, Color
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.range import IntegerRange
from cellprofiler_core.setting.text import Integer, Float

import cellprofiler.gui.help
import cellprofiler.gui.help.content
from cellprofiler.modules import _help, threshold

__doc__ = """\
IdentifyPrimaryObjects
======================

**IdentifyPrimaryObjects** identifies biological objects of interest.
It requires grayscale images containing bright objects on a dark background.
Incoming images must be 2D (including 2D slices of 3D images);
please use the **Watershed** module for identification of objects in 3D.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

See also
^^^^^^^^

See also **IdentifySecondaryObjects**, **IdentifyTertiaryObjects**,
**IdentifyObjectsManually**, and **Watershed** (for segmentation of 3D objects).

What is a primary object?
^^^^^^^^^^^^^^^^^^^^^^^^^

{DEFINITION_OBJECT}

We define an object as *primary* when it can be found in an image without needing the
assistance of another cellular feature as a reference. For example:

-  The nuclei of cells are usually more easily identifiable than whole-
   cell stains due to their
   more uniform morphology, high contrast relative to the background
   when stained, and good separation between adjacent nuclei. These
   qualities typically make them appropriate candidates for primary
   object identification.
-  In contrast, whole-cell stains often yield irregular intensity patterns
   and are lower-contrast with more diffuse staining, making them more
   challenging to identify than nuclei without some supplemental image
   information being provided. In addition, cells often touch or even overlap
   their neighbors making it harder to delineate the cell borders. For
   these reasons, cell bodies are better suited for *secondary object*
   identification, because they are best identified by using a
   previously-identified primary object (i.e, the nuclei) as a
   reference. See the **IdentifySecondaryObjects** module for details on
   how to do this.

What do I need as input?
^^^^^^^^^^^^^^^^^^^^^^^^

To use this module, you will need to make sure that your input image has
the following qualities:

-  The image should be grayscale.
-  The foreground (i.e, regions of interest) are lighter than the
   background.
-  The image should be 2D. 2D slices of 3D images are acceptable if the
   image has not been loaded as volumetric in the **NamesAndTypes**
   module. For volumetric analysis
   of 3D images, please see the **Watershed** module.

If this is not the case, other modules can be used to pre-process the
images to ensure they are in the proper form:

-  If the objects in your images are dark on a light background, you
   should invert the images using the Invert operation in the
   **ImageMath** module.
-  If you are working with color images, they must first be converted to
   grayscale using the **ColorToGray** module.
-  If your images are brightfield/phase/DIC, they may be processed with the
   **EnhanceOrSuppressFeatures** module with its "*Texture*" or "*DIC*" settings.
-  If you struggle to find effective settings for this module, you may
   want to check our `tutorial`_ on preprocessing these images with
   ilastik prior to using them in CellProfiler.

What are the advanced settings?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**IdentifyPrimaryObjects** allows you to tweak your settings in many ways;
so many that it can often become confusing where you should start. This is
typically the most important but complex step in creating a good pipeline,
so do not be discouraged: other modules are easier to configure!
Using **IdentifyPrimaryObjects** with *'Use advanced settings?'* set to *'No'*
allows you to quickly try to identify your objects based only their typical size;
CellProfiler will then use its built-in defaults to decide how to set the
threshold and how to break clumped objects apart. If you are happy with the
results produced by the default settings, you can then move on to
construct the rest of your pipeline; if not, you can set
*'Use advanced settings?'* to *'Yes'* which will allow you to fully tweak and
customize all the settings.

What do I get as output?
^^^^^^^^^^^^^^^^^^^^^^^^

A set of primary objects are produced by this module, which can be used
in downstream modules for measurement purposes or other operations. See
the section "Measurements made by this module" below
for the measurements that are produced directly by this module. Once the module
has finished processing, the module display window will show the
following panels:

-  *Upper left:* The raw, original image.
-  *Upper right:* The identified objects shown as a color image where
   connected pixels that belong to the same object are assigned the same
   color (*label image*). Note that assigned colors
   are arbitrary; they are used simply to help you distinguish the
   various objects.
-  *Lower left:* The raw image overlaid with the colored outlines of the
   identified objects. Each object is assigned one of three (default)
   colors:

   -  Green: Acceptable; passed all criteria
   -  Magenta: Discarded based on size
   -  Yellow: Discarded due to touching the border

   If you need to change the color defaults, you can make adjustments in
   *File > Preferences*.
-  *Lower right:* A table showing some of the settings used by the module
   in order to produce the objects shown. Some of these are as you
   specified in settings; others are calculated by the module itself.

{HELP_ON_SAVING_OBJECTS}

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Image measurements:**

-  *Count:* The number of primary objects identified.
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

-  *Location\_X, Location\_Y:* The pixel (X,Y) coordinates of the
   primary object centroids. The centroid is calculated as the center of
   mass of the binary representation of the object.

Technical notes
^^^^^^^^^^^^^^^

CellProfiler contains a modular three-step strategy to identify objects
even if they touch each other ("declumping"). It is based on previously
published
algorithms (*Malpica et al., 1997; Meyer and Beucher, 1990; Ortiz de
Solorzano et al., 1999; Wahlby, 2003; Wahlby et al., 2004*). Choosing
different options for each of these three steps allows CellProfiler to
flexibly analyze a variety of different types of objects. The module has
many options, which vary in terms of speed and sophistication. More
detail can be found in the Settings section below. Here are the three
steps, using an example where nuclei are the primary objects:

#. CellProfiler determines whether a foreground region is an individual
   nucleus or two or more clumped nuclei.
#. The edges of nuclei are identified, using thresholding if the object
   is a single, isolated nucleus, and using more advanced options if the
   object is actually two or more nuclei that touch each other.
#. Some identified objects are discarded or merged together if they fail
   to meet certain your specified criteria. For example, partial objects
   at the border of the image can be discarded, and small objects can be
   discarded or merged with nearby larger ones. A separate module,
   **FilterObjects**, can further refine the identified nuclei, if
   desired, by excluding objects that are a particular size, shape,
   intensity, or texture.

References
^^^^^^^^^^

-  Malpica N, de Solorzano CO, Vaquero JJ, Santos, A, Vallcorba I,
   Garcia-Sagredo JM, del Pozo F (1997) “Applying watershed algorithms
   to the segmentation of clustered nuclei.” *Cytometry* 28, 289-297.
   (`link`_)
-  Meyer F, Beucher S (1990) “Morphological segmentation.” *J Visual
   Communication and Image Representation* 1, 21-46.
   (`link <https://doi.org/10.1016/1047-3203(90)90014-M>`__)
-  Ortiz de Solorzano C, Rodriguez EG, Jones A, Pinkel D, Gray JW, Sudar
   D, Lockett SJ. (1999) “Segmentation of confocal microscope images of
   cell nuclei in thick tissue sections.” *Journal of Microscopy-Oxford*
   193, 212-226.
   (`link <https://doi.org/10.1046/j.1365-2818.1999.00463.x>`__)
-  Wählby C (2003) *Algorithms for applied digital image cytometry*,
   Ph.D., Uppsala University, Uppsala.
-  Wählby C, Sintorn IM, Erlandsson F, Borgefors G, Bengtsson E. (2004)
   “Combining intensity, edge and shape information for 2D and 3D
   segmentation of cell nuclei in tissue sections.” *J Microsc* 215,
   67-76.
   (`link <https://doi.org/10.1111/j.0022-2720.2004.01338.x>`__)

.. _link: https://doi.org/10.1002/(SICI)1097-0320(19970801)28:4%3C289::AID-CYTO3%3E3.0.CO;2-7
.. _tutorial: http://blog.cellprofiler.org/2017/01/19/cellprofiler-ilastik-superpowered-segmentation/

""".format(
    **{
        "DEFINITION_OBJECT": _help.DEFINITION_OBJECT,
        "HELP_ON_SAVING_OBJECTS": _help.HELP_ON_SAVING_OBJECTS,
    }
)


#################################################
#
# Ancient offsets into the settings for Matlab pipelines
#
#################################################
IMAGE_NAME_VAR = 0
OBJECT_NAME_VAR = 1
SIZE_RANGE_VAR = 2
EXCLUDE_SIZE_VAR = 3
MERGE_CHOICE_VAR = 4
EXCLUDE_BORDER_OBJECTS_VAR = 5
THRESHOLD_METHOD_VAR = 6
THRESHOLD_CORRECTION_VAR = 7
THRESHOLD_RANGE_VAR = 8
OBJECT_FRACTION_VAR = 9
UNCLUMP_METHOD_VAR = 10
WATERSHED_VAR = 11
SMOOTHING_SIZE_VAR = 12
MAXIMA_SUPPRESSION_SIZE_VAR = 13
LOW_RES_MAXIMA_VAR = 14
SAVE_OUTLINES_VAR = 15
FILL_HOLES_OPTION_VAR = 16
TEST_MODE_VAR = 17
AUTOMATIC_SMOOTHING_VAR = 18
AUTOMATIC_MAXIMA_SUPPRESSION = 19
MANUAL_THRESHOLD_VAR = 20
BINARY_IMAGE_VAR = 21
MEASUREMENT_THRESHOLD_VAR = 22

#################################################
#
# V10 introduced a more unified handling of
#     threshold settings.
#
#################################################
OFF_THRESHOLD_METHOD_V9 = 6
OFF_THRESHOLD_CORRECTION_V9 = 7
OFF_THRESHOLD_RANGE_V9 = 8
OFF_OBJECT_FRACTION_V9 = 9
OFF_MANUAL_THRESHOLD_V9 = 19
OFF_BINARY_IMAGE_V9 = 20
OFF_TWO_CLASS_OTSU_V9 = 24
OFF_USE_WEIGHTED_VARIANCE_V9 = 25
OFF_ASSIGN_MIDDLE_TO_FOREGROUND_V9 = 26
OFF_THRESHOLDING_MEASUREMENT_V9 = 31
OFF_ADAPTIVE_WINDOW_METHOD_V9 = 32
OFF_ADAPTIVE_WINDOW_SIZE_V9 = 33
OFF_FILL_HOLES_V10 = 12
OFF_N_SETTINGS = 16

"""The number of settings, exclusive of threshold settings"""
N_SETTINGS = 16

UN_INTENSITY = "Intensity"
UN_SHAPE = "Shape"
UN_LOG = "Laplacian of Gaussian"
UN_NONE = "None"

WA_INTENSITY = "Intensity"
WA_SHAPE = "Shape"
WA_PROPAGATE = "Propagate"
WA_NONE = "None"

LIMIT_NONE = "Continue"
LIMIT_TRUNCATE = "Truncate"
LIMIT_ERASE = "Erase"

DEFAULT_MAXIMA_COLOR = "Blue"

"""Never fill holes"""
FH_NEVER = "Never"
FH_THRESHOLDING = "After both thresholding and declumping"
FH_DECLUMP = "After declumping only"

FH_ALL = (FH_NEVER, FH_THRESHOLDING, FH_DECLUMP)

# Settings text which is referenced in various places in the help
SIZE_RANGE_SETTING_TEXT = "Typical diameter of objects, in pixel units (Min,Max)"
EXCLUDE_SIZE_SETTING_TEXT = "Discard objects outside the diameter range?"
AUTOMATIC_SMOOTHING_SETTING_TEXT = (
    "Automatically calculate size of smoothing filter for declumping?"
)
SMOOTHING_FILTER_SIZE_SETTING_TEXT = "Size of smoothing filter"
AUTOMATIC_MAXIMA_SUPPRESSION_SETTING_TEXT = (
    "Automatically calculate minimum allowed distance between local maxima?"
)

# Icons for use in the help
INTENSITY_DECLUMPING_ICON = cellprofiler.gui.help.content.image_resource(
    "IdentifyPrimaryObjects_IntensityDeclumping.png"
)
SHAPE_DECLUMPING_ICON = cellprofiler.gui.help.content.image_resource(
    "IdentifyPrimaryObjects_ShapeDeclumping.png"
)


class IdentifyPrimaryObjects(
    cellprofiler_core.module.image_segmentation.ImageSegmentation
):
    variable_revision_number = 15

    category = "Object Processing"

    module_name = "IdentifyPrimaryObjects"

    def __init__(self):
        self.threshold = threshold.Threshold()

        super(IdentifyPrimaryObjects, self).__init__()

    def volumetric(self):
        return False

    def create_settings(self):
        super(IdentifyPrimaryObjects, self).create_settings()

        self.x_name.text = "Select the input image"
        self.x_name.doc = "Select the image that you want to use to identify objects."

        self.y_name.text = "Name the primary objects to be identified"
        self.y_name.doc = "Enter the name that you want to call the objects identified by this module."

        self.size_range = IntegerRange(
            SIZE_RANGE_SETTING_TEXT,
            (10, 40),
            minval=1,
            doc="""\
This setting is crucial for two reasons: first, the module uses it to
calculate certain automatic settings in order to identify your objects
of interest properly (see below). Second, when used in conjunction with the
*{EXCLUDE_SIZE_SETTING_TEXT}* setting below, you can choose to remove
objects outside the size range you provide here.

|image0| The units used here are pixels so that it is easy to zoom in
on objects and determine typical diameters. {HELP_ON_MEASURING_DISTANCES}

A few important notes:

-  The other settings that make use of the minimum object size entered
   here (whether the "*{EXCLUDE_SIZE_SETTING_TEXT}*" setting is used or
   not) are:

   -  "*{AUTOMATIC_SMOOTHING_SETTING_TEXT}*"
   -  "*{AUTOMATIC_MAXIMA_SUPPRESSION_SETTING_TEXT}*"

-  For non-round objects, the diameter you should enter here is actually
   the “equivalent diameter”, i.e., the diameter of a circle with the
   same area as the object.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
            """.format(
                **{
                    "EXCLUDE_SIZE_SETTING_TEXT": EXCLUDE_SIZE_SETTING_TEXT,
                    "PROTIP_RECOMMEND_ICON": _help.PROTIP_RECOMMEND_ICON,
                    "HELP_ON_MEASURING_DISTANCES": _help.HELP_ON_MEASURING_DISTANCES,
                    "AUTOMATIC_SMOOTHING_SETTING_TEXT": AUTOMATIC_SMOOTHING_SETTING_TEXT,
                    "AUTOMATIC_MAXIMA_SUPPRESSION_SETTING_TEXT": AUTOMATIC_MAXIMA_SUPPRESSION_SETTING_TEXT,
                }
            ),
        )

        self.exclude_size = Binary(
            EXCLUDE_SIZE_SETTING_TEXT,
            True,
            doc="""\
Select "*{YES}*" to discard objects outside the range you specified in the
*{SIZE_RANGE_SETTING_TEXT}* setting. Select "*{NO}*" to ignore this
criterion.

Objects discarded based on size are outlined in magenta in the module’s
display. See also the **FilterObjects** module to further discard
objects based on some other measurement.

|image0| Select "*{YES}*" to exclude small objects (e.g.,
dust, noise, and debris) or large objects (e.g., large clumps) if
desired.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
            """.format(
                **{
                    "YES": "Yes",
                    "SIZE_RANGE_SETTING_TEXT": SIZE_RANGE_SETTING_TEXT,
                    "NO": "No",
                    "PROTIP_RECOMMEND_ICON": _help.PROTIP_RECOMMEND_ICON,
                }
            ),
        )

        self.exclude_border_objects = Binary(
            "Discard objects touching the border of the image?",
            True,
            doc="""\
Choose "*{YES}*" to discard objects that touch the border of the image.
Choose "*{NO}*" to ignore this criterion.

Objects discarded because they touch the border are outlined in yellow in the
module’s display. Note that if a per-object thresholding method is used
or if the image has been previously cropped or masked, objects that
touch the border of the cropped or masked region may also discarded.

|image0| Removing objects that touch the image border is useful when
you do not want to make downstream measurements of objects that are not
fully within the field of view. For example, measuring the area of a
partial object would not be accurate.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
            """.format(
                **{
                    "YES": "Yes",
                    "NO": "No",
                    "PROTIP_RECOMMEND_ICON": _help.PROTIP_RECOMMEND_ICON,
                }
            ),
        )

        self.unclump_method = Choice(
            "Method to distinguish clumped objects",
            [UN_INTENSITY, UN_SHAPE, UN_NONE],
            doc="""\
This setting allows you to choose the method that is used to distinguish
between individual objects that are touching each other (and not properly
delineated as two objects by thresholding alone). In other words, this
setting allows you to “declump” a large, merged object into individual objects
of interest. To decide between these methods, you can run Test mode to
see the results of each.

   +--------------------------------------+--------------------------------------+
   | *{UN_INTENSITY}:* For objects that        | |image1|                             |
   | tend to have only a single peak of   |                                      |
   | brightness (e.g., objects that are   |                                      |
   | brighter towards their interiors and |                                      |
   | dimmer towards their edges), this    |                                      |
   | option counts each intensity peak as |                                      |
   | a separate object. The objects can   |                                      |
   | be any shape, so they need not be    |                                      |
   | round and uniform in size as would   |                                      |
   | be required for the *{UN_SHAPE}*          |                                      |
   | option.                              |                                      |
   |                                      |                                      |
   | |image0|  This choice is more        |                                      |
   | successful when the objects have a   |                                      |
   | smooth texture. By default, the      |                                      |
   | image is automatically blurred to    |                                      |
   | attempt to achieve appropriate       |                                      |
   | smoothness (see *Smoothing filter*   |                                      |
   | options), but overriding the default |                                      |
   | value can improve the outcome on     |                                      |
   | lumpy-textured objects.              |                                      |
   |                                      |                                      |
   | |image2|  The object centers are     |                                      |
   | defined as local intensity maxima in |                                      |
   | the smoothed image.                  |                                      |
   +--------------------------------------+--------------------------------------+
   | *{UN_SHAPE}:* For cases when there        | |image4|                             |
   | are definite indentations separating |                                      |
   | objects. The image is converted to   |                                      |
   | black and white (binary) and the     |                                      |
   | shape determines whether clumped     |                                      |
   | objects will be distinguished. The   |                                      |
   | declumping results of this method    |                                      |
   | are affected by the thresholding     |                                      |
   | method you choose.                   |                                      |
   |                                      |                                      |
   | |image3|  This choice works best for |                                      |
   | objects that are round. In this      |                                      |
   | case, the intensity patterns         |                                      |
   | (i.e., lumpy texture) in the         |                                      |
   | original image are largely           |                                      |
   | irrelevant. Therefore, the cells     |                                      |
   | need not be brighter towards the     |                                      |
   | interior as is required for the      |                                      |
   | *{UN_INTENSITY}* option.                  |                                      |
   |                                      |                                      |
   | |image5|  The binary thresholded     |                                      |
   | image is distance-transformed and    |                                      |
   | object centers are defined as peaks  |                                      |
   | in this image. A distance-transform  |                                      |
   | gives each pixel a value equal to    |                                      |
   | the nearest pixel below a certain    |                                      |
   | threshold, so it indicates the       |                                      |
   | *{UN_SHAPE}* of the object.               |                                      |
   +--------------------------------------+--------------------------------------+
   | *{UN_NONE}:* If objects are well separated and bright relative to the            |
   | background, it may be unnecessary to attempt to separate clumped objects.   |
   | Using the very fast *{UN_NONE}* option, a simple threshold will be used to       |
   | identify objects.                                                           |
   +--------------------------------------+--------------------------------------+

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
.. |image1| image:: {INTENSITY_DECLUMPING_ICON}
.. |image2| image:: {TECH_NOTE_ICON}
.. |image3| image:: {PROTIP_RECOMMEND_ICON}
.. |image4| image:: {SHAPE_DECLUMPING_ICON}
.. |image5| image:: {TECH_NOTE_ICON}
            """.format(
                **{
                    "UN_INTENSITY": UN_INTENSITY,
                    "UN_SHAPE": UN_SHAPE,
                    "PROTIP_RECOMMEND_ICON": _help.PROTIP_RECOMMEND_ICON,
                    "INTENSITY_DECLUMPING_ICON": INTENSITY_DECLUMPING_ICON,
                    "TECH_NOTE_ICON": _help.TECH_NOTE_ICON,
                    "SHAPE_DECLUMPING_ICON": SHAPE_DECLUMPING_ICON,
                    "UN_NONE": UN_NONE,
                }
            ),
        )

        self.watershed_method = Choice(
            "Method to draw dividing lines between clumped objects",
            [WA_INTENSITY, WA_SHAPE, WA_PROPAGATE, WA_NONE],
            doc="""\
This setting allows you to choose the method that is used to draw the
line between segmented objects, provided that you have chosen to declump
the objects. To decide between these methods, you can run Test mode to
see the results of each.

-  *{WA_INTENSITY}:* Works best where the dividing lines between
   clumped objects are dimmer than the remainder of the objects.

   **Technical description:** Using the previously identified local
   maxima as seeds, this method is a watershed (*Vincent and Soille,
   1991*) on the intensity image.

-  *{WA_SHAPE}:* Dividing lines between clumped objects are based on
   the shape of the clump. For example, when a clump contains two
   objects, the dividing line will be placed where indentations occur
   between the two objects. The intensity patterns in the original image
   are largely irrelevant: the cells need not be dimmer along the lines
   between clumped objects. Technical description: Using the previously
   identified local maxima as seeds, this method is a watershed on the
   distance-transformed thresholded image.
-  *{WA_PROPAGATE}:* This method uses a propagation algorithm instead
   of a watershed. The image is ignored and the pixels are assigned to
   the objects by repeatedly adding unassigned pixels to the objects
   that are immediately adjacent to them. This method is suited in cases
   such as objects with branching extensions, for instance neurites,
   where the goal is to trace outward from the cell body along the
   branch, assigning pixels in the branch along the way. See the help
   for the **IdentifySecondaryObjects** module for more details on this
   method.
-  *{WA_NONE}*: If objects are well separated and bright relative to
   the background, it may be unnecessary to attempt to separate clumped
   objects. Using the very fast *{WA_NONE}* option, a simple threshold
   will be used to identify objects.
""".format(
                **{
                    "WA_INTENSITY": WA_INTENSITY,
                    "WA_SHAPE": WA_SHAPE,
                    "WA_PROPAGATE": WA_PROPAGATE,
                    "WA_NONE": WA_NONE,
                }
            ),
        )

        self.automatic_smoothing = Binary(
            AUTOMATIC_SMOOTHING_SETTING_TEXT,
            True,
            doc="""\
*(Used only when distinguishing between clumped objects)*

Select "*{YES}*" to automatically calculate the amount of smoothing
applied to the image to assist in declumping. Select "*{NO}*" to
manually enter the smoothing filter size.

This setting, along with the *Minimum allowed distance between local
maxima* setting, affects whether objects close to each other are
considered a single object or multiple objects. It does not affect the
dividing lines between an object and the background.

Please note that this smoothing setting is applied after thresholding,
and is therefore distinct from the threshold smoothing method setting
above, which is applied *before* thresholding.

The size of the smoothing filter is automatically calculated based on
the *{SIZE_RANGE_SETTING_TEXT}* setting above. If you see too many
objects merged that ought to be separate or too many objects split up
that ought to be merged, you may want to override the automatically
calculated value.""".format(
                **{
                    "YES": "Yes",
                    "NO": "No",
                    "SIZE_RANGE_SETTING_TEXT": SIZE_RANGE_SETTING_TEXT,
                }
            ),
        )

        self.smoothing_filter_size = Integer(
            SMOOTHING_FILTER_SIZE_SETTING_TEXT,
            10,
            doc="""\
*(Used only when distinguishing between clumped objects)*

If you see too many objects merged that ought to be separated
(under-segmentation), this value should be lower. If you see too many
objects split up that ought to be merged (over-segmentation), the
value should be higher.

Note that splitting and merging is also
affected by your choice of settings for the setting,
*{AUTOMATIC_MAXIMA_SUPPRESSION_SETTING_TEXT}* It is an art to balance
these two settings; read the help carefully for both.

Reducing the texture of objects by increasing the smoothing increases
the chance that each real, distinct object has only one peak of
intensity but also increases the chance that two distinct objects will
be recognized as only one object. Note that increasing the size of the
smoothing filter increases the processing time exponentially.

Enter 0 to prevent any image smoothing in certain cases; for example,
for low resolution images with small objects ( < ~5 pixels in
diameter).
""".format(
                **{
                    "AUTOMATIC_MAXIMA_SUPPRESSION_SETTING_TEXT": AUTOMATIC_MAXIMA_SUPPRESSION_SETTING_TEXT
                }
            ),
        )

        self.automatic_suppression = Binary(
            AUTOMATIC_MAXIMA_SUPPRESSION_SETTING_TEXT,
            True,
            doc="""\
*(Used only when distinguishing between clumped objects)*

Select "*{YES}*" to automatically calculate the distance between
intensity maxima to assist in declumping. Select "*{NO}*" to manually
enter the permissible maxima distance.

This setting, along with the *{SMOOTHING_FILTER_SIZE_SETTING_TEXT}*
setting, affects whether objects close to each other are considered a
single object or multiple objects. It does not affect the dividing lines
between an object and the background. Local maxima that are closer
together than the minimum allowed distance will be suppressed (the local
intensity histogram is smoothed to remove the peaks within that
distance).

The distance can be automatically calculated based on the
minimum entered for the *{SIZE_RANGE_SETTING_TEXT}* setting above,
but if you see too many objects merged that ought to be separate, or too
many objects split up that ought to be merged, you may want to override
the automatically calculated value.""".format(
                **{
                    "YES": "Yes",
                    "NO": "No",
                    "SMOOTHING_FILTER_SIZE_SETTING_TEXT": SMOOTHING_FILTER_SIZE_SETTING_TEXT,
                    "SIZE_RANGE_SETTING_TEXT": SIZE_RANGE_SETTING_TEXT,
                }
            ),
        )

        self.maxima_suppression_size = Float(
            "Suppress local maxima that are closer than this minimum allowed distance",
            7,
            minval=0,
            doc="""\
*(Used only when distinguishing between clumped objects)*

Enter a positive integer, in pixel units. If you see too many objects
merged that ought to be separated (under-segmentation), the value
should be lower. If you see too many objects split up that ought to be
merged (over-segmentation), the value should be higher.

The maxima suppression distance should be set to be roughly equivalent
to the radius of the smallest object of interest that you would expect
to see in the experiment. Any distinct
“objects” that are found but are within two times this distance from
each other will be assumed to be actually two lumpy parts of the same
object, and they will be merged.

Note that splitting and merging is also
affected by your choice of settings for the setting,
*{SMOOTHING_FILTER_SIZE_SETTING_TEXT}* It is an art to balance
these two settings; read the help carefully for both.
""".format(
                **{
                    "SMOOTHING_FILTER_SIZE_SETTING_TEXT": SMOOTHING_FILTER_SIZE_SETTING_TEXT
                }
            ),
        )

        self.low_res_maxima = Binary(
            "Speed up by using lower-resolution image to find local maxima?",
            True,
            doc="""\
*(Used only when distinguishing between clumped objects)*

Select "*{YES}*" to down-sample the image for declumping. This can be
helpful for saving processing time on large images.

Note that if you have entered a minimum object diameter of 10 or less,
checking this box will have no effect.""".format(
                **{"YES": "Yes"}
            ),
        )

        self.fill_holes = Choice(
            "Fill holes in identified objects?",
            FH_ALL,
            value=FH_THRESHOLDING,
            doc="""\
This option controls how holes (regions of background surrounded by one
or more objects) are filled in:

-  *{FH_THRESHOLDING}:* Fill in holes that are smaller than
   the maximum object size prior to declumping and to fill in any holes
   after declumping.
-  *{FH_DECLUMP}:* Fill in holes located within identified
   objects after declumping.
-  *{FH_NEVER}:* Leave holes within objects.
   Please note that if an object is located within a hole and
   this option is enabled, the object will be lost when the hole is
   filled in.""".format(
                **{
                    "FH_THRESHOLDING": FH_THRESHOLDING,
                    "FH_DECLUMP": FH_DECLUMP,
                    "FH_NEVER": FH_NEVER,
                }
            ),
        )

        self.limit_choice = Choice(
            "Handling of objects if excessive number of objects identified",
            [LIMIT_NONE, LIMIT_ERASE],
            doc="""\
This setting deals with images that are segmented into an unreasonable
number of objects. This might happen if the module calculates a low
threshold or if the image has unusual artifacts.
**IdentifyPrimaryObjects** can handle this condition in one of three
ways:

-  *{LIMIT_NONE}*: Continue processing regardless if large numbers of
   objects are found.
-  *{LIMIT_ERASE}*: Erase all objects if the number of objects exceeds
   the maximum. This results in an image with no primary objects. This
   option is a good choice if a large number of objects indicates that
   the image should not be processed; it can save a lot of time in
   subsequent **Measure** modules.""".format(
                **{"LIMIT_NONE": LIMIT_NONE, "LIMIT_ERASE": LIMIT_ERASE}
            ),
        )

        self.maximum_object_count = Integer(
            "Maximum number of objects",
            value=500,
            minval=2,
            doc="""\
*(Used only when handling images with large numbers of objects by
erasing)*

This setting limits the number of objects in the image. See the
documentation for the previous setting for details.""",
        )

        self.want_plot_maxima = Binary(
            "Display accepted local maxima?",
            False,
            doc="""\
*(Used only when distinguishing between clumped objects)*
            
Note: As this only effects figure previews, maxima display settings will not be saved to the pipeline.

Select "*{YES}*" to display detected local maxima on the object outlines plot. This can be
helpful for fine-tuning segmentation parameters.

Local maxima are small cluster of pixels from which objects are 'grown' during segmentation.
Each object in a declumped segmentation will have a single maxima.

For example, for intensity-based declumping, maxima should appear at the brightest points in an object.
If obvious intensity peaks are missing they were probably removed by the filters set above.""".format(
                **{"YES": "Yes"}
            ),
        )

        self.maxima_color = Color(
            "Select maxima color",
            DEFAULT_MAXIMA_COLOR,
            doc="Maxima will be displayed in this color.",
        )

        self.maxima_size = Integer(
            "Select maxima size",
            value=1,
            minval=1,
            doc="Radius of the visible marker for each maxima."
                "You may want to increase this when working with large images.",
        )

        self.use_advanced = Binary(
            "Use advanced settings?",
            value=False,
            doc="""\
Select "*{YES}*" to use advanced module settings.
If "*{NO}*" is selected, the following settings are used:

-  *{THRESHOLD_SCOPE_TEXT}*: {THRESHOLD_SCOPE_VALUE}
-  *{THRESHOLD_METHOD_TEXT}*: {THRESHOLD_METHOD_VALUE}
-  *{THRESHOLD_SMOOTHING_SCALE_TEXT}*:
   {THRESHOLD_SMOOTHING_SCALE_VALUE} (sigma = 1)
-  *{THRESHOLD_CORRECTION_FACTOR_TEXT}*:
   {THRESHOLD_CORRECTION_FACTOR_VALUE}
-  *{THRESHOLD_RANGE_TEXT}*: minimum {THRESHOLD_RANGE_MIN}, maximum
   {THRESHOLD_RANGE_MAX}
-  *{UNCLUMP_METHOD_TEXT}*: {UNCLUMP_METHOD_VALUE}
-  *{WATERSHED_METHOD_TEXT}*: {WATERSHED_METHOD_VALUE}
-  *{AUTOMATIC_SMOOTHING_TEXT}*: *{YES}*
-  *{AUTOMATIC_SUPPRESSION_TEXT}*: *{YES}*
-  *{LOW_RES_MAXIMA_TEXT}*: *{YES}*
-  *{FILL_HOLES_TEXT}*: {FILL_HOLES_VALUE}
-  *{LIMIT_CHOICE_TEXT}*: {LIMIT_CHOICE_VALUE}""".format(
                **{
                    "AUTOMATIC_SMOOTHING_TEXT": self.automatic_smoothing.get_text(),
                    "AUTOMATIC_SUPPRESSION_TEXT": self.automatic_suppression.get_text(),
                    "FILL_HOLES_TEXT": self.fill_holes.get_text(),
                    "FILL_HOLES_VALUE": FH_THRESHOLDING,
                    "LIMIT_CHOICE_TEXT": self.limit_choice.get_text(),
                    "LIMIT_CHOICE_VALUE": LIMIT_NONE,
                    "LOW_RES_MAXIMA_TEXT": self.low_res_maxima.get_text(),
                    "NO": "No",
                    "THRESHOLD_CORRECTION_FACTOR_TEXT": self.threshold.threshold_correction_factor.get_text(),
                    "THRESHOLD_CORRECTION_FACTOR_VALUE": 1.0,
                    "THRESHOLD_METHOD_TEXT": self.threshold.global_operation.get_text(),
                    "THRESHOLD_METHOD_VALUE": threshold.TM_LI,
                    "THRESHOLD_RANGE_MAX": 1.0,
                    "THRESHOLD_RANGE_MIN": 0.0,
                    "THRESHOLD_RANGE_TEXT": self.threshold.threshold_range.get_text(),
                    "THRESHOLD_SCOPE_TEXT": self.threshold.threshold_scope.get_text(),
                    "THRESHOLD_SCOPE_VALUE": threshold.TS_GLOBAL,
                    "THRESHOLD_SMOOTHING_SCALE_TEXT": self.threshold.threshold_smoothing_scale.get_text(),
                    "THRESHOLD_SMOOTHING_SCALE_VALUE": 1.3488,
                    "UNCLUMP_METHOD_TEXT": self.unclump_method.get_text(),
                    "UNCLUMP_METHOD_VALUE": UN_INTENSITY,
                    "WATERSHED_METHOD_TEXT": self.watershed_method.get_text(),
                    "WATERSHED_METHOD_VALUE": WA_INTENSITY,
                    "YES": "Yes",
                }
            ),
        )

        self.threshold_setting_version = Integer(
            "Threshold setting version", value=self.threshold.variable_revision_number
        )

        self.threshold.create_settings()

        self.threshold.threshold_smoothing_scale.value = 1.3488  # sigma = 1

    def settings(self):
        settings = super(IdentifyPrimaryObjects, self).settings()

        settings += [
            self.size_range,
            self.exclude_size,
            self.exclude_border_objects,
            self.unclump_method,
            self.watershed_method,
            self.smoothing_filter_size,
            self.maxima_suppression_size,
            self.low_res_maxima,
            self.fill_holes,
            self.automatic_smoothing,
            self.automatic_suppression,
            self.limit_choice,
            self.maximum_object_count,
            self.use_advanced,
        ]

        threshold_settings = self.threshold.settings()[2:]

        return settings + [self.threshold_setting_version] + threshold_settings

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number < 10:
            raise NotImplementedError(
                "Automatic upgrade for this module is not supported in CellProfiler 3."
            )

        if variable_revision_number == 10:
            setting_values = list(setting_values)
            if setting_values[OFF_FILL_HOLES_V10] == "No":
                setting_values[OFF_FILL_HOLES_V10] = FH_NEVER
            elif setting_values[OFF_FILL_HOLES_V10] == "Yes":
                setting_values[OFF_FILL_HOLES_V10] = FH_THRESHOLDING
            variable_revision_number = 11

        if variable_revision_number == 11:
            if setting_values[6] == UN_LOG:
                setting_values[6] = UN_INTENSITY

            if setting_values[20] == LIMIT_TRUNCATE:
                setting_values[20] = "None"

            new_setting_values = setting_values[:4]

            new_setting_values += setting_values[5:11]

            new_setting_values += setting_values[12:15]

            new_setting_values += setting_values[20:]

            setting_values = new_setting_values

            variable_revision_number = 12

        if variable_revision_number == 12:
            new_setting_values = setting_values[: OFF_N_SETTINGS - 1]
            new_setting_values += ["Yes"]
            new_setting_values += setting_values[OFF_N_SETTINGS - 1:]

            setting_values = new_setting_values

            variable_revision_number = 13

        if variable_revision_number == 13:
            # Added maxima settings
            new_setting_values = setting_values[: 15]
            new_setting_values += ["No", DEFAULT_MAXIMA_COLOR]
            new_setting_values += setting_values[15:]

            setting_values = new_setting_values

            variable_revision_number = 14

        if variable_revision_number == 14:
            # Removed maxima settings
            new_setting_values = setting_values[: 15]
            new_setting_values += setting_values[17:]

            setting_values = new_setting_values

            variable_revision_number = 15


        threshold_setting_values = setting_values[N_SETTINGS:]

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

        setting_values = setting_values[:N_SETTINGS] + threshold_upgrade_settings

        return setting_values, variable_revision_number

    def help_settings(self):
        threshold_help_settings = self.threshold.help_settings()[2:]

        return (
            [
                self.use_advanced,
                self.x_name,
                self.y_name,
                self.size_range,
                self.exclude_size,
                self.exclude_border_objects,
            ]
            + threshold_help_settings
            + [
                self.unclump_method,
                self.watershed_method,
                self.automatic_smoothing,
                self.smoothing_filter_size,
                self.automatic_suppression,
                self.maxima_suppression_size,
                self.low_res_maxima,
                self.fill_holes,
                self.limit_choice,
                self.maximum_object_count,
            ]
        )

    def visible_settings(self):
        visible_settings = [self.use_advanced]

        visible_settings += super(IdentifyPrimaryObjects, self).visible_settings()

        visible_settings += [
            self.size_range,
            self.exclude_size,
            self.exclude_border_objects,
        ]

        if self.use_advanced.value:
            visible_settings += self.threshold.visible_settings()[2:]

            visible_settings += [self.unclump_method, self.watershed_method]

            if self.unclump_method != UN_NONE and self.watershed_method != WA_NONE:
                visible_settings += [self.automatic_smoothing]

                if not self.automatic_smoothing.value:
                    visible_settings += [self.smoothing_filter_size]

                visible_settings += [self.automatic_suppression]

                if not self.automatic_suppression.value:
                    visible_settings += [self.maxima_suppression_size]

                visible_settings += [self.low_res_maxima, self.want_plot_maxima]

                if self.want_plot_maxima.value:
                    visible_settings += [self.maxima_color, self.maxima_size]

            else:  # self.unclump_method == UN_NONE or self.watershed_method == WA_NONE
                visible_settings = visible_settings[:-2]

                if self.unclump_method == UN_NONE:
                    visible_settings += [self.unclump_method]
                else:  # self.watershed_method == WA_NONE
                    visible_settings += [self.watershed_method]

            visible_settings += [self.fill_holes, self.limit_choice]

            if self.limit_choice != LIMIT_NONE:
                visible_settings += [self.maximum_object_count]

        return visible_settings

    @property
    def advanced(self):
        return self.use_advanced.value

    @property
    def basic(self):
        return not self.advanced

    def run(self, workspace):
        workspace.display_data.statistics = []
        input_image = workspace.image_set.get_image(
            self.x_name.value, must_be_grayscale=True
        )

        final_threshold, orig_threshold, guide_threshold, binary_image, sigma = self.threshold.get_threshold(
            input_image, workspace, automatic=self.basic
        )

        self.threshold.add_threshold_measurements(
            self.y_name.value,
            workspace.measurements,
            final_threshold,
            orig_threshold,
            guide_threshold,
        )

        self.threshold.add_fg_bg_measurements(
            self.y_name.value, workspace.measurements, input_image, binary_image
        )

        global_threshold = numpy.mean(numpy.atleast_1d(final_threshold))

        #
        # Fill background holes inside foreground objects
        #
        def size_fn(size, is_foreground):
            return size < self.size_range.max * self.size_range.max

        if self.basic or self.fill_holes.value == FH_THRESHOLDING:
            binary_image = centrosome.cpmorphology.fill_labeled_holes(
                binary_image, size_fn=size_fn
            )

        labeled_image, object_count = scipy.ndimage.label(
            binary_image, numpy.ones((3, 3), bool)
        )

        (
            labeled_image,
            object_count,
            maxima_suppression_size,
        ) = self.separate_neighboring_objects(workspace, labeled_image, object_count)

        unedited_labels = labeled_image.copy()

        # Filter out objects touching the border or mask
        border_excluded_labeled_image = labeled_image.copy()
        labeled_image = self.filter_on_border(input_image, labeled_image)
        border_excluded_labeled_image[labeled_image > 0] = 0

        # Filter out small and large objects
        size_excluded_labeled_image = labeled_image.copy()
        labeled_image, small_removed_labels = self.filter_on_size(
            labeled_image, object_count
        )
        size_excluded_labeled_image[labeled_image > 0] = 0

        #
        # Fill holes again after watershed
        #
        if self.basic or self.fill_holes != FH_NEVER:
            labeled_image = centrosome.cpmorphology.fill_labeled_holes(labeled_image)

        # Relabel the image
        labeled_image, object_count = centrosome.cpmorphology.relabel(labeled_image)

        if self.advanced and self.limit_choice.value == LIMIT_ERASE:
            if object_count > self.maximum_object_count.value:
                labeled_image = numpy.zeros(labeled_image.shape, int)
                border_excluded_labeled_image = numpy.zeros(labeled_image.shape, int)
                size_excluded_labeled_image = numpy.zeros(labeled_image.shape, int)
                object_count = 0

        # Make an outline image
        outline_image = centrosome.outline.outline(labeled_image)
        outline_size_excluded_image = centrosome.outline.outline(
            size_excluded_labeled_image
        )
        outline_border_excluded_image = centrosome.outline.outline(
            border_excluded_labeled_image
        )

        if self.show_window:
            statistics = workspace.display_data.statistics
            statistics.append(["# of accepted objects", "%d" % object_count])
            if object_count > 0:
                areas = scipy.ndimage.sum(
                    numpy.ones(labeled_image.shape),
                    labeled_image,
                    numpy.arange(1, object_count + 1),
                )
                areas.sort()
                low_diameter = (
                    math.sqrt(float(areas[object_count // 10]) / numpy.pi) * 2
                )
                median_diameter = (
                    math.sqrt(float(areas[object_count // 2]) / numpy.pi) * 2
                )
                high_diameter = (
                    math.sqrt(float(areas[object_count * 9 // 10]) / numpy.pi) * 2
                )
                statistics.append(
                    ["10th pctile diameter", "%.1f pixels" % low_diameter]
                )
                statistics.append(["Median diameter", "%.1f pixels" % median_diameter])
                statistics.append(
                    ["90th pctile diameter", "%.1f pixels" % high_diameter]
                )
                object_area = numpy.sum(areas)
                total_area = numpy.product(labeled_image.shape[:2])
                statistics.append(
                    [
                        "Area covered by objects",
                        "%.1f %%" % (100.0 * float(object_area) / float(total_area)),
                    ]
                )
                statistics.append(["Thresholding filter size", "%.1f" % sigma])
                statistics.append(["Threshold", "%0.3g" % global_threshold])
                if self.basic or self.unclump_method != UN_NONE:
                    statistics.append(
                        [
                            "Declumping smoothing filter size",
                            "%.1f" % (self.calc_smoothing_filter_size()),
                        ]
                    )
                    statistics.append(
                        ["Maxima suppression size", "%.1f" % maxima_suppression_size]
                    )
            else:
                statistics.append(["Threshold", "%0.3g" % global_threshold])
            workspace.display_data.image = input_image.pixel_data
            workspace.display_data.labeled_image = labeled_image
            workspace.display_data.size_excluded_labels = size_excluded_labeled_image
            workspace.display_data.border_excluded_labels = (
                border_excluded_labeled_image
            )

        # Add image measurements
        objname = self.y_name.value
        measurements = workspace.measurements

        # Add label matrices to the object set
        objects = cellprofiler_core.object.Objects()
        objects.segmented = labeled_image
        objects.unedited_segmented = unedited_labels
        objects.small_removed_segmented = small_removed_labels
        objects.parent_image = input_image

        workspace.object_set.add_objects(objects, self.y_name.value)

        self.add_measurements(workspace)

    def smooth_image(self, image, mask):
        """Apply the smoothing filter to the image"""

        filter_size = self.calc_smoothing_filter_size()
        if filter_size == 0:
            return image
        sigma = filter_size / 2.35
        #
        # We not only want to smooth using a Gaussian, but we want to limit
        # the spread of the smoothing to 2 SD, partly to make things happen
        # locally, partly to make things run faster, partly to try to match
        # the Matlab behavior.
        #
        filter_size = max(int(float(filter_size) / 2.0), 1)
        f = (
            1
            / numpy.sqrt(2.0 * numpy.pi)
            / sigma
            * numpy.exp(
                -0.5 * numpy.arange(-filter_size, filter_size + 1) ** 2 / sigma ** 2
            )
        )

        def fgaussian(image):
            output = scipy.ndimage.convolve1d(image, f, axis=0, mode="constant")
            return scipy.ndimage.convolve1d(output, f, axis=1, mode="constant")

        #
        # Use the trick where you similarly convolve an array of ones to find
        # out the edge effects, then divide to correct the edge effects
        #
        edge_array = fgaussian(mask.astype(float))
        masked_image = image.copy()
        masked_image[~mask] = 0
        smoothed_image = fgaussian(masked_image)
        masked_image[mask] = smoothed_image[mask] / edge_array[mask]
        return masked_image

    def separate_neighboring_objects(self, workspace, labeled_image, object_count):
        """Separate objects based on local maxima or distance transform

        workspace - get the image from here

        labeled_image - image labeled by scipy.ndimage.label

        object_count  - # of objects in image

        returns revised labeled_image, object count, maxima_suppression_size,
        LoG threshold and filter diameter
        """
        if self.advanced and (
            self.unclump_method == UN_NONE or self.watershed_method == WA_NONE
        ):
            return labeled_image, object_count, 7

        cpimage = workspace.image_set.get_image(
            self.x_name.value, must_be_grayscale=True
        )
        image = cpimage.pixel_data
        mask = cpimage.mask

        blurred_image = self.smooth_image(image, mask)
        if self.size_range.min > 10 and (self.basic or self.low_res_maxima.value):
            image_resize_factor = 10.0 / float(self.size_range.min)
            if self.basic or self.automatic_suppression.value:
                maxima_suppression_size = 7
            else:
                maxima_suppression_size = (
                    self.maxima_suppression_size.value * image_resize_factor + 0.5
                )
            reported_maxima_suppression_size = (
                maxima_suppression_size / image_resize_factor
            )
        else:
            image_resize_factor = 1.0
            if self.basic or self.automatic_suppression.value:
                maxima_suppression_size = self.size_range.min / 1.5
            else:
                maxima_suppression_size = self.maxima_suppression_size.value
            reported_maxima_suppression_size = maxima_suppression_size
        maxima_mask = centrosome.cpmorphology.strel_disk(
            max(1, maxima_suppression_size - 0.5)
        )
        distance_transformed_image = None
        if self.basic or self.unclump_method == UN_INTENSITY:
            # Remove dim maxima
            maxima_image = self.get_maxima(
                blurred_image, labeled_image, maxima_mask, image_resize_factor
            )
        elif self.unclump_method == UN_SHAPE:
            if self.fill_holes == FH_NEVER:
                # For shape, even if the user doesn't want to fill holes,
                # a point far away from the edge might be near a hole.
                # So we fill just for this part.
                foreground = (
                    centrosome.cpmorphology.fill_labeled_holes(labeled_image) > 0
                )
            else:
                foreground = labeled_image > 0
            distance_transformed_image = scipy.ndimage.distance_transform_edt(
                foreground
            )
            # randomize the distance slightly to get unique maxima
            numpy.random.seed(0)
            distance_transformed_image += numpy.random.uniform(
                0, 0.001, distance_transformed_image.shape
            )
            maxima_image = self.get_maxima(
                distance_transformed_image,
                labeled_image,
                maxima_mask,
                image_resize_factor,
            )
        else:
            raise ValueError(
                "Unsupported local maxima method: %s" % self.unclump_method.value
            )

        # Create the image for watershed
        if self.basic or self.watershed_method == WA_INTENSITY:
            # use the reverse of the image to get valleys at peaks
            watershed_image = 1 - image
        elif self.watershed_method == WA_SHAPE:
            if distance_transformed_image is None:
                distance_transformed_image = scipy.ndimage.distance_transform_edt(
                    labeled_image > 0
                )
            watershed_image = -distance_transformed_image
            watershed_image = watershed_image - numpy.min(watershed_image)
        elif self.watershed_method == WA_PROPAGATE:
            # No image used
            pass
        else:
            raise NotImplementedError(
                "Watershed method %s is not implemented" % self.watershed_method.value
            )
        #
        # Create a marker array where the unlabeled image has a label of
        # -(nobjects+1)
        # and every local maximum has a unique label which will become
        # the object's label. The labels are negative because that
        # makes the watershed algorithm use FIFO for the pixels which
        # yields fair boundaries when markers compete for pixels.
        #
        self.labeled_maxima, object_count = scipy.ndimage.label(
            maxima_image, numpy.ones((3, 3), bool)
        )
        if self.advanced and self.watershed_method == WA_PROPAGATE:
            watershed_boundaries, distance = centrosome.propagate.propagate(
                numpy.zeros(self.labeled_maxima.shape),
                self.labeled_maxima,
                labeled_image != 0,
                1.0,
            )
        else:
            markers_dtype = (
                numpy.int16
                if object_count < numpy.iinfo(numpy.int16).max
                else numpy.int32
            )
            markers = numpy.zeros(watershed_image.shape, markers_dtype)
            markers[self.labeled_maxima > 0] = -self.labeled_maxima[
                self.labeled_maxima > 0
            ]

            #
            # Some labels have only one maker in them, some have multiple and
            # will be split up.
            #

            watershed_boundaries = skimage.segmentation.watershed(
                connectivity=numpy.ones((3, 3), bool),
                image=watershed_image,
                markers=markers,
                mask=labeled_image != 0,
            )

            watershed_boundaries = -watershed_boundaries

        return watershed_boundaries, object_count, reported_maxima_suppression_size

    def get_maxima(self, image, labeled_image, maxima_mask, image_resize_factor):
        if image_resize_factor < 1.0:
            shape = numpy.array(image.shape) * image_resize_factor
            i_j = (
                numpy.mgrid[0 : shape[0], 0 : shape[1]].astype(float)
                / image_resize_factor
            )
            resized_image = scipy.ndimage.map_coordinates(image, i_j)
            resized_labels = scipy.ndimage.map_coordinates(
                labeled_image, i_j, order=0
            ).astype(labeled_image.dtype)

        else:
            resized_image = image
            resized_labels = labeled_image
        #
        # find local maxima
        #
        if maxima_mask is not None:
            binary_maxima_image = centrosome.cpmorphology.is_local_maximum(
                resized_image, resized_labels, maxima_mask
            )
            binary_maxima_image[resized_image <= 0] = 0
        else:
            binary_maxima_image = (resized_image > 0) & (labeled_image > 0)
        if image_resize_factor < 1.0:
            inverse_resize_factor = float(image.shape[0]) / float(
                binary_maxima_image.shape[0]
            )
            i_j = (
                numpy.mgrid[0 : image.shape[0], 0 : image.shape[1]].astype(float)
                / inverse_resize_factor
            )
            binary_maxima_image = (
                scipy.ndimage.map_coordinates(binary_maxima_image.astype(float), i_j)
                > 0.5
            )
            assert binary_maxima_image.shape[0] == image.shape[0]
            assert binary_maxima_image.shape[1] == image.shape[1]

        # Erode blobs of touching maxima to a single point

        shrunk_image = centrosome.cpmorphology.binary_shrink(binary_maxima_image)
        return shrunk_image

    def filter_on_size(self, labeled_image, object_count):
        """ Filter the labeled image based on the size range

        labeled_image - pixel image labels
        object_count - # of objects in the labeled image
        returns the labeled image, and the labeled image with the
        small objects removed
        """
        if self.exclude_size.value and object_count > 0:
            areas = scipy.ndimage.measurements.sum(
                numpy.ones(labeled_image.shape),
                labeled_image,
                numpy.array(list(range(0, object_count + 1)), dtype=numpy.int32),
            )
            areas = numpy.array(areas, dtype=int)
            min_allowed_area = (
                numpy.pi * (self.size_range.min * self.size_range.min) / 4
            )
            max_allowed_area = (
                numpy.pi * (self.size_range.max * self.size_range.max) / 4
            )
            # area_image has the area of the object at every pixel within the object
            area_image = areas[labeled_image]
            labeled_image[area_image < min_allowed_area] = 0
            small_removed_labels = labeled_image.copy()
            labeled_image[area_image > max_allowed_area] = 0
        else:
            small_removed_labels = labeled_image.copy()
        return labeled_image, small_removed_labels

    def filter_on_border(self, image, labeled_image):
        """Filter out objects touching the border

        In addition, if the image has a mask, filter out objects
        touching the border of the mask.
        """
        if self.exclude_border_objects.value:
            border_labels = list(labeled_image[0, :])
            border_labels.extend(labeled_image[:, 0])
            border_labels.extend(labeled_image[labeled_image.shape[0] - 1, :])
            border_labels.extend(labeled_image[:, labeled_image.shape[1] - 1])
            border_labels = numpy.array(border_labels)
            #
            # the following histogram has a value > 0 for any object
            # with a border pixel
            #
            histogram = scipy.sparse.coo_matrix(
                (
                    numpy.ones(border_labels.shape),
                    (border_labels, numpy.zeros(border_labels.shape)),
                ),
                shape=(numpy.max(labeled_image) + 1, 1),
            ).todense()
            histogram = numpy.array(histogram).flatten()
            if any(histogram[1:] > 0):
                histogram_image = histogram[labeled_image]
                labeled_image[histogram_image > 0] = 0
            elif image.has_mask:
                # The assumption here is that, if nothing touches the border,
                # the mask is a large, elliptical mask that tells you where the
                # well is. That's the way the old Matlab code works and it's duplicated here
                #
                # The operation below gets the mask pixels that are on the border of the mask
                # The erosion turns all pixels touching an edge to zero. The not of this
                # is the border + formerly masked-out pixels.
                mask_border = numpy.logical_not(
                    scipy.ndimage.binary_erosion(image.mask)
                )
                mask_border = numpy.logical_and(mask_border, image.mask)
                border_labels = labeled_image[mask_border]
                border_labels = border_labels.flatten()
                histogram = scipy.sparse.coo_matrix(
                    (
                        numpy.ones(border_labels.shape),
                        (border_labels, numpy.zeros(border_labels.shape)),
                    ),
                    shape=(numpy.max(labeled_image) + 1, 1),
                ).todense()
                histogram = numpy.array(histogram).flatten()
                if any(histogram[1:] > 0):
                    histogram_image = histogram[labeled_image]
                    labeled_image[histogram_image > 0] = 0
        return labeled_image

    def display(self, workspace, figure):
        if self.show_window:
            """Display the image and labeling"""
            figure.set_subplots((2, 2))

            orig_axes = figure.subplot(0, 0)
            label_axes = figure.subplot(1, 0, sharexy=orig_axes)
            outlined_axes = figure.subplot(0, 1, sharexy=orig_axes)

            title = "Input image, cycle #%d" % (workspace.measurements.image_number,)
            image = workspace.display_data.image
            labeled_image = workspace.display_data.labeled_image
            size_excluded_labeled_image = workspace.display_data.size_excluded_labels
            border_excluded_labeled_image = (
                workspace.display_data.border_excluded_labels
            )

            ax = figure.subplot_imshow_grayscale(0, 0, image, title)
            figure.subplot_imshow_labels(
                1, 0, labeled_image, self.y_name.value, sharexy=ax
            )

            cplabels = [
                dict(name=self.y_name.value, labels=[labeled_image]),
                dict(
                    name="Objects filtered out by size",
                    labels=[size_excluded_labeled_image],
                ),
                dict(
                    name="Objects touching border",
                    labels=[border_excluded_labeled_image],
                ),
            ]
            if (
                self.unclump_method != UN_NONE
                and self.watershed_method != WA_NONE
                and self.want_plot_maxima
            ):
                # Generate static colormap for alpha overlay
                from matplotlib.colors import ListedColormap

                cmap = ListedColormap(self.maxima_color.value)
                if self.maxima_size.value > 1:
                    strel = skimage.morphology.disk(self.maxima_size.value - 1)
                    labels = skimage.morphology.dilation(self.labeled_maxima, footprint=strel)
                else:
                    labels = self.labeled_maxima
                cplabels.append(
                    dict(
                        name="Detected maxima",
                        labels=[labels],
                        mode="alpha",
                        alpha_value=1,
                        alpha_colormap=cmap,
                    )
                )
            title = "%s outlines" % self.y_name.value
            figure.subplot_imshow_grayscale(
                0, 1, image, title, cplabels=cplabels, sharexy=ax
            )

            figure.subplot_table(
                1,
                1,
                [[x[1]] for x in workspace.display_data.statistics],
                row_labels=[x[0] for x in workspace.display_data.statistics],
            )

    def calc_smoothing_filter_size(self):
        """Return the size of the smoothing filter, calculating it if in automatic mode"""
        if self.automatic_smoothing.value:
            return 2.35 * self.size_range.min / 3.5
        else:
            return self.smoothing_filter_size.value

    def is_object_identification_module(self):
        return True

    def get_measurement_columns(self, pipeline):
        columns = super(IdentifyPrimaryObjects, self).get_measurement_columns(pipeline)

        columns += self.threshold.get_measurement_columns(
            pipeline, object_name=self.y_name.value
        )

        return columns

    def get_categories(self, pipeline, object_name):
        categories = self.threshold.get_categories(pipeline, object_name)

        categories += super(IdentifyPrimaryObjects, self).get_categories(
            pipeline, object_name
        )

        return categories

    def get_measurements(self, pipeline, object_name, category):
        measurements = self.threshold.get_measurements(pipeline, object_name, category)

        measurements += super(IdentifyPrimaryObjects, self).get_measurements(
            pipeline, object_name, category
        )

        return measurements

    def get_measurement_objects(self, pipeline, object_name, category, measurement):
        if measurement in self.threshold.get_measurements(
            pipeline, object_name, category
        ):
            return [self.y_name.value]

        return []
