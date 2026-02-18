"""
UntangleWorms
=============

**UntangleWorms** untangles overlapping worms.

This module either assembles a training set of sample worms in order to
create a worm model, or takes a binary image and the results of worm
training and labels the worms in the image, untangling them and
associating all of a worm’s pieces together. The results of untangling
the input image will be an object set that can be used with downstream
measurement modules. If using the *overlapping* style of objects, these
must be used within the pipeline as they cannot be saved.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

See also
^^^^^^^^

See also our `Worm Toolbox`_ page for sample images and pipelines, as
well as video tutorials.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Object measurements (for “Untangle” mode only)**:

-  *Length:* The length of the worm skeleton.
-  *Angle:* The angle at each of the control points
-  *ControlPointX\_N, ControlPointY\_N:* The X,Y coordinate of a control
   point *N*. A control point is a sampled location along the worm shape
   used to construct the model.

Technical notes
^^^^^^^^^^^^^^^

*Training* involves extracting morphological information from the sample
objects provided from the previous steps. Using the default training set
weights is recommended. Proper creation of the model is dependent on
providing a binary image as input consisting of single, separated
objects considered to be worms. You can the **Identify** modules to find
the tentative objects and then filter these objects to get individual
worms, whether by using **FilterObjects**, **EditObjectsManually** or
the size criteria in **IdentifyPrimaryObjects**. A binary image can be
obtained from an object set by using **ConvertObjectsToImage**.

At the end of the training run, a final display window is shown
displaying the following statistical data:

-  A boxplot of the direction angle shape costs. The direction angles
   (which are between -π and π) are the angles between lines joining
   consective control points. The angle 0 corresponds to the case when
   two adjacent line segments are parallel (and thus belong to the same
   line).
-  A cumulative boxplot of the worm lengths as determined by the model.
-  A cumulative boxplot of the worm angles as determined by the model.
-  A heatmap of the covariance matrix of the feature vectors. For *N*
   control points, the feature vector is of length *N*-1 and contains
   *N*-2 elements for each of the angles between them, plus an element
   representing the worm length.

*Untangling* involves untangles the worms using a provided worm model,
built from a large number of samples of single worms. If the result of
the untangling is not satisfactory (e.g., it is unable to detect long
worms or is too stringent about shape variation) and you do not wish to
re-train, you can adjust the provided worm model manually by opening the
.xml file in a text editor and changing the values for the fields
defining worm length, area etc. You may also want to adjust the “Maximum
Complexity” module setting which controls how complex clusters the
untangling will handle. Large clusters (> 6 worms) may be slow to
process.

References
^^^^^^^^^^

-  Wählby C, Kamentsky L, Liu ZH, Riklin-Raviv T, Conery AL, O’Rourke
   EJ, Sokolnicki KL, Visvikis O, Ljosa V, Irazoqui JE, Golland P,
   Ruvkun G, Ausubel FM, Carpenter AE (2012). "An image analysis toolbox
   for high-throughput *C. elegans* assays." *Nature Methods* 9(7):
   714-716. `(link) <https://doi.org/10.1038/nmeth.1984>`__

.. _Worm Toolbox: http://www.cellprofiler.org/wormtoolbox/
"""

import logging
import os
import xml.dom.minidom as DOM
from urllib.request import urlopen
from packaging.version import Version

import numpy
import scipy.ndimage
from scipy.interpolate import interp1d
from scipy.io import loadmat
from scipy.sparse import coo
from centrosome.outline import outline
from centrosome.propagate import propagate
import centrosome.cpmorphology

from cellprofiler_core.constants.measurement import C_LOCATION
from cellprofiler_core.constants.measurement import C_NUMBER
from cellprofiler_core.constants.measurement import FTR_CENTER_X
from cellprofiler_core.constants.measurement import FTR_CENTER_Y
from cellprofiler_core.constants.measurement import FTR_OBJECT_NUMBER
from cellprofiler_core.constants.measurement import IMAGE, COLTYPE_FLOAT, C_COUNT
from cellprofiler_core.constants.measurement import M_LOCATION_CENTER_X
from cellprofiler_core.constants.measurement import M_LOCATION_CENTER_Y
from cellprofiler_core.constants.measurement import M_NUMBER_OBJECT_NUMBER
from cellprofiler_core.constants.module import (
    USING_METADATA_GROUPING_HELP_REF,
    IO_FOLDER_CHOICE_HELP_TEXT,
)
from cellprofiler_core.image import Image
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.module import Module
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.object import Objects
from cellprofiler_core.preferences import DEFAULT_OUTPUT_FOLDER_NAME
from cellprofiler_core.preferences import URL_FOLDER_NAME
from cellprofiler_core.preferences import get_default_colormap
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting import ValidationError
from cellprofiler_core.setting.choice import Choice, Colormap
from cellprofiler_core.setting.text import Directory, OutlineImageName, Filename
from cellprofiler_core.setting.text import Float
from cellprofiler_core.setting.text import ImageName
from cellprofiler_core.setting.text import Integer
from cellprofiler_core.setting.text import LabelName
from cellprofiler_core.utilities.core.module.identify import (
    add_object_count_measurements,
    add_object_location_measurements,
    get_object_measurement_columns,
)

from cellprofiler import __version__ as cellprofiler_version


LOGGER = logging.getLogger(__name__)

RETAINING_OUTLINES_HELP = """\
Select *{YES}* to retain the outlines of the new objects for later use
in the pipeline. For example, a common use is for quality control
purposes by overlaying them on your image of choice using the
**OverlayOutlines** module and then saving the overlay image with the
**SaveImages** module.
""".format(
    **{"YES": "Yes"}
)

OO_WITH_OVERLAP = "With overlap"
OO_WITHOUT_OVERLAP = "Without overlap"
OO_BOTH = "Both"

MODE_TRAIN = "Train"
MODE_UNTANGLE = "Untangle"

"""Shape cost method = angle shape model for cluster paths selection"""
SCM_ANGLE_SHAPE_MODEL = "angle_shape_model"

"""Maximum # of sets of paths considered at any level"""
MAX_CONSIDERED = 50000
"""Maximum # of different paths considered for input"""
MAX_PATHS = 400

"""Name of the worm training data list inside the image set"""
TRAINING_DATA = "TrainingData"

"""An attribute on the object names that tags them as worm objects"""
ATTR_WORM_MEASUREMENTS = "WormMeasurements"
######################################################
#
# Features measured
#
######################################################

"""Worm untangling measurement category"""
C_WORM = "Worm"

"""The length of the worm skeleton"""
F_LENGTH = "Length"

"""The angle at each of the control points (Worm_Angle_1 for example)"""
F_ANGLE = "Angle"

"""The X coordinate of a control point (Worm_ControlPointX_14 for example)"""
F_CONTROL_POINT_X = "ControlPointX"

"""The Y coordinate of a control point (Worm_ControlPointY_14 for example)"""
F_CONTROL_POINT_Y = "ControlPointY"

######################################################
#
# Training file XML tags:
#
######################################################

T_NAMESPACE = "http://www.cellprofiler.org/linked_files/schemas/UntangleWorms.xsd"
T_TRAINING_DATA = "training-data"
T_VERSION = "version"
T_MIN_AREA = "min-area"
T_MAX_AREA = "max-area"
T_COST_THRESHOLD = "cost-threshold"
T_NUM_CONTROL_POINTS = "num-control-points"
T_MEAN_ANGLES = "mean-angles"
T_INV_ANGLES_COVARIANCE_MATRIX = "inv-angles-covariance-matrix"
T_MAX_SKEL_LENGTH = "max-skel-length"
T_MAX_RADIUS = "max-radius"
T_MIN_PATH_LENGTH = "min-path-length"
T_MAX_PATH_LENGTH = "max-path-length"
T_MEDIAN_WORM_AREA = "median-worm-area"
T_OVERLAP_WEIGHT = "overlap-weight"
T_LEFTOVER_WEIGHT = "leftover-weight"
T_RADII_FROM_TRAINING = "radii-from-training"
T_TRAINING_SET_SIZE = "training-set-size"
T_VALUES = "values"
T_VALUE = "value"

C_ALL = "Process all clusters"
C_ALL_VALUE = numpy.iinfo(int).max
C_MEDIUM = "Medium"
C_MEDIUM_VALUE = 200
C_HIGH = "High"
C_HIGH_VALUE = 600
C_VERY_HIGH = "Very high"
C_VERY_HIGH_VALUE = 1000
C_CUSTOM = "Custom"

complexity_limits = {
    C_ALL: C_ALL_VALUE,
    C_MEDIUM: C_MEDIUM_VALUE,
    C_HIGH: C_HIGH_VALUE,
    C_VERY_HIGH: C_VERY_HIGH_VALUE,
}


class UntangleWorms(Module):
    variable_revision_number = 2
    category = ["Worm Toolbox"]
    module_name = "UntangleWorms"

    def create_settings(self):
        """Create the settings that parameterize the module"""
        self.mode = Choice(
            "Train or untangle worms?",
            [MODE_UNTANGLE, MODE_TRAIN],
            doc="""\
**UntangleWorms** has two modes:

-  *%(MODE_TRAIN)s* creates one training set per image group, using all
   of the worms in the training set as examples. It then writes the
   training file at the end of each image group.
-  *%(MODE_UNTANGLE)s* uses the training file to untangle images of
   worms.

{grouping}
""".format(
                grouping=USING_METADATA_GROUPING_HELP_REF
            )
            % globals(),
        )

        self.image_name = ImageName(
            "Select the input binary image",
            "None",
            doc="""\
A binary image where the foreground indicates the worm
shapes. The binary image can be produced by the **ApplyThreshold**
module.""",
        )

        self.overlap = Choice(
            "Overlap style",
            [OO_BOTH, OO_WITH_OVERLAP, OO_WITHOUT_OVERLAP],
            doc="""\
This setting determines which style objects are output. If two worms
overlap, you have a choice of including the overlapping regions in both
worms or excluding the overlapping regions from both worms.

-  *%(OO_WITH_OVERLAP)s:* Save objects including overlapping regions.
-  *%(OO_WITHOUT_OVERLAP)s:* Save only the portions of objects that do
   not overlap.
-  *%(OO_BOTH)s:* Save two versions: with and without overlap.
"""
            % globals(),
        )

        self.overlap_objects = LabelName(
            "Name the output overlapping worm objects",
            "OverlappingWorms",
            provided_attributes={ATTR_WORM_MEASUREMENTS: True},
            doc="""\
*(Used only if “%(MODE_UNTANGLE)s” mode and “%(OO_BOTH)s” or
“%(OO_WITH_OVERLAP)s” overlap style are selected)*

This setting names the objects representing the overlapping worms. When
worms cross, they overlap and pixels are shared by both of the
overlapping worms. The overlapping worm objects share these pixels and
measurements of both overlapping worms will include these pixels in the
measurements of both worms.
"""
            % globals(),
        )

        self.wants_overlapping_outlines = Binary(
            "Retain outlines of the overlapping objects?",
            False,
            doc="""\
*(Used only if “%(MODE_UNTANGLE)s” mode and “%(OO_BOTH)s” or
“%(OO_WITH_OVERLAP)s” overlap style are selected)*

%(RETAINING_OUTLINES_HELP)s
"""
            % globals(),
        )

        self.overlapping_outlines_colormap = Colormap(
            "Outline colormap?",
            doc="""\
*(Used only if “%(MODE_UNTANGLE)s” mode, “%(OO_BOTH)s” or
“%(OO_WITH_OVERLAP)s” overlap style and retaining outlines are
selected )*

This setting controls the colormap used when drawing outlines. The
outlines are drawn in color to highlight the shapes of each worm in a
group of overlapping worms
"""
            % globals(),
        )

        self.overlapping_outlines_name = OutlineImageName(
            "Name the overlapped outline image",
            "OverlappedWormOutlines",
            doc="""\
*(Used only if “%(MODE_UNTANGLE)s” mode and “%(OO_BOTH)s” or
“%(OO_WITH_OVERLAP)s” overlap style are selected)*

This is the name of the outlines of the overlapped worms.
"""
            % globals(),
        )

        self.nonoverlapping_objects = LabelName(
            "Name the output non-overlapping worm objects",
            "NonOverlappingWorms",
            provided_attributes={ATTR_WORM_MEASUREMENTS: True},
            doc="""\
*(Used only if “%(MODE_UNTANGLE)s” mode and “%(OO_BOTH)s” or
“%(OO_WITH_OVERLAP)s” overlap style are selected)*

This setting names the objects representing the worms, excluding those
regions where the worms overlap. When worms cross, there are pixels that
cannot be unambiguously assigned to one worm or the other. These pixels
are excluded from both worms in the non-overlapping objects and will not
be a part of the measurements of either worm.
"""
            % globals(),
        )

        self.wants_nonoverlapping_outlines = Binary(
            "Retain outlines of the non-overlapping worms?",
            False,
            doc="""\
*(Used only if “%(MODE_UNTANGLE)s” mode and “%(OO_BOTH)s” or
“%(OO_WITH_OVERLAP)s” overlap style are selected)*

%(RETAINING_OUTLINES_HELP)s
"""
            % globals(),
        )

        self.nonoverlapping_outlines_name = OutlineImageName(
            "Name the non-overlapped outlines image",
            "NonoverlappedWormOutlines",
            doc="""\
*(Used only if “%(MODE_UNTANGLE)s” mode and “%(OO_BOTH)s” or
“%(OO_WITH_OVERLAP)s” overlap style are selected)*

This is the name of the of the outlines of the worms with the
overlapping sections removed.
"""
            % globals(),
        )

        self.training_set_directory = Directory(
            "Training set file location",
            support_urls=True,
            allow_metadata=False,
            doc="""\
Select the folder containing the training set to be loaded.
{folder_choice}

An additional option is the following:

-  *URL*: Use the path part of a URL. For instance, your training set
   might be hosted at
   ``http://my_institution.edu/server/my_username/TrainingSet.xml`` To
   access this file, you would choose *URL* and enter
   ``http://my_institution.edu/server/my_username/`` as the path
   location.
""".format(
                folder_choice=IO_FOLDER_CHOICE_HELP_TEXT
            ),
        )
        self.training_set_directory.dir_choice = DEFAULT_OUTPUT_FOLDER_NAME

        def get_directory_fn():
            """Get the directory for the CSV file name"""
            return self.training_set_directory.get_absolute_path()

        def set_directory_fn(path):
            dir_choice, custom_path = self.training_set_directory.get_parts_from_path(
                path
            )
            self.training_set_directory.join_parts(dir_choice, custom_path)

        self.training_set_file_name = Filename(
            "Training set file name",
            "TrainingSet.xml",
            doc="""This is the name of the training set file.""",
            get_directory_fn=get_directory_fn,
            set_directory_fn=set_directory_fn,
            browse_msg="Choose training set",
            exts=[("Worm training set (*.xml)", "*.xml"), ("All files (*.*)", "*.*")],
        )

        self.wants_training_set_weights = Binary(
            "Use training set weights?",
            True,
            doc="""\
Select "*Yes*" to use the overlap and leftover weights from the
training set.

Select "*No*" to override these weights with user-specified values.
"""
            % globals(),
        )

        self.override_overlap_weight = Float(
            "Overlap weight",
            5,
            0,
            doc="""\
*(Used only if not using training set weights)*

This setting controls how much weight is given to overlaps between
worms. **UntangleWorms** charges a penalty to a particular putative
grouping of worms that overlap equal to the length of the overlapping
region times the overlap weight.

-  Increase the overlap weight to make **UntangleWorms** avoid
   overlapping portions of worms.
-  Decrease the overlap weight to make **UntangleWorms** ignore
   overlapping portions of worms.
""",
        )

        self.override_leftover_weight = Float(
            "Leftover weight",
            10,
            0,
            doc="""\
*(Used only if not using training set weights)*

This setting controls how much weight is given to areas not covered by
worms. **UntangleWorms** charges a penalty to a particular putative
grouping of worms that fail to cover all of the foreground of a binary
image. The penalty is equal to the length of the uncovered region
times the leftover weight.

-  Increase the leftover weight to make **UntangleWorms** cover more
   foreground with worms.
-  Decrease the overlap weight to make **UntangleWorms** ignore
   uncovered foreground.
""",
        )

        self.min_area_percentile = Float(
            "Minimum area percentile",
            1,
            0,
            100,
            doc="""\
*(Used only if “%(MODE_TRAIN)s” mode is selected)*

**UntangleWorms** will discard single worms whose area is less than a
certain minimum. It ranks all worms in the training set according to
area and then picks the worm at this percentile. It then computes the
minimum area allowed as this worm’s area times the minimum area factor.
"""
            % globals(),
        )

        self.min_area_factor = Float(
            "Minimum area factor",
            0.85,
            0,
            doc="""\
*(Used only if “%(MODE_TRAIN)s” mode is selected)*

This setting is a multiplier that is applied to the area of the worm,
selected as described in the documentation for *Minimum area
percentile*.
"""
            % globals(),
        )

        self.max_area_percentile = Float(
            "Maximum area percentile",
            90,
            0,
            100,
            doc="""\
*(Used only if “%(MODE_TRAIN)s” mode is selected)*

**UntangleWorms** uses a maximum area to distinguish between single
worms and clumps of worms. Any blob whose area is less than the maximum
area is considered to be a single worm whereas any blob whose area is
greater is considered to be two or more worms. **UntangleWorms** orders
all worms in the training set by area and picks the worm at the
percentile given by this setting. It then multiplies this worm’s area by
the *Maximum area factor* (see below) to get the maximum area
"""
            % globals(),
        )

        self.max_area_factor = Float(
            "Maximum area factor",
            1.0,
            0,
            doc="""\
*(Used only if “%(MODE_TRAIN)s” mode is selected)*

The *Maximum area factor* setting is used to compute the maximum area as
described above in *Maximum area percentile*.
"""
            % globals(),
        )

        self.min_length_percentile = Float(
            "Minimum length percentile",
            1,
            0,
            100,
            doc="""\
*(Used only if “%(MODE_TRAIN)s” mode is selected)*

**UntangleWorms** uses the minimum length to restrict its search for
worms in a clump to worms of at least the minimum length.
**UntangleWorms** sorts all worms by length and picks the worm at the
percentile indicated by this setting. It then multiplies the length of
this worm by the *Minimum length factor* (see below) to get the minimum
length.
"""
            % globals(),
        )

        self.min_length_factor = Float(
            "Minimum length factor",
            0.9,
            0,
            doc="""\
*(Used only if “%(MODE_TRAIN)s” mode is selected)*

**UntangleWorms** uses the *Minimum length factor* to compute the
minimum length from the training set as described in the documentation
above for *Minimum length percentile*
"""
            % globals(),
        )

        self.max_length_percentile = Float(
            "Maximum length percentile",
            99,
            0,
            100,
            doc="""\
*(Used only if “%(MODE_TRAIN)s” mode is selected)*

**UntangleWorms** uses the maximum length to restrict its search for
worms in a clump to worms of at least the maximum length. It computes
this length by sorting all of the training worms by length. It then
selects the worm at the *Maximum length percentile* and multiplies that
worm’s length by the *Maximum length factor* to get the maximum length
"""
            % globals(),
        )

        self.max_length_factor = Float(
            "Maximum length factor",
            1.1,
            0,
            doc="""\
*(Used only if “%(MODE_TRAIN)s” mode is selected)*

**UntangleWorms** uses this setting to compute the maximum length as
described in *Maximum length percentile* above
"""
            % globals(),
        )

        self.max_cost_percentile = Float(
            "Maximum cost percentile",
            90,
            0,
            100,
            doc="""\
*(Used only if “%(MODE_TRAIN)s” mode is selected)*

**UntangleWorms** computes a shape-based cost for each worm it
considers. It will restrict the allowed cost to less than the cost
threshold. During training, **UntangleWorms** computes the shape cost of
every worm in the training set. It then orders them by cost and uses
*Maximum cost percentile* to pick the worm at the given percentile. It
them multiplies this worm’s cost by the *Maximum cost factor* to compute
the cost threshold.
"""
            % globals(),
        )

        self.max_cost_factor = Float(
            "Maximum cost factor",
            1.9,
            0,
            doc="""\
*(Used only “%(MODE_TRAIN)s” mode is selected)*

**UntangleWorms** uses this setting to compute the cost threshold as
described in *Maximum cost percentile* above.
"""
            % globals(),
        )

        self.num_control_points = Integer(
            "Number of control points",
            21,
            3,
            50,
            doc="""\
*(Used only if “%(MODE_TRAIN)s” mode is selected)*

This setting controls the number of control points that will be sampled
when constructing a worm shape from its skeleton.
"""
            % globals(),
        )

        self.max_radius_percentile = Float(
            "Maximum radius percentile",
            90,
            0,
            100,
            doc="""\
*(Used only if “%(MODE_TRAIN)s” mode is selected)*

**UntangleWorms** uses the maximum worm radius during worm
skeletonization. **UntangleWorms** sorts the radii of worms in
increasing size and selects the worm at this percentile. It then
multiplies this worm’s radius by the *Maximum radius factor* (see below)
to compute the maximum radius.
"""
            % globals(),
        )

        self.max_radius_factor = Float(
            "Maximum radius factor",
            1,
            0,
            doc="""\
*(Used only if “%(MODE_TRAIN)s” mode is selected)*

**UntangleWorms** uses this setting to compute the maximum radius as
described in *Maximum radius percentile* above.
"""
            % globals(),
        )

        self.complexity = Choice(
            "Maximum complexity",
            [C_MEDIUM, C_HIGH, C_VERY_HIGH, C_ALL, C_CUSTOM],
            value=C_HIGH,
            doc="""\
*(Used only if “%(MODE_UNTANGLE)s” mode is selected)*

This setting controls which clusters of worms are rejected as being
too time-consuming to process. **UntangleWorms** judges complexity
based on the number of segments in a cluster where a segment is the
piece of a worm between crossing points or from the head or tail to
the first or last crossing point. The choices are:

-  *%(C_MEDIUM)s*: %(C_MEDIUM_VALUE)d segments (takes up to several
   minutes to process)
-  *%(C_HIGH)s*: %(C_HIGH_VALUE)d segments (takes up to a
   quarter-hour to process)
-  *%(C_VERY_HIGH)s*: %(C_VERY_HIGH_VALUE)d segments (can take
   hours to process)
-  *%(C_CUSTOM)s*: allows you to enter a custom number of segments.
-  *%(C_ALL)s*: Process all worms, regardless of complexity
"""
            % globals(),
        )

        self.custom_complexity = Integer(
            "Custom complexity",
            400,
            20,
            doc="""\
*(Used only if “%(MODE_UNTANGLE)s” mode and “%(C_CUSTOM)s” complexity
are selected )*

Enter the maximum number of segments of any cluster that
should be processed.
"""
            % globals(),
        )

    def settings(self):
        return [
            self.image_name,
            self.overlap,
            self.overlap_objects,
            self.nonoverlapping_objects,
            self.training_set_directory,
            self.training_set_file_name,
            self.wants_training_set_weights,
            self.override_overlap_weight,
            self.override_leftover_weight,
            self.wants_overlapping_outlines,
            self.overlapping_outlines_colormap,
            self.overlapping_outlines_name,
            self.wants_nonoverlapping_outlines,
            self.nonoverlapping_outlines_name,
            self.mode,
            self.min_area_percentile,
            self.min_area_factor,
            self.max_area_percentile,
            self.max_area_factor,
            self.min_length_percentile,
            self.min_length_factor,
            self.max_length_percentile,
            self.max_length_factor,
            self.max_cost_percentile,
            self.max_cost_factor,
            self.num_control_points,
            self.max_radius_percentile,
            self.max_radius_factor,
            self.complexity,
            self.custom_complexity,
        ]

    def help_settings(self):
        return [
            self.mode,
            self.image_name,
            self.overlap,
            self.overlap_objects,
            self.nonoverlapping_objects,
            self.complexity,
            self.custom_complexity,
            self.training_set_directory,
            self.training_set_file_name,
            self.wants_training_set_weights,
            self.override_overlap_weight,
            self.override_leftover_weight,
            self.wants_overlapping_outlines,
            self.overlapping_outlines_colormap,
            self.overlapping_outlines_name,
            self.wants_nonoverlapping_outlines,
            self.nonoverlapping_outlines_name,
            self.min_area_percentile,
            self.min_area_factor,
            self.max_area_percentile,
            self.max_area_factor,
            self.min_length_percentile,
            self.min_length_factor,
            self.max_length_percentile,
            self.max_length_factor,
            self.max_cost_percentile,
            self.max_cost_factor,
            self.num_control_points,
            self.max_radius_percentile,
            self.max_radius_factor,
        ]

    def visible_settings(self):
        result = [self.mode, self.image_name]
        if self.mode == MODE_UNTANGLE:
            result += [self.overlap]
            if self.overlap in (OO_WITH_OVERLAP, OO_BOTH):
                result += [self.overlap_objects, self.wants_overlapping_outlines]
                if self.wants_overlapping_outlines:
                    result += [
                        self.overlapping_outlines_colormap,
                        self.overlapping_outlines_name,
                    ]
            if self.overlap in (OO_WITHOUT_OVERLAP, OO_BOTH):
                result += [
                    self.nonoverlapping_objects,
                    self.wants_nonoverlapping_outlines,
                ]
                if self.wants_nonoverlapping_outlines:
                    result += [self.nonoverlapping_outlines_name]
                result += [self.complexity]
                if self.complexity == C_CUSTOM:
                    result += [self.custom_complexity]
        result += [
            self.training_set_directory,
            self.training_set_file_name,
            self.wants_training_set_weights,
        ]
        if not self.wants_training_set_weights:
            result += [self.override_overlap_weight, self.override_leftover_weight]
            if self.mode == MODE_TRAIN:
                result += [
                    self.min_area_percentile,
                    self.min_area_factor,
                    self.max_area_percentile,
                    self.max_area_factor,
                    self.min_length_percentile,
                    self.min_length_factor,
                    self.max_length_percentile,
                    self.max_length_factor,
                    self.max_cost_percentile,
                    self.max_cost_factor,
                    self.num_control_points,
                    self.max_radius_percentile,
                    self.max_radius_factor,
                ]
        return result

    def overlap_weight(self, params):
        """The overlap weight to use in the cost calculation"""
        if not self.wants_training_set_weights:
            return self.override_overlap_weight.value
        elif params is None:
            return 2
        else:
            return params.overlap_weight

    def leftover_weight(self, params):
        """The leftover weight to use in the cost calculation"""
        if not self.wants_training_set_weights:
            return self.override_leftover_weight.value
        elif params is None:
            return 10
        else:
            return params.leftover_weight

    def ncontrol_points(self):
        """# of control points when making a training set"""
        if self.mode == MODE_UNTANGLE:
            params = self.read_params()
            return params.num_control_points
        if not self.wants_training_set_weights:
            return 21
        else:
            return self.num_control_points.value

    @property
    def max_complexity(self):
        if self.complexity != C_CUSTOM:
            return complexity_limits[self.complexity.value]
        return self.custom_complexity.value

    def prepare_group(self, workspace, grouping, image_numbers):
        """Prepare to process a group of worms"""
        d = self.get_dictionary(workspace.image_set_list)
        d[TRAINING_DATA] = []

    def get_dictionary_for_worker(self):
        """Don't share the training data dictionary between workers"""
        return {TRAINING_DATA: []}

    def run(self, workspace):
        """Run the module on the current image set"""
        if self.mode == MODE_TRAIN:
            self.run_train(workspace)
        else:
            self.run_untangle(workspace)

    class TrainingData(object):
        """One worm's training data"""

        def __init__(self, area, skel_length, angles, radial_profile):
            self.area = area
            self.skel_length = skel_length
            self.angles = angles
            self.radial_profile = radial_profile

    def run_train(self, workspace):
        """Train based on the current image set"""

        image_name = self.image_name.value
        image_set = workspace.image_set
        image = image_set.get_image(image_name, must_be_binary=True)
        num_control_points = self.ncontrol_points()
        labels, count = scipy.ndimage.label(
            image.pixel_data, centrosome.cpmorphology.eight_connect
        )
        skeleton = centrosome.cpmorphology.skeletonize(image.pixel_data)
        distances = scipy.ndimage.distance_transform_edt(image.pixel_data)
        worms = self.get_dictionary(workspace.image_set_list)[TRAINING_DATA]
        areas = numpy.bincount(labels.ravel())
        if self.show_window:
            dworms = workspace.display_data.worms = []
            workspace.display_data.input_image = image.pixel_data
        for i in range(1, count + 1):
            mask = labels == i
            graph = self.get_graph_from_binary(image.pixel_data & mask, skeleton & mask)
            path_coords, path = self.get_longest_path_coords(
                graph, numpy.iinfo(int).max
            )
            if len(path_coords) == 0:
                continue
            cumul_lengths = self.calculate_cumulative_lengths(path_coords)
            if cumul_lengths[-1] == 0:
                continue
            control_points = self.sample_control_points(
                path_coords, cumul_lengths, num_control_points
            )
            angles = self.get_angles(control_points)
            #
            # Interpolate in 2-d when looking up the distances
            #
            fi, fj = (control_points - numpy.floor(control_points)).transpose()
            ci, cj = control_points.astype(int).transpose()
            ci1 = numpy.minimum(ci + 1, labels.shape[0] - 1)
            cj1 = numpy.minimum(cj + 1, labels.shape[1] - 1)
            radial_profile = numpy.zeros(num_control_points)
            for ii, jj, f in (
                (ci, cj, (1 - fi) * (1 - fj)),
                (ci1, cj, fi * (1 - fj)),
                (ci, cj1, (1 - fi) * fj),
                (ci1, cj1, fi * fj),
            ):
                radial_profile += distances[ii, jj] * f
            worms.append(
                self.TrainingData(areas[i], cumul_lengths[-1], angles, radial_profile)
            )
            if self.show_window:
                dworms.append(control_points)

    def is_aggregation_module(self):
        """Building the model requires aggregation across image sets"""
        return self.mode == MODE_TRAIN

    def post_group(self, workspace, grouping):
        """Write the training data file as we finish grouping."""
        if self.mode == MODE_TRAIN:
            worms = self.get_dictionary(workspace.image_set_list)[TRAINING_DATA]
            #
            # Either get weights from our instance or instantiate
            # the default UntangleWorms to get the defaults
            #
            if self.wants_training_set_weights:
                this = self
            else:
                this = UntangleWorms()
            nworms = len(worms)
            num_control_points = self.ncontrol_points()
            areas = numpy.zeros(nworms)
            lengths = numpy.zeros(nworms)
            radial_profiles = numpy.zeros((num_control_points, nworms))
            angles = numpy.zeros((num_control_points - 2, nworms))
            for i, training_data in enumerate(worms):
                areas[i] = training_data.area
                lengths[i] = training_data.skel_length
                angles[:, i] = training_data.angles
                radial_profiles[:, i] = training_data.radial_profile
            areas.sort()
            lengths.sort()
            min_area = this.min_area_factor.value * numpy.percentile(
                areas, this.min_area_percentile.value
            )
            max_area = this.max_area_factor.value * numpy.percentile(
                areas, this.max_area_percentile.value
            )
            median_area = numpy.median(areas)
            min_length = this.min_length_factor.value * numpy.percentile(
                lengths, this.min_length_percentile.value
            )
            max_length = this.max_length_factor.value * numpy.percentile(
                lengths, this.max_length_percentile.value
            )
            max_skel_length = numpy.percentile(lengths, this.max_length_percentile.value)
            max_radius = this.max_radius_factor.value * numpy.percentile(
                radial_profiles.flatten(), this.max_radius_percentile.value
            )
            mean_radial_profile = numpy.mean(radial_profiles, 1)
            #
            # Mirror the angles by negating them. Flip heads and tails
            # because they are arbitrary.
            #
            angles = numpy.hstack((angles, -angles, angles[::-1, :], -angles[::-1, :]))
            lengths = numpy.hstack([lengths] * 4)
            feat_vectors = numpy.vstack((angles, lengths[numpy.newaxis, :]))
            mean_angles_length = numpy.mean(feat_vectors, 1)
            fv_adjusted = feat_vectors - mean_angles_length[:, numpy.newaxis]
            angles_covariance_matrix = numpy.cov(fv_adjusted)
            inv_angles_covariance_matrix = numpy.linalg.inv(angles_covariance_matrix)
            angle_costs = [
                numpy.dot(numpy.dot(fv, inv_angles_covariance_matrix), fv)
                for fv in fv_adjusted.transpose()
            ]
            max_cost = this.max_cost_factor.value * numpy.percentile(
                angle_costs, this.max_cost_percentile.value
            )
            #
            # Write it to disk
            #
            if workspace.pipeline.test_mode:
                return
            m = workspace.measurements
            assert isinstance(m, Measurements)
            path = self.training_set_directory.get_absolute_path(m)
            file_name = m.apply_metadata(self.training_set_file_name.value)
            fd = open(os.path.join(path, file_name), "w")
            doc = DOM.getDOMImplementation().createDocument(
                T_NAMESPACE, T_TRAINING_DATA, None
            )
            top = doc.documentElement
            top.setAttribute("xmlns", T_NAMESPACE)
            ver = Version(cellprofiler_version)
            for tag, value in (
                (T_VERSION, int(f"{ver.major}{ver.minor}{ver.micro}")),
                (T_MIN_AREA, min_area),
                (T_MAX_AREA, max_area),
                (T_COST_THRESHOLD, max_cost),
                (T_NUM_CONTROL_POINTS, num_control_points),
                (T_MAX_SKEL_LENGTH, max_skel_length),
                (T_MIN_PATH_LENGTH, min_length),
                (T_MAX_PATH_LENGTH, max_length),
                (T_MEDIAN_WORM_AREA, median_area),
                (T_MAX_RADIUS, max_radius),
                (T_OVERLAP_WEIGHT, this.override_overlap_weight.value),
                (T_LEFTOVER_WEIGHT, this.override_leftover_weight.value),
                (T_TRAINING_SET_SIZE, nworms),
            ):
                element = doc.createElement(tag)
                content = doc.createTextNode(str(value))
                element.appendChild(content)
                top.appendChild(element)
            for tag, values in (
                (T_MEAN_ANGLES, mean_angles_length),
                (T_RADII_FROM_TRAINING, mean_radial_profile),
            ):
                element = doc.createElement(tag)
                top.appendChild(element)
                for value in values:
                    value_element = doc.createElement(T_VALUE)
                    content = doc.createTextNode(str(value))
                    value_element.appendChild(content)
                    element.appendChild(value_element)
            element = doc.createElement(T_INV_ANGLES_COVARIANCE_MATRIX)
            top.appendChild(element)
            for row in inv_angles_covariance_matrix:
                values = doc.createElement(T_VALUES)
                element.appendChild(values)
                for col in row:
                    value = doc.createElement(T_VALUE)
                    content = doc.createTextNode(str(col))
                    value.appendChild(content)
                    values.appendChild(value)
            doc.writexml(fd, addindent="  ", newl="\n")
            fd.close()
            if self.show_window:
                workspace.display_data.angle_costs = angle_costs
                workspace.display_data.feat_vectors = feat_vectors
                workspace.display_data.angles_covariance_matrix = (
                    angles_covariance_matrix
                )

    def run_untangle(self, workspace):
        """Untangle based on the current image set"""
        params = self.read_params()
        image_name = self.image_name.value
        image_set = workspace.image_set
        image = image_set.get_image(image_name, must_be_binary=True)
        labels, count = scipy.ndimage.label(
            image.pixel_data, centrosome.cpmorphology.eight_connect
        )
        #
        # Skeletonize once, then remove any points in the skeleton
        # that are adjacent to the edge of the image, then skeletonize again.
        #
        # This gets rid of artifacts that cause combinatoric explosions:
        #
        #    * * * * * * * *
        #      *   *   *
        #    * * * * * * * *
        #
        skeleton = centrosome.cpmorphology.skeletonize(image.pixel_data)
        eroded = scipy.ndimage.binary_erosion(
            image.pixel_data, centrosome.cpmorphology.eight_connect
        )
        skeleton = centrosome.cpmorphology.skeletonize(skeleton & eroded)
        #
        # The path skeletons
        #
        all_path_coords = []
        if count != 0 and numpy.sum(skeleton) != 0:
            areas = numpy.bincount(labels.flatten())
            skeleton_areas = numpy.bincount(labels[skeleton])
            current_index = 1
            for i in range(1, count + 1):
                if (
                    areas[i] < params.min_worm_area
                    or i >= skeleton_areas.shape[0]
                    or skeleton_areas[i] == 0
                ):
                    # Completely exclude the worm
                    continue
                elif areas[i] <= params.max_area:
                    path_coords, path_struct = self.single_worm_find_path(
                        workspace, labels, i, skeleton, params
                    )
                    if len(path_coords) > 0 and self.single_worm_filter(
                        workspace, path_coords, params
                    ):
                        all_path_coords.append(path_coords)
                else:
                    graph = self.cluster_graph_building(
                        workspace, labels, i, skeleton, params
                    )
                    if len(graph.segments) > self.max_complexity:
                        LOGGER.warning(
                            "Warning: rejecting cluster of %d segments.\n"
                            % len(graph.segments)
                        )
                        continue
                    paths = self.get_all_paths(
                        graph, params.min_path_length, params.max_path_length
                    )
                    paths_selected = self.cluster_paths_selection(
                        graph, paths, labels, i, params
                    )
                    del graph
                    del paths
                    all_path_coords += paths_selected
        (
            ijv,
            all_lengths,
            all_angles,
            all_control_coords_x,
            all_control_coords_y,
        ) = self.worm_descriptor_building(all_path_coords, params, labels.shape)
        if self.show_window:
            workspace.display_data.input_image = image.pixel_data
        object_set = workspace.object_set
        assert isinstance(object_set, ObjectSet)
        measurements = workspace.measurements
        assert isinstance(measurements, Measurements)

        object_names = []
        if self.overlap in (OO_WITH_OVERLAP, OO_BOTH):
            o = Objects()
            o.ijv = ijv
            o.parent_image = image
            name = self.overlap_objects.value
            object_names.append(name)
            object_set.add_objects(o, name)
            add_object_count_measurements(measurements, name, o.count)
            if self.show_window:
                workspace.display_data.overlapping_labels = [
                    l for l, idx in o.get_labels()
                ]

            if o.count == 0:
                center_x = numpy.zeros(0)
                center_y = numpy.zeros(0)
            else:
                center_x = numpy.bincount(ijv[:, 2], ijv[:, 1])[o.indices] / o.areas
                center_y = numpy.bincount(ijv[:, 2], ijv[:, 0])[o.indices] / o.areas
            measurements.add_measurement(name, M_LOCATION_CENTER_X, center_x)
            measurements.add_measurement(name, M_LOCATION_CENTER_Y, center_y)
            measurements.add_measurement(name, M_NUMBER_OBJECT_NUMBER, o.indices)
            #
            # Save outlines
            #
            if self.wants_overlapping_outlines:
                from matplotlib.cm import ScalarMappable

                colormap = self.overlapping_outlines_colormap.value
                if colormap == "Default":
                    colormap = get_default_colormap()
                if len(ijv) == 0:
                    ishape = image.pixel_data.shape
                    outline_pixels = numpy.zeros((ishape[0], ishape[1], 3))
                else:
                    my_map = ScalarMappable(cmap=colormap)
                    colors = my_map.to_rgba(numpy.unique(ijv[:, 2]))
                    outline_pixels = o.make_ijv_outlines(colors[:, :3])
                outline_image = Image(outline_pixels, parent_image=image)
                image_set.add(self.overlapping_outlines_name.value, outline_image)

        if self.overlap in (OO_WITHOUT_OVERLAP, OO_BOTH):
            #
            # Sum up the number of overlaps using a sparse matrix
            #
            overlap_hits = coo.coo_matrix(
                (numpy.ones(len(ijv)), (ijv[:, 0], ijv[:, 1])), image.pixel_data.shape
            )
            overlap_hits = overlap_hits.toarray()
            mask = overlap_hits == 1
            labels = coo.coo_matrix((ijv[:, 2], (ijv[:, 0], ijv[:, 1])), mask.shape)
            labels = labels.toarray()
            labels[~mask] = 0
            o = Objects()
            o.segmented = labels
            o.parent_image = image
            name = self.nonoverlapping_objects.value
            object_names.append(name)
            object_set.add_objects(o, name)
            add_object_count_measurements(measurements, name, o.count)
            add_object_location_measurements(measurements, name, labels, o.count)
            if self.show_window:
                workspace.display_data.nonoverlapping_labels = [
                    l for l, idx in o.get_labels()
                ]

            if self.wants_nonoverlapping_outlines:
                outline_pixels = outline(labels) > 0
                outline_image = Image(outline_pixels, parent_image=image)
                image_set.add(self.nonoverlapping_outlines_name.value, outline_image)
        for name in object_names:
            measurements.add_measurement(
                name, "_".join((C_WORM, F_LENGTH)), all_lengths
            )
            for values, ftr in (
                (all_angles, F_ANGLE),
                (all_control_coords_x, F_CONTROL_POINT_X),
                (all_control_coords_y, F_CONTROL_POINT_Y),
            ):
                for i in range(values.shape[1]):
                    feature = "_".join((C_WORM, ftr, str(i + 1)))
                    measurements.add_measurement(name, feature, values[:, i])

    def display(self, workspace, figure):
        from cellprofiler.gui.constants.figure import CPLDM_ALPHA

        if self.mode == MODE_UNTANGLE:
            figure.set_subplots((1, 1))
            cplabels = []
            if self.overlap in (OO_BOTH, OO_WITH_OVERLAP):
                title = self.overlap_objects.value
                cplabels.append(
                    dict(
                        name=self.overlap_objects.value,
                        labels=workspace.display_data.overlapping_labels,
                        mode=CPLDM_ALPHA,
                    )
                )
            else:
                title = self.nonoverlapping_objects.value
            if self.overlap in (OO_BOTH, OO_WITHOUT_OVERLAP):
                cplabels.append(
                    dict(
                        name=self.nonoverlapping_objects.value,
                        labels=workspace.display_data.nonoverlapping_labels,
                    )
                )
            image = workspace.display_data.input_image
            if image.ndim == 2:
                figure.subplot_imshow_grayscale(
                    0, 0, image, title=title, cplabels=cplabels
                )
        else:
            figure.set_subplots((1, 1))
            figure.subplot_imshow_bw(
                0, 0, workspace.display_data.input_image, title=self.image_name.value
            )
            axes = figure.subplot(0, 0)
            for control_points in workspace.display_data.worms:
                axes.plot(
                    control_points[:, 1], control_points[:, 0], "ro-", markersize=4
                )

    def display_post_group(self, workspace, figure):
        """Display some statistical information about training, post-group

        workspace - holds the display data used to create the display

        figure - the module's figure.
        """
        if self.mode == MODE_TRAIN:
            from matplotlib.transforms import Bbox

            angle_costs = workspace.display_data.angle_costs
            feat_vectors = workspace.display_data.feat_vectors
            angles_covariance_matrix = workspace.display_data.angles_covariance_matrix
            figure = workspace.create_or_find_figure(
                subplots=(4, 1), window_name="UntangleWorms_PostGroup"
            )
            f = figure.figure
            f.clf()
            a = f.add_subplot(1, 4, 1)
            a.set_position((Bbox([[0.1, 0.1], [0.15, 0.9]])))
            a.boxplot(angle_costs)
            a.set_title("Costs")
            a = f.add_subplot(1, 4, 2)
            a.set_position((Bbox([[0.2, 0.1], [0.25, 0.9]])))
            a.boxplot(feat_vectors[-1, :])
            a.set_title("Lengths")
            a = f.add_subplot(1, 4, 3)
            a.set_position((Bbox([[0.30, 0.1], [0.60, 0.9]])))
            a.boxplot(feat_vectors[:-1, :].transpose() * 180 / numpy.pi)
            a.set_title("Angles")
            a = f.add_subplot(1, 4, 4)
            a.set_position((Bbox([[0.65, 0.1], [1, 0.45]])))
            a.imshow(angles_covariance_matrix[:-1, :-1], interpolation="nearest")
            a.set_title("Covariance")
            f.canvas.draw()
            figure.Refresh()

    def single_worm_find_path(self, workspace, labels, i, skeleton, params):
        """Finds the worm's skeleton  as a path.

        labels - the labels matrix, labeling single and clusters of worms

        i - the labeling of the worm of interest

        params - The parameter structure

        returns:

        path_coords: A 2 x n array, of coordinates for the path found. (Each
              point along the polyline path is represented by a column,
              i coordinates in the first row and j coordinates in the second.)

        path_struct: a structure describing the path
        """
        binary_im = labels == i
        skeleton = skeleton & binary_im
        graph_struct = self.get_graph_from_binary(binary_im, skeleton)
        return self.get_longest_path_coords(graph_struct, params.max_path_length)

    def get_graph_from_binary(
        self, binary_im, skeleton, max_radius=None, max_skel_length=None
    ):
        """Manufacture a graph of the skeleton of the worm

        Given a binary image containing a cluster of worms, returns a structure
        describing the graph structure of the skeleton of the cluster. This graph
        structure can later be used as input to e.g., get_all_paths().

        Input parameters:

        binary_im: A logical image, containing the cluster to be resolved. Must
        contain exactly one connected component.

        Output_parameters:

        graph_struct: An object with attributes

        image_size: Equal to size(binary_im).

        segments: A list describing the segments of
        the skeleton. Each element is an array of i,j coordinates
        of the pixels making up one segment, traced in the right order.

        branch_areas: A list describing the
        branch areas, i.e., the areas where different segments join. Each
        element is an array of i,j coordinates
        of the pixels making up one branch area, in no particular order.
        The branch areas will include all branchpoints,
        followed by a dilation. If max_radius is supplied, all pixels remaining
        after opening the binary image consisting of all pixels further
        than max_pix from the image background. This allows skeleton pixels
        in thick regions to be replaced by branchpoint regions, which increases
        the chance of connecting skeleton pieces correctly.

        incidence_matrix: A num_branch_areas x num_segments logical array,
        describing the incidence relations among the branch areas and
        segments. incidence_matrix(i, j) is set if and only if branch area
        i connects to segment j.

        incidence_directions: A num_branch_areas x num_segments logical
        array, intended to indicate the directions in which the segments
        are traced. incidence_directions(i,j) is set if and only if the
        "start end" (as in the direction in which the pixels are enumerated
        in graph_struct.segments) of segment j is connected to branch point
        i.

        Notes:

        1. Because of a dilatation step in obtaining them, the branch areas need
           not be (in fact, are never, unless binary_im contains all pixels)
           a subset of the foreground pixels of binary_im. However, they are a
           subset of the ones(3,3)-dilatation of binary_im.

        2. The segments are not considered to actually enter the branch areas;
           that is to say, the pixel set of the branch areas is disjoint from
           that of the segments.

        3. Even if one segment is only one pixel long (but still connects to
           two branch areas), its orientation is well-defined, i.e., one branch
           area will be chosen as starting end. (Even though in this case, the
           "positive direction" of the segment cannot be determined from the
           information in graph_struct.segments.)"""
        branch_areas_binary = centrosome.cpmorphology.branchpoints(skeleton)
        if max_radius is not None:
            #
            # Add any points that are more than the worm diameter to
            # the branchpoints. Exclude segments without supporting branchpoints:
            #
            # OK:
            #
            # * * *       * * *
            #       * * *
            # * * *       * * *
            #
            # Not OK:
            #
            # * * * * * * * * * *
            #
            strel = centrosome.cpmorphology.strel_disk(max_radius)
            far = scipy.ndimage.binary_erosion(binary_im, strel)
            far = scipy.ndimage.binary_opening(
                far, structure=centrosome.cpmorphology.eight_connect
            )
            far_labels, count = scipy.ndimage.label(far)
            far_counts = numpy.bincount(far_labels.ravel(), branch_areas_binary.ravel())
            far[far_counts[far_labels] < 2] = False
            branch_areas_binary |= far
            del far
            del far_labels
        branch_areas_binary = scipy.ndimage.binary_dilation(
            branch_areas_binary, structure=centrosome.cpmorphology.eight_connect
        )
        segments_binary = skeleton & ~branch_areas_binary
        if max_skel_length is not None and numpy.sum(segments_binary) > 0:
            max_skel_length = max(int(max_skel_length), 2)  # paranoia
            i, j, labels, order, distance, num_segments = self.trace_segments(
                segments_binary
            )
            #
            # Put breakpoints every max_skel_length, but not at end
            #
            max_order = numpy.array(
                scipy.ndimage.maximum(order, labels, numpy.arange(num_segments + 1))
            )
            big_segment = max_order >= max_skel_length
            segment_count = numpy.maximum(
                (max_order + max_skel_length - 1) / max_skel_length, 1
            ).astype(int)
            segment_length = ((max_order + 1) / segment_count).astype(int)
            new_bp_mask = (
                (order % segment_length[labels] == segment_length[labels] - 1)
                & (order != max_order[labels])
                & (big_segment[labels])
            )
            new_branch_areas_binary = numpy.zeros(segments_binary.shape, bool)
            new_branch_areas_binary[i[new_bp_mask], j[new_bp_mask]] = True
            new_branch_areas_binary = scipy.ndimage.binary_dilation(
                new_branch_areas_binary, structure=centrosome.cpmorphology.eight_connect
            )
            branch_areas_binary |= new_branch_areas_binary
            segments_binary &= ~new_branch_areas_binary
        return self.get_graph_from_branching_areas_and_segments(
            branch_areas_binary, segments_binary
        )

    def trace_segments(self, segments_binary):
        """Find distance of every point in a segment from a segment endpoint

        segments_binary - a binary mask of the segments in an image.

        returns a tuple of the following:
        i - the i coordinate of a point in the mask
        j - the j coordinate of a point in the mask
        label - the segment's label
        order - the ordering (from 0 to N-1 where N is the # of points in
                the segment.)
        distance - the propagation distance of the point from the endpoint
        num_segments - the # of labelled segments
        """
        #
        # Break long skeletons into pieces whose maximum length
        # is max_skel_length.
        #
        segments_labeled, num_segments = scipy.ndimage.label(
            segments_binary, structure=centrosome.cpmorphology.eight_connect
        )
        if num_segments == 0:
            return (
                numpy.array([], int),
                numpy.array([], int),
                numpy.array([], int),
                numpy.array([], int),
                numpy.array([]),
                0,
            )
        #
        # Get one endpoint per segment
        #
        endpoints = centrosome.cpmorphology.endpoints(segments_binary)
        #
        # Use a consistent order: pick with lowest i, then j.
        # If a segment loops upon itself, we pick an arbitrary point.
        #
        order = numpy.arange(numpy.prod(segments_binary.shape))
        order.shape = segments_binary.shape
        order[~endpoints] += numpy.prod(segments_binary.shape)
        labelrange = numpy.arange(num_segments + 1).astype(int)
        endpoint_loc = scipy.ndimage.minimum_position(
            order, segments_labeled, labelrange
        )
        endpoint_loc = numpy.array(endpoint_loc, int)
        endpoint_labels = numpy.zeros(segments_labeled.shape, numpy.int16)
        endpoint_labels[endpoint_loc[:, 0], endpoint_loc[:, 1]] = segments_labeled[
            endpoint_loc[:, 0], endpoint_loc[:, 1]
        ]
        #
        # A corner case - propagate will trace a loop around both ways. So
        # we have to find that last point and remove it so
        # it won't trace in that direction
        #
        loops = ~endpoints[endpoint_loc[1:, 0], endpoint_loc[1:, 1]]
        if numpy.any(loops):
            # Consider all points around the endpoint, finding the one
            # which is numbered last
            dilated_ep_labels = centrosome.cpmorphology.grey_dilation(
                endpoint_labels, footprint=numpy.ones((3, 3), bool)
            )
            dilated_ep_labels[dilated_ep_labels != segments_labeled] = 0
            loop_endpoints = scipy.ndimage.maximum_position(
                order, dilated_ep_labels.astype(int), labelrange[1:][loops]
            )
            loop_endpoints = numpy.array(loop_endpoints, int)
            segments_binary_temp = segments_binary.copy()
            segments_binary_temp[loop_endpoints[:, 0], loop_endpoints[:, 1]] = False
        else:
            segments_binary_temp = segments_binary
        #
        # Now propagate from the endpoints to get distances
        #
        _, distances = propagate(
            numpy.zeros(segments_binary.shape), endpoint_labels, segments_binary_temp, 1
        )
        if numpy.any(loops):
            # set the end-of-loop distances to be very large
            distances[loop_endpoints[:, 0], loop_endpoints[:, 1]] = numpy.inf
        #
        # Order points by label # and distance
        #
        i, j = numpy.mgrid[0 : segments_binary.shape[0], 0 : segments_binary.shape[1]]
        i = i[segments_binary]
        j = j[segments_binary]
        labels = segments_labeled[segments_binary]
        distances = distances[segments_binary]
        order = numpy.lexsort((distances, labels))
        i = i[order]
        j = j[order]
        labels = labels[order]
        distances = distances[order]
        #
        # Number each point in a segment consecutively. We determine
        # where each label starts. Then we subtract the start index
        # of each point's label from each point to get the order relative
        # to the first index of the label.
        #
        segment_order = numpy.arange(len(i))
        areas = numpy.bincount(labels.flatten())
        indexes = numpy.cumsum(areas) - areas
        segment_order -= indexes[labels]
        return i, j, labels, segment_order, distances, num_segments

    def get_graph_from_branching_areas_and_segments(
        self, branch_areas_binary, segments_binary
    ):
        """Turn branches + segments into a graph

        branch_areas_binary - binary mask of branch areas

        segments_binary - binary mask of segments != branch_areas

        Given two binary images, one containing "branch areas" one containing
        "segments", returns a structure describing the incidence relations
        between the branch areas and the segments.

        Output is same format as get_graph_from_binary(), so for details, see
        get_graph_from_binary
        """
        branch_areas_labeled, num_branch_areas = scipy.ndimage.label(
            branch_areas_binary, centrosome.cpmorphology.eight_connect
        )

        i, j, labels, order, distance, num_segments = self.trace_segments(
            segments_binary
        )

        ooo = numpy.lexsort((order, labels))
        i = i[ooo]
        j = j[ooo]
        labels = labels[ooo]
        order = order[ooo]
        distance = distance[ooo]
        counts = (
            numpy.zeros(0, int)
            if len(labels) == 0
            else numpy.bincount(labels.flatten())[1:]
        )

        branch_ij = numpy.argwhere(branch_areas_binary)
        if len(branch_ij) > 0:
            ooo = numpy.lexsort(
                [
                    branch_ij[:, 0],
                    branch_ij[:, 1],
                    branch_areas_labeled[branch_ij[:, 0], branch_ij[:, 1]],
                ]
            )
            branch_ij = branch_ij[ooo]
            branch_labels = branch_areas_labeled[branch_ij[:, 0], branch_ij[:, 1]]
            branch_counts = numpy.bincount(branch_areas_labeled.flatten())[1:]
        else:
            branch_labels = numpy.zeros(0, int)
            branch_counts = numpy.zeros(0, int)
        #
        # "find" the segment starts
        #
        starts = order == 0
        start_labels = numpy.zeros(segments_binary.shape, int)
        start_labels[i[starts], j[starts]] = labels[starts]
        #
        # incidence_directions = True for starts
        #
        incidence_directions = self.make_incidence_matrix(
            branch_areas_labeled, num_branch_areas, start_labels, num_segments
        )
        #
        # Get the incidence matrix for the ends
        #
        ends = numpy.cumsum(counts) - 1
        end_labels = numpy.zeros(segments_binary.shape, int)
        end_labels[i[ends], j[ends]] = labels[ends]
        incidence_matrix = self.make_incidence_matrix(
            branch_areas_labeled, num_branch_areas, end_labels, num_segments
        )
        incidence_matrix |= incidence_directions

        class Result(object):
            """A result graph:

            image_size: size of input image

            segments: a list for each segment of a forward (index = 0) and
                      reverse N x 2 array of coordinates of pixels in a segment

            segment_indexes: the index of label X into segments

            segment_counts: # of points per segment

            segment_order: for each pixel, its order when tracing

            branch_areas: an N x 2 array of branch point coordinates

            branch_area_indexes: index into the branch areas per branchpoint

            branch_area_counts: # of points in each branch

            incidence_matrix: matrix of areas x segments indicating connections

            incidence_directions: direction of each connection
            """

            def __init__(
                self,
                branch_areas_binary,
                counts,
                i,
                j,
                branch_ij,
                branch_counts,
                incidence_matrix,
                incidence_directions,
            ):
                self.image_size = tuple(branch_areas_binary.shape)
                self.segment_coords = numpy.column_stack((i, j))
                self.segment_indexes = numpy.cumsum(counts) - counts
                self.segment_counts = counts
                self.segment_order = order
                self.segments = [
                    (
                        self.segment_coords[
                            self.segment_indexes[i] : (
                                self.segment_indexes[i] + self.segment_counts[i]
                            )
                        ],
                        self.segment_coords[
                            self.segment_indexes[i] : (
                                self.segment_indexes[i] + self.segment_counts[i]
                            )
                        ][::-1],
                    )
                    for i in range(len(counts))
                ]

                self.branch_areas = branch_ij
                self.branch_area_indexes = numpy.cumsum(branch_counts) - branch_counts
                self.branch_area_counts = branch_counts
                self.incidence_matrix = incidence_matrix
                self.incidence_directions = incidence_directions

        return Result(
            branch_areas_binary,
            counts,
            i,
            j,
            branch_ij,
            branch_counts,
            incidence_matrix,
            incidence_directions,
        )

    def make_incidence_matrix(self, L1, N1, L2, N2):
        """Return an N1+1 x N2+1 matrix that marks all L1 and L2 that are 8-connected

        L1 - a labels matrix
        N1 - # of labels in L1
        L2 - a labels matrix
        N2 - # of labels in L2

        L1 and L2 should have no overlap

        Returns a matrix where M[n,m] is true if there is some pixel in
        L1 with value n that is 8-connected to a pixel in L2 with value m
        """
        #
        # Overlay the two labels matrix
        #
        L = L1.copy()
        L[L2 != 0] = L2[L2 != 0] + N1
        neighbor_count, neighbor_index, n2 = centrosome.cpmorphology.find_neighbors(L)
        if numpy.all(neighbor_count == 0):
            return numpy.zeros((N1, N2), bool)
        #
        # Keep the neighbors of L1 / discard neighbors of L2
        #
        neighbor_count = neighbor_count[:N1]
        neighbor_index = neighbor_index[:N1]
        n2 = n2[: (neighbor_index[-1] + neighbor_count[-1])]
        #
        # Get rid of blanks
        #
        label = numpy.arange(N1)[neighbor_count > 0]
        neighbor_index = neighbor_index[neighbor_count > 0]
        neighbor_count = neighbor_count[neighbor_count > 0]
        #
        # Correct n2 because we have formerly added N1 to its labels. Make
        # it zero-based.
        #
        n2 -= N1 + 1
        #
        # Create runs of n1 labels
        #
        n1 = numpy.zeros(len(n2), int)
        n1[0] = label[0]
        n1[neighbor_index[1:]] = label[1:] - label[:-1]
        n1 = numpy.cumsum(n1)
        incidence = coo.coo_matrix(
            (numpy.ones(n1.shape), (n1, n2)), shape=(N1, N2)
        ).toarray()
        return incidence != 0

    def get_longest_path_coords(self, graph_struct, max_length):
        """Given a graph describing the structure of the skeleton of an image,
        returns the longest non-self-intersecting (with some caveats, see
        get_all_paths.m) path through that graph, specified as a polyline.

        Inputs:

        graph_struct: A structure describing the graph. Same format as returned
        by get_graph_from_binary(), see that file for details.

        Outputs:

        path_coords: A n x 2 array, where successive columns contains the
        coordinates of successive points on the paths (which when joined with
        line segments form the path itself.)

        path_struct: A structure, with entries 'segments' and 'branch_areas',
        describing the path found, in relation to graph_struct. See
        get_all_paths.m for details."""

        path_list = self.get_all_paths(graph_struct, 0, max_length)
        current_longest_path_coords = []
        current_max_length = 0
        current_path = None
        for path in path_list:
            path_coords = self.path_to_pixel_coords(graph_struct, path)
            path_length = self.calculate_path_length(path_coords)
            if path_length >= current_max_length:
                current_longest_path_coords = path_coords
                current_max_length = path_length
                current_path = path
        return current_longest_path_coords, current_path

    def path_to_pixel_coords(self, graph_struct, path):
        """Given a structure describing paths in a graph, converts those to a
        polyline (i.e., successive coordinates) representation of the same graph.

        (This is possible because the graph_struct descriptor contains
        information on where the vertices and edges of the graph were initially
        located in the image plane.)

        Inputs:

        graph_struct: A structure describing the graph. Same format as returned
        by get_graph_from_binary(), so for details, see that file.

        path_struct: A structure which (in relation to graph_struct) describes
        a path through the graph. Same format as (each entry in the list)
        returned by get_all_paths(), so see further get_all_paths.m

        Outputs:

        pixel_coords: A n x 2 double array, where each column contains the
        coordinates of one point on the path. The path itself can be formed
        by joining these points successively to each other.

        Note that because of the way the graph is built, the points in pixel_coords are
        likely to contain segments consisting of runs of pixels where each is
        close to the next (in its 8-neighbourhood), but interleaved with
        reasonably long "jumps", where there is some distance between the end
        of one segment and the beginning of the next."""

        if len(path.segments) == 1:
            return graph_struct.segments[path.segments[0]][0]

        direction = graph_struct.incidence_directions[
            path.branch_areas[0], path.segments[0]
        ]
        result = [graph_struct.segments[path.segments[0]][direction]]
        for branch_area, segment in zip(path.branch_areas, path.segments[1:]):
            direction = not graph_struct.incidence_directions[branch_area, segment]
            result.append(graph_struct.segments[segment][direction])
        return numpy.vstack(result)

    def calculate_path_length(self, path_coords):
        """Return the path length, given path coordinates as Nx2"""
        if len(path_coords) < 2:
            return 0
        return numpy.sum(
            numpy.sqrt(numpy.sum((path_coords[:-1] - path_coords[1:]) ** 2, 1))
        )

    def calculate_cumulative_lengths(self, path_coords):
        """return a cumulative length vector given Nx2 path coordinates"""
        if len(path_coords) < 2:
            return numpy.array([0] * len(path_coords))
        return numpy.hstack(
            (
                [0],
                numpy.cumsum(
                    numpy.sqrt(numpy.sum((path_coords[:-1] - path_coords[1:]) ** 2, 1))
                ),
            )
        )

    def single_worm_filter(self, workspace, path_coords, params):
        """Given a path representing a single worm, calculates its shape cost, and
        either accepts it as a worm or rejects it, depending on whether or not
        the shape cost is higher than some threshold.

        Inputs:

        path_coords:  A N x 2 array giving the coordinates of the path.

        params: the parameters structure from which we use

            cost_theshold: Scalar double. The maximum cost possible for a worm;
            paths of shape cost higher than this are rejected.

            num_control_points. Scalar positive integer. The shape cost
            model uses control points sampled at equal intervals along the
            path.

            mean_angles: A (num_control_points-1) x
            1 double array. See calculate_angle_shape_cost() for how this is
            used.

            inv_angles_covariance_matrix: A
            (num_control_points-1)x(num_control_points-1) double matrix. See
            calculate_angle_shape_cost() for how this is used.

         Returns true if worm passes filter"""
        if len(path_coords) < 2:
            return False
        cumul_lengths = self.calculate_cumulative_lengths(path_coords)
        total_length = cumul_lengths[-1]
        control_coords = self.sample_control_points(
            path_coords, cumul_lengths, params.num_control_points
        )
        cost = self.calculate_angle_shape_cost(
            control_coords,
            total_length,
            params.mean_angles,
            params.inv_angles_covariance_matrix,
        )
        return cost < params.cost_threshold

    def sample_control_points(self, path_coords, cumul_lengths, num_control_points):
        """Sample equally-spaced control points from the Nx2 path coordinates

        Inputs:

        path_coords: A Nx2 double array, where each column specifies a point
        on the path (and the path itself is formed by joining successive
        points with line segments). Such as returned by
        path_struct_to_pixel_coords().

        cumul_lengths: A vector, where the ith entry indicates the
        length from the first point of the path to the ith in path_coords).
        In most cases, should be calculate_cumulative_lengths(path_coords).

        n: A positive integer. The number of control points to sample.

        Outputs:

        control_coords: A N x 2 double array, where the jth column contains the
        jth control point, sampled along the path. The first and last control
        points are equal to the first and last points of the path (i.e., the
        points whose coordinates are the first and last columns of
        path_coords), respectively."""
        assert num_control_points > 2
        #
        # Paranoia - eliminate any coordinates with length = 0, esp the last.
        #
        path_coords = path_coords.astype(float)
        cumul_lengths = cumul_lengths.astype(float)
        mask = numpy.hstack(([True], cumul_lengths[1:] != cumul_lengths[:-1]))
        path_coords = path_coords[mask]
        #
        # Create a function that maps control point index to distance
        #

        ncoords = len(path_coords)
        f = interp1d(cumul_lengths, numpy.linspace(0.0, float(ncoords - 1), ncoords))
        #
        # Sample points from f (for the ones in the middle)
        #
        first = float(cumul_lengths[-1]) / float(num_control_points - 1)
        last = float(cumul_lengths[-1]) - first
        findices = f(numpy.linspace(first, last, num_control_points - 2))
        indices = findices.astype(int)
        assert indices[-1] < ncoords - 1
        fracs = findices - indices
        sampled = (
            path_coords[indices, :] * (1 - fracs[:, numpy.newaxis])
            + path_coords[(indices + 1), :] * fracs[:, numpy.newaxis]
        )
        #
        # Tack on first and last
        #
        sampled = numpy.vstack((path_coords[:1, :], sampled, path_coords[-1:, :]))
        return sampled

    def calculate_angle_shape_cost(
        self, control_coords, total_length, mean_angles, inv_angles_covariance_matrix
    ):
        """% Calculates a shape cost based on the angle shape cost model.

        Given a set of N control points, calculates the N-2 angles between
        lines joining consecutive control points, forming them into a vector.
        The function then appends the total length of the path formed, as an
        additional value in the now (N-1)-dimensional feature
        vector.

        The returned value is the square of the Mahalanobis distance from
        this feature vector, v, to a training set with mean mu and covariance
        matrix C, calculated as

        cost = (v - mu)' * C^-1 * (v - mu)

        Input parameters:

        control_coords: A 2 x N double array, containing the coordinates of
        the control points; one control point in each column. In the same
        format as returned by sample_control_points().

        total_length: Scalar double. The total length of the path from which the control
        points are sampled. (I.e., the distance along the path from the
        first control point to the last, e.g., as returned by
        calculate_path_length().

        mean_angles: A (N-1) x 1 double array. The mu in the above formula,
        i.e., the mean of the feature vectors as calculated from the
        training set. Thus, the first N-2 entries are the means of the
        angles, and the last entry is the mean length of the training
        worms.

        inv_angles_covariance_matrix: A (N-1)x(N-1) double matrix. The
        inverse of the covariance matrix of the feature vectors in the
        training set. Thus, this is the C^-1 (nb: not just C) in the
        above formula.

        Output parameters:

        current_shape_cost: Scalar double. The squared Mahalanobis distance
        calculated. Higher values indicate that the path represented by
        the control points (and length) are less similar to the training
        set.

        Note that all the angles in question here are direction angles,
        constrained to lie between -pi and pi. The angle 0 corresponds to
        the case when two adjacnet line segments are parallel (and thus
        belong to the same line); the angles can be thought of as the
        (signed) angles through which the path "turns", and are thus not the
        angles between the line segments as such."""

        angles = self.get_angles(control_coords)
        feat_vec = numpy.hstack((angles, [total_length])) - mean_angles
        return numpy.dot(numpy.dot(feat_vec, inv_angles_covariance_matrix), feat_vec)

    def get_angles(self, control_coords):
        """Extract the angles at each interior control point

        control_coords - an Nx2 array of coordinates of control points

        returns an N-2 vector of angles between -pi and pi
        """
        segments_delta = control_coords[1:] - control_coords[:-1]
        segment_bearings = numpy.arctan2(segments_delta[:, 0], segments_delta[:, 1])
        angles = segment_bearings[1:] - segment_bearings[:-1]
        #
        # Constrain the angles to -pi <= angle <= pi
        #
        angles[angles > numpy.pi] -= 2 * numpy.pi
        angles[angles < -numpy.pi] += 2 * numpy.pi
        return angles

    def cluster_graph_building(self, workspace, labels, i, skeleton, params):
        binary_im = labels == i
        skeleton = skeleton & binary_im

        return self.get_graph_from_binary(
            binary_im, skeleton, params.max_radius, params.max_skel_length
        )

    class Path(object):
        def __init__(self, segments, branch_areas):
            self.segments = segments
            self.branch_areas = branch_areas

        def __repr__(self):
            return (
                "{ segments="
                + repr(self.segments)
                + " branch_areas="
                + repr(self.branch_areas)
                + " }"
            )

    def get_all_paths(self, graph_struct, min_length, max_length):
        """Given a structure describing a graph, returns a cell array containing
        a list of all paths through the graph.

        The format of graph_struct is exactly that outputted by
        get_graph_from_binary()

        Below, "vertex" refers to the "branch areas" of the
        graph_struct, and "edge" to refer to the "segments".

        For the purposes of this function, a path of length n is a sequence of n
        distinct edges

            e_1, ..., e_n

        together with a sequence of n-1 distinct vertices

            v_1, ..., v_{n-1}

        such that e_1 is incident to v_1, v_1 incident to e_2, and so on.

        Note that, since the ends are not considered parts of the paths, cyclic
        paths are allowed (i.e., ones starting and ending at the same vertex, but
        not self-crossing ones.)

        Furthermore, this function also considers two paths identical if one can
        be obtained by a simple reverse of the other.

        This function works by a simple depth-first search. It seems
        unnecessarily complicated compared to what it perhaps could have been;
        this is due to the fact that the endpoints are segments are not
        considered as vertices in the graph model used, and so each edge can be
        incident to less than 2 vertices.

        To explain how the function works, let me define an "unfinished path" to
        be a sequence of n edges e_1,...,e_n and n distinct vertices v_1, ..., v_n,
        where incidence relations e_1 - v_1 - e_2 - ... - e_n - v_n apply, and
        the intention is for the path to be continued through v_n. In constrast,
        call paths as defined in the previous paragraphs (where the last vertex
        is not included) "finished".

        The function first generates all unfinished paths of length 1 by looping
        through all possible edges, and for each edge at most 2 "continuation"
        vertices. It then calls get_all_paths_recur(), which, given an unfinished
        path, recursively generates a list of all possible finished paths
        beginning that unfinished path.

         To ensure that paths are only returned in one of the two possible
         directions, only 1-length paths and paths where the index of the
         first edge is less than that of the last edge are returned.

         To faciliate the processing in get_all_paths_recur, the function
         build_incidence_lists is used to calculate incidence tables in a list
         form.

         The output is a list of objects, "o" of the form

         o.segments - segment indices of the path
         o.branch_areas - branch area indices of the path"""

        (
            graph_struct.incident_branch_areas,
            graph_struct.incident_segments,
        ) = self.build_incidence_lists(graph_struct)
        n = len(graph_struct.segments)

        graph_struct.segment_lengths = numpy.array(
            [self.calculate_path_length(x[0]) for x in graph_struct.segments]
        )
        for j in range(n):
            current_length = graph_struct.segment_lengths[j]
            # Add all finished paths of length 1
            if current_length >= min_length:
                yield self.Path([j], [])
            #
            # Start the segment list for each branch area connected with
            # a segment with the segment.
            #
            segment_list = [j]
            branch_areas_list = [[k] for k in graph_struct.incident_branch_areas[j]]

            paths_list = self.get_all_paths_recur(
                graph_struct,
                segment_list,
                branch_areas_list,
                current_length,
                min_length,
                max_length,
            )
            for path in paths_list:
                yield path

    def build_incidence_lists(self, graph_struct):
        """Return a list of all branch areas incident to j for each segment

        incident_branch_areas{j} is a row array containing a list of all those
        branch areas incident to segment j; similarly, incident_segments{i} is a
        row array containing a list of all those segments incident to branch area
        i."""
        m = graph_struct.incidence_matrix.shape[1]
        n = graph_struct.incidence_matrix.shape[0]
        incident_segments = [
            numpy.arange(m)[graph_struct.incidence_matrix[i, :]] for i in range(n)
        ]
        incident_branch_areas = [
            numpy.arange(n)[graph_struct.incidence_matrix[:, i]] for i in range(m)
        ]
        return incident_branch_areas, incident_segments

    def get_all_paths_recur(
        self,
        graph,
        unfinished_segment,
        unfinished_branch_areas,
        current_length,
        min_length,
        max_length,
    ):
        """Recursively find paths

        incident_branch_areas - list of all branch areas incident on a segment
        incident_segments - list of all segments incident on a branch
        """
        if len(unfinished_segment) == 0:
            return
        last_segment = unfinished_segment[-1]
        for unfinished_branch in unfinished_branch_areas:
            end_branch_area = unfinished_branch[-1]
            #
            # Find all segments from the end branch
            #
            direction = graph.incidence_directions[end_branch_area, last_segment]

            last_coord = graph.segments[last_segment][int(direction)][-1]
            for j in graph.incident_segments[end_branch_area]:
                if j in unfinished_segment:
                    continue  # segment already in the path
                direction = not graph.incidence_directions[end_branch_area, j]
                first_coord = graph.segments[j][int(direction)][0]
                gap_length = numpy.sqrt(numpy.sum((last_coord - first_coord) ** 2))
                next_length = current_length + gap_length + graph.segment_lengths[j]
                if next_length > max_length:
                    continue
                next_segment = unfinished_segment + [j]
                if j > unfinished_segment[0] and next_length >= min_length:
                    # Only include if end segment index is greater
                    # than start
                    yield self.Path(next_segment, unfinished_branch)
                #
                # Can't loop back to "end_branch_area". Construct all of
                # possible branches otherwise
                #
                next_branch_areas = [
                    unfinished_branch + [k]
                    for k in graph.incident_branch_areas[j]
                    if (k != end_branch_area) and (k not in unfinished_branch)
                ]
                for path in self.get_all_paths_recur(
                    graph,
                    next_segment,
                    next_branch_areas,
                    next_length,
                    min_length,
                    max_length,
                ):
                    yield path

    def cluster_paths_selection(self, graph, paths, labels, i, params):
        """Select the best paths for worms from the graph

        Given a graph representing a worm cluster, and a list of paths in the
        graph, selects a subcollection of paths likely to represent the worms in
        the cluster.

        More specifically, finds (approximately, depending on parameters) a
        subset K of the set P paths, minimising

        Sum, over p in K, of shape_cost(K)
        +  a * Sum, over p,q distinct in K, of overlap(p, q)
        +  b * leftover(K)

        Here, shape_cost is a function which calculates how unlikely it is that
        the path represents a true worm.

        overlap(p, q) indicates how much overlap there is between paths p and q
        (we want to assign a cost to overlaps, to avoid picking out essentially
        the same worm, but with small variations, twice in K)

        leftover(K) is a measure of the amount of the cluster "unaccounted for"
        after all of the paths of P have been chosen. We assign a cost to this to
        make sure we pick out all the worms in the cluster.

        Shape model:'angle_shape_model'. More information
        can be found in calculate_angle_shape_cost(),

        Selection method

        'dfs_prune': searches
        through all the combinations of paths (view this as picking out subsets
        of P one element at a time, to make this a search tree) depth-first,
        but by keeping track of the best solution so far (and noting that the
        shape cost and overlap cost terms can only increase as paths are added
        to K), it can prune away large branches of the search tree guaranteed
        to be suboptimal.

        Furthermore, by setting the approx_max_search_n parameter to a finite
        value, this method adopts a "partially greedy" approach, at each step
        searching through only a set number of branches. Setting this parameter
        approx_max_search_n to 1 should in some sense give just the greedy
        algorithm, with the difference that this takes the leftover cost term
        into account in determining how many worms to find.

        Input parameters:

        graph_struct: A structure describing the graph. As returned from e.g.
        get_graph_from_binary().

        path_structs_list: A cell array of structures, each describing one path
        through the graph. As returned by cluster_paths_finding().

        params: The parameters structure. The parameters below should be
        in params.cluster_paths_selection

        min_path_length: Before performing the search, paths which are too
        short or too long are filtered away. This is the minimum length, in
        pixels.

        max_path_length: Before performing the search, paths which are too
        short or too long are filtered away. This is the maximum length, in
        pixels.

        shape_cost_method: 'angle_shape_cost'

        num_control_points: All shape cost models samples equally spaced
        control points along the paths whose shape cost are to be
        calculated. This is the number of such control points to sample.

        mean_angles: [Only for 'angle_shape_cost']

        inv_angles_covariance_matrix: [Only for 'angle_shape_cost']

        For these two parameters,  see calculate_angle_shape_cost().

        overlap_leftover_method:
        'skeleton_length'. The overlap/leftover calculation method to use.
        Note that if selection_method is 'dfs_prune', then this must be
        'skeleton_length'.

        selection_method: 'dfs_prune'. The search method
        to be used.

        median_worm_area: Scalar double. The approximate area of a typical
        worm.
        This approximates the number of worms in the
        cluster. Is only used to estimate the best branching factors in the
        search tree. If approx_max_search_n is infinite, then this is in
        fact not used at all.

        overlap_weight: Scalar double. The weight factor assigned to
        overlaps, i.e., the a in the formula of the cost to be minimised.
        the unit is (shape cost unit)/(pixels as a unit of
        skeleton length).

        leftover_weight:  The
        weight factor assigned to leftover pieces, i.e., the b in the
        formula of the cost to be minimised. In units of (shape cost
        unit)/(pixels of skeleton length).

        approx_max_search_n: [Only used if selection_method is 'dfs_prune']

        Outputs:

        paths_coords_selected: A cell array of worms selected. Each worm is
        represented as 2xm array of coordinates, specifying the skeleton of
        the worm as a polyline path.
"""
        min_path_length = params.min_path_length
        max_path_length = params.max_path_length
        median_worm_area = params.median_worm_area
        num_control_points = params.num_control_points

        mean_angles = params.mean_angles
        inv_angles_covariance_matrix = params.inv_angles_covariance_matrix

        component = labels == i
        max_num_worms = int(numpy.ceil(numpy.sum(component) / median_worm_area))

        # First, filter out based on path length
        # Simultaneously build a vector of shape costs and a vector of
        # reconstructed binaries for each of the (accepted) paths.

        #
        # List of tuples of path structs that pass filter + cost of shape
        #
        paths_and_costs = []
        for i, path in enumerate(paths):
            current_path_coords = self.path_to_pixel_coords(graph, path)
            cumul_lengths = self.calculate_cumulative_lengths(current_path_coords)
            total_length = cumul_lengths[-1]
            if total_length > max_path_length or total_length < min_path_length:
                continue
            control_coords = self.sample_control_points(
                current_path_coords, cumul_lengths, num_control_points
            )
            #
            # Calculate the shape cost
            #
            current_shape_cost = self.calculate_angle_shape_cost(
                control_coords, total_length, mean_angles, inv_angles_covariance_matrix
            )
            if current_shape_cost < params.cost_threshold:
                paths_and_costs.append((path, current_shape_cost))

        if len(paths_and_costs) == 0:
            return []

        path_segment_matrix = numpy.zeros(
            (len(graph.segments), len(paths_and_costs)), bool
        )
        for i, (path, cost) in enumerate(paths_and_costs):
            path_segment_matrix[path.segments, i] = True
        overlap_weight = self.overlap_weight(params)
        leftover_weight = self.leftover_weight(params)
        #
        # Sort by increasing cost
        #
        costs = numpy.array([cost for path, cost in paths_and_costs])
        order = numpy.lexsort([costs])
        if len(order) > MAX_PATHS:
            order = order[:MAX_PATHS]
        costs = costs[order]
        path_segment_matrix = path_segment_matrix[:, order]

        current_best_subset, current_best_cost = self.fast_selection(
            costs,
            path_segment_matrix,
            graph.segment_lengths,
            overlap_weight,
            leftover_weight,
            max_num_worms,
        )
        selected_paths = [paths_and_costs[order[i]][0] for i in current_best_subset]
        path_coords_selected = [
            self.path_to_pixel_coords(graph, path) for path in selected_paths
        ]
        return path_coords_selected

    def fast_selection(
        self,
        costs,
        path_segment_matrix,
        segment_lengths,
        overlap_weight,
        leftover_weight,
        max_num_worms,
    ):
        """Select the best subset of paths using a breadth-first search

        costs - the shape costs of every path

        path_segment_matrix - an N x M matrix where N are the segments
        and M are the paths. A cell is true if a path includes the segment

        segment_lengths - the length of each segment

        overlap_weight - the penalty per pixel of an overlap

        leftover_weight - the penalty per pixel of an excluded segment

        max_num_worms - maximum # of worms allowed in returned match.
        """
        current_best_subset = []
        current_best_cost = numpy.sum(segment_lengths) * leftover_weight
        current_costs = costs
        current_path_segment_matrix = path_segment_matrix.astype(int)
        current_path_choices = numpy.eye(len(costs), dtype=bool)
        for i in range(min(max_num_worms, len(costs))):
            (
                current_best_subset,
                current_best_cost,
                current_path_segment_matrix,
                current_path_choices,
            ) = self.select_one_level(
                costs,
                path_segment_matrix,
                segment_lengths,
                current_best_subset,
                current_best_cost,
                current_path_segment_matrix,
                current_path_choices,
                overlap_weight,
                leftover_weight,
            )
            if numpy.prod(current_path_choices.shape) == 0:
                break
        return current_best_subset, current_best_cost

    def select_one_level(
        self,
        costs,
        path_segment_matrix,
        segment_lengths,
        current_best_subset,
        current_best_cost,
        current_path_segment_matrix,
        current_path_choices,
        overlap_weight,
        leftover_weight,
    ):
        """Select from among sets of N paths

        Select the best subset from among all possible sets of N paths,
        then create the list of all sets of N+1 paths

        costs - shape costs of each path

        path_segment_matrix - a N x M boolean matrix where N are the segments
        and M are the paths and True means that a path has a given segment

        segment_lengths - the lengths of the segments (for scoring)

        current_best_subset - a list of the paths in the best collection so far

        current_best_cost - the total cost of that subset

        current_path_segment_matrix - a matrix giving the number of times
        a segment appears in each of the paths to be considered

        current_path_choices - an N x M matrix where N is the number of paths
        and M is the number of sets: the value at a cell is True if a path
        is included in that set.

        returns the current best subset, the current best cost and
        the current_path_segment_matrix and current_path_choices for the
        next round.
        """
        #
        # Compute the cost, not considering uncovered segments
        #
        partial_costs = (
            #
            # The sum of the individual costs of the chosen paths
            #
            numpy.sum(costs[:, numpy.newaxis] * current_path_choices, 0)
            +
            #
            # The sum of the multiply-covered segment lengths * penalty
            #
            numpy.sum(
                numpy.maximum(current_path_segment_matrix - 1, 0)
                * segment_lengths[:, numpy.newaxis],
                0,
            )
            * overlap_weight
        )
        total_costs = (
            partial_costs
            +
            #
            # The sum of the uncovered segments * the penalty
            #
            numpy.sum(
                (current_path_segment_matrix[:, :] == 0)
                * segment_lengths[:, numpy.newaxis],
                0,
            )
            * leftover_weight
        )

        order = numpy.lexsort([total_costs])
        if total_costs[order[0]] < current_best_cost:
            current_best_subset = (
                numpy.argwhere(current_path_choices[:, order[0]]).flatten().tolist()
            )
            current_best_cost = total_costs[order[0]]
        #
        # Weed out any that can't possibly be better
        #
        mask = partial_costs < current_best_cost
        if not numpy.any(mask):
            return (
                current_best_subset,
                current_best_cost,
                numpy.zeros((len(costs), 0), int),
                numpy.zeros((len(costs), 0), bool),
            )
        order = order[mask[order]]
        if len(order) * len(costs) > MAX_CONSIDERED:
            # Limit # to consider at next level
            order = order[: (1 + MAX_CONSIDERED // len(costs))]
        current_path_segment_matrix = current_path_segment_matrix[:, order]
        current_path_choices = current_path_choices[:, order]
        #
        # Create a matrix of disallowance - you can only add a path
        # that's higher than any existing path
        #
        i, j = numpy.mgrid[0 : len(costs), 0 : len(costs)]
        disallow = i >= j
        allowed = numpy.dot(disallow, current_path_choices) == 0
        if numpy.any(allowed):
            i, j = numpy.argwhere(allowed).transpose()
            current_path_choices = (
                numpy.eye(len(costs), dtype=bool)[:, i] | current_path_choices[:, j]
            )
            current_path_segment_matrix = (
                path_segment_matrix[:, i] + current_path_segment_matrix[:, j]
            )
            return (
                current_best_subset,
                current_best_cost,
                current_path_segment_matrix,
                current_path_choices,
            )
        else:
            return (
                current_best_subset,
                current_best_cost,
                numpy.zeros((len(costs), 0), int),
                numpy.zeros((len(costs), 0), bool),
            )

    def search_recur(
        self,
        path_segment_matrix,
        segment_lengths,
        path_raw_costs,
        overlap_weight,
        leftover_weight,
        current_subset,
        last_chosen,
        current_cost,
        current_segment_coverings,
        current_best_subset,
        current_best_cost,
        branching_factors,
        current_level,
    ):
        """Perform a recursive depth-first search on sets of paths

        Perform a depth-first search recursively,  keeping the best (so far)
        found subset of paths in current_best_subset, current_cost.

        path_segment_matrix, segment_lengths, path_raw_costs, overlap_weight,
        leftover_weight, branching_factor are essentially static.

        current_subset is the currently considered subset, as an array of
        indices, each index corresponding to a path in path_segment_matrix.

        To avoid picking out the same subset twice, we insist that in all
        subsets, indices are listed in increasing order.

        Note that the shape cost term and the overlap cost term need not be
        re-calculated each time, but can be calculated incrementally, as more
        paths are added to the subset in consideration. Thus, current_cost holds
        the sum of the shape cost and overlap cost terms for current_subset.

        current_segments_coverings, meanwhile, is a logical array of length equal
        to the number of segments in the graph, keeping track of the segments
        covered by paths in current_subset."""

        # The cost of current_subset, including the leftover cost term
        this_cost = current_cost + leftover_weight * numpy.sum(
            segment_lengths[~current_segment_coverings]
        )
        if this_cost < current_best_cost:
            current_best_cost = this_cost
            current_best_subset = current_subset
        if current_level < len(branching_factors):
            this_branch_factor = branching_factors[current_level]
        else:
            this_branch_factor = branching_factors[-1]
        # Calculate, for each path after last_chosen, how much cost would be added
        # to current_cost upon adding that path to the current_subset.
        current_overlapped_costs = (
            path_raw_costs[last_chosen:]
            + numpy.sum(
                current_segment_coverings[:, numpy.newaxis]
                * segment_lengths[:, numpy.newaxis]
                * path_segment_matrix[:, last_chosen:],
                0,
            )
            * overlap_weight
        )
        order = numpy.lexsort([current_overlapped_costs])
        #
        # limit to number of branches allowed at this level
        #
        order = order[numpy.arange(len(order)) + 1 < this_branch_factor]
        for index in order:
            new_cost = current_cost + current_overlapped_costs[index]
            if new_cost >= current_best_cost:
                break  # No chance of subsequent better cost
            path_index = last_chosen + index
            current_best_subset, current_best_cost = self.search_recur(
                path_segment_matrix,
                segment_lengths,
                path_raw_costs,
                overlap_weight,
                leftover_weight,
                current_subset + [path_index],
                path_index,
                new_cost,
                current_segment_coverings | path_segment_matrix[:, path_index],
                current_best_subset,
                current_best_cost,
                branching_factors,
                current_level + 1,
            )
        return current_best_subset, current_best_cost

    def worm_descriptor_building(self, all_path_coords, params, shape):
        """Return the coordinates of reconstructed worms in i,j,v form

        Given a list of paths found in an image, reconstructs labeled
        worms.

        Inputs:

        worm_paths: A list of worm paths, each entry an N x 2 array
        containing the coordinates of the worm path.

        params:  the params structure loaded using read_params()

        Outputs:

        * an Nx3 array where the first two indices are the i,j
          coordinate and the third is the worm's label.

        * the lengths of each worm
        * the angles for control points other than the ends
        * the coordinates of the control points
        """
        num_control_points = params.num_control_points
        if len(all_path_coords) == 0:
            return (
                numpy.zeros((0, 3), int),
                numpy.zeros(0),
                numpy.zeros((0, num_control_points - 2)),
                numpy.zeros((0, num_control_points)),
                numpy.zeros((0, num_control_points)),
            )

        worm_radii = params.radii_from_training
        all_i = []
        all_j = []
        all_lengths = []
        all_angles = []
        all_control_coords_x = []
        all_control_coords_y = []
        for path in all_path_coords:
            cumul_lengths = self.calculate_cumulative_lengths(path)
            control_coords = self.sample_control_points(
                path, cumul_lengths, num_control_points
            )
            ii, jj = self.rebuild_worm_from_control_points_approx(
                control_coords, worm_radii, shape
            )
            all_i.append(ii)
            all_j.append(jj)
            all_lengths.append(cumul_lengths[-1])
            all_angles.append(self.get_angles(control_coords))
            all_control_coords_x.append(control_coords[:, 1])
            all_control_coords_y.append(control_coords[:, 0])
        ijv = numpy.column_stack(
            (
                numpy.hstack(all_i),
                numpy.hstack(all_j),
                numpy.hstack(
                    [numpy.ones(len(ii), int) * (i + 1) for i, ii in enumerate(all_i)]
                ),
            )
        )
        all_lengths = numpy.array(all_lengths)
        all_angles = numpy.vstack(all_angles)
        all_control_coords_x = numpy.vstack(all_control_coords_x)
        all_control_coords_y = numpy.vstack(all_control_coords_y)
        return ijv, all_lengths, all_angles, all_control_coords_x, all_control_coords_y

    def rebuild_worm_from_control_points_approx(
        self, control_coords, worm_radii, shape
    ):
        """Rebuild a worm from its control coordinates

        Given a worm specified by some control points along its spline,
        reconstructs an approximate binary image representing the worm.

        Specifically, this function generates an image where successive control
        points have been joined by line segments, and then dilates that by a
        certain (specified) radius.

        Inputs:

        control_coords: A N x 2 double array, where each column contains the x
        and y coordinates for a control point.

        worm_radius: Scalar double. Approximate radius of a typical worm; the
        radius by which the reconstructed worm spline is dilated to form the
        final worm.

        Outputs:
        The coordinates of all pixels in the worm in an N x 2 array"""
        index, count, i, j = centrosome.cpmorphology.get_line_pts(
            control_coords[:-1, 0],
            control_coords[:-1, 1],
            control_coords[1:, 0],
            control_coords[1:, 1],
        )
        #
        # Get rid of the last point for the middle elements - these are
        # duplicated by the first point in the next line
        #
        i = numpy.delete(i, index[1:])
        j = numpy.delete(j, index[1:])
        index = index - numpy.arange(len(index))
        count -= 1
        #
        # Get rid of all segments that are 1 long. Those will be joined
        # by the segments around them.
        #
        index, count = index[count != 0], count[count != 0]
        #
        # Find the control point and within-control-point index of each point
        #
        label = numpy.zeros(len(i), int)
        label[index[1:]] = 1
        label = numpy.cumsum(label)
        order = numpy.arange(len(i)) - index[label]
        frac = order.astype(float) / count[label].astype(float)
        radius = worm_radii[label] * (1 - frac) + worm_radii[label + 1] * frac
        iworm_radius = int(numpy.max(numpy.ceil(radius)))
        #
        # Get dilation coordinates
        #
        ii, jj = numpy.mgrid[
            -iworm_radius : iworm_radius + 1, -iworm_radius : iworm_radius + 1
        ]
        dd = numpy.sqrt((ii * ii + jj * jj).astype(float))
        mask = ii * ii + jj * jj <= iworm_radius * iworm_radius
        ii = ii[mask]
        jj = jj[mask]
        dd = dd[mask]
        #
        # All points (with repeats)
        #
        i = (i[:, numpy.newaxis] + ii[numpy.newaxis, :]).flatten()
        j = (j[:, numpy.newaxis] + jj[numpy.newaxis, :]).flatten()
        #
        # We further mask out any dilation coordinates outside of
        # the radius at our point in question
        #
        m = (radius[:, numpy.newaxis] >= dd[numpy.newaxis, :]).flatten()
        i = i[m]
        j = j[m]
        #
        # Find repeats by sorting and comparing against next
        #
        order = numpy.lexsort((i, j))
        i = i[order]
        j = j[order]
        mask = numpy.hstack([[True], (i[:-1] != i[1:]) | (j[:-1] != j[1:])])
        i = i[mask]
        j = j[mask]
        mask = (i >= 0) & (j >= 0) & (i < shape[0]) & (j < shape[1])
        return i[mask], j[mask]

    def read_params(self):
        """Read the parameters file"""
        if not hasattr(self, "training_params"):
            self.training_params = {}
        return read_params(
            self.training_set_directory,
            self.training_set_file_name,
            self.training_params,
        )

    def validate_module(self, pipeline):
        if self.mode == MODE_UNTANGLE:
            if self.training_set_directory.dir_choice != URL_FOLDER_NAME:
                path = os.path.join(
                    self.training_set_directory.get_absolute_path(),
                    self.training_set_file_name.value,
                )
                if not os.path.exists(path):
                    raise ValidationError(
                        "Can't find file %s" % self.training_set_file_name.value,
                        self.training_set_file_name,
                    )

    def validate_module_warnings(self, pipeline):
        """Warn user re: Test mode """
        if pipeline.test_mode and self.mode == MODE_TRAIN:
            raise ValidationError(
                "UntangleWorms will not produce training set output in Test Mode",
                self.training_set_file_name,
            )

    def get_measurement_columns(self, pipeline):
        """Return a column of information for each measurement feature"""
        result = []
        if self.mode == MODE_UNTANGLE:
            object_names = []
            if self.overlap in (OO_WITH_OVERLAP, OO_BOTH):
                object_names.append(self.overlap_objects.value)
            if self.overlap in (OO_WITHOUT_OVERLAP, OO_BOTH):
                object_names.append(self.nonoverlapping_objects.value)
            for object_name in object_names:
                result += get_object_measurement_columns(object_name)
                all_features = (
                    [F_LENGTH]
                    + self.angle_features()
                    + self.control_point_features(True)
                    + self.control_point_features(False)
                )
                result += [
                    (object_name, "_".join((C_WORM, f)), COLTYPE_FLOAT)
                    for f in all_features
                ]
        return result

    def angle_features(self):
        """Return a list of angle feature names"""
        try:
            return [
                "_".join((F_ANGLE, str(n)))
                for n in range(1, self.ncontrol_points() - 1)
            ]
        except:
            LOGGER.error(
                "Failed to get # of control points from training file. Unknown number of angle measurements",
                exc_info=True,
            )
            return []

    def control_point_features(self, get_x):
        """Return a list of control point feature names

        get_x - return the X coordinate control point features if true, else y
        """
        try:
            return [
                "_".join((F_CONTROL_POINT_X if get_x else F_CONTROL_POINT_Y, str(n)))
                for n in range(1, self.ncontrol_points() + 1)
            ]
        except:
            LOGGER.error(
                "Failed to get # of control points from training file. Unknown number of control point features",
                exc_info=True,
            )
            return []

    def get_categories(self, pipeline, object_name):
        if object_name == IMAGE:
            return [C_COUNT]
        if (
            object_name == self.overlap_objects.value
            and self.overlap in (OO_BOTH, OO_WITH_OVERLAP)
        ) or (
            object_name == self.nonoverlapping_objects.value
            and self.overlap in (OO_BOTH, OO_WITHOUT_OVERLAP)
        ):
            return [
                C_LOCATION,
                C_NUMBER,
                C_WORM,
            ]
        return []

    def get_measurements(self, pipeline, object_name, category):
        wants_overlapping = self.overlap in (OO_BOTH, OO_WITH_OVERLAP)
        wants_nonoverlapping = self.overlap in (OO_BOTH, OO_WITHOUT_OVERLAP)
        result = []
        if object_name == IMAGE and category == C_COUNT:
            if wants_overlapping:
                result += [self.overlap_objects.value]
            if wants_nonoverlapping:
                result += [self.nonoverlapping_objects.value]
        if (wants_overlapping and object_name == self.overlap_objects) or (
            wants_nonoverlapping and object_name == self.nonoverlapping_objects
        ):
            if category == C_LOCATION:
                result += [
                    FTR_CENTER_X,
                    FTR_CENTER_Y,
                ]
            elif category == C_NUMBER:
                result += [FTR_OBJECT_NUMBER]
            elif category == C_WORM:
                result += [F_LENGTH, F_ANGLE, F_CONTROL_POINT_X, F_CONTROL_POINT_Y]
        return result

    def get_measurement_scales(
        self, pipeline, object_name, category, measurement, image_name
    ):
        wants_overlapping = self.overlap in (OO_BOTH, OO_WITH_OVERLAP)
        wants_nonoverlapping = self.overlap in (OO_BOTH, OO_WITHOUT_OVERLAP)
        scales = []
        if (
            (wants_overlapping and object_name == self.overlap_objects)
            or (wants_nonoverlapping and object_name == self.nonoverlapping_objects)
        ) and (category == C_WORM):
            if measurement == F_ANGLE:
                scales += [str(n) for n in range(1, self.ncontrol_points() - 1)]
            elif measurement in [F_CONTROL_POINT_X, F_CONTROL_POINT_Y]:
                scales += [str(n) for n in range(1, self.ncontrol_points() + 1)]
        return scales

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        """Prepare to create a batch file

        This function is called when CellProfiler is about to create a
        file for batch processing. It will pickle the image set list's
        "legacy_fields" dictionary. This callback lets a module prepare for
        saving.

        pipeline - the pipeline to be saved
        image_set_list - the image set list to be saved
        fn_alter_path - this is a function that takes a pathname on the local
                        host and returns a pathname on the remote host. It
                        handles issues such as replacing backslashes and
                        mapping mountpoints. It should be called for every
                        pathname stored in the settings or legacy fields.
        """
        self.training_set_directory.alter_for_create_batch_files(fn_alter_path)
        return True

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            # Added complexity
            setting_values = setting_values + [C_ALL, "400"]
            variable_revision_number = 2
        return setting_values, variable_revision_number


def read_params(training_set_directory, training_set_file_name, d):
    """Read a training set parameters  file

    training_set_directory - the training set directory setting

    training_set_file_name - the training set file name setting

    d - a dictionary that stores cached parameters
    """

    #
    # The parameters file is a .xml file with the following structure:
    #
    # initial_filter
    #     min_worm_area: float
    # single_worm_determination
    #     max_area: float
    # single_worm_find_path
    #     method: string (=? "dfs_longest_path")
    # single_worm_filter
    #     method: string (=? "angle_shape_cost")
    #     cost_threshold: float
    #     num_control_points: int
    #     mean_angles: float vector (num_control_points -1 entries)
    #     inv_angles_covariance_matrix: float matrix (num_control_points -1)**2
    # cluster_graph_building
    #     method: "large_branch_area_max_skel_length"
    #     max_radius: float
    #     max_skel_length: float
    # cluster_paths_finding
    #     method: string "dfs"
    # cluster_paths_selection
    #     shape_cost_method: "angle_shape_model"
    #     selection_method: "dfs_prune"
    #     overlap_leftover_method: "skeleton_length"
    #     min_path_length: float
    #     max_path_length: float
    #     median_worm__area: float
    #     worm_radius: float
    #     overlap_weight: int
    #     leftover_weight: int
    #     ---- the following are the same as for the single worm filter ---
    #     num_control_points: int
    #     mean_angles: float vector (num_control_points-1)
    #     inv_angles_covariance_matrix: (num_control_points-1)**2
    #     ----
    #     approx_max_search_n: int
    # worm_descriptor_building
    #     method: string = "default"
    #     radii_from_training: vector ?of length num_control_points?
    #
    class X(object):
        """This "class" is used as a vehicle for arbitrary dot notation

        For instance:
        > x = X()
        > x.foo = 1
        > x.foo
        1
        """

        pass

    path = training_set_directory.get_absolute_path()
    file_name = training_set_file_name.value
    if file_name in d:
        result, timestamp = d[file_name]
        if (
            timestamp == "URL"
            or timestamp == os.stat(os.path.join(path, file_name)).st_mtime
        ):
            return d[file_name][0]

    if training_set_directory.dir_choice == URL_FOLDER_NAME:
        url = file_name
        fd_or_file = urlopen(url)
        is_url = True
        timestamp = "URL"
    else:
        fd_or_file = os.path.join(path, file_name)
        is_url = False
        timestamp = os.stat(fd_or_file).st_mtime
    try:
        from xml.dom.minidom import parse

        doc = parse(fd_or_file)
        result = X()

        def f(tag, attribute, klass):
            elements = doc.documentElement.getElementsByTagName(tag)
            assert len(elements) == 1
            element = elements[0]
            text = "".join(
                [
                    text.data
                    for text in element.childNodes
                    if text.nodeType == doc.TEXT_NODE
                ]
            )
            setattr(result, attribute, klass(text.strip()))

        for tag, attribute, klass in (
            (T_VERSION, "version", int),
            (T_MIN_AREA, "min_worm_area", float),
            (T_MAX_AREA, "max_area", float),
            (T_COST_THRESHOLD, "cost_threshold", float),
            (T_NUM_CONTROL_POINTS, "num_control_points", int),
            (T_MAX_RADIUS, "max_radius", float),
            (T_MAX_SKEL_LENGTH, "max_skel_length", float),
            (T_MIN_PATH_LENGTH, "min_path_length", float),
            (T_MAX_PATH_LENGTH, "max_path_length", float),
            (T_MEDIAN_WORM_AREA, "median_worm_area", float),
            (T_OVERLAP_WEIGHT, "overlap_weight", float),
            (T_LEFTOVER_WEIGHT, "leftover_weight", float),
        ):
            f(tag, attribute, klass)
        elements = doc.documentElement.getElementsByTagName(T_MEAN_ANGLES)
        assert len(elements) == 1
        element = elements[0]
        result.mean_angles = numpy.zeros(result.num_control_points - 1)
        for index, value_element in enumerate(element.getElementsByTagName(T_VALUE)):
            text = "".join(
                [
                    text.data
                    for text in value_element.childNodes
                    if text.nodeType == doc.TEXT_NODE
                ]
            )
            result.mean_angles[index] = float(text.strip())
        elements = doc.documentElement.getElementsByTagName(T_RADII_FROM_TRAINING)
        assert len(elements) == 1
        element = elements[0]
        result.radii_from_training = numpy.zeros(result.num_control_points)
        for index, value_element in enumerate(element.getElementsByTagName(T_VALUE)):
            text = "".join(
                [
                    text.data
                    for text in value_element.childNodes
                    if text.nodeType == doc.TEXT_NODE
                ]
            )
            result.radii_from_training[index] = float(text.strip())
        result.inv_angles_covariance_matrix = numpy.zeros(
            [result.num_control_points - 1] * 2
        )
        elements = doc.documentElement.getElementsByTagName(
            T_INV_ANGLES_COVARIANCE_MATRIX
        )
        assert len(elements) == 1
        element = elements[0]
        for i, values_element in enumerate(element.getElementsByTagName(T_VALUES)):
            for j, value_element in enumerate(
                values_element.getElementsByTagName(T_VALUE)
            ):
                text = "".join(
                    [
                        text.data
                        for text in value_element.childNodes
                        if text.nodeType == doc.TEXT_NODE
                    ]
                )
                result.inv_angles_covariance_matrix[i, j] = float(text.strip())
    except:
        if is_url:
            fd_or_file = urlopen(url)

        mat_params = loadmat(fd_or_file)["params"][0, 0]
        field_names = list(mat_params.dtype.fields.keys())

        result = X()

        CLUSTER_PATHS_SELECTION = "cluster_paths_selection"
        CLUSTER_GRAPH_BUILDING = "cluster_graph_building"
        SINGLE_WORM_FILTER = "single_worm_filter"
        INITIAL_FILTER = "initial_filter"
        SINGLE_WORM_DETERMINATION = "single_worm_determination"
        CLUSTER_PATHS_FINDING = "cluster_paths_finding"
        WORM_DESCRIPTOR_BUILDING = "worm_descriptor_building"
        SINGLE_WORM_FIND_PATH = "single_worm_find_path"
        METHOD = "method"

        STRING = "string"
        SCALAR = "scalar"
        VECTOR = "vector"
        MATRIX = "matrix"

        def mp(*args, **kwargs):
            """Look up a field from mat_params"""
            x = mat_params
            for arg in args[:-1]:
                x = x[arg][0, 0]
            x = x[args[-1]]
            kind = kwargs.get("kind", SCALAR)
            if kind == SCALAR:
                return x[0, 0]
            elif kind == STRING:
                return x[0]
            elif kind == VECTOR:
                # Work-around for OS/X Numpy bug
                # Copy a possibly mis-aligned buffer
                b = numpy.array(
                    [v for v in numpy.frombuffer(x.data, numpy.uint8)], numpy.uint8
                )
                return numpy.frombuffer(b, x.dtype)
            return x

        result.min_worm_area = mp(INITIAL_FILTER, "min_worm_area")
        result.max_area = mp(SINGLE_WORM_DETERMINATION, "max_area")
        result.cost_threshold = mp(SINGLE_WORM_FILTER, "cost_threshold")
        result.num_control_points = mp(SINGLE_WORM_FILTER, "num_control_points")
        result.mean_angles = mp(SINGLE_WORM_FILTER, "mean_angles", kind=VECTOR)
        result.inv_angles_covariance_matrix = mp(
            SINGLE_WORM_FILTER, "inv_angles_covariance_matrix", kind=MATRIX
        )
        result.max_radius = mp(CLUSTER_GRAPH_BUILDING, "max_radius")
        result.max_skel_length = mp(CLUSTER_GRAPH_BUILDING, "max_skel_length")
        result.min_path_length = mp(CLUSTER_PATHS_SELECTION, "min_path_length")
        result.max_path_length = mp(CLUSTER_PATHS_SELECTION, "max_path_length")
        result.median_worm_area = mp(CLUSTER_PATHS_SELECTION, "median_worm_area")
        result.worm_radius = mp(CLUSTER_PATHS_SELECTION, "worm_radius")
        result.overlap_weight = mp(CLUSTER_PATHS_SELECTION, "overlap_weight")
        result.leftover_weight = mp(CLUSTER_PATHS_SELECTION, "leftover_weight")
        result.radii_from_training = mp(
            WORM_DESCRIPTOR_BUILDING, "radii_from_training", kind=VECTOR
        )
    d[file_name] = (result, timestamp)
    return result


def recalculate_single_worm_control_points(all_labels, ncontrolpoints):
    """Recalculate the control points for labeled single worms

    Given a labeling of single worms, recalculate the control points
    for those worms.

    all_labels - a sequence of label matrices

    ncontrolpoints - the # of desired control points

    returns a two tuple:

    an N x M x 2 array where the first index is the object number,
    the second index is the control point number and the third index is 0
    for the Y or I coordinate of the control point and 1 for the X or J
    coordinate.

    a vector of N lengths.
    """

    all_object_numbers = [
        list(filter((lambda n: n > 0), numpy.unique(l))) for l in all_labels
    ]
    if all([len(object_numbers) == 0 for object_numbers in all_object_numbers]):
        return numpy.zeros((0, ncontrolpoints, 2), int), numpy.zeros(0, int)
    module = UntangleWorms()
    module.create_settings()
    module.num_control_points.value = ncontrolpoints
    #
    # Put the module in training mode - assumes that the training file is
    # not present.
    #
    module.mode.value = MODE_TRAIN

    nobjects = numpy.max(numpy.hstack(all_object_numbers))
    result = numpy.ones((nobjects, ncontrolpoints, 2)) * numpy.nan
    lengths = numpy.zeros(nobjects)
    for object_numbers, labels in zip(all_object_numbers, all_labels):
        for object_number in object_numbers:
            mask = labels == object_number
            skeleton = centrosome.cpmorphology.skeletonize(mask)
            graph = module.get_graph_from_binary(mask, skeleton)
            path_coords, path = module.get_longest_path_coords(
                graph, numpy.iinfo(int).max
            )
            if len(path_coords) == 0:
                # return NaN for the control points
                continue
            cumul_lengths = module.calculate_cumulative_lengths(path_coords)
            if cumul_lengths[-1] == 0:
                continue
            control_points = module.sample_control_points(
                path_coords, cumul_lengths, ncontrolpoints
            )
            result[(object_number - 1), :, :] = control_points
            lengths[object_number - 1] = cumul_lengths[-1]
    return result, lengths
