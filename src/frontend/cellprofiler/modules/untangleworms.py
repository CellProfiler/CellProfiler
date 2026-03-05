from cellprofiler_library.modules._untangleworms import (
    get_angles,
    path_to_pixel_coords,
    calculate_angle_shape_cost,
    sample_control_points,
    calculate_cumulative_lengths,
    Path,
    get_all_paths_recur,
    build_incidence_lists,
    calculate_path_length,
    get_all_paths,
    get_longest_path_coords,
    make_incidence_matrix,
    trace_segments,
    get_graph_from_branching_areas_and_segments,
    get_graph_from_binary,
    single_worm_find_path,
    single_worm_filter,
    cluster_graph_building,
    select_one_level,
    fast_selection,
    rebuild_worm_from_control_points_approx,
    worm_descriptor_building,
    get_overlap_weight,
    get_leftover_weight,
    cluster_paths_selection,
)
from cellprofiler_library.opts.untangleworms import TrainingXMLTag
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
            (TrainingXMLTag.VERSION, "version", int),
            (TrainingXMLTag.MIN_AREA, "min_worm_area", float),
            (TrainingXMLTag.MAX_AREA, "max_area", float),
            (TrainingXMLTag.COST_THRESHOLD, "cost_threshold", float),
            (TrainingXMLTag.NUM_CONTROL_POINTS, "num_control_points", int),
            (TrainingXMLTag.MAX_RADIUS, "max_radius", float),
            (TrainingXMLTag.MAX_SKEL_LENGTH, "max_skel_length", float),
            (TrainingXMLTag.MIN_PATH_LENGTH, "min_path_length", float),
            (TrainingXMLTag.MAX_PATH_LENGTH, "max_path_length", float),
            (TrainingXMLTag.MEDIAN_WORM_AREA, "median_worm_area", float),
            (TrainingXMLTag.OVERLAP_WEIGHT, "overlap_weight", float),
            (TrainingXMLTag.LEFTOVER_WEIGHT, "leftover_weight", float),
        ):
            f(tag, attribute, klass)
        elements = doc.documentElement.getElementsByTagName(TrainingXMLTag.MEAN_ANGLES)
        assert len(elements) == 1
        element = elements[0]
        result.mean_angles = numpy.zeros(result.num_control_points - 1)
        for index, value_element in enumerate(element.getElementsByTagName(TrainingXMLTag.VALUE)):
            text = "".join(
                [
                    text.data
                    for text in value_element.childNodes
                    if text.nodeType == doc.TEXT_NODE
                ]
            )
            result.mean_angles[index] = float(text.strip())
        elements = doc.documentElement.getElementsByTagName(TrainingXMLTag.RADII_FROM_TRAINING)
        assert len(elements) == 1
        element = elements[0]
        result.radii_from_training = numpy.zeros(result.num_control_points)
        for index, value_element in enumerate(element.getElementsByTagName(TrainingXMLTag.VALUE)):
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
            TrainingXMLTag.INV_ANGLES_COVARIANCE_MATRIX
        )
        assert len(elements) == 1
        element = elements[0]
        for i, values_element in enumerate(element.getElementsByTagName(TrainingXMLTag.VALUES)):
            for j, value_element in enumerate(
                values_element.getElementsByTagName(TrainingXMLTag.VALUE)
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

    def overlap_weight(self, params, wants_training_set_weights, override_overlap_weight):
        """The overlap weight to use in the cost calculation"""
        if not wants_training_set_weights:
            return override_overlap_weight
        elif params is None:
            return 2
        else:
            return params.overlap_weight

    def leftover_weight(self, params, wants_training_set_weights, override_leftover_weight):
        """The leftover weight to use in the cost calculation"""
        if not wants_training_set_weights:
            return override_leftover_weight
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
            graph = get_graph_from_binary(image.pixel_data & mask, skeleton & mask)
            path_coords, path = get_longest_path_coords(
                graph, numpy.iinfo(int).max
            )
            if len(path_coords) == 0:
                continue
            cumul_lengths = calculate_cumulative_lengths(path_coords)
            if cumul_lengths[-1] == 0:
                continue
            control_points = sample_control_points(
                path_coords, cumul_lengths, num_control_points
            )
            angles = get_angles(control_points)
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
                TrainingXMLTag.NAMESPACE, TrainingXMLTag.TRAINING_DATA, None
            )
            top = doc.documentElement
            top.setAttribute("xmlns", TrainingXMLTag.NAMESPACE)
            ver = Version(cellprofiler_version)
            for tag, value in (
                (TrainingXMLTag.VERSION, int(f"{ver.major}{ver.minor}{ver.micro}")),
                (TrainingXMLTag.MIN_AREA, min_area),
                (TrainingXMLTag.MAX_AREA, max_area),
                (TrainingXMLTag.COST_THRESHOLD, max_cost),
                (TrainingXMLTag.NUM_CONTROL_POINTS, num_control_points),
                (TrainingXMLTag.MAX_SKEL_LENGTH, max_skel_length),
                (TrainingXMLTag.MIN_PATH_LENGTH, min_length),
                (TrainingXMLTag.MAX_PATH_LENGTH, max_length),
                (TrainingXMLTag.MEDIAN_WORM_AREA, median_area),
                (TrainingXMLTag.MAX_RADIUS, max_radius),
                (TrainingXMLTag.OVERLAP_WEIGHT, this.override_overlap_weight.value),
                (TrainingXMLTag.LEFTOVER_WEIGHT, this.override_leftover_weight.value),
                (TrainingXMLTag.TRAINING_SET_SIZE, nworms),
            ):
                element = doc.createElement(tag)
                content = doc.createTextNode(str(value))
                element.appendChild(content)
                top.appendChild(element)
            for tag, values in (
                (TrainingXMLTag.MEAN_ANGLES, mean_angles_length),
                (TrainingXMLTag.RADII_FROM_TRAINING, mean_radial_profile),
            ):
                element = doc.createElement(tag)
                top.appendChild(element)
                for value in values:
                    value_element = doc.createElement(TrainingXMLTag.VALUE)
                    content = doc.createTextNode(str(value))
                    value_element.appendChild(content)
                    element.appendChild(value_element)
            element = doc.createElement(TrainingXMLTag.INV_ANGLES_COVARIANCE_MATRIX)
            top.appendChild(element)
            for row in inv_angles_covariance_matrix:
                values = doc.createElement(TrainingXMLTag.VALUES)
                element.appendChild(values)
                for col in row:
                    value = doc.createElement(TrainingXMLTag.VALUE)
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
                    path_coords, path_struct = single_worm_find_path(
                        labels, i, skeleton, params
                    )
                    if len(path_coords) > 0 and single_worm_filter(
                        path_coords, params
                    ):
                        all_path_coords.append(path_coords)
                else:
                    graph = cluster_graph_building(
                        labels, i, skeleton, params
                    )
                    if len(graph.segments) > self.max_complexity:
                        LOGGER.warning(
                            "Warning: rejecting cluster of %d segments.\n"
                            % len(graph.segments)
                        )
                        continue
                    paths = get_all_paths(
                        graph, params.min_path_length, params.max_path_length
                    )
                    wants_training_set_weights = self.wants_training_set_weights
                    override_overlap_weight = self.override_overlap_weight.value
                    override_leftover_weight = self.override_leftover_weight.value
                    paths_selected = cluster_paths_selection(
                        graph, paths, labels, i, params, wants_training_set_weights, override_overlap_weight, override_leftover_weight
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
        ) = worm_descriptor_building(all_path_coords, params, labels.shape)
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
            graph = get_graph_from_binary(mask, skeleton)
            path_coords, path = get_longest_path_coords(
                graph, numpy.iinfo(int).max
            )
            if len(path_coords) == 0:
                # return NaN for the control points
                continue
            cumul_lengths = calculate_cumulative_lengths(path_coords)
            if cumul_lengths[-1] == 0:
                continue
            control_points = sample_control_points(
                path_coords, cumul_lengths, ncontrolpoints
            )
            result[(object_number - 1), :, :] = control_points
            lengths[object_number - 1] = cumul_lengths[-1]
    return result, lengths
