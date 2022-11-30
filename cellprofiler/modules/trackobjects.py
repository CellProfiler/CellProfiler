import numpy.ma
import scipy.ndimage
import scipy.sparse
from cellprofiler_core.constants.measurement import (
    COLTYPE_INTEGER,
    COLTYPE_FLOAT,
    GROUP_INDEX,
    GROUP_NUMBER,
    OBJECT_NUMBER,
    M_LOCATION_CENTER_X,
    M_LOCATION_CENTER_Y,
    MCA_AVAILABLE_POST_GROUP,
    EXPERIMENT,
    MCA_AVAILABLE_EACH_CYCLE,
    IMAGE_NUMBER,
)
from cellprofiler_core.constants.module import HELP_ON_MEASURING_DISTANCES
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.range import FloatRange
from cellprofiler_core.setting.subscriber import LabelSubscriber
from cellprofiler_core.setting.text import Integer, Float, ImageName

from cellprofiler.modules import _help
from cellprofiler.modules._help import PROTIP_RECOMMEND_ICON


__doc__ = """\
TrackObjects
============

**TrackObjects** allows tracking objects throughout sequential frames
of a series of images, so that from frame to frame each object maintains
a unique identity in the output measurements

This module must be placed downstream of a module that identifies
objects (e.g., **IdentifyPrimaryObjects**). **TrackObjects** will
associate each object with the same object in the frames before and
after. This allows the study of objects' lineages and the timing and
characteristics of dynamic events in movies.

Images in CellProfiler are processed sequentially by frame (whether
loaded as a series of images or a movie file). To process a collection
of images/movies, you will need to do the following:

-  Define each individual movie using metadata either contained within
   the image file itself or as part of the images nomenclature or folder
   structure.  Please see the **Metadata** module for more details on metadata
   collection and usage.
-  Group the movies to make sure that each image sequence is handled
   individually. Please see the **Groups** module for more details on the
   proper use of metadata for grouping.

For complete details, see *Help > Creating a Project > Loading Image Stacks and Movies*.

For an example pipeline using TrackObjects, see the CellProfiler
`Examples <http://cellprofiler.org/examples/#Tracking>`__
webpage.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

See also
^^^^^^^^

See also: Any of the **Measure** modules, **IdentifyPrimaryObjects**, **Groups**.

{HELP_ON_SAVING_OBJECTS}

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Object measurements**

-  *Label:* Each tracked object is assigned a unique identifier (label).
   Child objects resulting from a split or merge are assigned the label
   of the ancestor.
-  *ParentImageNumber, ParentObjectNumber:* The *ImageNumber* and
   *ObjectNumber* of the parent object in the prior frame. For a split,
   each child object will have the label of the object it split from.
   For a merge, the child will have the label of the closest parent.
-  *TrajectoryX, TrajectoryY:* The direction of motion (in x and y
   coordinates) of the object from the previous frame to the current
   frame.
-  *DistanceTraveled:* The distance traveled by the object from the
   previous frame to the current frame (calculated as the magnitude of
   the trajectory vectors).
-  *Displacement:* The shortest distance traveled by the object from its
   initial starting position to the position in the current frame. That
   is, it is the straight-line path between the two points.
-  *IntegratedDistance:* The total distance traveled by the object
   during the lifetime of the object.
-  *Linearity:* A measure of how linear the object trajectory is during
   the object lifetime. Calculated as (displacement from initial to
   final location)/(integrated object distance). Value is in range of
   [0,1].
-  *Lifetime:* The number of frames an objects has existed. The lifetime
   starts at 1 at the frame when an object appears, and is incremented
   with each frame that the object persists. At the final frame of the
   image set/movie, the lifetimes of all remaining objects are output.
-  *FinalAge:* Similar to *LifeTime* but is only output at the final
   frame of the object's life (or the movie ends, whichever comes
   first). At this point, the final age of the object is output; no
   values are stored for earlier frames.

   |TO_image0|  This value is useful if you want to plot a histogram of the
   object lifetimes; all but the final age can be ignored or filtered out.

The following object measurements are specific to the LAP
tracking method:

-  *LinkType:* The linking method used to link the object to its parent.
   Possible values are

   -  **0**: The object was not linked to a parent.
   -  **1**: The object was linked to a parent in the
      previous frame.
   -  **2**: The object is linked as the start of a split
      path.
   -  **3**: The object was linked to its parent as a
      daughter of a mitotic pair.
   -  **4**: The object was linked to a parent in a frame
      prior to the previous frame (a gap).

   Under some circumstances, multiple linking methods may apply to a
   given object, e.g, an object may be both the beginning of a split
   path and not have a parent. However, only one linking method is
   assigned.
-  *MovementModel:* The movement model used to track the object.

   -  **0**: The *Random* model was used.
   -  **1**: The *Velocity* model was used.
   -  **-1**: Neither model was used. This can occur under two
      circumstances:

      -  At the beginning of a trajectory, when there is no data to
         determine the model as yet.
      -  At the beginning of a closed gap, since a model was not
         actually applied to make the link in the first phase.

-  *LinkingDistance:* The difference between the propagated position of
   an object and the object to which it is matched.

   |TO_image1| A slowly decaying histogram of these distances indicates
   that the search radius is large enough. A cut-off histogram is a sign
   that the search radius is too small.

-  *StandardDeviation:* The Kalman filter maintains a running estimate
   of the variance of the error in estimated position for each model.
   This measurement records the linking distance divided by the standard
   deviation of the error when linking the object with its parent.

   |TO_image2| This value is multiplied by the
   "*Number of standard deviations for search radius*" setting to constrain the search
   distance. A histogram of this value can help determine if the
   "*Search radius limit, in pixel units (Min,Max)*" setting is appropriate.

-  *GapLength:* The number of frames between an object and its parent.
   For instance, an object in frame 3 with a parent in frame 1 has a gap
   length of 2.
-  *GapScore:* If an object is linked to its parent by bridging a gap,
   this value is the score for the gap.
-  *SplitScore:* If an object linked to its parent via a split, this
   value is the score for the split.
-  *MergeScore:* If an object linked to a child via a merge, this value
   is the score for the merge.
-  *MitosisScore:* If an object linked to two children via a mitosis,
   this value is the score for the mitosis.

**Image measurements**

-  *LostObjectCount:* Number of objects that appear in the previous
   frame but have no identifiable child in the current frame.
-  *NewObjectCount:* Number of objects that appear in the current frame
   but have no identifiable parent in the previous frame.
-  *SplitObjectCount:* Number of objects in the current frame that
   resulted from a split from a parent object in the previous frame.
-  *MergedObjectCount:* Number of objects in the current frame that
   resulted from the merging of child objects in the previous frame.

.. |TO_image0| image:: {PROTIP_RECOMMEND_ICON}
.. |TO_image1| image:: {PROTIP_RECOMMEND_ICON}
.. |TO_image2| image:: {PROTIP_RECOMMEND_ICON}
""".format(
    **{
        "PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON,
        "HELP_ON_SAVING_OBJECTS": _help.HELP_ON_SAVING_OBJECTS,
    }
)

TM_OVERLAP = "Overlap"
TM_DISTANCE = "Distance"
TM_MEASUREMENTS = "Measurements"
TM_LAP = "LAP"
TM_ALL = [TM_OVERLAP, TM_DISTANCE, TM_MEASUREMENTS, TM_LAP]
RADIUS_STD_SETTING_TEXT = "Number of standard deviations for search radius"
RADIUS_LIMIT_SETTING_TEXT = "Search radius limit, in pixel units (Min,Max)"
ONLY_IF_2ND_PHASE_LAP_TEXT = (
    """*(Used only if the %(TM_LAP)s tracking method is applied and the second phase is run)*"""
    % globals()
)

LT_NONE = 0
LT_PHASE_1 = 1
LT_SPLIT = 2
LT_MITOSIS = 3
LT_GAP = 4
KM_VEL = 1
KM_NO_VEL = 0
KM_NONE = -1

M_RANDOM = "Random"
M_VELOCITY = "Velocity"
M_BOTH = "Both"

import logging


import numpy as np
import numpy.ma
from scipy.ndimage import distance_transform_edt
import scipy.ndimage
import scipy.sparse
from cellprofiler_core.module import Module
from cellprofiler_core.image import Image
from cellprofiler_core.setting import (
    Measurement,
    Binary,
    ValidationError,
)
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.preferences import get_default_colormap
from centrosome.lapjv import lapjv
import centrosome.filter
from centrosome.cpmorphology import (
    fixup_scipy_ndimage_result,
    centers_of_labels,
    associate_by_distance,
    all_connected_components,
)
from centrosome.index import Indexes
from cellprofiler.modules._help import PROTIP_RECOMMEND_ICON

# if neighmovetrack is not available remove it from options
TM_ALL = ["Overlap", "Distance", "Measurements", "LAP", "Follow Neighbors"]

try:
    from centrosome.neighmovetrack import (
        NeighbourMovementTracking,
        NeighbourMovementTrackingParameters,
    )
except:
    TM_ALL.remove("Follow Neighbors")


LOGGER = logging.getLogger(__name__)

DT_COLOR_AND_NUMBER = "Color and Number"
DT_COLOR_ONLY = "Color Only"
DT_ALL = [DT_COLOR_AND_NUMBER, DT_COLOR_ONLY]

R_PARENT = "Parent"

F_PREFIX = "TrackObjects"
F_LABEL = "Label"
F_PARENT_OBJECT_NUMBER = "ParentObjectNumber"
F_PARENT_IMAGE_NUMBER = "ParentImageNumber"
F_TRAJECTORY_X = "TrajectoryX"
F_TRAJECTORY_Y = "TrajectoryY"
F_DISTANCE_TRAVELED = "DistanceTraveled"
F_DISPLACEMENT = "Displacement"
F_INTEGRATED_DISTANCE = "IntegratedDistance"
F_LINEARITY = "Linearity"
F_LIFETIME = "Lifetime"
F_FINAL_AGE = "FinalAge"
F_MOVEMENT_MODEL = "MovementModel"
F_LINK_TYPE = "LinkType"
F_LINKING_DISTANCE = "LinkingDistance"
F_STANDARD_DEVIATION = "StandardDeviation"
F_GAP_LENGTH = "GapLength"
F_GAP_SCORE = "GapScore"
F_MERGE_SCORE = "MergeScore"
F_SPLIT_SCORE = "SplitScore"
F_MITOSIS_SCORE = "MitosisScore"
F_KALMAN = "Kalman"
F_STATE = "State"
F_COV = "COV"
F_NOISE = "Noise"
F_VELOCITY_MODEL = "Vel"
F_STATIC_MODEL = "NoVel"
F_X = "X"
F_Y = "Y"
F_VX = "VX"
F_VY = "VY"
F_EXPT_ORIG_NUMTRACKS = "%s_OriginalNumberOfTracks" % F_PREFIX
F_EXPT_FILT_NUMTRACKS = "%s_FilteredNumberOfTracks" % F_PREFIX


def kalman_feature(model, matrix_or_vector, i, j=None):
    """Return the feature name for a Kalman feature

    model - model used for Kalman feature: velocity or static
    matrix_or_vector - the part of the Kalman state to save, vec, COV or noise
    i - the name for the first (or only for vec and noise) index into the vector
    j - the name of the second index into the matrix
    """
    pieces = [F_KALMAN, model, matrix_or_vector, i]
    if j is not None:
        pieces.append(j)
    return "_".join(pieces)


"""# of objects in the current frame without parents in the previous frame"""
F_NEW_OBJECT_COUNT = "NewObjectCount"
"""# of objects in the previous frame without parents in the new frame"""
F_LOST_OBJECT_COUNT = "LostObjectCount"
"""# of parents that split into more than one child"""
F_SPLIT_COUNT = "SplitObjectCount"
"""# of children that are merged from more than one parent"""
F_MERGE_COUNT = "MergedObjectCount"
"""Object area measurement for LAP method

The final part of the LAP method needs the object area measurement
which is stored using this name."""
F_AREA = "Area"

F_ALL_COLTYPE_ALL = [
    (F_LABEL, COLTYPE_INTEGER),
    (F_PARENT_OBJECT_NUMBER, COLTYPE_INTEGER),
    (F_PARENT_IMAGE_NUMBER, COLTYPE_INTEGER),
    (F_TRAJECTORY_X, COLTYPE_INTEGER),
    (F_TRAJECTORY_Y, COLTYPE_INTEGER),
    (F_DISTANCE_TRAVELED, COLTYPE_FLOAT),
    (F_DISPLACEMENT, COLTYPE_FLOAT),
    (F_INTEGRATED_DISTANCE, COLTYPE_FLOAT),
    (F_LINEARITY, COLTYPE_FLOAT),
    (F_LIFETIME, COLTYPE_INTEGER),
    (F_FINAL_AGE, COLTYPE_INTEGER),
]

F_IMAGE_COLTYPE_ALL = [
    (F_NEW_OBJECT_COUNT, COLTYPE_INTEGER),
    (F_LOST_OBJECT_COUNT, COLTYPE_INTEGER),
    (F_SPLIT_COUNT, COLTYPE_INTEGER),
    (F_MERGE_COUNT, COLTYPE_INTEGER),
]

F_ALL = [feature for feature, coltype in F_ALL_COLTYPE_ALL]

F_IMAGE_ALL = [feature for feature, coltype in F_IMAGE_COLTYPE_ALL]


class TrackObjects(Module):
    module_name = "TrackObjects"
    category = "Object Processing"
    variable_revision_number = 7

    def create_settings(self):
        self.tracking_method = Choice(
            "Choose a tracking method",
            TM_ALL,
            doc="""\
When trying to track an object in an image, **TrackObjects** will search
within a maximum specified distance (see the *distance within which to
search* setting) of the object's location in the previous image, looking
for a "match". Objects that match are assigned the same number, or
label, throughout the entire movie. There are several options for the
method used to find a match. Choose among these options based on which
is most consistent from frame to frame of your movie.

-  *Overlap:* Compares the amount of spatial overlap between identified
   objects in the previous frame with those in the current frame. The
   object with the greatest amount of spatial overlap will be assigned
   the same number (label).

   |image0| Recommended when there is a high degree of overlap of an
   object from one frame to the next, which is the case for movies with
   high frame rates relative to object motion.

-  *Distance:* Compares the distance between each identified object in
   the previous frame with that of the current frame. The closest
   objects to each other will be assigned the same number (label).
   Distances are measured from the perimeter of each object.

   |image1| Recommended for cases where the objects are not very
   crowded but where *Overlap* does not work sufficiently well, which is
   the case for movies with low frame rates relative to object motion.

-  *Measurements:* Compares each object in the current frame with
   objects in the previous frame based on a particular feature you have
   measured for the objects (for example, a particular intensity or
   shape measurement that can distinguish nearby objects). The object
   with the closest-matching measurement will be selected as a match and
   will be assigned the same number (label). This selection requires
   that you run the specified **Measure** module previous to this module
   in the pipeline so that the measurement values can be used to track
   the objects.
-  *Follow Neighbors:* Uses the multiobject tracking approach described
   by *Delgado-Gonzalo et al., 2010*. This approach assumes objects move
   in a coordinated way (contrary to LAP). An object's movement
   direction is more likely to be in agreement with the movement
   directions of its "neighbors". The problem is formulated as an
   optimization problem and solved using LAP algorithm (same as in LAP
   method).

   |image2| Recommended for cases where the objects are moving in
   synchronized way. In this case it may work better than *LAP*. This
   approach works well for yeast colonies grown on agar.

-  *LAP:* Uses the linear assignment problem (LAP) framework. The linear
   assignment problem (LAP) algorithm (*Jaqaman et al., 2008*) addresses
   the challenges of high object density, motion heterogeneity,
   temporary disappearances, and object merging and splitting. The
   algorithm first links objects between consecutive frames and then
   links the resulting partial trajectories into complete trajectories.
   Both steps are formulated as global combinatorial optimization
   problems whose solution identifies the overall most likely set of
   object trajectories throughout a movie.

   Tracks are constructed from an image sequence by detecting objects in
   each frame and linking objects between consecutive frames as a first
   step. This step alone may result in incompletely tracked objects due
   to the appearance and disappearance of objects, either in reality or
   apparently because of noise and imaging limitations. To correct this,
   you may apply an optional second step which closes temporal gaps
   between tracked objects and captures merging and splitting events.
   This step takes place at the end of the analysis run.

   |image3| Some recommendations on optimizing the LAP settings

   -  *Work with a minimal subset of your data:* Attempting to optimize
      these settings by examining a dataset containing many objects may
      be complicated and frustrating. Therefore, it is a good idea to
      work with a smaller portion of the data containing the behavior of
      interest.

      -  For example, if splits characterize your data, trying narrowing
         down to following just one cell that undergoes a split and
         examine a few frames before and after the event.
      -  You can insert the **Crop** module to zoom in a region of
         interest, optimize the settings and then either remove or
         disable the module when done.
      -  You can also use the **Input** modules to limit yourself to a
         few frames under consideration. For example, use the filtering
         settings in the **Images** module to use only certain files
         from the movie in the pipeline.

   -  *Begin by optimizing the settings for the first phase of the LAP:*
      The 2nd phase of the LAP method depends on the results of the
      first phase. Therefore, it is a good idea to optimize the first
      phase settings as the initial step.

      -  You can disable 2nd phase calculation by selecting *No* for
         "Run the second phase of the LAP algorithm?"
      -  By maximizing the number of correct frame-to-frame links in the
         first phase, the 2nd phase will have less candidates to
         consider for linking and have a better chance of closing gaps
         correctly.
      -  If tracks are not being linked in the first phase, you may need
         to adjust the number of standard deviations for the search
         radius and/or the radius limits (most likely the maximum
         limit). See the help for these settings for details.

   -  *Use any visualization tools at your disposal:* Visualizing the
      data often allows for easier decision making as opposed to sorting
      through tabular data alone.

      -  The `R <http://cran.r-project.org/>`__ open-source software
         package has analysis and visualization tools that can query a
         database.
      -  `CellProfiler Tracer <http://cellprofiler.org/tracer/>`__ is a
         version of CellProfiler Analyst that contains tools for
         visualizing time-lapse data that has been exported using the
         **ExportToDatabase** module.

   This Nearest Neighborhood method of this module was prepared by Filip
   Mroz, Adam Kaczmarek and Szymon Stoma. Please reach us at `Scopem,
   ETH <http://www.let-your-data-speak.com/>`__ for inquires.

References
^^^^^^^^^^

-  Jaqaman K, Loerke D, Mettlen M, Kuwata H, Grinstein S, Schmid SL,
  Danuser G. (2008) "Robust single-particle tracking in live-cell
  time-lapse sequences." *Nature Methods* 5(8),695-702.
  `(link) <https://doi.org/10.1038/nmeth.1237>`__
-  Jaqaman K, Danuser G. (2009) "Computational image analysis of
  cellular dynamics: a case study based on particle tracking." Cold
  Spring Harb Protoc. 2009(12):pdb.top65.
  `(link) <https://doi.org/10.1101/pdb.top65>`__

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
.. |image1| image:: {PROTIP_RECOMMEND_ICON}
.. |image2| image:: {PROTIP_RECOMMEND_ICON}
.. |image3| image:: {PROTIP_RECOMMEND_ICON}""".format(
                **{"PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON}
            ),
        )

        self.object_name = LabelSubscriber(
            "Select the objects to track",
            "None",
            doc="""Select the objects to be tracked by this module.""",
        )

        self.measurement = Measurement(
            "Select object measurement to use for tracking",
            lambda: self.object_name.value,
            doc="""\
*(Used only if "Measurements" is the tracking method)*

Select which type of measurement (category) and which specific feature
from the **Measure** module will be used for tracking. Select the
feature name from the popup box or see each **Measure** module’s help
for the list of the features measured by that module. If necessary, you
will also be asked to specify additional details such as the image from
which the measurements originated or the measurement scale.""",
        )

        self.pixel_radius = Integer(
            "Maximum pixel distance to consider matches",
            50,
            minval=1,
            doc="""\
Objects in the subsequent frame will be considered potential matches if
they are within this distance. To determine a suitable pixel distance,
you can look at the axis increments on each image (shown in pixel units)
or use the distance measurement tool.
{}
""".format(
                HELP_ON_MEASURING_DISTANCES
            ),
        )

        self.model = Choice(
            "Select the movement model",
            [M_RANDOM, M_VELOCITY, M_BOTH],
            value=M_BOTH,
            doc="""\
*(Used only if the "LAP" tracking method is applied)*

This setting controls how to predict an object’s position in the next
frame, assuming that each object moves randomly with a frame-to-frame
variance in position that follows a Gaussian distribution.

-  *{M_RANDOM}s:* A model in which objects move due to Brownian Motion
   or a similar process where the variance in position differs between
   objects.

   |image0|  Use this model if the objects move with some random jitter
   around a stationary location.

-  *Velocity:* A model in which the object moves with a velocity. Both
   velocity and position (after correcting for velocity) vary following
   a Gaussian distribution.

   |image1| Use this model if the objects move along a spatial
   trajectory in some direction over time.

-  *Both:* **TrackObjects** will predict each object’s position using
   both models and use the model with the lowest penalty to join an
   object in one frame with one in another.

   |image2| Use this option if both models above are applicable over
   time.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
.. |image1| image:: {PROTIP_RECOMMEND_ICON}
.. |image2| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                **{"M_RANDOM": M_RANDOM, "PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON}
            ),
        )

        self.radius_std = Float(
            "Number of standard deviations for search radius",
            3,
            minval=1,
            doc="""\
*(Used only if the "LAP" tracking method is applied)*

**TrackObjects** derives a search radius from an error estimation
based on (a) the standard deviation of the movement and (b) the
diameter of the object. The standard deviation is a measure of the
error between the observed and predicted positions of an object for
each movement model. The module will constrain the search for matching
objects from one frame to the next to the standard deviation of the
error times the number of standard deviations that you enter here.

|image0| Recommendations:

-  If the standard deviation is quite small, but the object makes a
   large spatial jump, this value may need to be set higher in order to
   increase the search area and thereby make the frame-to-frame linkage.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                **{"PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON}
            ),
        )

        self.radius_limit = FloatRange(
            "Search radius limit, in pixel units (Min,Max)",
            (2, 10),
            minval=0,
            doc="""\
*(Used only if the "LAP" tracking method is applied)*

**TrackObjects** derives a search radius from an error estimation
based on (a) the standard deviation of the movement and (b) the
diameter of the object. Potentially, the module can make an erroneous
assignment with a large error, leading to a large estimated error for
the object in the next frame. Conversely, the module can arrive at a
small estimated error by chance, leading to a maximum radius that does
not track the object in a subsequent frame. The radius limit
constrains the search radius to reasonable values.

|image0| Recommendations:

-  Special care must be taken to adjust the upper limit appropriate to
   the data.
-  The lower limit should be set to a radius (in pixels) that is a
   reasonable displacement for any object from one frame to the next.

   -  If you notice that a frame-to-frame linkage is not being made for
      a steadily-moving object, it may be that this value needs to be
      *decreased* such that the displacement falls above the lower
      limit.
   -  Alternately, if you notice that a frame-to-frame linkage is not
      being made for a roughly stationary object, this value may need to
      be *increased* so that the small displacement error is offset by
      the object diameter.

-  The upper limit should be set to the maximum reasonable displacement
   (in pixels) under any circumstances. Hence, if you notice that a
   frame-to-frame linkage is not being made in the case of a unusually
   large displacement, this value may need to be increased.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                **{"PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON}
            ),
        )

        self.wants_second_phase = Binary(
            "Run the second phase of the LAP algorithm?",
            True,
            doc="""\
*(Used only if the "LAP" tracking method is applied)*

Select "*Yes*" to run the second phase of the LAP algorithm after
processing all images. Select *No* to omit the second phase or to
perform the second phase when running the module as a data tool.

Since object tracks may start and end not only because of the true
appearance and disappearance of objects, but also because of apparent
disappearances due to noise and limitations in imaging, you may want to
run the second phase which attempts to close temporal gaps between
tracked objects and tries to capture merging and splitting events.

For additional details on optimizing the LAP settings, see the help for
each of the settings.

Note that if you use the second stage of the LAP algorithm, the output 
images generated by "*Save color-coded image?*" will NOT be accurate, 
as those images are generated before the second phase is run and not 
edited afterward.
""",
        )

        self.gap_cost = Integer(
            "Gap closing cost",
            40,
            minval=1,
            doc="""\
*(Used only if the "LAP" tracking method is applied and the second phase is run)*

This setting assigns a cost to keeping a gap caused when an object is
missing from one of the frames of a track (the alternative to keeping
the gap is to bridge it by connecting the tracks on either side of the
missing frames). The cost of bridging a gap is the distance, in
pixels, of the displacement of the object between frames.

|image0|  Recommendations:

-  Set the gap closing cost higher if tracks from objects in previous
   frames are being erroneously joined, across a gap, to tracks from
   objects in subsequent frames.
-  Set the gap closing cost lower if tracks are not properly joined due
   to gaps caused by mis-segmentation.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                **{"PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON}
            ),
        )

        self.split_cost = Integer(
            "Split alternative cost",
            40,
            minval=1,
            doc="""\
*(Used only if the "LAP" tracking method is applied and the second phase is run)*

This setting is the cost of keeping two tracks distinct when the
alternative is to make them into one track that splits. A split occurs
when an object in one frame is assigned to the same track as two
objects in a subsequent frame. The split cost takes two components
into account:

-  The area of the split object relative to the area of the resulting
   objects.
-  The displacement of the resulting objects relative to the position of
   the original object.

The split cost is roughly measured in pixels. The split alternative cost
is (conceptually) subtracted from the cost of making the split.

|image0|  Recommendations:

-  The split cost should be set lower if objects are being split that
   should not be split.
-  The split cost should be set higher if objects that should be split
   are not.
-  If you are confident that there should be no splits present in the
   data, the cost can be set to 1 (the minimum value possible)

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                **{"PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON}
            ),
        )

        self.merge_cost = Integer(
            "Merge alternative cost",
            40,
            minval=1,
            doc="""\
*(Used only if the "LAP" tracking method is applied and the second phase is run)*

This setting is the cost of keeping two tracks distinct when the
alternative is to merge them into one. A merge occurs when two objects
in one frame are assigned to the same track as a single object in a
subsequent frame. The merge score takes two components into account:

-  The area of the two objects to be merged relative to the area of the
   resulting objects.
-  The displacement of the original objects relative to the final
   object.

The merge cost is measured in pixels. The merge alternative cost is
(conceptually) subtracted from the cost of making the merge.

|image0|  Recommendations:

-  Set the merge alternative cost lower if objects are being merged when
   they should otherwise be kept separate.
-  Set the merge alternative cost higher if objects that are not merged
   should be merged.
-  If you are confident that there should be no merges present in the
   data, the cost can be set to 1 (the minimum value possible)

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                **{"PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON}
            ),
        )

        self.mitosis_cost = Integer(
            "Mitosis alternative cost",
            80,
            minval=1,
            doc="""\
*(Used only if the "LAP" tracking method is applied and the second phase is run)*

This setting is the cost of not linking a parent and two daughters via
the mitosis model. the LAP tracking method weighs this cost against
the score of a potential mitosis. The model expects the daughters to
be equidistant from the parent after mitosis, so the parent location
is expected to be midway between the daughters. In addition, the model
expects the daughters’ areas to be equal to the parent’s area. The
mitosis score is the distance error of the parent times the area
inequality ratio of the parent and daughters (the larger of
Area(daughters) / Area(parent) and Area(parent) / Area(daughters)).

|image0|  Recommendations:

-  An accepted mitosis closes two gaps, so all things being equal, the
   mitosis alternative cost should be approximately double the gap
   closing cost.
-  Increase the mitosis alternative cost to favor more mitoses and
   decrease it to prevent more mitoses candidates from being accepted.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                **{"PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON}
            ),
        )

        self.mitosis_max_distance = Integer(
            "Maximum mitosis distance, in pixel units",
            40,
            minval=1,
            doc="""\
*(Used only if the "LAP" tracking method is applied and the second phase is run)*

This setting is the maximum allowed distance in pixels of either of the
daughter candidate centroids after mitosis from the parent candidate."""
            % globals(),
        )

        self.max_gap_score = Integer(
            "Maximum gap displacement, in pixel units",
            5,
            minval=1,
            doc="""\
*(Used only if the "LAP" tracking method is applied and the second phase is run)*

This setting acts as a filter for unreasonably large displacements
during the second phase.

|image0|  Recommendations:

-  The maximum gap displacement should be set to roughly the maximum
   displacement of an object’s center from frame to frame. An object
   that makes large frame-to-frame jumps should have a higher value for
   this setting than one that only moves slightly.
-  Be aware that the LAP algorithm will run more slowly with a higher
   maximum gap displacement value, since the higher this value, the more
   objects that must be compared at each step.
-  Objects that would have been tracked between successive frames for a
   lower maximum displacement may not be tracked if the value is set
   higher.
-  This setting may be the culprit if an object is not tracked
   fame-to-frame despite optimizing the LAP first-pass settings.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                **{"PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON}
            ),
        )

        self.max_merge_score = Integer(
            "Maximum merge score",
            50,
            minval=1,
            doc="""\
*(Used only if the "LAP" tracking method is applied and the second phase is run)*

This setting acts as a filter for unreasonably large merge scores. The
merge score has two components:

-  The area of the resulting merged object relative to the area of the
   two objects to be merged.
-  The distances between the objects to be merged and the resulting
   object.

|image0|  Recommendations:

-  The LAP algorithm will run more slowly with a higher maximum merge
   score value.
-  Objects that would have been merged at a lower maximum merge score
   will not be considered for merging.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                **{"PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON}
            ),
        )

        self.max_split_score = Integer(
            "Maximum split score",
            50,
            minval=1,
            doc="""\
*(Used only if the "LAP" tracking method is applied and the second phase is run)*

This setting acts as a filter for unreasonably large split scores. The
split score has two components:

-  The area of the initial object relative to the area of the two
   objects resulting from the split.
-  The distances between the original and resulting objects.

|image0|  Recommendations:

-  The LAP algorithm will run more slowly with a maximum split score
   value.
-  Objects that would have been split at a lower maximum split score
   will not be considered for splitting.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                **{"PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON}
            ),
        )

        self.max_frame_distance = Integer(
            "Maximum temporal gap, in frames",
            5,
            minval=1,
            doc="""\
*(Used only if the "LAP" tracking method is applied and the second phase is run)*

**Care must be taken to adjust this setting appropriate to the data.**

This setting controls the maximum number of frames that can be skipped
when merging a temporal gap caused by an unsegmented object. These
gaps occur when an image is mis-segmented and identification fails to
find an object in one or more frames.

|image0|  Recommendations:

-  Set the maximum gap higher in order to have more chance of correctly
   recapturing an object after erroneously losing the original for a few
   frames.
-  Set the maximum gap lower to reduce the chance of erroneously
   connecting to the wrong object after correctly losing the original
   object (e.g., if the cell dies or moves off-screen).

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                **{"PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON}
            ),
        )

        self.average_cell_diameter = Float(
            "Average cell diameter in pixels",
            35.0,
            minval=5,
            doc="""\
*(Used only if "Follow Neighbors" tracking method is applied)*

The average cell diameter is used to scale many Follow Neighbors
algorithm parameters. %(HELP_ON_MEASURING_DISTANCES)s"""
            % globals(),
        )

        self.advanced_parameters = Binary(
            "Use advanced configuration parameters",
            False,
            doc="""\
*(Used only if "Follow Neighbors" tracking method is applied)*

Do you want to use advanced parameters to configure plugin? The default
values should be sufficient in most cases. You may want to use advanced
parameters when cells are incorrectly marked missing between frames or
cells of different sizes are falsely matched.""",
        )

        self.drop_cost = Float(
            "Cost of cell to empty matching",
            15,
            minval=1,
            maxval=200,
            doc="""\
*(Used only if "Follow Neighbors" tracking method is applied)*

The cost of considering cell (from frame t) not present in frame t+1.
Increasing this value leads to more cells (from t) being matched with
cells (from t+1) rather then classified as missing.

|image0|  Recommendations:

-  A value which is too high might cause incorrect cells to match
   between the frames.
-  A value which is too low might make the algorithm not to match cells
   between the frames.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                **{"PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON}
            ),
        )

        self.area_weight = Float(
            "Weight of area difference in function matching cost",
            25,
            minval=1,
            doc="""\
*(Used only if "Follow Neighbors" tracking method is applied)*
Increasing this value will make differences in position favored over
differences in area when identifying objects between frames.""",
        )

        self.wants_lifetime_filtering = Binary(
            "Filter objects by lifetime?",
            False,
            doc="""\
Select "*Yes*" if you want objects to be filtered by their lifetime,
i.e., total duration in frames. This is useful for marking objects
which transiently appear and disappear, such as the results of a
mis-segmentation.

You MUST use ExportToSpreadsheet, not ExportToDatabase, for 
lifetime filtering to work. 

|image0|  Recommendations:

-  This operation does not actually delete the filtered object, but
   merely removes its label from the tracked object list; the filtered
   object’s per-object measurements are retained.
-  An object can be filtered only if it is tracked as an unique object.
   Splits continue the lifetime count from their parents, so the minimum
   lifetime value does not apply to them.
   
Note that if you use lifetime filtering the output images generated by 
"*Save color-coded image?*" will NOT be accurate, as those images are 
generated before filtering is done and not edited afterward.

.. |image0| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                **{"PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON}
            ),
        )

        self.wants_minimum_lifetime = Binary(
            "Filter using a minimum lifetime?",
            True,
            doc="""\
*(Used only if objects are filtered by lifetime)*

Select "*Yes*" to filter the object on the basis of a minimum number
of frames.""".format(
                **{"PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON}
            ),
        )

        self.min_lifetime = Integer(
            "Minimum lifetime",
            1,
            minval=1,
            doc="""\
Enter the minimum number of frames an object is permitted to persist. Objects
which last this number of frames or lower are filtered out.""",
        )

        self.wants_maximum_lifetime = Binary(
            "Filter using a maximum lifetime?",
            False,
            doc="""\
*(Used only if objects are filtered by lifetime)*

Select "*Yes*" to filter the object on the basis of a maximum number
of frames."""
            % globals(),
        )

        self.max_lifetime = Integer(
            "Maximum lifetime",
            100,
            doc="""\
Enter the maximum number of frames an object is permitted to persist. Objects
which last this number of frames or more are filtered out.""",
        )

        self.display_type = Choice(
            "Select display option",
            DT_ALL,
            doc="""\
The output image can be saved as:

-  *%(DT_COLOR_ONLY)s:* A color-labeled image, with each tracked
   object assigned a unique color
-  *%(DT_COLOR_AND_NUMBER)s:* Same as above but with the tracked
   object number superimposed."""
            % globals(),
        )

        self.wants_image = Binary(
            "Save color-coded image?",
            False,
            doc="""\
Select "*Yes*" to retain the image showing the tracked objects for
later use in the pipeline. For example, a common use is for quality
control purposes saving the image with the **SaveImages** module.

Please note that if you are using the second phase of the LAP method
OR filtering by track lifetime, the final labels are not assigned until 
*after* the pipeline has completed processing of a particular timepoint. 
That means that saving the color-coded image will only show the an 
intermediate result and not the final product."""
            % globals(),
        )

        self.image_name = ImageName(
            "Name the output image",
            "TrackedCells",
            doc="""\
*(Used only if saving the color-coded image)*

Enter a name to give the color-coded image of tracked labels.""",
        )

    def settings(self):
        return [
            self.tracking_method,
            self.object_name,
            self.measurement,
            self.pixel_radius,
            self.display_type,
            self.wants_image,
            self.image_name,
            self.model,
            self.radius_std,
            self.radius_limit,
            self.wants_second_phase,
            self.gap_cost,
            self.split_cost,
            self.merge_cost,
            self.max_gap_score,
            self.max_split_score,
            self.max_merge_score,
            self.max_frame_distance,
            self.wants_lifetime_filtering,
            self.wants_minimum_lifetime,
            self.min_lifetime,
            self.wants_maximum_lifetime,
            self.max_lifetime,
            self.mitosis_cost,
            self.mitosis_max_distance,
            self.average_cell_diameter,
            self.advanced_parameters,
            self.drop_cost,
            self.area_weight,
        ]

    def validate_module(self, pipeline):
        """Make sure that the user has selected some limits when filtering"""
        if (
            self.tracking_method == "LAP"
            and self.wants_lifetime_filtering.value
            and (
                self.wants_minimum_lifetime.value == False
                and self.wants_minimum_lifetime.value == False
            )
        ):
            raise ValidationError(
                "Please enter a minimum and/or maximum lifetime limit",
                self.wants_lifetime_filtering,
            )

    def visible_settings(self):
        result = [self.tracking_method, self.object_name]
        if self.tracking_method == "Measurements":
            result += [self.measurement]
        if self.tracking_method == "LAP":
            result += [self.model, self.radius_std, self.radius_limit]
            result += [self.wants_second_phase]
            if self.wants_second_phase:
                result += [
                    self.gap_cost,
                    self.split_cost,
                    self.merge_cost,
                    self.mitosis_cost,
                    self.max_gap_score,
                    self.max_split_score,
                    self.max_merge_score,
                    self.max_frame_distance,
                    self.mitosis_max_distance,
                ]
        else:
            result += [self.pixel_radius]

        if self.tracking_method == "Follow Neighbors":
            result += [self.average_cell_diameter, self.advanced_parameters]
            if self.advanced_parameters:
                result += [self.drop_cost, self.area_weight]
        result += [self.wants_lifetime_filtering]

        if self.wants_lifetime_filtering:
            result += [self.wants_minimum_lifetime]
            if self.wants_minimum_lifetime:
                result += [self.min_lifetime]
            result += [self.wants_maximum_lifetime]
            if self.wants_maximum_lifetime:
                result += [self.max_lifetime]

        result += [self.display_type, self.wants_image]
        if self.wants_image.value:
            result += [self.image_name]
        return result

    @property
    def static_model(self):
        return self.model in (M_RANDOM, M_BOTH)

    @property
    def velocity_model(self):
        return self.model in (M_VELOCITY, M_BOTH)

    def get_ws_dictionary(self, workspace):
        return self.get_dictionary(workspace.image_set_list)

    def __get(self, field, workspace, default):
        if field in self.get_ws_dictionary(workspace):
            return self.get_ws_dictionary(workspace)[field]
        return default

    def __set(self, field, workspace, value):
        self.get_ws_dictionary(workspace)[field] = value

    def get_group_image_numbers(self, workspace):
        m = workspace.measurements
        assert isinstance(m, Measurements)
        d = self.get_ws_dictionary(workspace)
        group_number = m.get_group_number()
        if "group_number" not in d or d["group_number"] != group_number:
            d["group_number"] = group_number
            group_indexes = np.array(
                [
                    (m.get_measurement("Image", GROUP_INDEX, i), i)
                    for i in m.get_image_numbers()
                    if m.get_measurement("Image", GROUP_NUMBER, i) == group_number
                ],
                int,
            )
            order = np.lexsort([group_indexes[:, 0]])
            d["group_image_numbers"] = group_indexes[order, 1]
        return d["group_image_numbers"]

    def get_saved_measurements(self, workspace):
        return self.__get("measurements", workspace, np.array([], float))

    def set_saved_measurements(self, workspace, value):
        self.__set("measurements", workspace, value)

    def get_saved_coordinates(self, workspace):
        return self.__get("coordinates", workspace, np.zeros((2, 0), int))

    def set_saved_coordinates(self, workspace, value):
        self.__set("coordinates", workspace, value)

    def get_orig_coordinates(self, workspace):
        """The coordinates of the first occurrence of an object's ancestor"""
        return self.__get("orig coordinates", workspace, np.zeros((2, 0), int))

    def set_orig_coordinates(self, workspace, value):
        self.__set("orig coordinates", workspace, value)

    def get_saved_labels(self, workspace):
        return self.__get("labels", workspace, None)

    def set_saved_labels(self, workspace, value):
        self.__set("labels", workspace, value)

    def get_saved_object_numbers(self, workspace):
        return self.__get("object_numbers", workspace, np.array([], int))

    def set_saved_object_numbers(self, workspace, value):
        return self.__set("object_numbers", workspace, value)

    def get_saved_ages(self, workspace):
        return self.__get("ages", workspace, np.array([], int))

    def set_saved_ages(self, workspace, values):
        self.__set("ages", workspace, values)

    def get_saved_distances(self, workspace):
        return self.__get("distances", workspace, np.zeros((0,)))

    def set_saved_distances(self, workspace, values):
        self.__set("distances", workspace, values)

    def get_max_object_number(self, workspace):
        return self.__get("max_object_number", workspace, 0)

    def set_max_object_number(self, workspace, value):
        self.__set("max_object_number", workspace, value)

    def get_kalman_states(self, workspace):
        return self.__get("kalman_states", workspace, None)

    def set_kalman_states(self, workspace, value):
        self.__set("kalman_states", workspace, value)

    def prepare_group(self, workspace, grouping, image_numbers):
        """Erase any tracking information at the start of a run"""
        d = self.get_dictionary(workspace.image_set_list)
        d.clear()

        return True

    def measurement_name(self, feature):
        """Return a measurement name for the given feature"""
        if self.tracking_method == "LAP":
            return "%s_%s" % (F_PREFIX, feature)
        return "%s_%s_%s" % (F_PREFIX, feature, str(self.pixel_radius.value))

    def image_measurement_name(self, feature):
        """Return a measurement name for an image measurement"""
        if self.tracking_method == "LAP":
            return "%s_%s_%s" % (F_PREFIX, feature, self.object_name.value)
        return "%s_%s_%s_%s" % (
            F_PREFIX,
            feature,
            self.object_name.value,
            str(self.pixel_radius.value),
        )

    def add_measurement(self, workspace, feature, values):
        """Add a measurement to the workspace's measurements

        workspace - current image set's workspace
        feature - name of feature being measured
        values - one value per object
        """
        workspace.measurements.add_measurement(
            self.object_name.value, self.measurement_name(feature), values
        )

    def add_image_measurement(self, workspace, feature, value):
        measurement_name = self.image_measurement_name(feature)
        workspace.measurements.add_image_measurement(measurement_name, value)

    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.object_name.value)
        if self.tracking_method == "Distance":
            self.run_distance(workspace, objects)
        elif self.tracking_method == "Overlap":
            self.run_overlap(workspace, objects)
        elif self.tracking_method == "Measurements":
            self.run_measurements(workspace, objects)
        elif self.tracking_method == "LAP":
            self.run_lapdistance(workspace, objects)
        elif self.tracking_method == "Follow Neighbors":
            self.run_followneighbors(workspace, objects)
        else:
            raise NotImplementedError(
                "Unimplemented tracking method: %s" % self.tracking_method.value
            )
        if self.wants_image.value:
            import matplotlib.figure
            import matplotlib.axes
            import matplotlib.backends.backend_agg
            import matplotlib.transforms
            from cellprofiler.gui.tools import figure_to_image, only_display_image

            figure = matplotlib.figure.Figure()
            canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(figure)
            ax = figure.add_subplot(1, 1, 1)
            self.draw(objects.segmented, ax, self.get_saved_object_numbers(workspace))
            #
            # This is the recipe for just showing the axis
            #
            only_display_image(figure, objects.segmented.shape)
            image_pixels = figure_to_image(figure, dpi=figure.dpi)
            image = Image(image_pixels)
            workspace.image_set.add(self.image_name.value, image)
        if self.show_window:
            workspace.display_data.labels = objects.segmented
            workspace.display_data.object_numbers = self.get_saved_object_numbers(
                workspace
            )

    def display(self, workspace, figure):
        if hasattr(workspace.display_data, "labels"):
            figure.set_subplots((1, 1))
            subfigure = figure.figure
            subfigure.clf()
            ax = subfigure.add_subplot(1, 1, 1)
            self.draw(
                workspace.display_data.labels, ax, workspace.display_data.object_numbers
            )
        else:
            # We get here after running as a data tool
            figure.figure.text(0.5, 0.5, "Analysis complete", ha="center", va="center")

    def draw(self, labels, ax, object_numbers):
        import matplotlib.cm
        import matplotlib.colors

        indexer = np.zeros(len(object_numbers) + 1, int)
        indexer[1:] = object_numbers
        #
        # We want to keep the colors stable, but we also want the
        # largest possible separation between adjacent colors. So, here
        # we reverse the significance of the bits in the indices so
        # that adjacent number (e.g., 0 and 1) differ by 128, roughly
        #
        pow_of_2 = 2 ** np.mgrid[0:8, 0 : len(indexer)][0]
        bits = (indexer & pow_of_2).astype(bool)
        indexer = np.sum(bits.transpose() * (2 ** np.arange(7, -1, -1)), 1)
        recolored_labels = indexer[labels]
        cm = matplotlib.cm.get_cmap(get_default_colormap())
        cm.set_bad((0, 0, 0))
        norm = matplotlib.colors.BoundaryNorm(list(range(256)), 256)
        img = ax.imshow(
            numpy.ma.array(recolored_labels, mask=(labels == 0)), cmap=cm, norm=norm
        )
        if self.display_type == DT_COLOR_AND_NUMBER:
            i, j = centers_of_labels(labels)
            for n, x, y in zip(object_numbers, j, i):
                if np.isnan(x) or np.isnan(y):
                    # This happens if there are missing labels
                    continue
                ax.annotate(
                    str(n), xy=(x, y), color="white", arrowprops=dict(visible=False)
                )

    def run_followneighbors(self, workspace, objects):
        """Track objects based on following neighbors"""

        def calculate_iteration_value(param, initial_value):
            iteration_default = NeighbourMovementTrackingParameters.parameters_cost_iteration[
                param
            ]
            initial_default = NeighbourMovementTrackingParameters.parameters_cost_initial[
                param
            ]
            return float(iteration_default) / initial_default * initial_value

        tracker = NeighbourMovementTracking()
        tracker.parameters_tracking[
            "avgCellDiameter"
        ] = self.average_cell_diameter.value
        tracker.parameters_tracking["max_distance"] = self.pixel_radius.value

        tracker.parameters_cost_initial["default_empty_cost"] = self.drop_cost.value
        tracker.parameters_cost_iteration[
            "default_empty_cost"
        ] = calculate_iteration_value("default_empty_cost", self.drop_cost.value)

        tracker.parameters_cost_initial["area_weight"] = self.area_weight.value
        tracker.parameters_cost_iteration["area_weight"] = calculate_iteration_value(
            "area_weight", self.area_weight.value
        )

        old_labels = self.get_saved_labels(workspace)
        if old_labels is None:
            i, j = (centers_of_labels(objects.segmented) + 0.5).astype(int)
            count = len(i)
            self.map_objects(workspace, np.zeros((0,), int), np.zeros(count, int), i, j)
        else:
            old_i, old_j = (centers_of_labels(old_labels) + 0.5).astype(int)
            old_count = len(old_i)

            i, j = (centers_of_labels(objects.segmented) + 0.5).astype(int)
            count = len(i)

            new_labels = objects.segmented
            # Matching is (expected to be) a injective function of old labels to new labels so we can inverse it.
            matching = tracker.run_tracking(old_labels, new_labels)

            new_object_numbers = np.zeros(count, int)
            old_object_numbers = np.zeros(old_count, int)
            for old, new in matching:
                new_object_numbers[new - 1] = old
                old_object_numbers[old - 1] = new

            self.map_objects(workspace, old_object_numbers, new_object_numbers, i, j)
        self.set_saved_labels(workspace, objects.segmented)

    def run_distance(self, workspace, objects):
        """Track objects based on distance"""
        old_i, old_j = self.get_saved_coordinates(workspace)
        if len(old_i):
            distances, (i, j) = distance_transform_edt(
                objects.segmented == 0, return_indices=True
            )
            #
            # Look up the coordinates of the nearest new object (given by
            # the transform i,j), then look up the label at that coordinate
            # (objects.segmented[#,#])
            #
            new_object_numbers = objects.segmented[i[old_i, old_j], j[old_i, old_j]]
            #
            # Mask out any objects at too great of a distance
            #
            new_object_numbers[distances[old_i, old_j] > self.pixel_radius.value] = 0
            #
            # Do the same with the new centers and old objects
            #
            i, j = (centers_of_labels(objects.segmented) + 0.5).astype(int)
            old_labels = self.get_saved_labels(workspace)
            distances, (old_i, old_j) = distance_transform_edt(
                old_labels == 0, return_indices=True
            )
            old_object_numbers = old_labels[old_i[i, j], old_j[i, j]]
            old_object_numbers[distances[i, j] > self.pixel_radius.value] = 0
            self.map_objects(workspace, new_object_numbers, old_object_numbers, i, j)
        else:
            i, j = (centers_of_labels(objects.segmented) + 0.5).astype(int)
            count = len(i)
            self.map_objects(workspace, np.zeros((0,), int), np.zeros(count, int), i, j)
        self.set_saved_labels(workspace, objects.segmented)

    def run_lapdistance(self, workspace, objects):
        """Track objects based on distance"""
        m = workspace.measurements

        old_i, old_j = self.get_saved_coordinates(workspace)
        n_old = len(old_i)
        #
        # Automatically set the cost of birth and death above
        # that of the largest allowable cost.
        #
        costBorn = costDie = self.radius_limit.max * 1.10
        kalman_states = self.get_kalman_states(workspace)
        if kalman_states is None:
            if self.static_model:
                kalman_states = [centrosome.filter.static_kalman_model()]
            else:
                kalman_states = []
            if self.velocity_model:
                kalman_states.append(centrosome.filter.velocity_kalman_model())
        areas = fixup_scipy_ndimage_result(
            scipy.ndimage.sum(
                np.ones(objects.segmented.shape),
                objects.segmented,
                np.arange(1, np.max(objects.segmented) + 1, dtype=np.int32),
            )
        )
        areas = areas.astype(int)
        model_types = np.array(
            [
                m
                for m, s in (
                    (KM_NO_VEL, self.static_model),
                    (KM_VEL, self.velocity_model),
                )
                if s
            ],
            int,
        )

        if n_old > 0:
            new_i, new_j = centers_of_labels(objects.segmented)
            n_new = len(new_i)
            i, j = np.mgrid[0:n_old, 0:n_new]
            ##############################
            #
            #  Kalman filter prediction
            #
            #
            # We take the lowest cost among all possible models
            #
            minDist = np.ones((n_old, n_new)) * self.radius_limit.max
            d = np.ones((n_old, n_new)) * np.inf
            sd = np.zeros((n_old, n_new))
            # The index of the Kalman filter used: -1 means not used
            kalman_used = -np.ones((n_old, n_new), int)
            for nkalman, kalman_state in enumerate(kalman_states):
                assert isinstance(kalman_state, centrosome.filter.KalmanState)
                obs = kalman_state.predicted_obs_vec
                dk = np.sqrt((obs[i, 0] - new_i[j]) ** 2 + (obs[i, 1] - new_j[j]) ** 2)
                noise_sd = np.sqrt(np.sum(kalman_state.noise_var[:, 0:2], 1))
                radius = np.maximum(
                    np.minimum(noise_sd * self.radius_std.value, self.radius_limit.max),
                    self.radius_limit.min,
                )

                is_best = (dk < d) & (dk < radius[:, np.newaxis])
                d[is_best] = dk[is_best]
                minDist[is_best] = radius[i][is_best]
                kalman_used[is_best] = nkalman
            minDist = np.maximum(
                np.minimum(minDist, self.radius_limit.max), self.radius_limit.min
            )
            #
            #############################
            #
            # Linear assignment setup
            #
            t = np.argwhere((d < minDist))
            x = np.sqrt(
                (old_i[t[0 : t.size, 0]] - new_i[t[0 : t.size, 1]]) ** 2
                + (old_j[t[0 : t.size, 0]] - new_j[t[0 : t.size, 1]]) ** 2
            )
            t = t + 1
            t = np.column_stack((t, x))
            a = np.arange(len(old_i)) + 2
            x = np.searchsorted(t[0 : (t.size // 2), 0], a)
            a = np.arange(len(old_i)) + 1
            b = np.arange(len(old_i)) + len(new_i) + 1
            c = np.zeros(len(old_i)) + costDie
            b = np.column_stack((a, b, c))
            t = np.insert(t, x, b, 0)

            i, j = np.mgrid[0 : len(new_i), 0 : len(old_i) + 1]
            i = i + len(old_i) + 1
            j = j + len(new_i)
            j[0 : len(new_i) + 1, 0] = i[0 : len(new_i) + 1, 0] - len(old_i)
            x = np.zeros((len(new_i), len(old_i) + 1))
            x[0 : len(new_i) + 1, 0] = costBorn
            i = i.flatten()
            j = j.flatten()
            x = x.flatten()
            x = np.column_stack((i, j, x))
            t = np.vstack((t, x))

            # Tack 0 <-> 0 at the start because object #s start at 1
            i = np.hstack([0, t[:, 0].astype(int)])
            j = np.hstack([0, t[:, 1].astype(int)])
            c = np.hstack([0, t[:, 2]])
            x, y = lapjv(i, j, c)

            a = np.argwhere(x > len(new_i))
            b = np.argwhere(y > len(old_i))
            x[a[0 : len(a)]] = 0
            y[b[0 : len(b)]] = 0
            a = np.arange(len(old_i)) + 1
            b = np.arange(len(new_i)) + 1
            new_object_numbers = x[a[0 : len(a)]].astype(int)
            old_object_numbers = y[b[0 : len(b)]].astype(int)

            ###############################
            #
            #  Kalman filter update
            #
            model_idx = np.zeros(len(old_object_numbers), int)
            linking_distance = np.ones(len(old_object_numbers)) * np.NaN
            standard_deviation = np.ones(len(old_object_numbers)) * np.NaN
            model_type = np.ones(len(old_object_numbers), int) * KM_NONE
            link_type = np.ones(len(old_object_numbers), int) * LT_NONE
            mask = old_object_numbers > 0
            old_idx = old_object_numbers - 1
            model_idx[mask] = kalman_used[old_idx[mask], mask]
            linking_distance[mask] = d[old_idx[mask], mask]
            standard_deviation[mask] = linking_distance[mask] / noise_sd[old_idx[mask]]
            model_type[mask] = model_types[model_idx[mask]]
            link_type[mask] = LT_PHASE_1
            #
            # The measurement covariance is the square of the
            # standard deviation of the measurement error. Assume
            # that the measurement error comes from not knowing where
            # the center is within the cell, then the error is
            # proportional to the radius and the square to the area.
            #
            measurement_variance = areas.astype(float) / np.pi
            #
            # Broadcast the measurement error into a diagonal matrix
            #
            r = (
                measurement_variance[:, np.newaxis, np.newaxis]
                * np.eye(2)[np.newaxis, :, :]
            )
            new_kalman_states = []
            for kalman_state in kalman_states:
                #
                # The process noise covariance is a diagonal of the
                # state noise variance.
                #
                state_len = kalman_state.state_len
                q = np.zeros((len(old_idx), state_len, state_len))
                if np.any(mask):
                    #
                    # Broadcast into the diagonal
                    #
                    new_idx = np.arange(len(old_idx))[mask]
                    matching_idx = old_idx[new_idx]
                    i, j = np.mgrid[0 : len(matching_idx), 0:state_len]
                    q[new_idx[i], j, j] = kalman_state.noise_var[matching_idx[i], j]
                new_kalman_state = centrosome.filter.kalman_filter(
                    kalman_state, old_idx, np.column_stack((new_i, new_j)), q, r
                )
                new_kalman_states.append(new_kalman_state)
            self.set_kalman_states(workspace, new_kalman_states)

            i, j = (centers_of_labels(objects.segmented) + 0.5).astype(int)
            self.map_objects(workspace, new_object_numbers, old_object_numbers, i, j)
        else:
            i, j = centers_of_labels(objects.segmented)
            count = len(i)
            link_type = np.ones(count, int) * LT_NONE
            model_type = np.ones(count, int) * KM_NONE
            linking_distance = np.ones(count) * np.NaN
            standard_deviation = np.ones(count) * np.NaN
            #
            # Initialize the kalman_state with the new objects
            #
            new_kalman_states = []
            r = np.zeros((count, 2, 2))
            for kalman_state in kalman_states:
                q = np.zeros((count, kalman_state.state_len, kalman_state.state_len))
                new_kalman_state = centrosome.filter.kalman_filter(
                    kalman_state, -np.ones(count), np.column_stack((i, j)), q, r
                )
                new_kalman_states.append(new_kalman_state)
            self.set_kalman_states(workspace, new_kalman_states)

            i = (i + 0.5).astype(int)
            j = (j + 0.5).astype(int)
            self.map_objects(workspace, np.zeros((0,), int), np.zeros(count, int), i, j)
        m = workspace.measurements
        assert isinstance(m, Measurements)
        m.add_measurement(self.object_name.value, self.measurement_name(F_AREA), areas)
        m[
            self.object_name.value, self.measurement_name(F_LINKING_DISTANCE)
        ] = linking_distance
        m[
            self.object_name.value, self.measurement_name(F_STANDARD_DEVIATION)
        ] = standard_deviation
        m[self.object_name.value, self.measurement_name(F_MOVEMENT_MODEL)] = model_type
        m[self.object_name.value, self.measurement_name(F_LINK_TYPE)] = link_type
        self.save_kalman_measurements(workspace)
        self.set_saved_labels(workspace, objects.segmented)

    def get_kalman_models(self):
        """Return tuples of model and names of the vector elements"""
        if self.static_model:
            models = [(F_STATIC_MODEL, (F_Y, F_X))]
        else:
            models = []
        if self.velocity_model:
            models.append((F_VELOCITY_MODEL, (F_Y, F_X, F_VY, F_VX)))
        return models

    def save_kalman_measurements(self, workspace):
        """Save the first-pass state_vec, state_cov and state_noise"""

        m = workspace.measurements
        object_name = self.object_name.value
        for (model, elements), kalman_state in zip(
            self.get_kalman_models(), self.get_kalman_states(workspace)
        ):
            assert isinstance(kalman_state, centrosome.filter.KalmanState)
            nobjs = len(kalman_state.state_vec)
            if nobjs > 0:
                #
                # Get the last state_noise entry for each object
                #
                # scipy.ndimage.maximum probably should return NaN if
                # no index exists, but, in 0.8.0, returns 0. So stack
                # a bunch of -1 values so every object will have a "-1"
                # index.
                last_idx = scipy.ndimage.maximum(
                    np.hstack(
                        (-np.ones(nobjs), np.arange(len(kalman_state.state_noise_idx)))
                    ),
                    np.hstack((np.arange(nobjs), kalman_state.state_noise_idx)),
                    np.arange(nobjs),
                )
                last_idx = last_idx.astype(int)
            for i, element in enumerate(elements):
                #
                # state_vec
                #
                mname = self.measurement_name(kalman_feature(model, F_STATE, element))
                values = np.zeros(0) if nobjs == 0 else kalman_state.state_vec[:, i]
                m.add_measurement(object_name, mname, values)
                #
                # state_noise
                #
                mname = self.measurement_name(kalman_feature(model, F_NOISE, element))
                values = np.zeros(nobjs)
                if nobjs > 0:
                    values[last_idx == -1] = np.NaN
                    values[last_idx > -1] = kalman_state.state_noise[
                        last_idx[last_idx > -1], i
                    ]
                m.add_measurement(object_name, mname, values)
                #
                # state_cov
                #
                for j, el2 in enumerate(elements):
                    mname = self.measurement_name(
                        kalman_feature(model, F_COV, element, el2)
                    )
                    values = kalman_state.state_cov[:, i, j]
                    m.add_measurement(object_name, mname, values)

    def run_overlap(self, workspace, objects):
        """Track objects by maximum # of overlapping pixels"""
        current_labels = objects.segmented
        old_labels = self.get_saved_labels(workspace)
        i, j = (centers_of_labels(objects.segmented) + 0.5).astype(int)
        if old_labels is None:
            count = len(i)
            self.map_objects(workspace, np.zeros((0,), int), np.zeros(count, int), i, j)
        else:
            mask = (current_labels > 0) & (old_labels > 0)
            cur_count = np.max(current_labels)
            old_count = np.max(old_labels)
            count = np.sum(mask)
            if count == 0:
                # There's no overlap.
                self.map_objects(
                    workspace, np.zeros(old_count, int), np.zeros(cur_count, int), i, j
                )
            else:
                cur = current_labels[mask]
                old = old_labels[mask]
                histogram = scipy.sparse.coo_matrix(
                    (np.ones(count), (cur, old)), shape=(cur_count + 1, old_count + 1)
                ).toarray()
                old_of_new = np.argmax(histogram, 1)[1:]
                new_of_old = np.argmax(histogram, 0)[1:]
                #
                # The cast here seems to be needed to make scipy.ndimage.sum
                # work. See http://projects.scipy.org/numpy/ticket/1012
                #
                old_of_new = np.array(old_of_new, np.int16)
                old_of_new = np.array(old_of_new, np.int32)
                new_of_old = np.array(new_of_old, np.int16)
                new_of_old = np.array(new_of_old, np.int32)
                self.map_objects(workspace, new_of_old, old_of_new, i, j)
        self.set_saved_labels(workspace, current_labels)

    def run_measurements(self, workspace, objects):
        current_labels = objects.segmented
        new_measurements = workspace.measurements.get_current_measurement(
            self.object_name.value, self.measurement.value
        )
        old_measurements = self.get_saved_measurements(workspace)
        old_labels = self.get_saved_labels(workspace)
        i, j = (centers_of_labels(objects.segmented) + 0.5).astype(int)
        if old_labels is None:
            count = len(i)
            self.map_objects(workspace, np.zeros((0,), int), np.zeros(count, int), i, j)
        else:
            associations = associate_by_distance(
                old_labels, current_labels, self.pixel_radius.value
            )
            best_child = np.zeros(len(old_measurements), int)
            best_parent = np.zeros(len(new_measurements), int)
            best_child_measurement = (
                np.ones(len(old_measurements), int) * np.finfo(float).max
            )
            best_parent_measurement = (
                np.ones(len(new_measurements), int) * np.finfo(float).max
            )
            for old, new in associations:
                diff = abs(old_measurements[old - 1] - new_measurements[new - 1])
                if diff < best_child_measurement[old - 1]:
                    best_child[old - 1] = new
                    best_child_measurement[old - 1] = diff
                if diff < best_parent_measurement[new - 1]:
                    best_parent[new - 1] = old
                    best_parent_measurement[new - 1] = diff
            self.map_objects(workspace, best_child, best_parent, i, j)
        self.set_saved_labels(workspace, current_labels)
        self.set_saved_measurements(workspace, new_measurements)

    def run_as_data_tool(self, workspace):
        m = workspace.measurements
        assert isinstance(m, Measurements)
        group_numbers = {}
        for i in m.get_image_numbers():
            group_number = m.get_measurement("Image", GROUP_NUMBER, i)
            group_index = m.get_measurement("Image", GROUP_INDEX, i)
            if (group_number not in group_numbers) or (
                group_numbers[group_number][1] > group_index
            ):
                group_numbers[group_number] = (i, group_index)

        for group_number in sorted(group_numbers.keys()):
            m.image_set_number = group_numbers[group_number][0]
            self.post_group(workspace, {})

    def flood(self, i, at, a, b, c, d, z):
        z[i] = at
        if a[i] != -1 and z[a[i]] == 0:
            z = self.flood(a[i], at, a, b, c, d, z)
        if b[i] != -1 and z[b[i]] == 0:
            z = self.flood(b[i], at, a, b, c, d, z)
        if c[i] != -1 and z[c[i]] == 0:
            z = self.flood(c[i], at, a, b, c, d, z)
        if c[i] != -1 and z[c[i]] == 0:
            z = self.flood(c[i], at, a, b, c, d, z)
        return z

    def is_aggregation_module(self):
        """We connect objects across imagesets within a group = aggregation"""
        return True

    def post_group(self, workspace, grouping):
        # If any tracking method other than LAP, recalculate measurements
        # (Really, only the final age needs to be re-done)
        image_numbers = self.get_group_image_numbers(workspace)
        if self.tracking_method != "LAP":
            m = workspace.measurements
            assert isinstance(m, Measurements)
            self.recalculate_group(workspace, image_numbers)
            return

        self.recalculate_kalman_filters(workspace, image_numbers)
        if not self.wants_second_phase:
            return

        gap_cost = float(self.gap_cost.value)
        split_alternative_cost = float(self.split_cost.value) / 2
        merge_alternative_cost = float(self.merge_cost.value)
        mitosis_alternative_cost = float(self.mitosis_cost.value)

        max_gap_score = self.max_gap_score.value
        max_merge_score = self.max_merge_score.value
        max_split_score = self.max_split_score.value / 2  # to match legacy
        max_frame_difference = self.max_frame_distance.value

        m = workspace.measurements
        assert isinstance(m, Measurements)
        image_numbers = self.get_group_image_numbers(workspace)
        object_name = self.object_name.value
        (
            label,
            object_numbers,
            a,
            b,
            Area,
            parent_object_numbers,
            parent_image_numbers,
        ) = [
            [
                m.get_measurement(object_name, feature, i).astype(mtype)
                for i in image_numbers
            ]
            for feature, mtype in (
                (self.measurement_name(F_LABEL), int),
                (OBJECT_NUMBER, int),
                (M_LOCATION_CENTER_X, float),
                (M_LOCATION_CENTER_Y, float),
                (self.measurement_name(F_AREA), float),
                (self.measurement_name(F_PARENT_OBJECT_NUMBER), int),
                (self.measurement_name(F_PARENT_IMAGE_NUMBER), int),
            )
        ]
        group_indices, new_object_count, lost_object_count, merge_count, split_count = [
            np.array(
                [m.get_measurement("Image", feature, i) or 0 for i in image_numbers], int,
            )
            for feature in (
                GROUP_INDEX,
                self.image_measurement_name(F_NEW_OBJECT_COUNT),
                self.image_measurement_name(F_LOST_OBJECT_COUNT),
                self.image_measurement_name(F_MERGE_COUNT),
                self.image_measurement_name(F_SPLIT_COUNT),
            )
        ]
        #
        # Map image number to group index and vice versa
        #
        image_number_group_index = np.zeros(np.max(image_numbers) + 1, int)
        image_number_group_index[image_numbers] = np.array(group_indices, int)
        group_index_image_number = np.zeros(np.max(group_indices) + 1, int)
        group_index_image_number[group_indices] = image_numbers

        if all([len(lll) == 0 for lll in label]):
            return  # Nothing to do

        # sets up the arrays F, L, P, and Q
        # F is an array of all the cells that are the starts of segments
        #  F[:, :2] are the coordinates
        #  F[:, 2] is the image index
        #  F[:, 3] is the object index
        #  F[:, 4] is the object number
        #  F[:, 5] is the label
        #  F[:, 6] is the area
        #  F[:, 7] is the index into P
        # L is the ends
        # P includes all cells

        X = 0
        Y = 1
        IIDX = 2
        OIIDX = 3
        ONIDX = 4
        LIDX = 5
        AIDX = 6
        PIDX = 7
        P = np.vstack(
            [
                np.column_stack(
                    (
                        x,
                        y,
                        np.ones(len(x)) * i,
                        np.arange(len(x)),
                        o,
                        l,
                        area,
                        np.zeros(len(x)),
                    )
                )
                for i, (x, y, o, l, area) in enumerate(
                    zip(a, b, object_numbers, label, Area)
                )
            ]
        )
        count_per_label = np.bincount(P[:, LIDX].astype(int))
        idx = np.hstack([0, np.cumsum(count_per_label)])
        unique_label = np.unique(P[:, LIDX].astype(int))
        order = np.lexsort((P[:, OIIDX], P[:, IIDX], P[:, LIDX]))
        P = P[order, :]
        P[:, PIDX] = np.arange(len(P))
        F = P[idx[unique_label], :]
        L = P[idx[unique_label + 1] - 1, :]

        # Creates P1 and P2, which is P without the starts and ends
        # of segments respectively, representing possible
        # points of merges and splits respectively

        P1 = np.delete(P, idx[:-1], 0)
        P2 = np.delete(P, idx[idx > 0] - 1, 0)

        ##################################################
        #
        # Addresses of supplementary nodes:
        #
        # The LAP array is composed of the following ranges
        #
        # Count | node type
        # ------------------
        # T     | segment starts and ends
        # T     | gaps
        # OB    | split starts
        # OB    | merge ends
        # M     | mitoses
        #
        # T = # tracks
        # OB = # of objects that can serve as merge or split points
        # M = # of mitoses
        #
        # The graph:
        #
        # Gap Alternatives (in other words, do nothing)
        # ----------------------------------------------
        # End[i] <----> Gap alternative[i]
        # Gap alternative[i] <----> Start[i]
        # Split[i] <----> Split[i]
        # Merge[j] <----> Merge[j]
        # Mitosis[i] <----> Mitosis[i]
        #
        #
        # Bridge gaps:
        # -----------------------------------------------
        #
        # End[i] <---> Start[j]
        # Gap alternative[i] <----> Gap alternative[j]
        #
        # Splits
        # -----------------------------------------------
        #
        # Split[i] <----> Start[j]
        # Gap alternative[j] <----> Split[i]
        #
        # Merges
        # -----------------------------------------------
        # End[i] <----> Merge[j]
        # Merge[j] <----> Gap alternative[i]
        #
        # Mitoses
        # -----------------------------------------------
        # The mitosis model is somewhat imperfect. The mitosis
        # caps the parent and makes it unavailable as a candidate
        # for a gap closing. In the best case, there is only one
        # mitosis candidate for the left and right child and
        # the left and right child are connected to gap alternatives,
        # but there may be competing splits, gap closings or
        # other mitoses.
        #
        # We take a greedy approach, ordering the mitoses by their
        # scores and fulfilling them. After processing the mitoses,
        # we run LAP again, keeping only the parent nodes of untaken
        # mitoses and child nodes connected to gap alternatives
        #
        # End[i] <----> Mitosis[j]
        #
        ##################################################

        end_nodes = []
        start_nodes = []
        scores = []
        #
        # The offsets and lengths of the start/end node ranges
        #
        start_end_off = 0
        start_end_len = len(L)
        gap_off = start_end_end = start_end_len
        gap_end = gap_off + start_end_len
        # -------------------------------------------
        #
        # Null model (do nothing)
        #
        # -------------------------------------------

        for first, second in ((end_nodes, start_nodes), (start_nodes, end_nodes)):
            first.append(np.arange(start_end_len))
            second.append(np.arange(start_end_len) + gap_off)
            scores.append(np.ones(start_end_len) * gap_cost / 2)

        # ------------------------------------------
        #
        # Gap-closing model
        #
        # ------------------------------------------

        #
        # Create the edges between ends and starts.
        # The edge weight is the gap pair cost.
        #
        a, gap_scores = self.get_gap_pair_scores(F, L, max_frame_difference)
        # filter by max gap score
        mask = gap_scores <= max_gap_score
        if np.sum(mask) > 0:
            a, gap_scores = a[mask], gap_scores[mask]
            end_nodes.append(a[:, 0])
            start_nodes.append(a[:, 1])
            scores.append(gap_scores)
            #
            # Hook the gap alternative ends of the starts to
            # the gap alternative starts of the ends
            #
            end_nodes.append(a[:, 1] + gap_off)
            start_nodes.append(a[:, 0] + gap_off)
            scores.append(np.zeros(len(gap_scores)))

        # ---------------------------------------------------
        #
        # Merge model
        #
        # ---------------------------------------------------

        #
        # The first column of z is the index of the track that ends. The second
        # is the index into P2 of the object to be merged into
        #
        merge_off = gap_end
        if len(P1) > 0:
            # Do the initial winnowing in chunks of 10m pairs
            lchunk_size = 10000000 // len(P1)
            chunks = []
            for lstart in range(0, len(L), lchunk_size):
                lend = min(len(L), lstart + lchunk_size)
                merge_p1idx, merge_lidx = [
                    _.flatten() for _ in np.mgrid[0 : len(P1), lstart:lend]
                ]
                z = (P1[merge_p1idx, IIDX] - L[merge_lidx, IIDX]).astype(np.int32)
                mask = (z <= max_frame_difference) & (z > 0)
                if np.sum(mask) > 0:
                    chunks.append([_[mask] for _ in (merge_p1idx, merge_lidx, z)])
            if len(chunks) > 0:
                merge_p1idx, merge_lidx, z = [
                    np.hstack([_[i] for _ in chunks]) for i in range(3)
                ]
            else:
                merge_p1idx = merge_lidx = z = np.zeros(0, np.int32)
        else:
            merge_p1idx = merge_lidx = z = np.zeros(0, np.int32)

        if len(z) > 0:
            # Calculate penalty = distance * area penalty
            AreaLast = L[merge_lidx, AIDX]
            AreaBeforeMerge = P[P1[merge_p1idx, PIDX].astype(int) - 1, AIDX]
            AreaAtMerge = P1[merge_p1idx, AIDX]
            rho = self.calculate_area_penalty(AreaLast + AreaBeforeMerge, AreaAtMerge)
            d = np.sqrt(np.sum((L[merge_lidx, :2] - P2[merge_p1idx, :2]) ** 2, 1))
            merge_scores = d * rho
            mask = merge_scores <= max_merge_score
            merge_p1idx, merge_lidx, merge_scores = [
                _[mask] for _ in (merge_p1idx, merge_lidx, merge_scores)
            ]
            merge_len = np.sum(mask)
            if merge_len > 0:
                #
                # The end nodes are the ends being merged to the intermediates
                # The start nodes are the intermediates and have node #s
                # that start at merge_off
                #
                end_nodes.append(merge_lidx)
                start_nodes.append(merge_off + np.arange(merge_len))
                scores.append(merge_scores)
                #
                # Hook the gap alternative starts for the ends to
                # the merge nodes
                #
                end_nodes.append(merge_off + np.arange(merge_len))
                start_nodes.append(merge_lidx + gap_off)
                scores.append(np.ones(merge_len) * gap_cost / 2)
                #
                # The alternative hypothesis is represented by merges hooked
                # to merges
                #
                end_nodes.append(merge_off + np.arange(merge_len))
                start_nodes.append(merge_off + np.arange(merge_len))
                scores.append(np.ones(merge_len) * merge_alternative_cost)
        else:
            merge_len = 0
        merge_end = merge_off + merge_len

        # ------------------------------------------------------
        #
        # Split model
        #
        # ------------------------------------------------------

        split_off = merge_end
        if len(P2) > 0:
            lchunk_size = 10000000 // len(P2)
            chunks = []
            for fstart in range(0, len(L), lchunk_size):
                fend = min(len(L), fstart + lchunk_size)
                split_p2idx, split_fidx = [
                    _.flatten() for _ in np.mgrid[0 : len(P2), fstart:fend]
                ]
                z = (F[split_fidx, IIDX] - P2[split_p2idx, IIDX]).astype(np.int32)
                mask = (z <= max_frame_difference) & (z > 0)
                if np.sum(mask) > 0:
                    chunks.append([_[mask] for _ in (split_p2idx, split_fidx, z)])
            if len(chunks) > 0:
                split_p2idx, split_fidx, z = [
                    np.hstack([_[i] for _ in chunks]) for i in range(3)
                ]
            else:
                split_p2idx = split_fidx = z = np.zeros(0, np.int32)
        else:
            split_p2idx = split_fidx = z = np.zeros(0, int)

        if len(z) > 0:
            AreaFirst = F[split_fidx, AIDX]
            AreaAfterSplit = P[P2[split_p2idx, PIDX].astype(int) + 1, AIDX]
            AreaAtSplit = P2[split_p2idx, AIDX]
            d = np.sqrt(np.sum((F[split_fidx, :2] - P2[split_p2idx, :2]) ** 2, 1))
            rho = self.calculate_area_penalty(AreaFirst + AreaAfterSplit, AreaAtSplit)
            split_scores = d * rho
            mask = split_scores <= max_split_score
            split_p2idx, split_fidx, split_scores = [
                _[mask] for _ in (split_p2idx, split_fidx, split_scores)
            ]
            split_len = np.sum(mask)
            if split_len > 0:
                #
                # The end nodes are the intermediates (starting at split_off)
                # The start nodes are the F
                #
                end_nodes.append(np.arange(split_len) + split_off)
                start_nodes.append(split_fidx)
                scores.append(split_scores)
                #
                # Hook the alternate ends to the split starts
                #
                end_nodes.append(split_fidx + gap_off)
                start_nodes.append(np.arange(split_len) + split_off)
                scores.append(np.ones(split_len) * gap_cost / 2)
                #
                # The alternate hypothesis is split nodes hooked to themselves
                #
                end_nodes.append(np.arange(split_len) + split_off)
                start_nodes.append(np.arange(split_len) + split_off)
                scores.append(np.ones(split_len) * split_alternative_cost)
        else:
            split_len = 0
        split_end = split_off + split_len

        # ----------------------------------------------------------
        #
        # Mitosis model
        #
        # ----------------------------------------------------------

        mitoses, mitosis_scores = self.get_mitotic_triple_scores(F, L)
        n_mitoses = len(mitosis_scores)
        if n_mitoses > 0:
            order = np.argsort(mitosis_scores)
            mitoses, mitosis_scores = mitoses[order], mitosis_scores[order]
        MDLIDX = 0  # index of left daughter
        MDRIDX = 1  # index of right daughter
        MPIDX = 2  # index of parent
        mitoses_parent_lidx = mitoses[:, MPIDX]
        mitoses_left_child_findx = mitoses[:, MDLIDX]
        mitoses_right_child_findx = mitoses[:, MDRIDX]
        #
        # Create the ranges for mitoses
        #
        mitosis_off = split_end
        mitosis_len = n_mitoses
        mitosis_end = mitosis_off + mitosis_len
        if n_mitoses > 0:
            #
            # Taking the mitosis score will cost us the parent gap at least.
            #
            end_nodes.append(mitoses_parent_lidx)
            start_nodes.append(np.arange(n_mitoses) + mitosis_off)
            scores.append(mitosis_scores)
            #
            # Balance the mitosis against the gap alternative.
            #
            end_nodes.append(np.arange(n_mitoses) + mitosis_off)
            start_nodes.append(mitoses_parent_lidx + gap_off)
            scores.append(np.ones(n_mitoses) * gap_cost / 2)
            #
            # The alternative hypothesis links mitosis to mitosis
            # We charge the alternative hypothesis the mitosis_alternative
            # cost.
            #
            end_nodes.append(np.arange(n_mitoses) + mitosis_off)
            start_nodes.append(np.arange(n_mitoses) + mitosis_off)
            scores.append(np.ones(n_mitoses) * mitosis_alternative_cost)

        i = np.hstack(end_nodes)
        j = np.hstack(start_nodes)
        c = scores = np.hstack(scores)
        # -------------------------------------------------------
        #
        #      LAP Processing # 1
        #
        x, y = lapjv(i, j, c)
        score_matrix = scipy.sparse.coo.coo_matrix((c, (i, j))).tocsr()

        # ---------------------------
        #
        # Useful debugging diagnostics
        #
        def desc(node):
            """Describe a node for graphviz"""
            fl = F
            if node < start_end_end:
                fmt = "N%d:%d"
                idx = node
            elif node < gap_end:
                fmt = "G%d:%d"
                idx = node - gap_off
            elif node < merge_end:
                fmt = "M%d:%d"
                idx = merge_p1idx[node - merge_off]
                fl = P1
            elif node < split_end:
                fmt = "S%d:%d"
                idx = split_p2idx[node - split_off]
                fl = P2
            else:
                mitosis = mitoses[node - mitosis_off]
                (lin, lon), (rin, ron), (pin, pon) = [
                    (image_numbers[fl[idx, IIDX]], fl[idx, ONIDX])
                    for idx, fl in zip(mitosis, (F, F, L))
                ]
                return 'n%d[label="MIT%d:%d->%d:%d+%d:%d"]' % (
                    node,
                    pin,
                    pon,
                    lin,
                    lon,
                    rin,
                    ron,
                )
            return 'n%d[label="%s"]' % (
                node,
                fmt % (image_numbers[int(fl[idx, IIDX])], int(fl[idx, ONIDX])),
            )

        def write_graph(path, x, y):
            """Write a graphviz DOT file"""
            with open(path, "w") as fd:
                fd.write("digraph trackobjects {\n")
                graph_idx = np.where(
                    (x != np.arange(len(x))) & (y != np.arange(len(y)))
                )[0]
                for idx in graph_idx:
                    fd.write(desc(idx) + ";\n")
                for idx in graph_idx:
                    fd.write(
                        "n%d -> n%d [label=%0.2f];\n"
                        % (idx, x[idx], score_matrix[idx, x[idx]])
                    )
                fd.write("}\n")

        #
        # --------------------------------------------------------
        #
        # Mitosis fixup.
        #
        good_mitoses = np.zeros(len(mitoses), bool)
        for midx, (lidx, ridx, pidx) in enumerate(mitoses):
            #
            # If the parent was not accepted or either of the children
            # have been assigned to a mitosis, skip
            #
            if x[pidx] == midx + mitosis_off and not any(
                [mitosis_off <= y[idx] < mitosis_end for idx in (lidx, ridx)]
            ):
                alt_score = sum([score_matrix[y[idx], idx] for idx in (lidx, ridx)])
                #
                # Taking the alt score would cost us a mitosis alternative
                # cost, but would remove half of a gap alternative.
                #
                alt_score += mitosis_alternative_cost - gap_cost / 2
                #
                # Alternatively, taking the mitosis score would cost us
                # the gap alternatives of the left and right.
                #
                if alt_score > mitosis_scores[midx] + gap_cost:
                    for idx in lidx, ridx:
                        old_y = y[idx]
                        if old_y < start_end_end:
                            x[old_y] = old_y + gap_off
                        else:
                            x[old_y] = old_y
                    y[lidx] = midx + mitosis_off
                    y[ridx] = midx + mitosis_off
                    good_mitoses[midx] = True
                    continue
            x[pidx] = pidx + gap_off
            y[pidx + gap_off] = pidx
            x[midx + mitosis_off] = midx + mitosis_off
            y[midx + mitosis_off] = midx + mitosis_off
        if np.sum(good_mitoses) == 0:
            good_mitoses = np.zeros((0, 3), int)
            good_mitosis_scores = np.zeros(0)
        else:
            good_mitoses, good_mitosis_scores = (
                mitoses[good_mitoses],
                mitosis_scores[good_mitoses],
            )
        #
        # -------------------------------------
        #
        # Rerun to see if reverted mitoses could close gaps.
        #
        if np.any(x[mitoses[:, MPIDX]] != np.arange(len(mitoses)) + mitosis_off):
            rerun_end = np.ones(mitosis_end, bool)
            rerun_start = np.ones(mitosis_end, bool)
            rerun_end[:start_end_end] = x[:start_end_end] < mitosis_off
            rerun_end[mitosis_off:] = False
            rerun_start[:start_end_end] = y[:start_end_end] < mitosis_off
            rerun_start[mitosis_off:] = False
            mask = rerun_end[i] & rerun_start[j]
            i, j, c = i[mask], j[mask], c[mask]
            i = np.hstack(
                (
                    i,
                    good_mitoses[:, MPIDX],
                    good_mitoses[:, MDLIDX] + gap_off,
                    good_mitoses[:, MDRIDX] + gap_off,
                )
            )
            j = np.hstack(
                (
                    j,
                    good_mitoses[:, MPIDX] + gap_off,
                    good_mitoses[:, MDLIDX],
                    good_mitoses[:, MDRIDX],
                )
            )
            c = np.hstack((c, np.zeros(len(good_mitoses) * 3)))
            x, y = lapjv(i, j, c)
        #
        # Fixups to measurements
        #
        # fixup[N] gets the fixup dictionary for image set, N
        #
        # fixup[N][FEATURE] gets a tuple of a list of object numbers and
        #                   values.
        #
        fixups = {}

        def add_fixup(feature, image_number, object_number, value):
            if image_number not in fixups:
                fixups[image_number] = {feature: ([object_number], [value])}
            else:
                fid = fixups[image_number]
                if feature not in fid:
                    fid[feature] = ([object_number], [value])
                else:
                    object_numbers, values = fid[feature]
                    object_numbers.append(object_number)
                    values.append(value)

        # attaches different segments together if they are matches through the IAP
        a = -np.ones(len(F) + 1, dtype="int32")
        b = -np.ones(len(F) + 1, dtype="int32")
        c = -np.ones(len(F) + 1, dtype="int32")
        d = -np.ones(len(F) + 1, dtype="int32")
        z = np.zeros(len(F) + 1, dtype="int32")

        # relationships is a list of parent-child relationships. Each element
        # is a two-tuple of parent and child and each parent/child is a
        # two-tuple of image index and object number:
        #
        # [((<parent-image-index>, <parent-object-number>),
        #   (<child-image-index>, <child-object-number>))...]
        #
        relationships = []
        #
        # Starts can be linked to the following:
        #    ends             (start_end_off <= j < start_end_off+start_end_len)
        #    gap alternatives (gap_off <= j < merge_off+merge_len)
        #    splits           (split_off <= j < split_off+split_len)
        #    mitosis left     (mitosis_left_child_off <= j < ....)
        #    mitosis right    (mitosis_right_child_off <= j < ....)
        #
        # Discard starts linked to self = "do nothing"
        #
        start_idxs = np.where(y[:start_end_end] != np.arange(gap_off, gap_end))[0]
        for i in start_idxs:
            my_image_index = int(F[i, IIDX])
            my_image_number = image_numbers[my_image_index]
            my_object_index = int(F[i, OIIDX])
            my_object_number = int(F[i, ONIDX])
            yi = y[i]
            if yi < gap_end:
                # -------------------------------
                #
                #     GAP
                #
                # y[i] gives index of last hooked to first
                #
                b[i + 1] = yi + 1
                c[yi + 1] = i + 1
                #
                # Hook our parent image/object number to found parent
                #
                parent_image_index = int(L[yi, IIDX])
                parent_object_number = int(L[yi, ONIDX])
                parent_image_number = image_numbers[parent_image_index]
                parent_image_numbers[my_image_index][
                    my_object_index
                ] = parent_image_number
                parent_object_numbers[my_image_index][
                    my_object_index
                ] = parent_object_number
                relationships.append(
                    (
                        (parent_image_index, parent_object_number),
                        (my_image_index, my_object_number),
                    )
                )
                add_fixup(F_LINK_TYPE, my_image_number, my_object_number, LT_GAP)
                add_fixup(
                    F_GAP_LENGTH,
                    my_image_number,
                    my_object_number,
                    my_image_index - parent_image_index,
                )
                add_fixup(F_GAP_SCORE, my_image_number, my_object_number, scores[yi])
                #
                # One less new object
                #
                new_object_count[my_image_index] -= 1
                #
                # One less lost object (the lost object is recorded in
                # the image set after the parent)
                #
                lost_object_count[parent_image_index + 1] -= 1
                LOGGER.debug(
                    "Gap closing: %d:%d to %d:%d, score=%f"
                    % (
                        parent_image_number,
                        parent_object_number,
                        image_numbers[my_image_index],
                        object_numbers[my_image_index][my_object_index],
                        score_matrix[yi, i],
                    )
                )
            elif split_off <= yi < split_end:
                # ------------------------------------
                #
                #     SPLIT
                #
                p2_idx = split_p2idx[yi - split_off]
                parent_image_index = int(P2[p2_idx, IIDX])
                parent_image_number = image_numbers[parent_image_index]
                parent_object_number = int(P2[p2_idx, ONIDX])
                b[i + 1] = P2[p2_idx, LIDX]
                c[b[i + 1]] = i + 1
                parent_image_numbers[my_image_index][
                    my_object_index
                ] = parent_image_number
                parent_object_numbers[my_image_index][
                    my_object_index
                ] = parent_object_number
                relationships.append(
                    (
                        (parent_image_index, parent_object_number),
                        (my_image_index, my_object_number),
                    )
                )
                add_fixup(F_LINK_TYPE, my_image_number, my_object_number, LT_SPLIT)
                add_fixup(
                    F_SPLIT_SCORE,
                    my_image_number,
                    my_object_number,
                    split_scores[yi - split_off],
                )
                #
                # one less new object
                #
                new_object_count[my_image_index] -= 1
                #
                # one more split object
                #
                split_count[my_image_index] += 1
                LOGGER.debug(
                    "split: %d:%d to %d:%d, score=%f"
                    % (
                        parent_image_number,
                        parent_object_number,
                        image_numbers[my_image_index],
                        object_numbers[my_image_index][my_object_index],
                        split_scores[y[i] - split_off],
                    )
                )
        # ---------------------
        #
        # Process ends (parents)
        #
        end_idxs = np.where(x[:start_end_end] != np.arange(gap_off, gap_end))[0]
        for i in end_idxs:
            if x[i] < start_end_end:
                a[i + 1] = x[i] + 1
                d[a[i + 1]] = i + 1
            elif merge_off <= x[i] < merge_end:
                # -------------------
                #
                #    MERGE
                #
                # Handle merged objects. A merge hooks the end (L) of
                # a segment (the parent) to a gap alternative in P1 (the child)
                #
                p1_idx = merge_p1idx[x[i] - merge_off]
                a[i + 1] = P1[p1_idx, LIDX]
                d[a[i + 1]] = i + 1
                parent_image_index = int(L[i, IIDX])
                parent_object_number = int(L[i, ONIDX])
                parent_image_number = image_numbers[parent_image_index]
                child_image_index = int(P1[p1_idx, IIDX])
                child_object_number = int(P1[p1_idx, ONIDX])
                relationships.append(
                    (
                        (parent_image_index, parent_object_number),
                        (child_image_index, child_object_number),
                    )
                )
                add_fixup(
                    F_MERGE_SCORE,
                    parent_image_number,
                    parent_object_number,
                    merge_scores[x[i] - merge_off],
                )
                lost_object_count[parent_image_index + 1] -= 1
                merge_count[child_image_index] += 1
                LOGGER.debug(
                    "Merge: %d:%d to %d:%d, score=%f"
                    % (
                        image_numbers[parent_image_index],
                        parent_object_number,
                        image_numbers[child_image_index],
                        child_object_number,
                        merge_scores[x[i] - merge_off],
                    )
                )

        for (mlidx, mridx, mpidx), score in zip(good_mitoses, good_mitosis_scores):
            #
            # The parent is attached, one less lost object
            #
            lost_object_count[int(L[mpidx, IIDX]) + 1] -= 1
            a[mpidx + 1] = F[mlidx, LIDX]
            d[a[mpidx + 1]] = mpidx + 1
            parent_image_index = int(L[mpidx, IIDX])
            parent_image_number = image_numbers[parent_image_index]
            parent_object_number = int(L[mpidx, ONIDX])
            split_count[int(F[lidx, IIDX])] += 1
            for idx in mlidx, mridx:
                # --------------------------------------
                #
                #     MITOSIS child
                #
                my_image_index = int(F[idx, IIDX])
                my_image_number = image_numbers[my_image_index]
                my_object_index = int(F[idx, OIIDX])
                my_object_number = int(F[idx, ONIDX])

                b[idx + 1] = int(L[mpidx, LIDX])
                c[b[idx + 1]] = idx + 1
                parent_image_numbers[my_image_index][
                    my_object_index
                ] = parent_image_number
                parent_object_numbers[my_image_index][
                    my_object_index
                ] = parent_object_number
                relationships.append(
                    (
                        (parent_image_index, parent_object_number),
                        (my_image_index, my_object_number),
                    )
                )
                add_fixup(F_LINK_TYPE, my_image_number, my_object_number, LT_MITOSIS)
                add_fixup(F_MITOSIS_SCORE, my_image_number, my_object_number, score)
                new_object_count[my_image_index] -= 1
            LOGGER.debug(
                "Mitosis: %d:%d to %d:%d and %d, score=%f"
                % (
                    parent_image_number,
                    parent_object_number,
                    image_numbers[int(F[int(mlidx), int(IIDX)])],
                    F[mlidx, ONIDX],
                    F[mridx, ONIDX],
                    score,
                )
            )
        #
        # At this point a gives the label # of the track that connects
        # to the end of the indexed track. b gives the label # of the
        # track that connects to the start of the indexed track.
        # We convert these into edges.
        #
        # aa and bb are the vertices of an edge list and aa[i],bb[i]
        # make up an edge
        #
        connect_mask = a != -1
        aa = a[connect_mask]
        bb = np.argwhere(connect_mask).flatten()
        connect_mask = b != -1
        aa = np.hstack((aa, b[connect_mask]))
        bb = np.hstack((bb, np.argwhere(connect_mask).flatten()))
        #
        # Connect self to self for indices that do not connect
        #
        disconnect_mask = (a == -1) & (b == -1)
        aa = np.hstack((aa, np.argwhere(disconnect_mask).flatten()))
        bb = np.hstack((bb, np.argwhere(disconnect_mask).flatten()))
        z = all_connected_components(aa, bb)
        newlabel = [z[label[i]] for i in range(len(label))]
        #
        # Replace the labels for the image sets in the group
        # inside the list retrieved from the measurements
        #
        m_link_type = self.measurement_name(F_LINK_TYPE)
        for i, image_number in enumerate(image_numbers):
            n_objects = len(newlabel[i])
            m.add_measurement(
                "Image",
                self.image_measurement_name(F_LOST_OBJECT_COUNT),
                lost_object_count[i],
                image_set_number=image_number,
            )
            m.add_measurement(
                "Image",
                self.image_measurement_name(F_NEW_OBJECT_COUNT),
                new_object_count[i],
                image_set_number=image_number,
            )
            m.add_measurement(
                "Image",
                self.image_measurement_name(F_MERGE_COUNT),
                merge_count[i],
                image_set_number=image_number,
            )
            m.add_measurement(
                "Image",
                self.image_measurement_name(F_SPLIT_COUNT),
                split_count[i],
                image_set_number=image_number,
            )
            if n_objects == 0:
                continue
            m.add_measurement(
                object_name,
                self.measurement_name(F_LABEL),
                newlabel[i],
                image_set_number=image_number,
            )
            m.add_measurement(
                object_name,
                self.measurement_name(F_PARENT_IMAGE_NUMBER),
                parent_image_numbers[i],
                image_set_number=image_number,
            )
            m.add_measurement(
                object_name,
                self.measurement_name(F_PARENT_OBJECT_NUMBER),
                parent_object_numbers[i],
                image_set_number=image_number,
            )
            is_fixups = fixups.get(image_number, None)
            if (is_fixups is not None) and (F_LINK_TYPE in is_fixups):
                link_types = m[object_name, m_link_type, image_number]
                object_numbers, values = [np.array(_) for _ in is_fixups[F_LINK_TYPE]]
                link_types[object_numbers - 1] = values
                m[object_name, m_link_type, image_number] = link_types
            for feature, data_type in (
                (F_GAP_LENGTH, np.int32),
                (F_GAP_SCORE, np.float32),
                (F_MERGE_SCORE, np.float32),
                (F_SPLIT_SCORE, np.float32),
                (F_MITOSIS_SCORE, np.float32),
            ):
                if data_type == np.int32:
                    values = np.zeros(n_objects, data_type)
                else:
                    values = np.ones(n_objects, data_type) * np.NaN
                if (is_fixups is not None) and (feature in is_fixups):
                    object_numbers, fixup_values = [
                        np.array(_) for _ in is_fixups[feature]
                    ]
                    values[object_numbers - 1] = fixup_values.astype(data_type)
                m[object_name, self.measurement_name(feature), image_number] = values
        #
        # Write the relationships.
        #
        if len(relationships) > 0:
            relationships = np.array(relationships)
            parent_image_numbers = image_numbers[relationships[:, 0, 0]]
            child_image_numbers = image_numbers[relationships[:, 1, 0]]
            parent_object_numbers = relationships[:, 0, 1]
            child_object_numbers = relationships[:, 1, 1]
            m.add_relate_measurement(
                self.module_num,
                R_PARENT,
                object_name,
                object_name,
                parent_image_numbers,
                parent_object_numbers,
                child_image_numbers,
                child_object_numbers,
            )

        self.recalculate_group(workspace, image_numbers)

    def calculate_area_penalty(self, a1, a2):
        """Calculate a penalty for areas that don't match

        Ideally, area should be conserved while tracking. We divide the larger
        of the two by the smaller of the two to get the area penalty
        which is then multiplied by the distance.

        Note that this differs from Jaqaman eqn 5 which has an asymmetric
        penalty (sqrt((a1 + a2) / b) for a1+a2 > b and b / (a1 + a2) for
        a1+a2 < b. I can't think of a good reason why they should be
        asymmetric.
        """
        result = a1 / a2
        result[result < 1] = 1 / result[result < 1]
        result[np.isnan(result)] = np.inf
        return result

    def get_gap_pair_scores(self, F, L, max_gap):
        """Compute scores for matching last frame with first to close gaps

        F - an N x 3 (or more) array giving X, Y and frame # of the first object
            in each track

        L - an N x 3 (or more) array giving X, Y and frame # of the last object
            in each track

        max_gap - the maximum allowed # of frames between the last and first

        Returns: an M x 2 array of M pairs where the first element of the array
                 is the index of the track whose last frame is to be joined to
                 the track whose index is the second element of the array.

                 an M-element vector of scores.
        """
        #
        # There have to be at least two things to match
        #
        nothing = (np.zeros((0, 2), int), np.zeros(0))

        if F.shape[0] <= 1:
            return nothing

        X = 0
        Y = 1
        IIDX = 2
        AIDX = 6

        #
        # Create an indexing ordered by the last frame index and by the first
        #
        i = np.arange(len(F))
        j = np.arange(len(F))
        f_iidx = F[:, IIDX].astype(int)
        l_iidx = L[:, IIDX].astype(int)

        i_lorder = np.lexsort((i, l_iidx))
        j_forder = np.lexsort((j, f_iidx))
        i = i[i_lorder]
        j = j[j_forder]
        i_counts = np.bincount(l_iidx)
        j_counts = np.bincount(f_iidx)
        i_indexes = Indexes([i_counts])
        j_indexes = Indexes([j_counts])
        #
        # The lowest possible F for each L is 1+L
        #
        j_self = np.minimum(np.arange(len(i_counts)), len(j_counts) - 1)
        j_first_idx = j_indexes.fwd_idx[j_self] + j_counts[j_self]
        #
        # The highest possible F for each L is L + max_gap. j_end is the
        # first illegal value... just past that.
        #
        j_last = np.minimum(np.arange(len(i_counts)) + max_gap, len(j_counts) - 1)
        j_end_idx = j_indexes.fwd_idx[j_last] + j_counts[j_last]
        #
        # Structure the i and j block ranges
        #
        ij_counts = j_end_idx - j_first_idx
        ij_indexes = Indexes([i_counts, ij_counts])
        if ij_indexes.length == 0:
            return nothing
        #
        # The index into L of the first element of the pair
        #
        ai = i[i_indexes.fwd_idx[ij_indexes.rev_idx] + ij_indexes.idx[0]]
        #
        # The index into F of the second element of the pair
        #
        aj = j[j_first_idx[ij_indexes.rev_idx] + ij_indexes.idx[1]]
        #
        # The distances
        #
        d = np.sqrt((L[ai, X] - F[aj, X]) ** 2 + (L[ai, Y] - F[aj, Y]) ** 2)
        #
        # Rho... the area penalty
        #
        rho = self.calculate_area_penalty(L[ai, AIDX], F[aj, AIDX])
        return np.column_stack((ai, aj)), d * rho

    def get_mitotic_triple_scores(self, F, L):
        """Compute scores for matching a parent to two daughters

        F - an N x 3 (or more) array giving X, Y and frame # of the first object
            in each track

        L - an N x 3 (or more) array giving X, Y and frame # of the last object
            in each track

        Returns: an M x 3 array of M triples where the first column is the
                 index in the L array of the parent cell and the remaining
                 columns are the indices of the daughters in the F array

                 an M-element vector of distances of the parent from the expected
        """
        X = 0
        Y = 1
        IIDX = 2
        AIDX = 6

        if len(F) <= 1:
            return np.zeros((0, 3), np.int32), np.zeros(0, np.int32)

        max_distance = self.mitosis_max_distance.value

        # Find all daughter pairs within same frame
        i, j = np.where(F[:, np.newaxis, IIDX] == F[np.newaxis, :, IIDX])
        i, j = i[i < j], j[i < j]  # get rid of duplicates and self-compares

        #
        # Calculate the maximum allowed distance before one or the other
        # daughter is farther away than the maximum allowed from the center
        #
        # That's the max_distance * 2 minus the distance
        #
        dmax = max_distance * 2 - np.sqrt(np.sum((F[i, :2] - F[j, :2]) ** 2, 1))
        mask = dmax >= 0
        i, j = i[mask], j[mask]
        if len(i) == 0:
            return np.zeros((0, 3), np.int32), np.zeros(0, np.int32)
        center_x = (F[i, X] + F[j, X]) / 2
        center_y = (F[i, Y] + F[j, Y]) / 2
        frame = F[i, IIDX]

        # Find all parent-daughter pairs where the parent
        # is in the frame previous to the daughters
        ij, k = [_.flatten() for _ in np.mgrid[0 : len(i), 0 : len(L)]]
        mask = F[i[ij], IIDX] == L[k, IIDX] + 1
        ij, k = ij[mask], k[mask]
        if len(ij) == 0:
            return np.zeros((0, 3), np.int32), np.zeros(0, np.int32)

        d = np.sqrt((center_x[ij] - L[k, X]) ** 2 + (center_y[ij] - L[k, Y]) ** 2)
        mask = d <= dmax[ij]
        ij, k, d = ij[mask], k[mask], d[mask]
        if len(ij) == 0:
            return np.zeros((0, 3), np.int32), np.zeros(0, np.int32)

        rho = self.calculate_area_penalty(F[i[ij], AIDX] + F[j[ij], AIDX], L[k, AIDX])
        return np.column_stack((i[ij], j[ij], k)), d * rho

    def recalculate_group(self, workspace, image_numbers):
        """Recalculate all measurements once post_group has run

        workspace - the workspace being operated on
        image_numbers - the image numbers of the group's image sets' measurements
        """
        m = workspace.measurements
        object_name = self.object_name.value

        assert isinstance(m, Measurements)

        image_index = np.zeros(np.max(image_numbers) + 1, int)
        image_index[image_numbers] = np.arange(len(image_numbers))
        image_index[0] = -1
        index_to_imgnum = np.array(image_numbers)

        parent_image_numbers, parent_object_numbers = [
            [
                m.get_measurement(
                    object_name, self.measurement_name(feature), image_number
                )
                for image_number in image_numbers
            ]
            for feature in (F_PARENT_IMAGE_NUMBER, F_PARENT_OBJECT_NUMBER)
        ]

        #
        # Do all_connected_components on the graph of parents to find groups
        # that share the same ancestor
        #
        count = np.array([len(x) for x in parent_image_numbers])
        idx = Indexes(count)
        if idx.length == 0:
            # Nothing to do
            return
        parent_image_numbers = np.hstack(parent_image_numbers).astype(int)
        parent_object_numbers = np.hstack(parent_object_numbers).astype(int)
        parent_image_indexes = image_index[parent_image_numbers]
        parent_object_indexes = parent_object_numbers - 1
        i = np.arange(idx.length)
        i = i[parent_image_numbers != 0]
        j = idx.fwd_idx[parent_image_indexes[i]] + parent_object_indexes[i]
        # Link self to self too
        i = np.hstack((i, np.arange(idx.length)))
        j = np.hstack((j, np.arange(idx.length)))
        labels = all_connected_components(i, j)
        nlabels = np.max(labels) + 1
        #
        # Set the ancestral index for each label
        #
        ancestral_index = np.zeros(nlabels, int)
        ancestral_index[labels[parent_image_numbers == 0]] = (
            np.argwhere(parent_image_numbers == 0).flatten().astype(int)
        )
        ancestral_image_index = idx.rev_idx[ancestral_index]
        ancestral_object_index = ancestral_index - idx.fwd_idx[ancestral_image_index]
        #
        # Blow these up to one per object for convenience
        #
        ancestral_index = ancestral_index[labels]
        ancestral_image_index = ancestral_image_index[labels]
        ancestral_object_index = ancestral_object_index[labels]

        def start(image_index):
            """Return the start index in the array for the given image index"""
            return idx.fwd_idx[image_index]

        def end(image_index):
            """Return the end index in the array for the given image index"""
            return start(image_index) + idx.counts[0][image_index]

        def slyce(image_index):
            return slice(start(image_index), end(image_index))

        class wrapped(object):
            """make an indexable version of a measurement, with parent and ancestor fetching"""

            def __init__(self, feature_name):
                self.feature_name = feature_name
                self.backing_store = np.hstack(
                    [
                        m.get_measurement(object_name, feature_name, i)
                        for i in image_numbers
                    ]
                )

            def __getitem__(self, index):
                return self.backing_store[slyce(index)]

            def __setitem__(self, index, val):
                self.backing_store[slyce(index)] = val
                m.add_measurement(
                    object_name,
                    self.feature_name,
                    val,
                    image_set_number=image_numbers[index],
                )

            def get_parent(self, index, no_parent=None):
                result = np.zeros(idx.counts[0][index], self.backing_store.dtype)
                my_slice = slyce(index)
                mask = parent_image_numbers[my_slice] != 0
                if not np.all(mask):
                    if np.isscalar(no_parent) or (no_parent is None):
                        result[~mask] = no_parent
                    else:
                        result[~mask] = no_parent[~mask]
                if np.any(mask):
                    result[mask] = self.backing_store[
                        idx.fwd_idx[parent_image_indexes[my_slice][mask]]
                        + parent_object_indexes[my_slice][mask]
                    ]
                return result

            def get_ancestor(self, index):
                return self.backing_store[ancestral_index[slyce(index)]]

        #
        # Recalculate the trajectories
        #
        x = wrapped(M_LOCATION_CENTER_X)
        y = wrapped(M_LOCATION_CENTER_Y)
        trajectory_x = wrapped(self.measurement_name(F_TRAJECTORY_X))
        trajectory_y = wrapped(self.measurement_name(F_TRAJECTORY_Y))
        integrated = wrapped(self.measurement_name(F_INTEGRATED_DISTANCE))
        dists = wrapped(self.measurement_name(F_DISTANCE_TRAVELED))
        displ = wrapped(self.measurement_name(F_DISPLACEMENT))
        linearity = wrapped(self.measurement_name(F_LINEARITY))
        lifetimes = wrapped(self.measurement_name(F_LIFETIME))
        label = wrapped(self.measurement_name(F_LABEL))
        final_age = wrapped(self.measurement_name(F_FINAL_AGE))

        age = {}  # Dictionary of per-label ages
        if self.wants_lifetime_filtering.value:
            minimum_lifetime = (
                self.min_lifetime.value
                if self.wants_minimum_lifetime.value
                else -np.Inf
            )
            maximum_lifetime = (
                self.max_lifetime.value if self.wants_maximum_lifetime.value else np.Inf
            )

        for image_number in image_numbers:
            index = image_index[image_number]
            this_x = x[index]
            if len(this_x) == 0:
                continue
            this_y = y[index]
            last_x = x.get_parent(index, no_parent=this_x)
            last_y = y.get_parent(index, no_parent=this_y)
            x_diff = this_x - last_x
            y_diff = this_y - last_y
            #
            # TrajectoryX,Y = X,Y distances traveled from step to step
            #
            trajectory_x[index] = x_diff
            trajectory_y[index] = y_diff
            #
            # DistanceTraveled = Distance traveled from step to step
            #
            dists[index] = np.sqrt(x_diff * x_diff + y_diff * y_diff)
            #
            # Integrated distance = accumulated distance for lineage
            #
            integrated[index] = integrated.get_parent(index, no_parent=0) + dists[index]
            #
            # Displacement = crow-fly distance from initial ancestor
            #
            x_tot_diff = this_x - x.get_ancestor(index)
            y_tot_diff = this_y - y.get_ancestor(index)
            tot_distance = np.sqrt(x_tot_diff * x_tot_diff + y_tot_diff * y_tot_diff)
            displ[index] = tot_distance
            #
            # Linearity = ratio of displacement and integrated
            # distance. NaN for new cells is ok.
            #
            linearity[index] = tot_distance / integrated[index]
            #
            # Add 1 to lifetimes / one for new
            #
            lifetimes[index] = lifetimes.get_parent(index, no_parent=0) + 1

            #
            # Age = overall lifetime of each label
            #
            for this_label, this_lifetime in zip(label[index], lifetimes[index]):
                age[this_label] = this_lifetime

        all_labels = list(age.keys())
        all_ages = list(age.values())
        if self.wants_lifetime_filtering.value:
            labels_to_filter = [
                k
                for k, v in list(age.items())
                if v <= minimum_lifetime or v >= maximum_lifetime
            ]
        for image_number in image_numbers:
            index = image_index[image_number]

            # Fill in final object ages
            this_label = label[index]
            this_lifetime = lifetimes[index]
            this_age = final_age[index]
            ind = np.array(all_labels).searchsorted(this_label)
            i = np.array(all_ages)[ind] == this_lifetime
            this_age[i] = this_lifetime[i]
            final_age[index] = this_age

            # Filter object ages below the minimum
            if self.wants_lifetime_filtering.value:
                if len(labels_to_filter) > 0:
                    this_label = label[index].astype(float)
                    this_label[np.in1d(this_label, np.array(labels_to_filter))] = np.NaN
                    label[index] = this_label
        m.add_experiment_measurement(F_EXPT_ORIG_NUMTRACKS, nlabels)
        if self.wants_lifetime_filtering.value:
            m.add_experiment_measurement(
                F_EXPT_FILT_NUMTRACKS, nlabels - len(labels_to_filter)
            )

    def map_objects(self, workspace, new_of_old, old_of_new, i, j):
        """Record the mapping of old to new objects and vice-versa

        workspace - workspace for current image set
        new_of_old - an array of the new labels for every old label
        old_of_new - an array of the old labels for every new label
        i, j - the coordinates for each new object.
        """
        m = workspace.measurements
        assert isinstance(m, Measurements)
        image_number = m.get_current_image_measurement(IMAGE_NUMBER)
        new_of_old = new_of_old.astype(int)
        old_of_new = old_of_new.astype(int)
        old_object_numbers = self.get_saved_object_numbers(workspace).astype(int)
        max_object_number = self.get_max_object_number(workspace)
        old_count = len(new_of_old)
        new_count = len(old_of_new)
        #
        # Record the new objects' parents
        #
        parents = old_of_new.copy()
        parents[parents != 0] = old_object_numbers[
            (old_of_new[parents != 0] - 1)
        ].astype(parents.dtype)
        self.add_measurement(workspace, F_PARENT_OBJECT_NUMBER, old_of_new)
        parent_image_numbers = np.zeros(len(old_of_new))
        parent_image_numbers[parents != 0] = image_number - 1
        self.add_measurement(workspace, F_PARENT_IMAGE_NUMBER, parent_image_numbers)
        #
        # Assign object IDs to the new objects
        #
        mapping = np.zeros(new_count, int)
        if old_count > 0 and new_count > 0:
            mapping[old_of_new != 0] = old_object_numbers[
                old_of_new[old_of_new != 0] - 1
            ]
            miss_count = np.sum(old_of_new == 0)
            lost_object_count = np.sum(new_of_old == 0)
        else:
            miss_count = new_count
            lost_object_count = old_count
        nunmapped = np.sum(mapping == 0)
        new_max_object_number = max_object_number + nunmapped
        mapping[mapping == 0] = np.arange(
            max_object_number + 1, new_max_object_number + 1
        )
        self.set_max_object_number(workspace, new_max_object_number)
        self.add_measurement(workspace, F_LABEL, mapping)
        self.set_saved_object_numbers(workspace, mapping)
        #
        # Compute distances and trajectories
        #
        diff_i = np.zeros(new_count)
        diff_j = np.zeros(new_count)
        distance = np.zeros(new_count)
        integrated_distance = np.zeros(new_count)
        displacement = np.zeros(new_count)
        linearity = np.ones(new_count)
        orig_i = i.copy()
        orig_j = j.copy()
        old_i, old_j = self.get_saved_coordinates(workspace)
        old_distance = self.get_saved_distances(workspace)
        old_orig_i, old_orig_j = self.get_orig_coordinates(workspace)
        has_old = old_of_new != 0
        if np.any(has_old):
            old_indexes = old_of_new[has_old] - 1
            orig_i[has_old] = old_orig_i[old_indexes]
            orig_j[has_old] = old_orig_j[old_indexes]
            diff_i[has_old] = i[has_old] - old_i[old_indexes]
            diff_j[has_old] = j[has_old] - old_j[old_indexes]
            distance[has_old] = np.sqrt(diff_i[has_old] ** 2 + diff_j[has_old] ** 2)
            integrated_distance[has_old] = old_distance[old_indexes] + distance[has_old]
            displacement[has_old] = np.sqrt(
                (i[has_old] - orig_i[has_old]) ** 2
                + (j[has_old] - orig_j[has_old]) ** 2
            )
            linearity[has_old] = displacement[has_old] / integrated_distance[has_old]
        self.add_measurement(workspace, F_TRAJECTORY_X, diff_j)
        self.add_measurement(workspace, F_TRAJECTORY_Y, diff_i)
        self.add_measurement(workspace, F_DISTANCE_TRAVELED, distance)
        self.add_measurement(workspace, F_DISPLACEMENT, displacement)
        self.add_measurement(workspace, F_INTEGRATED_DISTANCE, integrated_distance)
        self.add_measurement(workspace, F_LINEARITY, linearity)
        self.set_saved_distances(workspace, integrated_distance)
        self.set_orig_coordinates(workspace, (orig_i, orig_j))
        self.set_saved_coordinates(workspace, (i, j))
        #
        # Update the ages
        #
        age = np.ones(new_count, int)
        if np.any(has_old):
            old_age = self.get_saved_ages(workspace)
            age[has_old] = old_age[old_of_new[has_old] - 1] + 1
        self.add_measurement(workspace, F_LIFETIME, age)
        final_age = np.NaN * np.ones(
            new_count, float
        )  # Initialize to NaN; will re-calc later
        self.add_measurement(workspace, F_FINAL_AGE, final_age)
        self.set_saved_ages(workspace, age)
        self.set_saved_object_numbers(workspace, mapping)
        #
        # Add image measurements
        #
        self.add_image_measurement(workspace, F_NEW_OBJECT_COUNT, np.sum(parents == 0))
        self.add_image_measurement(workspace, F_LOST_OBJECT_COUNT, lost_object_count)
        #
        # Find parents with more than one child. These are the progenitors
        # for daughter cells.
        #
        if np.any(parents != 0):
            h = np.bincount(parents[parents != 0])
            split_count = np.sum(h > 1)
        else:
            split_count = 0
        self.add_image_measurement(workspace, F_SPLIT_COUNT, split_count)
        #
        # Find children with more than one parent. These are the merges
        #
        if np.any(new_of_old != 0):
            h = np.bincount(new_of_old[new_of_old != 0])
            merge_count = np.sum(h > 1)
        else:
            merge_count = 0
        self.add_image_measurement(workspace, F_MERGE_COUNT, merge_count)
        #########################################
        #
        # Compile the relationships between children and parents
        #
        #########################################
        last_object_numbers = np.arange(1, len(new_of_old) + 1)
        new_object_numbers = np.arange(1, len(old_of_new) + 1)
        r_parent_object_numbers = np.hstack(
            (old_of_new[old_of_new != 0], last_object_numbers[new_of_old != 0])
        )
        r_child_object_numbers = np.hstack(
            (new_object_numbers[parents != 0], new_of_old[new_of_old != 0])
        )
        if len(r_child_object_numbers) > 0:
            #
            # Find unique pairs
            #
            order = np.lexsort((r_child_object_numbers, r_parent_object_numbers))
            r_child_object_numbers = r_child_object_numbers[order]
            r_parent_object_numbers = r_parent_object_numbers[order]
            to_keep = np.hstack(
                (
                    [True],
                    (r_parent_object_numbers[1:] != r_parent_object_numbers[:-1])
                    | (r_child_object_numbers[1:] != r_child_object_numbers[:-1]),
                )
            )
            r_child_object_numbers = r_child_object_numbers[to_keep]
            r_parent_object_numbers = r_parent_object_numbers[to_keep]
            r_image_numbers = (
                np.ones(r_parent_object_numbers.shape[0], r_parent_object_numbers.dtype)
                * image_number
            )
            if len(r_child_object_numbers) > 0:
                m.add_relate_measurement(
                    self.module_num,
                    R_PARENT,
                    self.object_name.value,
                    self.object_name.value,
                    r_image_numbers - 1,
                    r_parent_object_numbers,
                    r_image_numbers,
                    r_child_object_numbers,
                )

    def recalculate_kalman_filters(self, workspace, image_numbers):
        """Rerun the kalman filters to improve the motion models"""
        m = workspace.measurements
        object_name = self.object_name.value
        object_number = m[object_name, OBJECT_NUMBER, image_numbers]

        # ########################
        #
        # Create an indexer that lets you do the following
        #
        # parent_x = x[idx.fwd_idx[image_number - fi] + object_number - 1]
        # parent_y = y[idx.fwd_idx[image_number - fi] + object_number - 1]
        #
        # #######################
        x = m[object_name, M_LOCATION_CENTER_X, image_numbers]
        fi = np.min(image_numbers)
        max_image = np.max(image_numbers) + 1
        counts = np.zeros(max_image - fi, int)
        counts[image_numbers - fi] = np.array([len(xx) for xx in x])
        idx = Indexes(counts)
        x = np.hstack(x)
        y = np.hstack(m[object_name, M_LOCATION_CENTER_Y, image_numbers])
        area = np.hstack(m[object_name, self.measurement_name(F_AREA), image_numbers])
        parent_image_number = np.hstack(
            m[object_name, self.measurement_name(F_PARENT_IMAGE_NUMBER), image_numbers]
        ).astype(int)
        parent_object_number = np.hstack(
            m[object_name, self.measurement_name(F_PARENT_OBJECT_NUMBER), image_numbers]
        ).astype(int)
        link_type = np.hstack(
            m[object_name, self.measurement_name(F_LINK_TYPE), image_numbers]
        )
        link_distance = np.hstack(
            m[object_name, self.measurement_name(F_LINKING_DISTANCE), image_numbers]
        )
        movement_model = np.hstack(
            m[object_name, self.measurement_name(F_MOVEMENT_MODEL), image_numbers]
        )

        models = self.get_kalman_models()
        kalman_models = [
            centrosome.filter.static_kalman_model()
            if model == F_STATIC_MODEL
            else centrosome.filter.velocity_kalman_model()
            for model, elements in models
        ]
        kalman_states = [
            centrosome.filter.KalmanState(
                kalman_model.observation_matrix, kalman_model.translation_matrix
            )
            for kalman_model in kalman_models
        ]
        #
        # Initialize the last image set's states using no information
        #
        # TO_DO - use the kalman state information in the measurements
        #         to construct the kalman models that will best predict
        #         the penultimate image set.
        #
        n_objects = counts[-1]
        if n_objects > 0:
            this_slice = slice(idx.fwd_idx[-1], idx.fwd_idx[-1] + n_objects)
            ii = y[this_slice]
            jj = x[this_slice]
            new_kalman_states = []
            r = np.column_stack(
                (
                    area[this_slice].astype(float) / np.pi,
                    np.zeros(n_objects),
                    np.zeros(n_objects),
                    area[this_slice].astype(float),
                )
            ).reshape(n_objects, 2, 2)
            for kalman_state in kalman_states:
                new_kalman_states.append(
                    centrosome.filter.kalman_filter(
                        kalman_state,
                        -np.ones(n_objects, int),
                        np.column_stack((ii, jj)),
                        np.zeros(n_objects),
                        r,
                    )
                )
            kalman_states = new_kalman_states
        else:
            this_slice = slice(idx.fwd_idx[-1], idx.fwd_idx[-1])
        #
        # Update the kalman states and take any new linkage distances
        # and movement models that are better
        #
        for image_number in reversed(sorted(image_numbers)[:-1]):
            i = image_number - fi
            n_objects = counts[i]
            child_object_number = np.zeros(n_objects, int)
            next_slice = this_slice
            this_slice = slice(idx.fwd_idx[i], idx.fwd_idx[i] + counts[i])
            next_links = link_type[next_slice]
            next_has_link = next_links == LT_PHASE_1
            if any(next_has_link):
                next_parents = parent_object_number[next_slice]
                next_object_number = np.arange(counts[i + 1]) + 1
                child_object_number[
                    next_parents[next_has_link] - 1
                ] = next_object_number[next_has_link]
            has_child = child_object_number != 0
            if np.any(has_child):
                kid_idx = child_object_number[has_child] - 1
            ii = y[this_slice]
            jj = x[this_slice]
            r = np.column_stack(
                (
                    area[this_slice].astype(float) / np.pi,
                    np.zeros(n_objects),
                    np.zeros(n_objects),
                    area[this_slice].astype(float),
                )
            ).reshape(n_objects, 2, 2)
            new_kalman_states = []
            errors = link_distance[next_slice]
            model_used = movement_model[next_slice]
            for (model, elements), kalman_state in zip(models, kalman_states):
                assert isinstance(kalman_state, centrosome.filter.KalmanState)
                n_elements = len(elements)
                q = np.zeros((n_objects, n_elements, n_elements))
                if np.any(has_child):
                    obs = kalman_state.predicted_obs_vec
                    dk = np.sqrt(
                        (obs[kid_idx, 0] - ii[has_child]) ** 2
                        + (obs[kid_idx, 1] - jj[has_child]) ** 2
                    )
                    this_model = np.where(dk < errors[kid_idx])[0]
                    if len(this_model) > 0:
                        km_model = KM_NO_VEL if model == F_STATIC_MODEL else KM_VEL
                        model_used[kid_idx[this_model]] = km_model
                        errors[kid_idx[this_model]] = dk[this_model]

                    for j in range(n_elements):
                        q[has_child, j, j] = kalman_state.noise_var[kid_idx, j]
                updated_state = centrosome.filter.kalman_filter(
                    kalman_state,
                    child_object_number - 1,
                    np.column_stack((ii, jj)),
                    q,
                    r,
                )
                new_kalman_states.append(updated_state)
            if np.any(has_child):
                # fix child linking distances and models
                mname = self.measurement_name(F_LINKING_DISTANCE)
                m[object_name, mname, image_number + 1] = errors
                mname = self.measurement_name(F_MOVEMENT_MODEL)
                m[object_name, mname, image_number + 1] = model_used
            kalman_states = new_kalman_states

    def get_kalman_feature_names(self):
        if self.tracking_method != "LAP":
            return []
        return sum(
            [
                sum(
                    [
                        [
                            kalman_feature(model, F_STATE, element),
                            kalman_feature(model, F_NOISE, element),
                        ]
                        + [kalman_feature(model, F_COV, element, e2) for e2 in elements]
                        for element in elements
                    ],
                    [],
                )
                for model, elements in self.get_kalman_models()
            ],
            [],
        )

    def get_measurement_columns(self, pipeline):
        result = [
            (self.object_name.value, self.measurement_name(feature), coltype)
            for feature, coltype in F_ALL_COLTYPE_ALL
        ]
        result += [
            ("Image", self.image_measurement_name(feature), coltype)
            for feature, coltype in F_IMAGE_COLTYPE_ALL
        ]
        attributes = {MCA_AVAILABLE_POST_GROUP: True}
        if self.tracking_method == "LAP":
            result += [
                (self.object_name.value, self.measurement_name(name), coltype)
                for name, coltype in (
                    (F_AREA, COLTYPE_INTEGER),
                    (F_LINK_TYPE, COLTYPE_INTEGER),
                    (F_LINKING_DISTANCE, COLTYPE_FLOAT),
                    (F_STANDARD_DEVIATION, COLTYPE_FLOAT),
                    (F_MOVEMENT_MODEL, COLTYPE_INTEGER),
                )
            ]
            result += [
                (self.object_name.value, self.measurement_name(name), COLTYPE_FLOAT,)
                for name in list(self.get_kalman_feature_names())
            ]
            if self.wants_second_phase:
                result += [
                    (self.object_name.value, self.measurement_name(name), coltype)
                    for name, coltype in (
                        (F_GAP_LENGTH, COLTYPE_INTEGER),
                        (F_GAP_SCORE, COLTYPE_FLOAT),
                        (F_MERGE_SCORE, COLTYPE_FLOAT),
                        (F_SPLIT_SCORE, COLTYPE_FLOAT),
                        (F_MITOSIS_SCORE, COLTYPE_FLOAT),
                    )
                ]
                # Add the post-group attribute to all measurements
                result = [(c[0], c[1], c[2], attributes) for c in result]
            else:
                pg_meas = [
                    self.measurement_name(feature)
                    for feature in (F_LINKING_DISTANCE, F_MOVEMENT_MODEL)
                ]
                result = [
                    c if c[1] not in pg_meas else (c[0], c[1], c[2], attributes)
                    for c in result
                ]

        return result

    def get_object_relationships(self, pipeline):
        """Return the object relationships produced by this module"""
        object_name = self.object_name.value
        if self.wants_second_phase and self.tracking_method == "LAP":
            when = MCA_AVAILABLE_POST_GROUP
        else:
            when = MCA_AVAILABLE_EACH_CYCLE
        return [(R_PARENT, object_name, object_name, when)]

    def get_categories(self, pipeline, object_name):
        if object_name in (self.object_name.value, "Image"):
            return [F_PREFIX]
        elif object_name == EXPERIMENT:
            return [F_PREFIX]
        else:
            return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.object_name.value and category == F_PREFIX:
            result = list(F_ALL)
            if self.tracking_method == "LAP":
                result += [
                    F_AREA,
                    F_LINKING_DISTANCE,
                    F_STANDARD_DEVIATION,
                    F_LINK_TYPE,
                    F_MOVEMENT_MODEL,
                ]
                if self.wants_second_phase:
                    result += [
                        F_GAP_LENGTH,
                        F_GAP_SCORE,
                        F_MERGE_SCORE,
                        F_SPLIT_SCORE,
                        F_MITOSIS_SCORE,
                    ]
                result += self.get_kalman_feature_names()
            return result
        if object_name == "Image":
            result = F_IMAGE_ALL
            return result
        if object_name == EXPERIMENT and category == F_PREFIX:
            return [F_EXPT_ORIG_NUMTRACKS, F_EXPT_FILT_NUMTRACKS]
        return []

    def get_measurement_objects(self, pipeline, object_name, category, measurement):
        if (
            object_name == "Image"
            and category == F_PREFIX
            and measurement in F_IMAGE_ALL
        ):
            return [self.object_name.value]
        return []

    def get_measurement_scales(
        self, pipeline, object_name, category, feature, image_name
    ):
        if self.tracking_method == "LAP":
            return []

        if feature in self.get_measurements(pipeline, object_name, category):
            return [str(self.pixel_radius.value)]
        return []

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            setting_values = setting_values + ["100", "100"]
            variable_revision_number = 2
        if variable_revision_number == 2:
            # Added phase 2 parameters
            setting_values = setting_values + ["40", "40", "40", "50", "50", "50", "5"]
            variable_revision_number = 3
        if variable_revision_number == 3:
            # Added Kalman choices:
            # Model
            # radius std
            # radius limit
            setting_values = (
                setting_values[:7] + [M_BOTH, "3", "2,10"] + setting_values[9:]
            )
            variable_revision_number = 4

        if variable_revision_number == 4:
            # Added lifetime filtering: Wants filtering + min/max allowed lifetime
            setting_values = setting_values + ["No", "Yes", "1", "No", "100"]
            variable_revision_number = 5

        if variable_revision_number == 5:
            # Added mitosis alternative score + mitosis_max_distance
            setting_values = setting_values + ["80", "40"]
            variable_revision_number = 6

        # added after integration of FOLLOWNEIGHBORS
        if variable_revision_number == 6:
            # adding new settings for FOLLOWNEIGHBORS
            setting_values = setting_values + [30.0, False, 15.0, 25.0]
            # order of params in settings
            # self.average_cell_diameter, self.advanced_parameters,self.drop_cost, self.area_weight
            variable_revision_number = 7

        return setting_values, variable_revision_number
