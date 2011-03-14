"""<b>Track Objects</b> allows tracking objects throughout sequential 
frames of a series of images, so that from frame to frame
each object maintains a unique identity in the output measurements
<hr>

This module must be placed downstream of a module that identifies objects
(e.g., <b>IdentifyPrimaryObjects</b>). <b>TrackObjects</b> will associate each
object with the same object in the frames before and after. This allows the study 
of objects' lineages and the timing and characteristics of dynamic events in 
movies.

Images in CellProfiler are processed 
sequentially by frame (whether loaded as a series of images or a movie file). 
To process a collection of images/movies, you will need to 
group the input using grouping options in <b>LoadImages</b> to make sure 
that each image sequence is
handled individually. See the help in that module, and CellProfiler Help > General Help > Using MetaData in CellProfiler for more information. If you are only processing a single movie in each analysis 
run, you do not need to set up image grouping.


For an example pipeline using TrackObjects, see the CellProfiler <a href="http://www.cellprofiler.org/examples.htm">Examples</a> webpage.

<h4>Available measurements</h4>
<ul>
<li><i>Object features</i>
<ul>
<li><i>Label:</i> Each tracked object is assigned a unique identifier (label). 
Results of splits or merges are seen as new objects and assigned a new
label.</li>
<li><i>Parent:</i> The label of the object in the last frame. For a split, each
child object will have the label of the object it split from. For a merge,
the child will have the label of the closest parent.</li>
<li><i>TrajectoryX, TrajectoryY:</i> The direction of motion (in x and y coordinates) of the 
object from the previous frame to the current frame.</li>
<li><i>DistanceTraveled:</i> The distance traveled by the object from the 
previous frame to the current frame (calculated as the magnitude of 
the distance traveled vector).</li>
<li><i>IntegratedDistance:</i> The total distance traveled by the object during
the lifetime of the object.</li>
<li><i>Linearity:</i> A measure of how linear the object trajectity is during the
object lifetime. Calculated as (distance from initial to final 
location)/(integrated object distance). Value is in range of [0,1].</li>
<li><i>Lifetime:</i> The duration (in frames) of the object. The lifetime begins 
at the frame when an object appears and is output as a measurement when
the object disappears. At the final frame of the image set/movie, the 
lifetimes of all remaining objects are output.</li>
</ul>
</li>
<li><i>Image features</i>
<ul>
<li><i>LostObjectCount:</i> Number of objects that appear in the previous frame
but have no identifiable child in the current frame.</li>
<li><i>NewObjectCount:</i> Number of objects that appear in the current frame but
have no identifiable parent in the previous frame. </li>
<li><i>DaughterObjectCount:</i> Number of objects in the current frame that 
resulted from a split from a parent object in the previous frame.</li>
<li><i>MergedObjectCount:</i> Number of objects in the current frame that 
resulted from the merging of child objects in the previous frame.</li>
</ul>
</li>
</ul>

See also: Any of the <b>Measure</b> modules, <b>IdentifyPrimaryObjects</b>, <b>LoadImages</b>.
"""
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2011
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

__version__="$Revision$"

import numpy as np
import numpy.ma
import matplotlib.figure
import matplotlib.axes
import matplotlib.backends.backend_agg
from scipy.ndimage import distance_transform_edt
import scipy.ndimage
import scipy.sparse

import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas
import cellprofiler.preferences as cpprefs
import contrib.LAP as LAP
import cellprofiler.cpmath.filter as cpfilter
from cellprofiler.cpmath.cpmorphology import fixup_scipy_ndimage_result as fix
from cellprofiler.cpmath.cpmorphology import centers_of_labels
from cellprofiler.cpmath.cpmorphology import associate_by_distance
from cellprofiler.cpmath.cpmorphology import all_connected_components
from identify import M_LOCATION_CENTER_X, M_LOCATION_CENTER_Y
from cellprofiler.gui.help import HELP_ON_MEASURING_DISTANCES

TM_OVERLAP = 'Overlap'
TM_DISTANCE = 'Distance'
TM_MEASUREMENTS = 'Measurements'
TM_LAP = "LAP"
TM_ALL = [TM_OVERLAP, TM_DISTANCE, TM_MEASUREMENTS,TM_LAP]

DT_COLOR_AND_NUMBER = 'Color and Number'
DT_COLOR_ONLY = 'Color Only'
DT_ALL = [DT_COLOR_AND_NUMBER, DT_COLOR_ONLY]

F_PREFIX = "TrackObjects"
F_LABEL = "Label"
F_PARENT_OBJECT_NUMBER = "ParentObjectNumber"
'''The group index gives the index of the object's image within the group

The ImageNumber is unstable, with regard to groups - it could jump around
'''
F_PARENT_GROUP_INDEX = "ParentGroupIndex"
F_TRAJECTORY_X = "TrajectoryX"
F_TRAJECTORY_Y = "TrajectoryY"
F_DISTANCE_TRAVELED = "DistanceTraveled"
F_INTEGRATED_DISTANCE = "IntegratedDistance"
F_LINEARITY = "Linearity"
F_LIFETIME = "Lifetime"
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
def kalman_feature(model, matrix_or_vector, i, j=None):
    '''Return the feature name for a Kalman feature
    
    model - model used for Kalman feature: velocity or static
    matrix_or_vector - the part of the Kalman state to save, vec, COV or noise
    i - the name for the first (or only for vec and noise) index into the vector
    j - the name of the second index into the matrix
    '''
    pieces = [F_KALMAN, model, matrix_or_vector, i]
    if j is not None:
        pieces.append(j)
    return "_".join(pieces)

'''# of objects in the current frame without parents in the previous frame'''
F_NEW_OBJECT_COUNT = "NewObjectCount"
'''# of objects in the previous frame without parents in the new frame'''
F_LOST_OBJECT_COUNT = "LostObjectCount"
'''# of parents that split into more than one child'''
F_SPLIT_COUNT = "SplitCount"
'''# of children that are merged from more than one parent'''
F_MERGE_COUNT = "MergeCount"
'''Object area measurement for LAP method

The final part of the LAP method needs the object area measurement
which is stored using this name.'''
F_AREA = "Area"

F_ALL_COLTYPE_ALL = [(F_LABEL, cpmeas.COLTYPE_INTEGER),
                     (F_PARENT_OBJECT_NUMBER, cpmeas.COLTYPE_INTEGER),
                     (F_PARENT_GROUP_INDEX, cpmeas.COLTYPE_INTEGER),
                     (F_TRAJECTORY_X, cpmeas.COLTYPE_INTEGER),
                     (F_TRAJECTORY_Y, cpmeas.COLTYPE_INTEGER),
                     (F_DISTANCE_TRAVELED, cpmeas.COLTYPE_FLOAT),
                     (F_INTEGRATED_DISTANCE, cpmeas.COLTYPE_FLOAT),
                     (F_LINEARITY, cpmeas.COLTYPE_FLOAT),
                     (F_LIFETIME, cpmeas.COLTYPE_INTEGER)]

F_IMAGE_COLTYPE_ALL = [(F_NEW_OBJECT_COUNT, cpmeas.COLTYPE_INTEGER),
                       (F_LOST_OBJECT_COUNT, cpmeas.COLTYPE_INTEGER),
                       (F_SPLIT_COUNT, cpmeas.COLTYPE_INTEGER),
                       (F_MERGE_COUNT, cpmeas.COLTYPE_INTEGER)]

F_ALL = [feature for feature, coltype in F_ALL_COLTYPE_ALL]

F_IMAGE_ALL = [feature for feature, coltype in F_IMAGE_COLTYPE_ALL]

'''Random motion model, for instance Brownian motion'''
M_RANDOM = "Random"
'''Velocity motion model, object position depends on prior velocity'''
M_VELOCITY = "Velocity"
'''Random and velocity models'''
M_BOTH = "Both"

class TrackObjects(cpm.CPModule):

    module_name = 'TrackObjects'
    category = "Object Processing"
    variable_revision_number = 4

    def create_settings(self):
        self.tracking_method = cps.Choice('Choose a tracking method',
                                          TM_ALL, doc="""\
            When trying to track an object in an image, 
            <b>TrackObjects</b> will search within a maximum 
            specified distance (see the <i>distance within which to search</i> setting)
            of the object's location in the previous image, looking for a "match".
            Objects that match are assigned the same number, or label, throughout the 
            entire movie.
            There are several options for the method used to find a match. Choose 
            among these options based on which is most consistent from frame
            to frame of your movie.
            <ul>
            <li><i>Overlap:</i> Compares the amount of spatial overlap between identified objects in 
            the previous frame with those in the current frame. The object with the
            greatest amount of spatial overlap will be assigned the same number (label). Recommended
            when there is a high degree of overlap of an object from one frame to the next, 
            which is the case for movies with high frame rates relative to object motion.</li>

            <li><i>Distance:</i> Compares the distance between each identified
            object in the previous frame with that of the current frame. The 
            closest objects to each other will be assigned the same number (label).
            Distances are measured from the perimeter of each object. Recommended
            for cases where the objects are not very crowded but where <i>Overlap</i> 
            does not work sufficiently well, which is the case
            for movies with low frame rates relative to object motion.</li>

            <li><i>Measurement:</i> Compares each object in the 
            current frame with objects in the previous frame based on a particular 
            feature you have measured for the objects (for example, a particular intensity or shape measurement that can distinguish nearby objects). The object 
            with the closest-matching measurement will be selected as a match and will be 
            assigned the same number (label). This selection requires that you run the 
            specified <b>Measure</b> module previous to this module in the pipeline so
            that the measurement values can be used to track the objects.</li>
            
            <li><i>LAP:</i> Uses the linear assignment problem (LAP) framework. The
            linear assignment problem (LAP) algorithm (<i>Jaqaman et al., 2008</i>) 
            addresses the challenges of high object density, motion heterogeneity, 
            temporary disappearances, and object merging and splitting. 
            The algorithm first links objects between consecutive frames and then links 
            the resulting partial trajectories into complete trajectories. Both steps are formulated 
            as global combinatorial optimization problems whose solution identifies the overall 
            most likely set of object trajectories throughout a movie.

            Tracks are constructed from an image sequence by detecting objects in each 
            frame and linking objects between consecutive frames as a first step. This step alone
            may result in incompletely tracked objects due to the appearance and disappearance
            of objects, either in reality or apparently because of noise and imaging limitations.
            To correct this, you may apply an optional second step which closes temporal gaps 
            between tracked objects and captures merging and splitting events. This step takes
            place at the end of the analysis run.  Reference:
            <ul>
            <li>Jaqaman K, Loerke D, Mettlen M, Kuwata H, Grinstein S, Schmid SL, Danuser G. (2008)
            "Robust single-particle tracking in live-cell time-lapse sequences."
            <i>Nature Methods</i> 5(8),695-702.</li>
            </ul>
            </li>
            </ul>""")

        self.object_name = cps.ObjectNameSubscriber(
            'Select the objects to track','None', doc="""What did you call the objects you want to track?""")

        self.measurement = cps.Measurement(
            'Select object measurement to use for tracking',
            lambda : self.object_name.value, doc="""
            <i>(Used only if Measurements is the tracking method)</i><br>
            What measurement do you want to use for tracking?
            Choose which type of measurement (category) and which specific feature from the
            <b>Measure</b> module will be used for tracking. Select the feature name from 
            the popup box or see each <b>Measure</b> module's help for the list of
            the features measured by that module. If necessary, you will also be asked 
            to specify additional details such as the 
            image from which the measurements originated or the measurement scale.""")

        self.pixel_radius = cps.Integer(
            'Maximum pixel distance to consider matches',50,minval=1,doc="""
            Objects in the subsequent frame will be considered potential matches if 
            they are within this distance. To determine a suitable pixel distance, you can look
            at the axis increments on each image (shown in pixel units) or
            use the distance measurement tool. %(HELP_ON_MEASURING_DISTANCES)s"""%globals())

        self.model = cps.Choice(
            "Select the motion model",[M_RANDOM, M_VELOCITY, M_BOTH], value=M_BOTH,
            doc = """<i>(Used only if the LAP tracking method is applied)</i><br>
            This setting controls how to predict an object's position in
            the next frame, assuming that each object moves randomly with
            a frame-to-frame variance in position that follows a Gaussian
            distribution.<br>
            <ul><li><i>%(M_RANDOM)s:</i> A model in which objects move due to 
            Brownian Motion or a similar process where the variance in position
            differs between objects. Use this model if the objects move with some
            random jitter around a stationary location.</li>
            <li><i>%(M_VELOCITY)s:</i> A model in which the object moves with
            a velocity. Both velocity and position (after correcting for
            velocity) vary following a Gaussian distribution. Use this model if
            the objects move along a spatial trajectory in some direction over time.</li>
            <li><i>%(M_BOTH)s:</i> <b>TrackObjects</b> will predict each
            object's position using both models and use the model with the
            lowest penalty to join an object in one frame with one in another. Use this
            option if both models above are applicable over time.
            </li></ul>""" % globals())
        
        self.radius_std = cps.Float(
            'Number of standard deviations for search radius', 3, minval=1,
            doc = """<i>(Used only if the LAP tracking method is applied)</i>
            <br>
            <b>TrackObjects</b> will estimate the variance of the error
            between the observed and predicted positions of an object for
            each movement model. It will constrain the search for matching
            objects from one frame to the next to the standard deviation
            of the error times the number of standard
            deviations that you enter here.""")
        
        self.radius_limit = cps.FloatRange(
            'Search radius limit, in pixel units (Min,Max)', (2, 10), minval = 0,
            doc = """<i>(Used only if the LAP tracking method is applied)</i>
            <br>
            <b>TrackObjects</b> derives a search radius based on the error
            estimation. Potentially, the module can make an erroneous assignment
            with a large error, leading to a large estimated error for
            the object in the next frame. Conversely, the module can arrive
            at a small estimated error by chance, leading to a maximum radius
            that does not track the object in a subsequent frame. The radius
            limit constrains the maximum radius to reasonable values. 
            <p>The lower limit should be set to a radius (in pixels) that is a
            reasonable displacement for any object from one frame to the next.
            The upper limit should be set to the maximum reasonable 
            displacement under any circumstances.</p>""")
        
        self.wants_second_phase = cps.Binary(
            "Run the second phase of the LAP algorithm?", True, doc="""
            <i>(Used only if the LAP tracking method is applied)</i><br>
            Check this box to run the second phase of the LAP algorithm
            after processing all images. Leave the box unchecked to omit the
            second phase or to perform the second phase when running as a data
            tool.
            <p>Since object tracks may start and end not only because of the true appearance 
            and disappearance of objects, but also because of apparent disappearances due
            to noise and limitations in imaging, you may want to run the second phase 
            which attempts to close temporal gaps between tracked objects and tries to
            capture merging and splitting events.</p>
            <p>For additional details on optimizing the LAP settings, refer to Jaqaman K, Danuser G. 
            "Computational image analysis of cellular dynamics: a case study based on particle 
            tracking." <i>Cold Spring Harb Protocols</i> 2009(12) 
            <a href="http://cshprotocols.cshlp.org/cgi/content/full/2009/12/pdb.top65">[link]</a>,
            in particular the section "Adjustment of control parameters and diagnostics for track evaluation."</p>""")
        
        self.gap_cost = cps.Integer(
            'Gap cost', 40, minval=1, doc = '''
            <i>(Used only if the LAP tracking method is applied and the second phase is run)</i><br>
            This setting assigns a cost to keeping a gap caused
            when an object is missing from one of the frames of a track (the
            alternative to keeping the gap is to bridge it by connecting
            the tracks on either side of the missing frames).
            The cost of bridging a gap is the distance, in pixels, of the 
            displacement of the object between frames.<br><br>
            Set the gap cost higher if tracks from objects in previous
            frames are being erroneously joined, across a gap, to tracks from 
            objects in subsequent frames. Set the cost lower if tracks
            are not properly joined due to gaps caused by mis-segmentation.''')
        
        self.split_cost = cps.Integer(
            'Split alternative cost', 40, minval=1, doc = '''
            <i>(Used only if the LAP tracking method is applied and the second phase is run)</i><br>
            This setting is the cost of keeping two tracks distinct
            when the alternative is to make them into one track that
            splits. A split occurs when an object in one frame is assigned
            to the same track as two objects in a subsequent frame.
            The split score takes into
            account the area of the split object relative to the area of
            the resulting objects and the displacement of the resulting
            objects relative to the position of the original object and is
            roughly measured in pixels. The split alternative cost is 
            (conceptually) subtracted from the cost of making the split.<br>
            The split cost should be set lower if objects are being split
            that should not be split. It should be set higher if objects
            that should be split are not.''')
        
        self.merge_cost = cps.Integer(
            'Merge alternative cost', 40, minval=1,doc = '''
            <i>(Used only if the LAP tracking method is applied and the second phase is run)</i><br>
            This setting is the cost of keeping two tracks
            distinct when the alternative is to merge them into one.
            A merge occurs when two objects in one frame are assigned to
            the same track as a single object in a subsequent frame.
            The merge score takes into account the area of the two objects
            to be merged relative to the area of the resulting objects and
            the displacement of the original objects relative to the final
            object. The merge cost is measured in pixels. The merge
            alternative cost is (conceptually) subtracted from the
            cost of making the merge.<br>
            Set the merge alternative cost lower if objects are being
            merged when they should otherwise be kept separate. Set the cost
            higher if objects that are not merged should be merged.''')
        
        self.max_gap_score = cps.Integer(
            'Maximum gap displacement', 50, minval=1, doc = '''
            <i>(Used only if the LAP tracking method is applied and the second phase is run)</i><br>
            This setting acts as a filter for unreasonably large
            displacements during the second phase. The measurement is roughly
            the maximum displacement of an object's center from frame to frame.
            The algorithm will run more slowly with a higher value. The
            algorithm will not consider objects that would otherwise be
            tracked between frames if set to a lower value.''')
        
        self.max_merge_score = cps.Integer(
            'Maximum merge score', 50, minval=1, doc = '''
            <i>(Used only if the LAP tracking method is applied and the second phase is run)</i><br>
            This setting acts as a filter for unreasonably large
            merge scores. The merge score has two components: the area
            of the resulting merged object relative to the area of the
            two objects to be merged and the distances between the objects
            to be merged and the resulting object. The algorithm will run
            more slowly with a higher value. The algorithm will exclude
            objects that would otherwise be merged if it is set to a lower
            value.''')
        
        self.max_split_score = cps.Integer(
            'Maximum split score', 50, minval=1, doc = '''
            <i>(Used only if the LAP tracking method is applied and the second phase is run)</i><br>
            This setting acts as a filter for unreasonably large
            split scores. The split score has two components: the area
            of the initial object relative to the area of the
            two objects resulting from the split and the distances between the 
            original and resulting objects. The algorithm will run
            more slowly with a higher value. The algorithm will exclude
            objects that would otherwise be split if it is set to a lower
            value.''')
        
        self.max_frame_distance = cps.Integer(
            'Maximum gap', 5, minval=1, doc = '''
            <i>(Used only if the LAP tracking method is applied and the second phase is run)</i><br>
            This setting controls the maximum number of frames that can
            be skipped when merging a gap caused by an unsegmented object.
            These gaps occur when an image is mis-segmented and identification
            fails to find an object in one or more frames.''')

        self.display_type = cps.Choice(
            'Select display option', DT_ALL, doc="""
            How do you want to display the tracked objects?
            The output image can be saved as either a color-labeled image, with each tracked
            object assigned a unique color, or as a color-labeled image with the tracked object 
            number superimposed.""")

        self.wants_image = cps.Binary(
            "Save color-coded image?", False, doc="""
            Do you want to save the image with tracked, color-coded objects?
            Specify a name to give the image showing the tracked objects. This image
            can be saved with a <b>SaveImages</b> module placed after this module.""")

        self.image_name = cps.ImageNameProvider(
            "Name the output image", "TrackedCells", doc = '''
            <i>(Used only if saving the color-coded image)</i><br>
            What do you want to call the color-coded image, which will be available for downstream modules, such as <b>SaveImages</b>?''')

    def settings(self):
        return [self.tracking_method, self.object_name, self.measurement,
                self.pixel_radius, self.display_type, self.wants_image,
                self.image_name, self.model,
                self.radius_std, self.radius_limit,
                self.wants_second_phase,
                self.gap_cost, self.split_cost, self.merge_cost,
                self.max_gap_score, self.max_split_score,
                self.max_merge_score, self.max_frame_distance]

    def visible_settings(self):
        result = [self.tracking_method, self.object_name]
        if self.tracking_method == TM_MEASUREMENTS:
            result += [ self.measurement]
        if self.tracking_method == TM_LAP:
            result += [self.model, self.radius_std, self.radius_limit]
            result += [self.wants_second_phase]
            if self.wants_second_phase:
                result += [ 
                    self.gap_cost, self.split_cost, self.merge_cost,
                    self.max_gap_score, self.max_split_score,
                    self.max_merge_score, self.max_frame_distance]
        else:
            result += [self.pixel_radius]
            
        result +=[ self.display_type, self.wants_image]
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
        if self.get_ws_dictionary(workspace).has_key(field):
            return self.get_ws_dictionary(workspace)[field]
        return default

    def __set(self, field, workspace, value):
        self.get_ws_dictionary(workspace)[field] = value

    def get_group_number(self, workspace):
        '''get the current group's group number'''
        return self.__get("group_number", workspace, None)
    
    def set_group_number(self, workspace, value):
        self.__set("group_number", workspace, value)
    
    def get_group_indexes(self, workspace):
        '''get the group indexes for the current group'''
        result = self.__get("group_indexes", workspace, None)
        if result is None:
            result = []
            self.__set("group_indexes", workspace, result)
        return result
    
    def get_saved_measurements(self, workspace):
        return self.__get("measurements", workspace, np.array([], float))

    def set_saved_measurements(self, workspace, value):
        self.__set("measurements", workspace, value)

    def get_saved_coordinates(self, workspace):
        return self.__get("coordinates", workspace, np.zeros((2,0), int))

    def set_saved_coordinates(self, workspace, value):
        self.__set("coordinates", workspace, value)

    def get_orig_coordinates(self, workspace):
        '''The coordinates of the first occurrence of an object's ancestor'''
        return self.__get("orig coordinates", workspace, np.zeros((2,0), int))

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

    def prepare_group(self, pipeline, image_set_list, grouping, image_numbers):
        '''Erase any tracking information at the start of a run'''
        d = self.get_dictionary(image_set_list)
        d.clear()
        
        return True

    def measurement_name(self, feature):
        '''Return a measurement name for the given feature'''
        if self.tracking_method == TM_LAP:
            return "%s_%s" % (F_PREFIX, feature)
        return "%s_%s_%s" % (F_PREFIX, feature, str(self.pixel_radius.value))
    
    def image_measurement_name(self, feature):
        '''Return a measurement name for an image measurement'''
        if self.tracking_method == TM_LAP:
            return "%s_%s_%s" % (F_PREFIX, feature, self.object_name.value)
        return "%s_%s_%s_%s" % (F_PREFIX, feature, self.object_name.value,
                               str(self.pixel_radius.value))

    def add_measurement(self, workspace, feature, values):
        '''Add a measurement to the workspace's measurements

        workspace - current image set's workspace
        feature - name of feature being measured
        values - one value per object
        '''
        workspace.measurements.add_measurement(
            self.object_name.value,
            self.measurement_name(feature),
            values)

    def add_image_measurement(self, workspace, feature, value):
        measurement_name = self.image_measurement_name(feature)
        workspace.measurements.add_image_measurement(measurement_name, value)
        
    def is_interactive(self):
        return False
    
    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.object_name.value)
        if self.tracking_method == TM_DISTANCE:
            self.run_distance(workspace, objects)
        elif self.tracking_method == TM_OVERLAP:
            self.run_overlap(workspace, objects)
        elif self.tracking_method == TM_MEASUREMENTS:
            self.run_measurements(workspace, objects)
        elif self.tracking_method == TM_LAP:
            self.run_lapdistance(workspace, objects)
        else:
            raise NotImplementedError("Unimplemented tracking method: %s" %
                                      self.tracking_method.value)
        if self.wants_image.value:
            import matplotlib.transforms
            from cellprofiler.gui.cpfigure import figure_to_image, only_display_image
            
            figure = matplotlib.figure.Figure()
            canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(figure)
            ax = figure.add_subplot(1,1,1)
            self.draw(objects.segmented, ax, 
                      self.get_saved_object_numbers(workspace))
            #
            # This is the recipe for just showing the axis
            #
            only_display_image(figure, objects.segmented.shape)
            image_pixels = figure_to_image(figure, dpi=figure.dpi)
            image = cpi.Image(image_pixels)
            workspace.image_set.add(self.image_name.value, image)
        if workspace.frame is not None:
            workspace.display_data.labels = objects.segmented
            workspace.display_data.object_numbers = \
                     self.get_saved_object_numbers(workspace)
            
    def display(self, workspace):
        frame = workspace.create_or_find_figure(title="TrackObjects, image cycle #%d"%(
                workspace.measurements.image_set_number),subplots=(1,1))
        figure = frame.figure
        figure.clf()
        ax = figure.add_subplot(1,1,1)
        self.draw(workspace.display_data.labels, ax, 
                  workspace.display_data.object_numbers)

    def draw(self, labels, ax, object_numbers):
        indexer = np.zeros(len(object_numbers)+1,int)
        indexer[1:] = object_numbers
        #
        # We want to keep the colors stable, but we also want the
        # largest possible separation between adjacent colors. So, here
        # we reverse the significance of the bits in the indices so
        # that adjacent number (e.g. 0 and 1) differ by 128, roughly
        #
        pow_of_2 = 2**np.mgrid[0:8,0:len(indexer)][0]
        bits = (indexer & pow_of_2).astype(bool)
        indexer = np.sum(bits.transpose() * (2 ** np.arange(7,-1,-1)), 1)
        recolored_labels = indexer[labels]
        cm = matplotlib.cm.get_cmap(cpprefs.get_default_colormap())
        cm.set_bad((0,0,0))
        norm = matplotlib.colors.BoundaryNorm(range(256), 256)
        img = ax.imshow(numpy.ma.array(recolored_labels, mask=(labels==0)),
                        cmap=cm, norm=norm)
        if self.display_type == DT_COLOR_AND_NUMBER:
            i,j = centers_of_labels(labels)
            for n, x, y in zip(object_numbers, j, i):
                if np.isnan(x) or np.isnan(y):
                    # This happens if there are missing labels
                    continue
                ax.annotate(str(n), xy=(x,y),color='white',
                            arrowprops=dict(visible=False))

    def run_distance(self, workspace, objects):
        '''Track objects based on distance'''
        old_i, old_j = self.get_saved_coordinates(workspace)
        if len(old_i):
            distances, (i,j) = distance_transform_edt(objects.segmented == 0,
                                                      return_indices=True)
            #
            # Look up the coordinates of the nearest new object (given by
            # the transform i,j), then look up the label at that coordinate
            # (objects.segmented[#,#])
            #
            new_object_numbers = objects.segmented[i[old_i, old_j],
                                                   j[old_i, old_j]]
            #
            # Mask out any objects at too great of a distance
            #
            new_object_numbers[distances[old_i, old_j] >
                               self.pixel_radius.value] = 0
            #
            # Do the same with the new centers and old objects
            #
            i,j = (centers_of_labels(objects.segmented)+.5).astype(int)
            old_labels = self.get_saved_labels(workspace)
            distances, (old_i,old_j) = distance_transform_edt(
                old_labels == 0,
                return_indices=True)
            old_object_numbers = old_labels[old_i[i, j],
                                            old_j[i, j]]
            old_object_numbers[distances[i, j] > self.pixel_radius.value] = 0
            self.map_objects(workspace, 
                             new_object_numbers,
                             old_object_numbers, 
                             i,j)
        else:
            i,j = (centers_of_labels(objects.segmented)+.5).astype(int)
            count = len(i)
            self.map_objects(workspace, np.zeros((0,),int), 
                             np.zeros(count,int), i,j)
        self.set_saved_labels(workspace, objects.segmented)

    def run_lapdistance(self, workspace, objects):
        '''Track objects based on distance'''
        m = workspace.measurements
        group_number = m.get_current_image_measurement(cpp.GROUP_NUMBER)
        self.set_group_number(workspace, group_number)
        group_index = m.get_current_image_measurement(cpp.GROUP_INDEX)
        self.get_group_indexes(workspace).append(group_index)

        old_i, old_j = self.get_saved_coordinates(workspace)
        n_old = len(old_i)
        #
        # Automatically set the cost of birth and death above
        # that of the largest allowable cost.
        #
        costBorn = costDie = self.radius_limit.max * 1.10
        kalman_states = self.get_kalman_states(workspace)
        if kalman_states == None:
            if self.static_model:
                kalman_states = [ cpfilter.static_kalman_model()]
            else:
                kalman_states = []
            if self.velocity_model:
                kalman_states.append(cpfilter.velocity_kalman_model())
        areas = fix(scipy.ndimage.sum(
            np.ones(objects.segmented.shape), objects.segmented, 
            np.arange(1, np.max(objects.segmented) + 1,dtype=np.int32)))
        areas = areas.astype(int)

        if n_old > 0:
            new_i, new_j = centers_of_labels(objects.segmented)
            n_new = len(new_i)
            i,j = np.mgrid[0:n_old, 0:n_new]
            ##############################
            #
            #  Kalman filter prediction
            #
            #
            # We take the lowest cost among all possible models
            #
            minDist = np.ones((n_old, n_new)) * self.radius_limit.max
            d = np.ones((n_old, n_new)) * np.inf
            # The index of the Kalman filter used: -1 means not used
            kalman_used = -np.ones((n_old, n_new), int)
            for nkalman, kalman_state in enumerate(kalman_states):
                assert isinstance(kalman_state, cpfilter.KalmanState)
                obs = kalman_state.predicted_obs_vec
                dk = np.sqrt((obs[i,0] - new_i[j])**2 +
                             (obs[i,1] - new_j[j])**2)
                noise_sd = np.sqrt(np.sum(kalman_state.noise_var[:,0:2], 1))
                radius = np.maximum(np.minimum(noise_sd * self.radius_std.value, 
                                               self.radius_limit.max),
                                    self.radius_limit.min)
                                    
                is_best = ((dk < d) & (dk < radius[:, np.newaxis]))
                d[is_best] = dk[is_best]
                minDist[is_best] = radius[i][is_best]
                kalman_used[is_best] = nkalman
            minDist = np.maximum(np.minimum(minDist, self.radius_limit.max),
                                 self.radius_limit.min)
            #
            #############################
            #
            # Linear assignment setup
            #
            n = len(old_i)+len(new_i)
            kk = np.zeros((n+10)*(n+10), np.int32)
            first = np.zeros(n+10, np.int32)
            cc = np.zeros((n+10)*(n+10), np.float)
            t = np.argwhere((d < minDist))
            x = np.sqrt((old_i[t[0:t.size, 0]]-new_i[t[0:t.size, 1]])**2 + (old_j[t[0:t.size, 0]]-new_j[t[0:t.size, 1]])**2)
            t = t+1
            t = np.column_stack((t, x))
            a = np.arange(len(old_i))+2
            x = np.searchsorted(t[0:(t.size/2),0], a)
            a = np.arange(len(old_i))+1
            b = np.arange(len(old_i))+len(new_i)+1
            c = np.zeros(len(old_i))+costDie
            b = np.column_stack((a, b, c))
            t = np.insert(t, x, b, 0)

            i,j = np.mgrid[0:len(new_i),0:len(old_i)+1]
            i = i+len(old_i)+1
            j = j+len(new_i)
            j[0:len(new_i)+1,0] = i[0:len(new_i)+1,0]-len(old_i)
            x = np.zeros((len(new_i),len(old_i)+1))
            x[0:len(new_i)+1,0] = costBorn
            i = i.flatten()
            j = j.flatten()
            x = x.flatten()
            x = np.column_stack((i, j, x))
            t = np.vstack((t, x))

            kk = np.ndarray.astype(t[0:(t.size/3),1], 'int32')
            cc = t[0:(t.size/3),2]

            a = np.arange(len(old_i)+len(new_i)+2)
            first = np.bincount(np.ndarray.astype(t[0:(t.size/3),0], 'int32')+1)
            first = np.cumsum(first)+1

            first[0] = 0
            kk = np.hstack((np.array((0)), kk))
            cc = np.hstack((np.array((0.0)), cc))

            x, y =  LAP.LAP(np.ascontiguousarray(kk, np.int32),
                            np.ascontiguousarray(first, np.int32),
                            np.ascontiguousarray(cc, np.float), n)
            a = np.argwhere(x > len(new_i))
            b = np.argwhere(y >len(old_i))
            x[a[0:len(a)]] = 0
            y[b[0:len(b)]] = 0
            a = np.arange(len(old_i))+1
            b = np.arange(len(new_i))+1
            new_object_numbers = x[a[0:len(a)]]
            old_object_numbers = y[b[0:len(b)]]

            ###############################
            # 
            #  Kalman filter update
            #
            model_idx = np.zeros(len(old_object_numbers), int)
            mask = old_object_numbers > 0
            old_idx = old_object_numbers - 1
            model_idx[mask] =\
                kalman_used[old_idx[mask], mask]
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
            r = (measurement_variance[:, np.newaxis, np.newaxis] * 
                 np.eye(2)[np.newaxis,:,:])
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
                    i,j = np.mgrid[0:len(matching_idx),0:state_len]
                    q[new_idx[i], j, j] = \
                        kalman_state.noise_var[matching_idx[i],j]
                new_kalman_state = cpfilter.kalman_filter(
                    kalman_state,
                    old_idx,
                    np.column_stack((new_i, new_j)),
                    q,r)
                new_kalman_states.append(new_kalman_state)
            self.set_kalman_states(workspace, new_kalman_states)
                    
            i,j = (centers_of_labels(objects.segmented)+.5).astype(int)
            self.map_objects(workspace, 
                             new_object_numbers,
                             old_object_numbers, 
                             i,j)
        else:
            i,j = centers_of_labels(objects.segmented)
            count = len(i)
            #
            # Initialize the kalman_state with the new objects
            #
            new_kalman_states = []
            r = np.zeros((count, 2, 2))
            for kalman_state in kalman_states:
                q = np.zeros((count, kalman_state.state_len, kalman_state.state_len))
                new_kalman_state = cpfilter.kalman_filter(
                    kalman_state, -np.ones(count),
                    np.column_stack((i,j)), q, r)
                new_kalman_states.append(new_kalman_state)
            self.set_kalman_states(workspace, new_kalman_states)
                                        
            i = (i+.5).astype(int)
            j = (j+.5).astype(int)
            self.map_objects(workspace, np.zeros((0,),int), 
                             np.zeros(count,int), i,j)
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        m.add_measurement(self.object_name.value,
                          self.measurement_name(F_AREA),
                          areas)
        self.save_kalman_measurements(workspace)
        self.set_saved_labels(workspace, objects.segmented)
        
    def get_kalman_models(self):
        '''Return tuples of model and names of the vector elements'''
        if self.static_model:
            models = [ (F_STATIC_MODEL, (F_Y, F_X))]
        else:
            models = []
        if self.velocity_model:
            models.append((F_VELOCITY_MODEL, (F_Y, F_X, F_VY, F_VX)))
        return models
    
    def save_kalman_measurements(self, workspace):
        '''Save the first-pass state_vec, state_cov and state_noise'''
        
        m = workspace.measurements
        object_name = self.object_name.value
        for (model, elements), kalman_state in zip(
            self.get_kalman_models(), self.get_kalman_states(workspace)):
            assert isinstance(kalman_state, cpfilter.KalmanState)
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
                    np.hstack((
                        -np.ones(nobjs),
                        np.arange(len(kalman_state.state_noise_idx)))),
                    np.hstack((
                        np.arange(nobjs), kalman_state.state_noise_idx)),
                    np.arange(nobjs))
                last_idx = last_idx.astype(int)
            for i, element in enumerate(elements):
                #
                # state_vec
                #
                mname = self.measurement_name(
                    kalman_feature(model, F_STATE, element))
                values = np.zeros(0) if nobjs == 0 else kalman_state.state_vec[:,i]
                m.add_measurement(object_name, mname, values)
                #
                # state_noise
                #
                mname = self.measurement_name(
                    kalman_feature(model, F_NOISE, element))
                values = np.zeros(nobjs)
                if nobjs > 0:
                    values[last_idx == -1] = np.NaN
                    values[last_idx > -1] = kalman_state.state_noise[last_idx[last_idx > -1], i]
                m.add_measurement(object_name, mname, values)
                #
                # state_cov
                #
                for j, el2 in enumerate(elements):
                    mname = self.measurement_name(
                        kalman_feature(model, F_COV, element, el2))
                    values = kalman_state.state_cov[:, i, j]
                    m.add_measurement(object_name, mname, values)

    def run_overlap(self, workspace, objects):
        '''Track objects by maximum # of overlapping pixels'''
        current_labels = objects.segmented
        old_labels = self.get_saved_labels(workspace)
        i,j = (centers_of_labels(objects.segmented)+.5).astype(int)
        if old_labels is None:
            count = len(i)
            self.map_objects(workspace, np.zeros((0,),int), 
                             np.zeros(count,int), i,j)
        else:
            mask = ((current_labels > 0) & (old_labels > 0))
            cur_count = np.max(current_labels)
            old_count = np.max(old_labels)
            count = np.sum(mask)
            if count == 0:
                # There's no overlap.
                self.map_objects(workspace,
                                 np.zeros(old_count, int),
                                 np.zeros(cur_count,int),
                                 i,j)
            else:
                cur = current_labels[mask]
                old = old_labels[mask]
                histogram = scipy.sparse.coo_matrix(
                    (np.ones(count),(cur, old)),
                    shape=(cur_count+1,old_count+1)).toarray()
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
                self.map_objects(workspace,
                                 new_of_old,
                                 old_of_new,
                                 i,j)
        self.set_saved_labels(workspace, current_labels)

    def run_measurements(self, workspace, objects):
        current_labels = objects.segmented
        new_measurements = workspace.measurements.get_current_measurement(
            self.object_name.value,
            self.measurement.value)
        old_measurements = self.get_saved_measurements(workspace)
        old_labels = self.get_saved_labels(workspace)
        i,j = (centers_of_labels(objects.segmented)+.5).astype(int)
        if old_labels is None:
            count = len(i)
            self.map_objects(workspace, np.zeros((0,),int), 
                             np.zeros(count,int), i,j)
        else:
            associations = associate_by_distance(old_labels, current_labels,
                                                 self.pixel_radius.value)
            best_child = np.zeros(len(old_measurements), int)
            best_parent = np.zeros(len(new_measurements), int)
            best_child_measurement = (np.ones(len(old_measurements), int) *
                                      np.finfo(float).max)
            best_parent_measurement = (np.ones(len(new_measurements), int) *
                                       np.finfo(float).max)
            for old, new in associations:
                diff = abs(old_measurements[old-1] - new_measurements[new-1])
                if diff < best_child_measurement[old-1]:
                    best_child[old-1] = new
                    best_child_measurement[old-1] = diff
                if diff < best_parent_measurement[new-1]:
                    best_parent[new-1] = old
                    best_parent_measurement[new-1] = diff
            self.map_objects(workspace, best_child, best_parent, i,j)
        self.set_saved_labels(workspace,current_labels)
        self.set_saved_measurements(workspace, new_measurements)

    def run_as_data_tool(self, workspace):
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        #
        # The parent object numbers, image numbers and labels all
        # come from the .mat file as floats. Convert them into ints.
        #
        for feature in (F_PARENT_GROUP_INDEX, F_PARENT_OBJECT_NUMBER,
                        F_LABEL, F_LIFETIME):
            values = m.get_all_measurements(self.object_name.value,
                                            self.measurement_name(feature))
            for i in range(len(values)):
                values[i] = values[i].astype(int)
        group_numbers = np.array(
            m.get_all_measurements(cpmeas.IMAGE, cpp.GROUP_NUMBER))
        group_indexes = np.array(
            m.get_all_measurements(cpmeas.IMAGE, cpp.GROUP_INDEX))
        for group_number in np.unique(group_numbers):
            self.set_group_number(workspace, group_number)
            gi = group_indexes[group_numbers == group_number]
            glist = self.get_group_indexes(workspace)
            del glist[:]
            glist += gi.tolist()
            self.post_group(workspace, {})

    def flood(self, i, at, a, b, c, d, z):
        z[i] = at
        if(a[i] != -1 and z[a[i]] == 0):
            z = self.flood(a[i], at, a, b, c, d, z)
        if(b[i] != -1 and z[b[i]] == 0):
            z = self.flood(b[i], at, a, b, c, d, z)
        if(c[i] != -1 and z[c[i]] == 0):
            z = self.flood(c[i], at, a, b, c, d, z)
        if(c[i] != -1 and z[c[i]] == 0):
            z = self.flood(c[i], at, a, b, c, d, z)
        return z

    def post_group(self, workspace, grouping):
        if (self.tracking_method != TM_LAP or
            not self.wants_second_phase):
            return
        ############################################
        #
        # All of the scores going into the LAP must be positive
        # so we have to balance the positive costs instead of
        # doing something simpler, like subtracting the cost of a gap
        # from the cost of bridging the gap.
        #
        # Variables we have to play with:
        # Gap initiation - cost applied to a start
        # Gap termination - cost applied to an end
        # Split termination - alternative cost to splitting.
        # Merge initiation - alternative cost to merging.
        #
        # Cost of gap = 2*gap_termination + 2*gap_initiation
        # Alternative cost = gap_termination + displacement across gap + gap_initiation
        #
        # Cost of split = split displacement + area + split_termination
        # Cost of alternative = gap_initiation + gap_termination
        # split_termination = split_cost - gap_initiation + gap_termination
        #
        # Cost of merge = gap_initiation + merge displacement + area + gap_termination
        # Cost of alternative = merge_initiation + gap_initiation
        # merge_initiation = merge_cost - gap_initiation
        ############################################
        
        gap_cost = float(self.gap_cost.value)
        split_alternative_cost = float(self.split_cost.value)
        merge_alternative_cost = float(self.merge_cost.value)
        
        # Make the gap closing cost high enough so that 
        # gap_initiation + gap_termination > merge or split alternative costs
        gap_closing_cost = split_alternative_cost + merge_alternative_cost
        gap_initiation_cost = (gap_cost + gap_closing_cost) / 2
        gap_termination_cost = (gap_cost + gap_closing_cost) / 2
        split_termination_cost = split_alternative_cost + gap_initiation_cost + gap_termination_cost
        merge_initiation_cost = merge_alternative_cost
        
        para1 = self.max_gap_score.value #max upper-left
        para2 = self.max_merge_score.value #max upper-middle
        para3 = gap_termination_cost #value for upper-right
        para4 = self.max_split_score.value #max for middle-left
        para5 = split_termination_cost #value for middle-right
        para6 = gap_initiation_cost #value for lower-left
        para7 = merge_initiation_cost #value for lower-middle
        para8 = self.max_frame_distance.value #max frame difference

        m = workspace.measurements
        label = m.get_all_measurements(self.object_name.value, 
                                       self.measurement_name(F_LABEL))
        orig_label = label
        a = m.get_all_measurements(self.object_name.value, M_LOCATION_CENTER_X)
        b = m.get_all_measurements(self.object_name.value, M_LOCATION_CENTER_Y)
        Area = m.get_all_measurements(self.object_name.value, 
                                      self.measurement_name(F_AREA))
        group_numbers = np.array(
            m.get_all_measurements(cpmeas.IMAGE, cpp.GROUP_NUMBER), int)
        group_indices = np.array(
            m.get_all_measurements(cpmeas.IMAGE, cpp.GROUP_INDEX), int)
        parent_object_numbers = m.get_all_measurements(
            self.object_name.value, self.measurement_name(F_PARENT_OBJECT_NUMBER))
        parent_image_numbers = m.get_all_measurements(
            self.object_name.value, self.measurement_name(F_PARENT_GROUP_INDEX))
        new_object_count = m.get_all_measurements(
            cpmeas.IMAGE, self.image_measurement_name(F_NEW_OBJECT_COUNT))
        lost_object_count = m.get_all_measurements(
            cpmeas.IMAGE, self.image_measurement_name(F_LOST_OBJECT_COUNT))
        merge_count = m.get_all_measurements(
            cpmeas.IMAGE, self.image_measurement_name(F_MERGE_COUNT))
        split_count = m.get_all_measurements(
            cpmeas.IMAGE, self.image_measurement_name(F_SPLIT_COUNT))
        my_group_number = self.get_group_number(workspace)
        my_group_indices = np.array(self.get_group_indexes(workspace))
        indexes = np.arange(len(group_numbers))
        mask = group_numbers == my_group_number
        group_indices = group_indices[mask]
        indexes = indexes[mask]
        assert np.all(group_indices == my_group_indices)
        #
        # Reduce the lists to only the ones in the group
        #
        label = [label[ix] for ix in indexes]
        a = [a[ix] for ix in indexes]
        b = [b[ix] for ix in indexes]
        Area = [Area[ix] for ix in indexes]
        numFrames = len(b)
        parent_image_numbers = [parent_image_numbers[ix] for ix in indexes]
        parent_object_numbers = [parent_object_numbers[ix] for ix in indexes]

        #Calculates the maximum number of cells in a single frame

        i = 0
        mlength = 0
        while i<numFrames:
            if(mlength < len(label[i])):
                mlength = len(label[i])
            i = i+1
        #
        # Quit if there's nothing to do
        #
        if mlength == 0:
            return

        #converts the ragged array into a two-dimensional array    

        labelprime = np.zeros((numFrames, mlength), dtype=np.int)
        aprime =  np.zeros((numFrames, mlength))
        bprime =  np.zeros((numFrames, mlength))
        Areaprime = np.zeros((numFrames, mlength))

        for i, (ll, aa, bb, AA) in enumerate(zip(label, a, b, Area)):
            llength = len(ll)
            labelprime[i,:llength] = ll
            aprime[i,:llength] = aa
            bprime[i,:llength] = bb
            Areaprime[i, :llength] = AA

        #sets up the arrays F, L, P, and Q
        #F is an array of all the cells that are the starts of segments
        #  F[:,:2] are the coordinates
        #  F[:,2] is the image index
        #  F[:,3] is the object index
        #L is the ends
        #P includes all cells
        #Q[i] is the segment that P[i] belongs to

        N = np.amax(labelprime)
        length = np.zeros(N, dtype=np.int)
        F = np.zeros((N, 4), dtype=np.int)
        L = np.zeros((N, 4), dtype=np.int)
        P = np.zeros((0, 4), dtype=np.int)

        Q = np.zeros(0)

        i = 1
        while i <= N:
            l = np.argwhere(labelprime == i)
            j = np.arange(len(l))
            x = aprime[l[j, 0], l[j, 1]]
            y = bprime[l[j, 0], l[j, 1]]

            t = np.column_stack((x, y, l))

            F[i-1] = t[0]
            L[i-1] = t[len(l)-1]
            length[i-1] = len(l)

            P = np.vstack((P, t))
            Q = np.hstack((Q, np.zeros(length[i-1], dtype=np.int)+i))

            i = i+1

        #Creates P1 and P2, which is P without the starts and ends of segments respectively, representing possible
        #points of merges and splits respectively

        Q = np.arange(len(P))
        t = np.cumsum(length)-length
        P1 = np.delete(P, t, 0)
        Q1 = np.delete(Q, t, 0)
        t = t+length-1
        P2 = np.delete(P, t, 0)
        Q2 = np.delete(Q, t, 0)
        
        ##################################################
        #
        # Addresses of supplementary nodes:
        # The LAP array is composed of six address ranges.
        # 
        # 1 to T      = segment starts and ends
        # T+1 to T+OB = split starts
        # T+OB+1 to T * 2 + OB = gap alternatives
        # T * 2 + OB + 1 to T * 2 + OB * 2 = merge ends
        # T * 2 + OB * 2 + 1 to T * 2 + OB * 2 = split alternatives
        # T * 2 + OB * 3 + 1 to T * 2 + OB * 3 = merge alternatives
        #
        # T = # tracks
        # OB = # of objects that can serve as merge or split points
        ##################################################
        
        ss_off = len(F)
        ga_off = len(F) + len(P1)
        me_off = len(F) * 2 + len(P1)
        sa_off = len(F) * 2 + len(P1) * 2
        ma_off = len(F) * 2 + len(P1) * 3

        #creates the upper-left block

        i, j = np.mgrid[0:len(F),0:len(F)]
        d = np.sqrt((L[i, 0]-F[j, 0])**2 + (L[i, 1]-F[j, 1])**2)

        #removes the possibility of gaps that have too large of a frame difference

        y = F[j, 2]-L[i, 2]
        x = np.argwhere(y > para8)
        i = np.arange(len(x))
        d[x[i,0], x[i,1]] = para1+1
        x = np.argwhere(y <= 0)
        i = np.arange(len(x))
        d[x[i,0], x[i,1]] = para1+1

        # Filter out costs that are too high
        a = np.argwhere(d <= para1)
        #
        # Add the gap closing cost which is just an offset that guarantees
        # that the gap initiation and gap termination are higher than
        # split and merge coss
        #
        d += gap_closing_cost
        d = np.column_stack((a, d[a[0:len(a), 0], a[0:len(a), 1]]))

        #creates the transpose for the lower-right block

        w = np.column_stack((d[:, 1]+len(F)+len(P1), 
                             d[:, 0]+len(F)+len(P1), 
                             np.zeros(len(d))+0.0001))
        d = np.vstack((d, w))

        #upper-right block (which provides terminating alternatives for gaps)

        f = np.column_stack((np.arange(len(F)), 
                             np.arange(len(F))+len(F)+len(P1), 
                             np.zeros(len(F))+gap_termination_cost))
        d = np.vstack((d, f))
        
        #lower-left (which provides initiating alternatives for gaps)

        a = np.column_stack((np.arange(len(F))+len(F)+len(P1), 
                             np.arange(len(F)), 
                             np.zeros(len(F))+gap_initiation_cost))

        d = np.vstack((d, a))

        #finds possible merge points with a small enough gap difference, for upper-middle block

        i = 0
        j = np.arange(len(P1))
        #
        # The first column of z is the index of the track that ends. The second
        # is the index into P2 of the object to be merged into
        #
        z = np.zeros((0,2), np.int32)
        while i <len(F):
            y = P1[j, 2]-L[i, 2]
            y = y.astype("int32")
            x = np.argwhere((y <= para8) & (y > 0))
            y = np.column_stack((np.zeros(len(x), dtype="int32")+i, x))
            z = np.vstack((z, y))
            i = i+1

        #calculates actual cost according to the formula given in the supplmenetary notes    
        AreaLast = Areaprime[L[z[:, 0], 2].astype("int32"), 
                             L[z[:, 0], 3].astype("int32")]
        AreaBeforeMerge = Areaprime[P[Q1[z[:, 1]]-1, 2].astype("int32"), 
                                    P[Q1[z[:, 1]]-1, 3].astype("int32")]
        AreaAtMerge = Areaprime[P1[z[:, 1], 2].astype("int32"), 
                                P1[z[:, 1], 3].astype("int32")]
        rho = ((AreaLast+AreaBeforeMerge)/AreaAtMerge)**2
        px = np.argwhere(rho < 1)
        if(len(px) > 0):
            rho[px] = np.sqrt((1/rho[px]))
        if len(z) > 0:
            rho = np.sqrt(np.sum((L[z[:, 0], :2]-P2[z[:, 1], :2])**2, 1)) * rho
        else:
            rho = np.zeros(0)
        e = rho

        #filters out the costs that are too high

        b = np.argwhere(e <= para2)

        #puts together all the upper blocks

        if len(b) > 0:
            z = z[b].reshape((len(b), 2))
            e = e[b].reshape((len(b)))
        else:
            z = np.zeros((0,2),z.dtype)
            e = np.zeros((0,),e.dtype)
        e = np.column_stack((z, e))

        # link the alternative cost of merging to the merge-end node
        # with a cost of zero (bookkeeping)

        f = np.column_stack((np.arange(len(P1))+ma_off,
                             np.arange(len(P1))+me_off,
                             np.zeros(len(P1))))
        #
        # Link the alternative cost of merging to the gap node
        # with a cost that's equal to the gap termination cost minus
        # the alternative penalty to merging
        #
        
        g = np.column_stack((e[:,1] + ma_off,
                             e[:,0] + ga_off,
                             np.zeros(len(e))+ gap_termination_cost - merge_alternative_cost))
        #
        # We also need a path from every merge to every merge 
        # initiator so that every merge can have an end. This is just
        # bookkeeping, so again no cost.
        #
        h = np.column_stack((np.arange(len(P1))+me_off,
                             np.arange(len(P1))+ma_off,
                             np.zeros(len(P1))))

        # Mark the first index as an index into P1 by moving it past the
        # track number indices.
        e[0:len(e), 1] = e[0:len(e), 1]+me_off
        d = np.vstack((d, e, f, g, h))
        
        #similar process for the middle-left block as the upper-middle left block

        if len(P1) > 0:
            i = 0
            j = np.arange(len(F))
            # The first column of Z is the index of the object being split
            # The second is the index of the track that results from
            # the split.
            #
            z = np.zeros((0,2), np.int32)
            while i < len(P1):
                y = F[j, 2]-P2[i, 2]
                y = y.astype("int32")
                x = np.argwhere((y <= para8) & (y > 0))
                y = np.column_stack((np.zeros(len(x), dtype="int32")+i, x))
                z = np.vstack((z, y))
                i = i+1
    
            AreaFirst = Areaprime[F[z[:, 1], 2].astype("int32"), 
                                  F[z[:, 1], 3].astype("int32")]
            AreaAfterSplit = Areaprime[P[Q2[z[:, 0]]+1, 2].astype("int32"), 
                                       P[Q2[z[:, 0]]+1, 3].astype("int32")]
            AreaAtSplit = Areaprime[P2[z[:, 0], 2].astype("int32"), 
                                    P2[z[:, 0], 3].astype("int32")]
            rho = ((AreaFirst+AreaAfterSplit)/AreaAtSplit)**2
            x = np.argwhere(rho < 1)
            if(len(x) > 1):
                rho[x] = (1/rho[x])*(1/rho[x])
            if len(z):
                rho = np.sqrt(np.sum((F[z[:, 1], :2]-P1[z[:, 0], :2])**2, 1)) * rho
            else:
                rho = np.zeros(0)
            e = rho
    
            b = np.argwhere(e <= para4)
            if len(b) > 0:
                z = z[b].reshape((len(b), 2))
                e = e[b].reshape((len(b)))
                e = np.column_stack((z, e))
            else:
                e = np.zeros((0,3))
        else:
            e = np.zeros((0,3))

        #middle-right block - the alternative for each split is that it
        # terminates with the split cost.

        f = np.column_stack((np.arange(len(P1))+ss_off,
                             np.arange(len(P1))+sa_off,
                             np.zeros(len(P1))))
        #
        # For bookkeeping, we need to make a path from each segment start's
        # terminator to each of these alternatives.
        
        g = np.column_stack((e[:,1] + ga_off,
                             e[:,0] + sa_off,
                             np.zeros(len(e))+gap_initiation_cost - split_alternative_cost))
        #
        # We also need a path from every split terminators to 
        # every split so that every split can have a start. This is just
        # bookkeeping, so again no cost.
        #
        h = np.column_stack((np.arange(len(P1))+sa_off,
                             np.arange(len(P1))+ss_off,
                             np.zeros(len(P1))))

        # Add the # of tracks to the first column of Z (now E) in order
        # to mark it as an index into P1.
        e[0:len(e), 0] = e[0:len(e), 0]+ss_off
 
        d = np.vstack((d, e, f, g, h))
        #sorts the list of costs, and add one to indices for costs to fit in with LAP

        indices = np.lexsort((d[:,1], d[:,0]))
        d = d[indices]
        d[:, 1] = d[:, 1]+1
        d[:, 0] = d[:, 0]+1

        #gets first, kk, and cc ready for LAP

        counts = np.bincount(np.ndarray.astype(d[:,0], np.int32))
        first = np.ascontiguousarray(np.hstack((np.cumsum(counts) - counts+1,
                                                [len(d)+1])), np.int32)
        kk = np.ascontiguousarray(np.hstack(([0],d[:, 1])), np.int32)
        cc = np.ascontiguousarray(np.hstack(([0.0], d[:, 2])), np.float)

        x, y =  LAP.LAP(kk, first, cc, len(F)*2+len(P1)*4)

        #attaches different segments together if they are matches through the IAP
        a = np.zeros(len(F)+1, dtype="int32")
        b = np.zeros(len(F)+1, dtype="int32")
        c = np.zeros(len(F)+1, dtype="int32")-1
        d = np.zeros(len(F)+1, dtype="int32")-1
        z = np.zeros(len(F)+1, dtype="int32")
        i = 1
        while i<=len(F):
            if(y[i] <= len(F)):
                #
                # y[i] gives index of last hooked to first
                #
                b[i] = y[i]
                c[b[i]] = i
                #
                # Hook our parent image/object number to found parent
                #
                my_image_index, my_object_index = F[i-1, 2:]
                parent_image_index, parent_object_index = L[y[i]-1, 2:]
                parent_image_numbers[my_image_index][my_object_index] = \
                                    group_indices[parent_image_index]
                parent_object_numbers[my_image_index][my_object_index] = \
                                     parent_object_index + 1
                #
                # One less new object
                #
                new_object_count[indexes[my_image_index]] -= 1
                #
                # One less lost object (the lost object is recorded in
                # the image set after the parent)
                #
                lost_object_count[indexes[parent_image_index+1]] -= 1
            elif(y[i] > ss_off and y[i] <= ss_off+len(P1)):
                #
                # Hook split objects to their parent
                #
                my_image_index, my_object_index = F[i-1, 2:]
                parent_image_index, parent_object_index = P2[y[i]-1-ss_off][2:]
                b[i] = labelprime[parent_image_index, parent_object_index]
                c[b[i]] = i
                parent_image_numbers[my_image_index][my_object_index] = \
                                    group_indices[parent_image_index]
                parent_object_numbers[my_image_index][my_object_index] = \
                                     parent_object_index + 1
                #
                # one less new object
                #
                new_object_count[indexes[my_image_index]] -= 1
                #
                # one more split object
                #
                split_count[indexes[my_image_index]] += 1
            else:
                b[i] = -1

            if(x[i] <= len(F)):
                a[i] = x[i]
                d[a[i]] = i
            elif(x[i] > me_off and x[i] <= me_off+len(P1)):
                #
                # Handle merged objects
                # 
                my_image_index, my_object_index = P1[x[i]-1-me_off][2:]
                a[i] = labelprime[my_image_index, my_object_index]
                d[a[i]] = i
                lost_object_count[indexes[my_image_index]] -= 1
                merge_count[indexes[my_image_index]] += 1
            else:
                a[i] = -1
            i = i+1

        #
        # At this point a gives the label # of the track that connects
        # to the end of the indexed track. b gives the label # of the
        # track that connects to the start of the indexed track.
        # We convert these into edges.
        #
        # aa and bb are the vertices of an edge list and aa[i],bb[i]
        # make up an edge
        #
        connect_mask = (a != -1)
        aa = a[connect_mask]
        bb = np.argwhere(connect_mask).flatten()
        connect_mask = (b != -1)
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
        for i, image_number in enumerate(indexes):
            orig_label[image_number] = newlabel[i]

        self.recalculate_group(workspace, indexes)
        
    def recalculate_group(self, workspace, indexes):
        '''Recalculate all measurements once post_group has run
        
        workspace - the workspace being operated on
        indexes - the indexes of the group's image sets' measurements
        '''
        m = workspace.measurements
        object_name = self.object_name.value
        assert isinstance(m, cpmeas.Measurements)
        parent_object_numbers = m.get_all_measurements(
            object_name, self.measurement_name(F_PARENT_OBJECT_NUMBER))
        parent_image_numbers = m.get_all_measurements(
            object_name, self.measurement_name(F_PARENT_GROUP_INDEX))
        group_indexes = m.get_all_measurements( cpmeas.IMAGE,
                                                cpp.GROUP_INDEX)
        max_object_count = np.max([len(x) for x in parent_object_numbers])
        if max_object_count == 0:
            return
        max_image_number = np.max(m.get_image_number_from_index(indexes))
        def w(a):
            '''Wrap a measurement array as a numpy sparse array
            
            Elements in the array can be addressed correctly by image number
            and object number.
            '''
            result = scipy.sparse.lil_matrix((max_image_number+1, 
                                              max_object_count+2), 
                                             dtype = a[0].dtype)
            for index in indexes:
                row = a[index]
                nobjects = len(row)
                if nobjects > 0:
                    group_index = group_indexes[index]
                    result[group_index, 1:(len(row)+1)] = row
            return result
            
        #
        # Recalculate the trajectories
        #
        old_dists = m.get_all_measurements(
            object_name, self.measurement_name(F_DISTANCE_TRAVELED))
        old_integrated = m.get_all_measurements(
            object_name, self.measurement_name(F_INTEGRATED_DISTANCE))
        w_integrated = w(old_integrated)
        x = w(m.get_all_measurements(object_name, M_LOCATION_CENTER_X))
        y = w(m.get_all_measurements(object_name, M_LOCATION_CENTER_Y))
        old_trajectory_x = m.get_all_measurements(
            object_name, self.measurement_name(F_TRAJECTORY_X))
        old_trajectory_y = m.get_all_measurements(
            object_name, self.measurement_name(F_TRAJECTORY_Y))
        old_linearity = m.get_all_measurements(
            object_name, self.measurement_name(F_LINEARITY))
        w_linearity = w(old_linearity)
        old_lifetime = m.get_all_measurements(
            object_name, self.measurement_name(F_LIFETIME))
        w_lifetime = w(old_lifetime)
        #
        # Maintain the first position for a lineage in first_x and first_y
        #
        first_x = scipy.sparse.lil_matrix((max_image_number+1,
                                           max_object_count+2), 
                                          dtype = float)
        first_y = scipy.sparse.lil_matrix((max_image_number+1,
                                           max_object_count+2),
                                          dtype = float)
        lifetime = scipy.sparse.lil_matrix((max_image_number+1,
                                            max_object_count+2), 
                                           dtype = float)
        for index in indexes:
            po = parent_object_numbers[index]
            pi = parent_image_numbers[index]
            has_parent = (po != 0)
            pi = pi[has_parent]
            po = po[has_parent]
            
            group_index = group_indexes[index]
            nobjects = len(po)

            has_parent_objnum = np.argwhere(has_parent).flatten() + 1
            has_no_parent_objnum = np.argwhere(~ has_parent).flatten() + 1
            #
            # Parent = 0 indicates first appearance in the lineage
            #            otherwise copy from last parent.
            #
            if len(has_no_parent_objnum) > 0:
                first_x[group_index, has_no_parent_objnum] = \
                       x[group_index, has_no_parent_objnum]
                first_y[group_index, has_no_parent_objnum] = \
                       y[group_index, has_no_parent_objnum]
            if len(has_parent_objnum) == 0:
                continue
            this_first_x = first_x[pi, po]
            first_x[group_index, has_parent_objnum] = this_first_x
            this_first_y = first_y[pi, po]
            first_y[group_index, has_parent_objnum] = this_first_y
            #
            # Distances traveled from step to step
            #
            this_x = x[group_index, has_parent_objnum].toarray()[0]
            this_y = y[group_index, has_parent_objnum].toarray()[0]
            last_x = x[pi, po].toarray()[0]
            last_y = y[pi, po].toarray()[0]
            x_diff = this_x - last_x
            y_diff = this_y - last_y
            otx = old_trajectory_x[index]
            otx[~ has_parent] = 0
            otx[has_parent] = x_diff
            oty = old_trajectory_y[index]
            oty[~ has_parent] = 0
            oty[has_parent] = y_diff
            #
            # Integrated distance = accumulated distance for lineage
            #
            oid = old_integrated[index]
            lid = w_integrated[pi, po].toarray().flatten()
            oid[~ has_parent] = 0
            integrated_distance = np.sqrt(x_diff * x_diff + y_diff * y_diff) + lid
            oid[has_parent] = integrated_distance
            w_integrated[group_index, has_parent_objnum] = integrated_distance
            #
            # Total distance = crow-fly distance from initial ancestor
            #
            x_tot_diff = this_x - this_first_x.toarray()[0]
            y_tot_diff = this_y - this_first_y.toarray()[0]
            tot_distance = np.sqrt(x_tot_diff * x_tot_diff + 
                                   y_tot_diff * y_tot_diff)
            old_dist = old_dists[index]
            old_dist[~ has_parent] = 0
            old_dist[has_parent] = tot_distance
            #
            # Linearity = ratio of crow-fly distance and integrated
            # distance. NaN for new cells is ok.
            #
            linearity = old_linearity[index]
            linearity[~ has_parent] = np.NaN
            linearity[has_parent] = tot_distance / integrated_distance
            #
            # Add 1 to lifetimes / zero for new
            #
            lifetime = old_lifetime[index]
            lifetime[~ has_parent] = 0
            this_lifetime = w_lifetime[pi, po].toarray()[0] + 1
            lifetime[has_parent] = this_lifetime
            w_lifetime[group_index, has_parent_objnum] = this_lifetime
        
    def map_objects(self, workspace, new_of_old, old_of_new, i,j):
        '''Record the mapping of old to new objects and vice-versa

        workspace - workspace for current image set
        new_to_old - an array of the new labels for every old label
        old_to_new - an array of the old labels for every new label
        score - a score that is higher for better mappings: we use this
                score to assign new label numbers so that the new objects
                that are "better" inherit the old objects' label number
        '''
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        group_index = m.get_current_image_measurement(cpp.GROUP_INDEX)
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
        parents[parents != 0] =\
               old_object_numbers[(old_of_new[parents!=0]-1)].astype(parents.dtype)
        self.add_measurement(workspace, F_PARENT_OBJECT_NUMBER, old_of_new)
        parent_group_indexes = np.zeros(len(old_of_new))
        parent_group_indexes[parents != 0] = group_index-1
        self.add_measurement(workspace, F_PARENT_GROUP_INDEX, parent_group_indexes)
        #
        # Assign object IDs to the new objects
        #
        mapping = np.zeros(new_count, int)
        if old_count > 0 and new_count > 0:
            mapping[old_of_new != 0] = \
                   old_object_numbers[old_of_new[old_of_new != 0] - 1]
            miss_count = np.sum(old_of_new == 0)
            lost_object_count = np.sum(new_of_old == 0)
        else:
            miss_count = new_count
            lost_object_count = old_count
        nunmapped = np.sum(mapping==0)
        new_max_object_number = max_object_number + nunmapped
        mapping[mapping == 0] = np.arange(max_object_number+1,
                                          new_max_object_number + 1)
        self.set_max_object_number(workspace, new_max_object_number)
        self.add_measurement(workspace, F_LABEL, mapping)
        self.set_saved_object_numbers(workspace, mapping)
        #
        # Compute distances and trajectories
        #
        diff_i = np.zeros(new_count)
        diff_j = np.zeros(new_count)
        distance = np.zeros(new_count)
        idistance = np.zeros(new_count)
        odistance = np.zeros(new_count)
        distance = np.zeros(new_count)
        linearity = np.ones(new_count)
        orig_i = i.copy()
        orig_j = j.copy()
        old_i, old_j = self.get_saved_coordinates(workspace)
        old_distance = self.get_saved_distances(workspace)
        old_orig_i, old_orig_j = self.get_orig_coordinates(workspace)
        has_old = (old_of_new != 0)
        if np.any(has_old):
            old_indexes = old_of_new[has_old]-1
            orig_i[has_old] = old_orig_i[old_indexes]
            orig_j[has_old] = old_orig_j[old_indexes]
            diff_i[has_old] = i[has_old] - old_i[old_indexes]
            diff_j[has_old] = j[has_old] - old_j[old_indexes]
            distance[has_old] = np.sqrt(diff_i[has_old]**2 + diff_j[has_old]**2)

            idistance[has_old] = (old_distance[old_indexes] + 
                                  distance[has_old])
            odistance = np.sqrt((i-orig_i)**2 + (j-orig_j)**2)
            linearity[has_old] = odistance[has_old] / idistance[has_old]
        self.add_measurement(workspace, F_TRAJECTORY_X, diff_j)
        self.add_measurement(workspace, F_TRAJECTORY_Y, diff_i)
        self.add_measurement(workspace, F_DISTANCE_TRAVELED, distance)
        self.add_measurement(workspace, F_INTEGRATED_DISTANCE, idistance)
        self.add_measurement(workspace, F_LINEARITY, linearity)
        self.set_saved_distances(workspace, idistance)
        self.set_orig_coordinates(workspace, (orig_i, orig_j))
        self.set_saved_coordinates(workspace, (i,j))
        #
        # Update the ages
        #
        age = np.zeros(new_count, int)
        if np.any(has_old):
            old_age = self.get_saved_ages(workspace)
            age[has_old] = old_age[old_of_new[has_old]-1]+1
        self.add_measurement(workspace, F_LIFETIME, age)
        self.set_saved_ages(workspace, age)
        self.set_saved_object_numbers(workspace, mapping)
        #
        # Add image measurements
        #
        self.add_image_measurement(workspace, F_NEW_OBJECT_COUNT, 
                                   np.sum(parents==0))
        self.add_image_measurement(workspace, F_LOST_OBJECT_COUNT, 
                                   lost_object_count)
        #
        # Find parents with more than one child. These are the progenetors
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

    def get_kalman_feature_names(self):
        if self.tracking_method != TM_LAP:
            return []
        return sum(
            [sum(
                [[ kalman_feature(model, F_STATE, element),
                   kalman_feature(model, F_NOISE, element)] + 
                 [ kalman_feature(model, F_COV, element, e2)
                   for e2 in elements]
                 for element in elements],[])
             for model, elements  in self.get_kalman_models()], [])
    
    def get_measurement_columns(self, pipeline):
        result =  [(self.object_name.value,
                    self.measurement_name(feature),
                    coltype)
                   for feature, coltype in F_ALL_COLTYPE_ALL]
        result += [(cpmeas.IMAGE, self.image_measurement_name(feature), coltype)
                   for feature, coltype in F_IMAGE_COLTYPE_ALL]
        if self.tracking_method == TM_LAP:
            result += [( self.object_name.value,
                         self.measurement_name(F_AREA),
                         cpmeas.COLTYPE_INTEGER)]
            result += [( self.object_name.value,
                         self.measurement_name(name),
                         coltype) for name in self.get_kalman_feature_names()]
            if self.wants_second_phase:
                # Add the post-group attribute to all measurements
                attributes = { cpmeas.MCA_AVAILABLE_POST_GROUP: True }
                result = [ ( c[0], c[1], c[2], attributes) for c in result]
        return result

    def get_categories(self, pipeline, object_name):
        if object_name in (self.object_name.value, cpmeas.IMAGE):
            return [F_PREFIX]
        else:
            return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == self.object_name.value and category == F_PREFIX:
            result = list(F_ALL)
            if self.tracking_method == TM_LAP:
                result += [F_AREA]
                result += self.get_kalman_feature_names()
            return result
        if object_name == cpmeas.IMAGE:
            result = F_IMAGE_ALL
            return result
        return []

    def get_measurement_objects(self, pipeline, object_name, category, 
                                measurement):
        if (object_name == cpmeas.IMAGE and category == F_PREFIX and
            measurement in F_IMAGE_ALL):
            return [ self.object_name.value]
        return []
        
    def get_measurement_scales(self, pipeline, object_name, category, feature,image_name):
        if self.tracking_method == TM_LAP:
            return []
        
        if feature in self.get_measurements(pipeline, object_name, category):
            return [str(self.pixel_radius.value)]
        return []

    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
        if from_matlab and variable_revision_number == 3:
            wants_image = setting_values[10] != cps.DO_NOT_USE
            measurement =  '_'.join(setting_values[2:6])
            setting_values = [ setting_values[0], # tracking method
                               setting_values[1], # object name
                               measurement,
                               setting_values[6], # pixel_radius
                               setting_values[7], # display_type
                               wants_image,
                               setting_values[10]]
            variable_revision_number = 1
            from_matlab = False
        if (not from_matlab) and variable_revision_number == 1:
            setting_values = setting_values + ["100","100"]
            variable_revision_number = 2
        if (not from_matlab) and variable_revision_number == 2:
            # Added phase 2 parameters
            setting_values = setting_values + [
                "40","40","40","50","50","50","5"]
            variable_revision_number = 3
        if (not from_matlab) and variable_revision_number == 3:
            # Added Kalman choices:
            # Model
            # radius std
            # radius limit
            setting_values = (setting_values[:7] + 
                              [ M_BOTH, "3", "2,10"] +
                              setting_values[9:])
        return setting_values, variable_revision_number, from_matlab


