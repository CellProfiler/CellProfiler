import cellprofiler.icons
from cellprofiler.gui.help import PROTIP_RECOMEND_ICON, PROTIP_AVOID_ICON, TECH_NOTE_ICON

__doc__ = '''<b>Identify Secondary Objects</b> identifies objects (e.g., cell edges) using
objects identified by another module (e.g., nuclei) as a starting point.
<hr>
<h4>What is a secondary object?</h4>
In CellProfiler, we use the term <i>object</i> as a generic term to refer to an identifed
feature in an image, usually a cellular subcompartment of some kind (for example,
nuclei, cells, colonies, worms). We define an object as <i>secondary</i> when it can be
found in an image by using another cellular feature as a reference for guiding detection.

<p>For densely-packed cells
(such as those in a confluent monolayer), determining the cell borders using a cell body stain
can be quite difficult since they often have irregular intensity patterns and are lower-contrast
with more diffuse staining. In addition, cells often touch their neighbors making it harder to
delineate the cell borders. It is often easier to identify an organelle which is well separated
spatially (such as the nucleus) as an object first and then use that object to guide the detection
of the cell borders. See the <b>IdentifyPrimaryObjects</b> module for details on how to identify
a primary object.</p>

In order to identify the edges of secondary objects, this module performs two tasks:
<ol>
<li>Finds the dividing lines between secondary objects which touch each other.</li>
<li>Finds the dividing lines between the secondary objects and the
background of the image. In most cases, this is done by thresholding the image stained
for the secondary objects.</li>
</ol>

<h4>What do I need as input?</h4>
This module identifies secondary objects based on two types of input:
<ol>
<li>An <i>object</i> (e.g., nuclei) identified from a prior module. These are typically produced
by an <b>IdentifyPrimaryObjects</b> module, but any object produced by another module may be
selected for this purpose.</li>
<li>An <i>image</i> highlighting the image features defining the cell edges. This is typically
a fluorescent stain for the cell body, membrane or cytoskeleton (e.g., phalloidin staining for actin).
However, any image which produces these features can be used for this purpose. For example, an image
processing module might be used to transform a brightfield image into one which captures the characteristics of a cell
body flourescent stain.</li>
</ol>

<h4>What do the settings mean?</h4>
See below for help on the individual settings. The following icons are used to call attention to
key items:
<ul>
<li><img src="memory:%(PROTIP_RECOMEND_ICON)s">&nbsp;Our recommendation or example use case
for which a particular setting is best used.</li>
<li><img src="memory:%(PROTIP_AVOID_ICON)s">&nbsp;Indicates a condition under which
a particular setting may not work well.</li>
<li><img src="memory:%(TECH_NOTE_ICON)s">&nbsp;Technical note. Provides more
detailed information on the setting, if interested.</li>
</ul>

<h4>What do I get as output?</h4>
A set of secondary objects are produced by this module, which can be used in downstream modules
for measurement purposes or other operations. Because each primary object is used as the starting point
for producing a corresponding secondary object, keep in mind the following points:
<ul>
<li>The primary object will always be completely contained
within a secondary object. For example, nuclei are completely enclosed within identified cells
stained for actin.</li>
<li>There will always be at most one secondary object for each primary object.</li>
</ul>
See the section <a href="#Available_measurements">"Available measurements"</a> below for
the measurements that are produced by this module.

Once the module has finished processing, the module display window
will show the following panels:
<ul>
<li><i>Upper left:</i> The raw, original image.</li>
<li><i>Upper right:</i> The identified objects shown as a color
image where connected pixels that belong to the same object are assigned the same
color (<i>label image</i>). It is important to note that assigned colors are
arbitrary; they are used simply to help you distingush the various objects. </li>
<li><i>Lower left:</i> The raw image overlaid with the colored outlines of the
identified secondary objects. The objects are shown with the following colors:
<ul>
<li>Magenta: Secondary objects</li>
<li>Green: Primary objects</li>
</ul>
If you need to change the color defaults, you can
make adjustments in <i>File > Preferences</i>.</li>
<li><i>Lower right:</i> A table showing some of the settings selected by the user, as well as
those calculated by the module in order to produce the objects shown.</li>
</ul>


<a name="Available_measurements">
<h4>Available measurements</h4>
<b>Image measurements:</b>
<ul>
<li><i>Count:</i> The number of secondary objects identified.</li>
<li><i>OriginalThreshold:</i> The global threshold for the image.</li>
<li><i>FinalThreshold:</i> For the global threshold methods, this value is the
same as <i>OriginalThreshold</i>. For the adaptive or per-object methods, this
value is the mean of the local thresholds.</li>
<li><i>WeightedVariance:</i> The sum of the log-transformed variances of the
foreground and background pixels, weighted by the number of pixels in
each distribution.</li>
<li><i>SumOfEntropies:</i> The sum of entropies computed from the foreground and
background distributions.</li>
</ul>

<b>Object measurements:</b>
<ul>
<li><i>Parent:</i> The identity of the primary object associated with each secondary
object.</li>
<li><i>Location_X, Location_Y:</i> The pixel (X,Y) coordinates of the center of
mass of the identified secondary objects.</li>
</ul>

<h4>Technical notes</h4>
The <i>Propagation</i> algorithm is the default approach for secondary object creation,
creating each primary object as a "seed" guided by the input image and limited to the
foreground region as determined by the chosen thresholding method. &lambda; is
a regularization parameter; see the help for the setting for more details. Propagation
of secondary object labels is by the shortest path to an adjacent primary object
from the starting ("seeding") primary object. The seed-to-pixel distances are
calculated as the sum of absolute differences in a 3x3 (8-connected) image
neighborhood, combined with &lambda; via sqrt(differences<sup>2</sup> + &lambda;<sup>2</sup>).

<p>See also the other <b>Identify</b> modules.</p>
''' % globals()

import numpy as np
import os
import scipy.ndimage as scind
import scipy.misc as scimisc

import cellprofiler.module as cpm
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.preferences as cpprefs
import cellprofiler.workspace as cpw
import cellprofiler.setting as cps
from cellprofiler.setting import YES, NO
import identify as cpmi
from identify import FI_IMAGE_SIZE
import centrosome.threshold as cpthresh
import centrosome.otsu
from centrosome.propagate import propagate
from centrosome.cpmorphology import fill_labeled_holes
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
import skimage.morphology.watershed
from centrosome.filter import stretch
from centrosome.outline import outline
from cellprofiler.gui.help import RETAINING_OUTLINES_HELP, NAMING_OUTLINES_HELP

M_PROPAGATION = "Propagation"
M_WATERSHED_G = "Watershed - Gradient"
M_WATERSHED_I = "Watershed - Image"
M_DISTANCE_N = "Distance - N"
M_DISTANCE_B = "Distance - B"

'''# of setting values other than thresholding ones'''
N_SETTING_VALUES = 14

'''Parent (seed) relationship of input objects to output objects'''
R_PARENT = "Parent"


class IdentifySecondaryObjects(cpmi.Identify):
    module_name = "IdentifySecondaryObjects"
    variable_revision_number = 9
    category = "Object Processing"

    def create_settings(self):
        self.primary_objects = cps.ObjectNameSubscriber(
                "Select the input objects", "Nuclei", doc="""
            What did you call the objects you want to use as "seeds" to identify a secondary
            object around each one? By definition, each primary object must be associated with exactly one
            secondary object and completely contained within it.""")

        self.objects_name = cps.ObjectNameProvider(
                "Name the objects to be identified", "Cells", doc="""
            Enter the name that you want to call the objects identified by this module.""")

        self.method = cps.Choice(
                "Select the method to identify the secondary objects",
                [M_PROPAGATION, M_WATERSHED_G, M_WATERSHED_I, M_DISTANCE_N, M_DISTANCE_B],
                M_PROPAGATION, doc="""
            <p>There are several methods available to find the dividing lines
            between secondary objects which touch each other:
            <ul>
            <li><i>%(M_PROPAGATION)s:</i> This method will find dividing lines
            between clumped objects where the image stained for secondary objects
            shows a change in staining (i.e., either a dimmer or a brighter line).
            Smoother lines work better, but unlike the Watershed method, small gaps
            are tolerated. This method is considered an improvement on the
            traditional <i>Watershed</i> method. The dividing lines between objects are
            determined by a combination of the distance to the nearest primary object
            and intensity gradients. This algorithm uses local image similarity to
            guide the location of boundaries between cells. Boundaries are
            preferentially placed where the image's local appearance changes
            perpendicularly to the boundary (<i>Jones et al, 2005</i>).</li>

            <li><i>%(M_WATERSHED_G)s:</i> This method uses the watershed algorithm
            (<i>Vincent and Soille, 1991</i>) to assign
            pixels to the primary objects which act as seeds for the watershed.
            In this variant, the watershed algorithm operates on the Sobel
            transformed image which computes an intensity gradient. This method
            works best when the image intensity drops off or increases rapidly
            near the boundary between cells.
            </li>
            <li><i>%(M_WATERSHED_I)s:</i> This method is similar to the above,
            but it uses the inverted intensity of the image for the watershed.
            The areas of lowest intensity will form the boundaries between
            cells. This method works best when there is a saddle of relatively
            low intensity at the cell-cell boundary.
            </li>
            <li><i>Distance:</i> In this method, the edges of the primary
            objects are expanded a specified distance to create the secondary
            objects. For example, if nuclei are labeled but there is no stain to help
            locate cell edges, the nuclei can simply be expanded in order to estimate
            the cell's location. This is often called the "doughnut" or "annulus" or
            "ring" approach for identifying the cytoplasm.
            There are two methods that can be used:
            <ul>
            <li><i>%(M_DISTANCE_N)s</i>: In this method, the image of the secondary
            staining is not used at all; the expanded objects are the
            final secondary objects.</li>
            <li><i>%(M_DISTANCE_B)s</i>: Thresholding of the secondary staining image is used to eliminate background
            regions from the secondary objects. This allows the extent of the
            secondary objects to be limited to a certain distance away from the edge
            of the primary objects without including regions of background.</li></ul></li>
            </ul>
            <b>References</b>
            <ul>
            <li>Jones TR, Carpenter AE, Golland P (2005) "Voronoi-Based Segmentation of Cells on Image Manifolds",
            <i>ICCV Workshop on Computer Vision for Biomedical Image Applications</i>, 535-543.
            (<a href="http://www.cellprofiler.org/linked_files/Papers/JonesCVBIA2005.pdf">link</a>)</li>
            <li>(Vincent L, Soille P (1991) "Watersheds in Digital Spaces: An Efficient Algorithm Based on Immersion
            Simulations", <i>IEEE Transactions of Pattern Analysis and Machine
            Intelligence</i>, 13(6): 583-598
            (<a href="http://dx.doi.org/10.1109/34.87344">link</a>)</li>
            </ul>""" % globals())

        self.image_name = cps.ImageNameSubscriber(
                "Select the input image",
                cps.NONE, doc="""
            The selected image will be used to find the edges of the secondary objects.
            For <i>%(M_DISTANCE_N)s</i> this will not affect object identification,
            only the final display.""" % globals())

        self.create_threshold_settings()

        self.distance_to_dilate = cps.Integer(
                "Number of pixels by which to expand the primary objects", 10, minval=1)

        self.regularization_factor = cps.Float(
                "Regularization factor", 0.05, minval=0, doc="""
            <i>(Used only if %(M_PROPAGATION)s method is selected)</i> <br>
            The regularization factor &lambda; can be anywhere in the range 0 to infinity.
            This method takes two factors into account when deciding where to draw
            the dividing line between two touching secondary objects: the distance to
            the nearest primary object, and the intensity of the secondary object
            image. The regularization factor controls the balance between these two
            considerations:
            <ul>
            <li>A &lambda; value of 0 means that the distance to the nearest
            primary object is ignored and the decision is made entirely on the
            intensity gradient between the two competing primary objects. </li>
            <li>Larger values of &lambda; put more and more weight on the distance between the two objects.
            This relationship is such that small changes in &lambda; will have fairly different
            results (e.,g 0.01 vs 0.001). However, the intensity image is almost completely
            ignored at &lambda; much greater than 1.</li>
            <li>At infinity, the result will look like %(M_DISTANCE_B)s, masked to the
            secondary staining image.</li>
            </ul>""" % globals())

        self.use_outlines = cps.Binary(
                "Retain outlines of the identified secondary objects?", False, doc="""
            %(RETAINING_OUTLINES_HELP)s""" % globals())

        self.outlines_name = cps.OutlineNameProvider(
                'Name the outline image', "SecondaryOutlines", doc="""
            %(NAMING_OUTLINES_HELP)s""" % globals())

        self.wants_discard_edge = cps.Binary(
                "Discard secondary objects touching the border of the image?",
                False, doc="""
            Select <i>%(YES)s</i> to discard secondary objects which touch
            the image border. Select <i>%(NO)s</i> to retain objects regardless
            of whether they touch the image edge or not.
            <p>The objects are discarded
            with respect to downstream measurement modules, but they are retained in memory
            as "unedited objects"; this allows them to be considered in downstream modules that modify the
            segmentation.</p>""" % globals())

        self.fill_holes = cps.Binary(
                "Fill holes in identified objects?", True, doc="""
            Select <i>%(YES)s</i> to fill any holes inside objects.""" % globals())

        self.wants_discard_primary = cps.Binary(
                "Discard the associated primary objects?", False, doc="""
            <i>(Used only if discarding secondary objects touching the image border)</i> <br>
            It might be appropriate to discard the primary object
            for any secondary object that touches the edge of the image.
            <p>Select <i>%(YES)s</i> to create a new set of objects that are identical
            to the original primary objects set, minus the objects for which the associated
            secondary object touches the image edge.</p>""" % globals())

        self.new_primary_objects_name = cps.ObjectNameProvider(
                "Name the new primary objects", "FilteredNuclei", doc="""
            <i>(Used only if associated primary objects are discarded)</i> <br>
            You can name the primary objects that remain after the discarding step.
            These objects will all have secondary objects
            that do not touch the edge of the image. Note that any primary object
            whose secondary object touches the edge will be retained in memory as an
            "unedited object"; this allows them to be considered in downstream modules that modify the
            segmentation.""")

        self.wants_primary_outlines = cps.Binary(
                "Retain outlines of the new primary objects?", False, doc="""
            <i>(Used only if associated primary objects are discarded)</i><br>
            %(RETAINING_OUTLINES_HELP)s""" % globals())

        self.new_primary_outlines_name = cps.OutlineNameProvider(
                "Name the new primary object outlines", "FilteredNucleiOutlines", doc="""
            <i>(Used only if associated primary objects are discarded and saving outlines of new primary objects)</i><br>
            Enter a name for the outlines of the identified
            objects. The outlined image can be selected in downstream modules by selecting
            them from any drop-down image list.""")

    def settings(self):
        return [
                   self.primary_objects, self.objects_name, self.method, self.image_name,
                   self.distance_to_dilate, self.regularization_factor, self.outlines_name,
                   self.use_outlines, self.wants_discard_edge, self.wants_discard_primary,
                   self.new_primary_objects_name, self.wants_primary_outlines,
                   self.new_primary_outlines_name, self.fill_holes] + \
               self.get_threshold_settings()

    def visible_settings(self):
        result = [self.image_name, self.primary_objects, self.objects_name,
                  self.method]
        if self.method != M_DISTANCE_N:
            result += self.get_threshold_visible_settings()
        if self.method in (M_DISTANCE_B, M_DISTANCE_N):
            result.append(self.distance_to_dilate)
        elif self.method == M_PROPAGATION:
            result.append(self.regularization_factor)
        result += [self.fill_holes, self.wants_discard_edge]
        if self.wants_discard_edge:
            result.append(self.wants_discard_primary)
            if self.wants_discard_primary:
                result += [self.new_primary_objects_name,
                           self.wants_primary_outlines]
                if self.wants_primary_outlines:
                    result.append(self.new_primary_outlines_name)
        result.append(self.use_outlines)
        if self.use_outlines.value:
            result.append(self.outlines_name)
        return result

    def help_settings(self):
        return [self.primary_objects, self.objects_name,
                self.method, self.image_name] + \
               self.get_threshold_visible_settings() + \
               [self.distance_to_dilate,
                self.regularization_factor,
                self.fill_holes, self.wants_discard_edge, self.wants_discard_primary,
                self.new_primary_objects_name, self.wants_primary_outlines,
                self.new_primary_outlines_name,
                self.use_outlines, self.outlines_name]

    def upgrade_settings(self,
                         setting_values,
                         variable_revision_number,
                         module_name,
                         from_matlab):
        if from_matlab and variable_revision_number == 1:
            NotImplementedError("Sorry, Matlab variable revision # 1 is not supported")
        if from_matlab and variable_revision_number == 2:
            # Added test mode - default = no
            setting_values = list(setting_values)
            setting_values.append(cps.NO)
            variable_revision_number = 3
        if from_matlab and variable_revision_number == 3:
            new_setting_values = list(setting_values)
            if setting_values[4].isdigit():
                # User entered manual threshold
                new_setting_values[4] = cpthresh.TM_MANUAL
                new_setting_values.append(setting_values[4])
                new_setting_values.append(cps.DO_NOT_USE)
            elif (not setting_values[4] in
                (cpthresh.TM_OTSU_GLOBAL, cpthresh.TM_OTSU_ADAPTIVE,
                 cpthresh.TM_OTSU_PER_OBJECT, cpthresh.TM_MOG_GLOBAL,
                 cpthresh.TM_MOG_ADAPTIVE, cpthresh.TM_MOG_PER_OBJECT,
                 cpthresh.TM_BACKGROUND_GLOBAL, cpthresh.TM_BACKGROUND_ADAPTIVE,
                 cpthresh.TM_BACKGROUND_PER_OBJECT, cpthresh.TM_ROBUST_BACKGROUND,
                 cpthresh.TM_ROBUST_BACKGROUND_GLOBAL,
                 cpthresh.TM_ROBUST_BACKGROUND_ADAPTIVE,
                 cpthresh.TM_ROBUST_BACKGROUND_PER_OBJECT,
                 cpthresh.TM_RIDLER_CALVARD_GLOBAL, cpthresh.TM_RIDLER_CALVARD_ADAPTIVE,
                 cpthresh.TM_RIDLER_CALVARD_PER_OBJECT, cpthresh.TM_KAPUR_GLOBAL,
                 cpthresh.TM_KAPUR_ADAPTIVE, cpthresh.TM_KAPUR_PER_OBJECT)):
                # User entered an image name -  guess
                new_setting_values[4] = cpthresh.TM_BINARY_IMAGE
                new_setting_values.append('0')
                new_setting_values.append(setting_values[4])
            else:
                new_setting_values.append('0')
                new_setting_values.append(cps.DO_NOT_USE)
            if setting_values[10] == cps.DO_NOT_USE:
                new_setting_values.append(cps.NO)
            else:
                new_setting_values.append(cps.YES)
            setting_values = new_setting_values
            from_matlab = False
            variable_revision_number = 1
        if from_matlab:
            NotImplementedError(
                    "Don't know how to convert Matlab IdentifySecondary revision # %d" % variable_revision_number)
        if variable_revision_number != self.variable_revision_number:
            NotImplementedError("Don't know how to handle IdentifySecondary revision # %d" % variable_revision_number)
        if (not from_matlab) and variable_revision_number == 1:
            # Removed test mode
            # added Otsu parameters.
            setting_values = setting_values[:11] + setting_values[12:]
            setting_values += [cpmi.O_TWO_CLASS, cpmi.O_WEIGHTED_VARIANCE,
                               cpmi.O_FOREGROUND]
            variable_revision_number = 2

        if (not from_matlab) and variable_revision_number == 2:
            # Added discarding touching
            setting_values = setting_values + [cps.NO, cps.NO, "FilteredNuclei"]
            variable_revision_number = 3

        if (not from_matlab) and variable_revision_number == 3:
            # Added new primary outlines
            setting_values = setting_values + [cps.NO, "FilteredNucleiOutlines"]
            variable_revision_number = 4

        if (not from_matlab) and variable_revision_number == 4:
            # Added measurements to threshold methods
            setting_values = setting_values + [cps.NONE]
            variable_revision_number = 5

        if (not from_matlab) and variable_revision_number == 5:
            # Change name of watershed option
            if setting_values[2] == "Watershed":
                setting_values[2] = M_WATERSHED_G
            variable_revision_number = 6

        if (not from_matlab) and variable_revision_number == 6:
            # Fill labeled holes added
            fill_holes = (cps.NO
                          if setting_values[2] in (M_DISTANCE_B, M_DISTANCE_N)
                          else cps.YES)
            setting_values = setting_values + [fill_holes]
            variable_revision_number = 7

        if (not from_matlab) and variable_revision_number == 7:
            # Added adaptive thresholding settings
            setting_values += [FI_IMAGE_SIZE, "10"]
            variable_revision_number = 8

        if (not from_matlab) and variable_revision_number == 8:
            primary_objects, objects_name, method, image_name, \
            threshold_method, threshold_correction_factor, \
            threshold_range, object_fraction, distance_to_dilate, \
            regularization_factor, outlines_name, manual_threshold, \
            binary_image, use_outlines, two_class_otsu, \
            use_weighted_variance, assign_middle_to_foreground, \
            wants_discard_edge, wants_discard_primary, \
            new_primary_objects_name, wants_primary_outlines, \
            new_primary_outlines_name, thresholding_measurement, \
            fill_holes, adaptive_window_method, \
            adaptive_window_size = setting_values
            setting_values = [
                                 primary_objects, objects_name, method, image_name,
                                 distance_to_dilate, regularization_factor, outlines_name,
                                 use_outlines, wants_discard_edge, wants_discard_primary,
                                 new_primary_objects_name, wants_primary_outlines,
                                 new_primary_outlines_name, fill_holes] + \
                             self.upgrade_legacy_threshold_settings(
                                     threshold_method, cpmi.TSM_NONE,
                                     threshold_correction_factor, threshold_range,
                                     object_fraction, manual_threshold, thresholding_measurement,
                                     binary_image, two_class_otsu, use_weighted_variance,
                                     assign_middle_to_foreground, adaptive_window_method,
                                     adaptive_window_size)
            variable_revision_number = 9
        setting_values = setting_values[:N_SETTING_VALUES] + \
                         self.upgrade_threshold_settings(setting_values[N_SETTING_VALUES:])
        return setting_values, variable_revision_number, from_matlab

    def run(self, workspace):
        assert isinstance(workspace, cpw.Workspace)
        image_name = self.image_name.value
        image = workspace.image_set.get_image(image_name,
                                              must_be_grayscale=True)
        workspace.display_data.statistics = []
        img = image.pixel_data
        mask = image.mask
        objects = workspace.object_set.get_objects(self.primary_objects.value)
        global_threshold = None
        if self.method == M_DISTANCE_N:
            has_threshold = False
        else:
            thresholded_image = self.threshold_image(image_name, workspace)
            has_threshold = True

        #
        # Get the following labels:
        # * all edited labels
        # * labels touching the edge, including small removed
        #
        labels_in = objects.unedited_segmented.copy()
        labels_touching_edge = np.hstack(
                (labels_in[0, :], labels_in[-1, :], labels_in[:, 0], labels_in[:, -1]))
        labels_touching_edge = np.unique(labels_touching_edge)
        is_touching = np.zeros(np.max(labels_in) + 1, bool)
        is_touching[labels_touching_edge] = True
        is_touching = is_touching[labels_in]

        labels_in[(~ is_touching) & (objects.segmented == 0)] = 0
        #
        # Stretch the input labels to match the image size. If there's no
        # label matrix, then there's no label in that area.
        #
        if tuple(labels_in.shape) != tuple(img.shape):
            tmp = np.zeros(img.shape, labels_in.dtype)
            i_max = min(img.shape[0], labels_in.shape[0])
            j_max = min(img.shape[1], labels_in.shape[1])
            tmp[:i_max, :j_max] = labels_in[:i_max, :j_max]
            labels_in = tmp

        if self.method in (M_DISTANCE_B, M_DISTANCE_N):
            if self.method == M_DISTANCE_N:
                distances, (i, j) = scind.distance_transform_edt(labels_in == 0,
                                                                 return_indices=True)
                labels_out = np.zeros(labels_in.shape, int)
                dilate_mask = distances <= self.distance_to_dilate.value
                labels_out[dilate_mask] = \
                    labels_in[i[dilate_mask], j[dilate_mask]]
            else:
                labels_out, distances = propagate(img, labels_in,
                                                  thresholded_image,
                                                  1.0)
                labels_out[distances > self.distance_to_dilate.value] = 0
                labels_out[labels_in > 0] = labels_in[labels_in > 0]
            if self.fill_holes:
                small_removed_segmented_out = fill_labeled_holes(labels_out)
            else:
                small_removed_segmented_out = labels_out
            #
            # Create the final output labels by removing labels in the
            # output matrix that are missing from the segmented image
            #
            segmented_labels = objects.segmented
            segmented_out = self.filter_labels(small_removed_segmented_out,
                                               objects, workspace)
        elif self.method == M_PROPAGATION:
            labels_out, distance = propagate(img, labels_in,
                                             thresholded_image,
                                             self.regularization_factor.value)
            if self.fill_holes:
                small_removed_segmented_out = fill_labeled_holes(labels_out)
            else:
                small_removed_segmented_out = labels_out.copy()
            segmented_out = self.filter_labels(small_removed_segmented_out,
                                               objects, workspace)
        elif self.method == M_WATERSHED_G:
            #
            # First, apply the sobel filter to the image (both horizontal
            # and vertical). The filter measures gradient.
            #
            sobel_image = np.abs(scind.sobel(img))
            #
            # Combine the image mask and threshold to mask the watershed
            #
            watershed_mask = np.logical_or(thresholded_image, labels_in > 0)
            watershed_mask = np.logical_and(watershed_mask, mask)

            #
            # Perform the first watershed
            #

            labels_out = skimage.morphology.watershed(
                connectivity=np.ones((3, 3), bool),
                image=sobel_image,
                markers=labels_in,
                mask=watershed_mask
            )

            if self.fill_holes:
                small_removed_segmented_out = fill_labeled_holes(labels_out)
            else:
                small_removed_segmented_out = labels_out.copy()
            segmented_out = self.filter_labels(small_removed_segmented_out,
                                               objects, workspace)
        elif self.method == M_WATERSHED_I:
            #
            # invert the image so that the maxima are filled first
            # and the cells compete over what's close to the threshold
            #
            inverted_img = 1 - img
            #
            # Same as above, but perform the watershed on the original image
            #
            watershed_mask = np.logical_or(thresholded_image, labels_in > 0)
            watershed_mask = np.logical_and(watershed_mask, mask)
            #
            # Perform the watershed
            #

            labels_out = skimage.morphology.watershed(
                connectivity=np.ones((3, 3), bool),
                image=inverted_img,
                markers=labels_in,
                mask=watershed_mask
            )

            if self.fill_holes:
                small_removed_segmented_out = fill_labeled_holes(labels_out)
            else:
                small_removed_segmented_out = labels_out
            segmented_out = self.filter_labels(small_removed_segmented_out,
                                               objects, workspace)

        if self.wants_discard_edge and self.wants_discard_primary:
            #
            # Make a new primary object
            #
            lookup = scind.maximum(segmented_out,
                                   objects.segmented,
                                   range(np.max(objects.segmented) + 1))
            lookup = fix(lookup)
            lookup[0] = 0
            lookup[lookup != 0] = np.arange(np.sum(lookup != 0)) + 1
            segmented_labels = lookup[objects.segmented]
            segmented_out = lookup[segmented_out]
            new_objects = cpo.Objects()
            new_objects.segmented = segmented_labels
            if objects.has_unedited_segmented:
                new_objects.unedited_segmented = objects.unedited_segmented
            if objects.has_small_removed_segmented:
                new_objects.small_removed_segmented = objects.small_removed_segmented
            new_objects.parent_image = objects.parent_image
            primary_outline = outline(segmented_labels)
            if self.wants_primary_outlines:
                out_img = cpi.Image(primary_outline.astype(bool),
                                    parent_image=image)
                workspace.image_set.add(self.new_primary_outlines_name.value,
                                        out_img)
        else:
            primary_outline = outline(objects.segmented)
        secondary_outline = outline(segmented_out)

        #
        # Add the objects to the object set
        #
        objects_out = cpo.Objects()
        objects_out.unedited_segmented = small_removed_segmented_out
        objects_out.small_removed_segmented = small_removed_segmented_out
        objects_out.segmented = segmented_out
        objects_out.parent_image = image
        objname = self.objects_name.value
        workspace.object_set.add_objects(objects_out, objname)
        if self.use_outlines.value:
            out_img = cpi.Image(secondary_outline.astype(bool),
                                parent_image=image)
            workspace.image_set.add(self.outlines_name.value, out_img)
        object_count = np.max(segmented_out)
        #
        # Add measurements
        #
        measurements = workspace.measurements
        cpmi.add_object_count_measurements(measurements, objname, object_count)
        cpmi.add_object_location_measurements(measurements, objname,
                                              segmented_out)
        #
        # Relate the secondary objects to the primary ones and record
        # the relationship.
        #
        children_per_parent, parents_of_children = \
            objects.relate_children(objects_out)
        measurements.add_measurement(self.primary_objects.value,
                                     cpmi.FF_CHILDREN_COUNT % objname,
                                     children_per_parent)
        measurements.add_measurement(objname,
                                     cpmi.FF_PARENT % self.primary_objects.value,
                                     parents_of_children)
        image_numbers = np.ones(len(parents_of_children), int) * \
                        measurements.image_set_number
        mask = parents_of_children > 0
        measurements.add_relate_measurement(
                self.module_num, R_PARENT,
                self.primary_objects.value, self.objects_name.value,
                image_numbers[mask], parents_of_children[mask],
                image_numbers[mask],
                np.arange(1, len(parents_of_children) + 1)[mask])
        #
        # If primary objects were created, add them
        #
        if self.wants_discard_edge and self.wants_discard_primary:
            workspace.object_set.add_objects(new_objects,
                                             self.new_primary_objects_name.value)
            cpmi.add_object_count_measurements(measurements,
                                               self.new_primary_objects_name.value,
                                               np.max(new_objects.segmented))
            cpmi.add_object_location_measurements(measurements,
                                                  self.new_primary_objects_name.value,
                                                  new_objects.segmented)
            for parent_objects, parent_name, child_objects, child_name in (
                    (objects, self.primary_objects.value,
                     new_objects, self.new_primary_objects_name.value),
                    (new_objects, self.new_primary_objects_name.value,
                     objects_out, objname)):
                children_per_parent, parents_of_children = \
                    parent_objects.relate_children(child_objects)
                measurements.add_measurement(parent_name,
                                             cpmi.FF_CHILDREN_COUNT % child_name,
                                             children_per_parent)
                measurements.add_measurement(child_name,
                                             cpmi.FF_PARENT % parent_name,
                                             parents_of_children)
        if self.show_window:
            object_area = np.sum(segmented_out > 0)
            workspace.display_data.object_pct = \
                100 * object_area / np.product(segmented_out.shape)
            workspace.display_data.img = img
            workspace.display_data.segmented_out = segmented_out
            workspace.display_data.primary_labels = objects.segmented
            workspace.display_data.global_threshold = global_threshold
            workspace.display_data.object_count = object_count

    def display(self, workspace, figure):
        from identify import TS_BINARY_IMAGE

        object_pct = workspace.display_data.object_pct
        img = workspace.display_data.img
        primary_labels = workspace.display_data.primary_labels
        segmented_out = workspace.display_data.segmented_out
        global_threshold = workspace.display_data.global_threshold
        object_count = workspace.display_data.object_count
        statistics = workspace.display_data.statistics

        if global_threshold is not None:
            statistics.append(["Threshold", "%.3f" % global_threshold])

        if object_count > 0:
            areas = scind.sum(np.ones(segmented_out.shape), segmented_out, np.arange(1, object_count + 1))
            areas.sort()
            low_diameter = (np.sqrt(float(areas[object_count / 10]) / np.pi) * 2)
            median_diameter = (np.sqrt(float(areas[object_count / 2]) / np.pi) * 2)
            high_diameter = (np.sqrt(float(areas[object_count * 9 / 10]) / np.pi) * 2)
            statistics.append(["10th pctile diameter",
                               "%.1f pixels" % low_diameter])
            statistics.append(["Median diameter",
                               "%.1f pixels" % median_diameter])
            statistics.append(["90th pctile diameter",
                               "%.1f pixels" % high_diameter])
            if self.method != M_DISTANCE_N and self.threshold_scope != TS_BINARY_IMAGE:
                statistics.append(["Thresholding filter size",
                                   "%.1f" % workspace.display_data.threshold_sigma])
            statistics.append(["Area covered by objects", "%.1f %%" % object_pct])
        workspace.display_data.statistics = statistics

        figure.set_subplots((2, 2))
        title = "Input image, cycle #%d" % workspace.measurements.image_number
        figure.subplot_imshow_grayscale(0, 0, img, title)
        figure.subplot_imshow_labels(1, 0, segmented_out, "%s objects" % self.objects_name.value,
                                     sharexy=figure.subplot(0, 0))

        cplabels = [
            dict(name=self.primary_objects.value,
                 labels=[primary_labels]),
            dict(name=self.objects_name.value,
                 labels=[segmented_out])]
        title = "%s and %s outlines" % (
            self.primary_objects.value, self.objects_name.value)
        figure.subplot_imshow_grayscale(
                0, 1, img, title=title, cplabels=cplabels,
                sharexy=figure.subplot(0, 0))
        figure.subplot_table(
                1, 1,
                [[x[1]] for x in workspace.display_data.statistics],
                row_labels=[x[0] for x in workspace.display_data.statistics])

    def filter_labels(self, labels_out, objects, workspace):
        """Filter labels out of the output

        Filter labels that are not in the segmented input labels. Optionally
        filter labels that are touching the edge.

        labels_out - the unfiltered output labels
        objects    - the objects thing, containing both segmented and
                     small_removed labels
        """
        segmented_labels = objects.segmented
        max_out = np.max(labels_out)
        if max_out > 0:
            segmented_labels, m1 = cpo.size_similarly(labels_out, segmented_labels)
            segmented_labels[~m1] = 0
            lookup = scind.maximum(segmented_labels, labels_out,
                                   range(max_out + 1))
            lookup = np.array(lookup, int)
            lookup[0] = 0
            segmented_labels_out = lookup[labels_out]
        else:
            segmented_labels_out = labels_out.copy()
        if self.wants_discard_edge:
            image = workspace.image_set.get_image(self.image_name.value)
            if image.has_mask:
                mask_border = (image.mask & ~ scind.binary_erosion(image.mask))
                edge_labels = segmented_labels_out[mask_border]
            else:
                edge_labels = np.hstack((segmented_labels_out[0, :],
                                         segmented_labels_out[-1, :],
                                         segmented_labels_out[:, 0],
                                         segmented_labels_out[:, -1]))
            edge_labels = np.unique(edge_labels)
            #
            # Make a lookup table that translates edge labels to zero
            # but translates everything else to itself
            #
            lookup = np.arange(max(max_out, np.max(segmented_labels)) + 1)
            lookup[edge_labels] = 0
            #
            # Run the segmented labels through this to filter out edge
            # labels
            segmented_labels_out = lookup[segmented_labels_out]

        return segmented_labels_out

    def is_object_identification_module(self):
        '''IdentifySecondaryObjects makes secondary objects sets so it's a identification module'''
        return True

    def get_measurement_columns(self, pipeline):
        '''Return column definitions for measurements made by this module'''
        columns = cpmi.get_object_measurement_columns(self.objects_name.value)
        columns += [(self.primary_objects.value,
                     cpmi.FF_CHILDREN_COUNT % self.objects_name.value,
                     cpmeas.COLTYPE_INTEGER),
                    (self.objects_name.value,
                     cpmi.FF_PARENT % self.primary_objects.value,
                     cpmeas.COLTYPE_INTEGER)]
        if self.method != M_DISTANCE_N:
            columns += cpmi.get_threshold_measurement_columns(self.objects_name.value)
        if self.wants_discard_edge and self.wants_discard_primary:
            columns += cpmi.get_object_measurement_columns(self.new_primary_objects_name.value)
            columns += [(self.new_primary_objects_name.value,
                         cpmi.FF_CHILDREN_COUNT % self.objects_name.value,
                         cpmeas.COLTYPE_INTEGER),
                        (self.objects_name.value,
                         cpmi.FF_PARENT % self.new_primary_objects_name.value,
                         cpmeas.COLTYPE_INTEGER)]
            columns += [(self.primary_objects.value,
                         cpmi.FF_CHILDREN_COUNT % self.new_primary_objects_name.value,
                         cpmeas.COLTYPE_INTEGER),
                        (self.new_primary_objects_name.value,
                         cpmi.FF_PARENT % self.primary_objects.value,
                         cpmeas.COLTYPE_INTEGER)]

        return columns

    def get_categories(self, pipeline, object_name):
        """Return the categories of measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        """
        object_dictionary = self.get_object_dictionary()
        categories = []
        if self.method != M_DISTANCE_N:
            categories += self.get_threshold_categories(pipeline, object_name)
        categories += self.get_object_categories(pipeline, object_name,
                                                 object_dictionary)
        return categories

    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        object_dictionary = self.get_object_dictionary()
        result = []
        if self.method != M_DISTANCE_N:
            result += self.get_threshold_measurements(pipeline, object_name,
                                                      category)
        result += self.get_object_measurements(pipeline, object_name,
                                               category, object_dictionary)
        return result

    def get_object_dictionary(self):
        '''Get the dictionary of parent child relationships

        see Identify.get_object_categories, Identify.get_object_measurements
        '''
        object_dictionary = {
            self.objects_name.value: [self.primary_objects.value]
        }
        if self.wants_discard_edge and self.wants_discard_primary:
            object_dictionary[self.objects_name.value] += \
                [self.new_primary_objects_name.value]
            object_dictionary[self.new_primary_objects_name.value] = \
                [self.primary_objects.value]
        return object_dictionary

    def get_measurement_objects(self, pipeline, object_name, category, measurement):
        if self.method != M_DISTANCE_N:
            return self.get_threshold_measurement_objects(
                    pipeline, object_name, category, measurement)
        return []

    def get_measurement_objects_name(self):
        return self.objects_name.value


IdentifySecondary = IdentifySecondaryObjects
