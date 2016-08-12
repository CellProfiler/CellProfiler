"""<b>UntangleWorms</b> untangles overlapping worms.
<hr>
This module either assembles a training set of sample worms in order to create a worm
model, or takes a binary image and the results of worm training and
labels the worms in the image, untangling them and associating all of a
worm's pieces together.

The results of untangling the input image will be an object set that can be used with
downstream measurment modules. If using the <i>overlapping</i> style of objects, these
can be saved as images using <b>SaveImages</b> to create a multi-page TIF file by
specifying "Objects" as the type of image to save.

<h4>Available measurements</h4>

<b>Object measurements (for "Untangle" mode only)</b>:
<ul>
<li><i>Length:</i> The length of the worm skeleton. </li>
<li><i>Angle:</i> The angle at each of the control points</li>
<li><i>ControlPointX_N, ControlPointY_N:</i> The X,Y coordinate of a control point <i>N</i>.
A control point is a sampled location along the worm shape used to construct the model.</li>
</ul>

<h4>Technical notes</h4>

<i>Training</i> involves extracting morphological information from the sample objects
provided from the previous steps. Using the default training set weights is recommended.
Proper creation of the model is dependent on providing a binary image as input consisting
of single, separated objects considered to be worms. You can the <b>Identify</b> modules
to find the tentative objects and then filter these objects to get individual worms, whether
by using <b>FilterObjects</b>, <b>EditObjectsManually</b> or the size criteria in
<b>IdentifyPrimaryObjects</b>. A binary image can be obtained from an object set by using
<b>ConvertObjectsToImage</b>.

<p>At the end of the training run, a final display window is shown displaying the following
statistical data:
<ul>
<li>A boxplot of the direction angle shape costs. The direction angles (which are between -&pi; and &pi;)
are the angles between lines joining consective control points. The angle 0 corresponds to
the case when two adjacent line segments are parallel (and thus belong to the same line).</li>
<li>A cumulative boxplot of the worm lengths as determined by the model.</li>
<li>A cumulative boxplot of the worm angles as determined by the model.</li>
<li>A heatmap of the covariance matrix of the feature vectors. For <i>N</i> control points,
the feature vector is of length <i>N</i>-1 and contains <i>N</i>-2 elements for each of the
angles between them, plus an element representing the worm length.</li>
</ul></p>

<p><i>Untangling</i> involves untangles the worms using a provided worm model, built
from a large number of samples of single worms. If the result of the untangling is
not satisfactory (e.g., it is unable to detect long worms or is too stringent about
shape variation) and you do not wish to re-train, you can adjust the provided worm model
manually by opening the .xml file in a text editor
and changing the values for the fields defining worm length, area etc. You may also want to adjust the
"Maximum Complexity" module setting which controls how complex clusters the untangling will handle.
Large clusters (&gt; 6 worms) may be slow to process.</p>

<h4>References</h4>
<ul>
<li>W&auml;hlby C, Kamentsky L, Liu ZH, Riklin-Raviv T, Conery AL, O'Rourke EJ,
Sokolnicki KL, Visvikis O, Ljosa V, Irazoqui JE, Golland P, Ruvkun G,
Ausubel FM, Carpenter AE (2012). "An image analysis toolbox for high-throughput
<i>C. elegans</i> assays." <i>Nature Methods</i> 9(7): 714-716.
<a href="http://dx.doi.org/10.1038/nmeth.1984">(link)</a></li>
</ul>

<p>See also: Our <a href="http://www.cellprofiler.org/wormtoolbox/">Worm
Toolbox</a> page for sample images and pipelines, as well
as video tutorials.</p>
"""

import logging
import os
import xml.dom.minidom

import cellprofiler.gui.help
import cellprofiler.identify
import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.setting
import cellprofiler.worms
import centrosome.cpmorphology
import centrosome.outline
import centrosome.propagate
import matplotlib.mlab
import numpy
import scipy.interpolate
import scipy.io
import scipy.ndimage
import scipy.sparse
from cellprofiler.worms import read_params

logger = logging.getLogger(__name__)


class UntangleWorms(cellprofiler.module.Module):
    variable_revision_number = 2
    category = ["Object Processing", "Worm Toolbox"]
    module_name = "UntangleWorms"

    def create_settings(self):
        """Create the settings that parameterize the module"""
        self.mode = cellprofiler.setting.Choice(
            "Train or untangle worms?",
            [cellprofiler.worms.MODE_UNTANGLE, cellprofiler.worms.MODE_TRAIN],
            doc="""
            <b>UntangleWorms</b> has two modes:
            <ul>
            <li><i>{train}</i> creates one training set per image group,
            using all of the worms in the training set as examples. It then writes
            the training file at the end of each image group.</li>
            <li><i>{untangle}</i> uses the training file to untangle images of worms.</li>
            </ul>
            {using_metadata_grouping_help}""".format(**{
                'train': cellprofiler.worms.MODE_TRAIN,
                'untangle': cellprofiler.worms.MODE_UNTANGLE,
                'using_metadata_grouping_help': cellprofiler.gui.help.USING_METADATA_GROUPING_HELP_REF
            })
        )

        self.image_name = cellprofiler.setting.ImageNameSubscriber(
                "Select the input binary image", cellprofiler.setting.NONE, doc="""
            A binary image where the foreground indicates the worm
            shapes. The binary image can be produced by the <b>ApplyThreshold</b>
            module.""")

        self.overlap = cellprofiler.setting.Choice(
                "Overlap style", [cellprofiler.worms.OO_BOTH, cellprofiler.worms.OO_WITH_OVERLAP, cellprofiler.worms.OO_WITHOUT_OVERLAP], doc="""
            This setting determines which style objects are output.
            If two worms overlap, you have a choice of including the overlapping
            regions in both worms or excluding the overlapping regions from
            both worms.
            <ul>
            <li><i>%(OO_WITH_OVERLAP)s:</i> Save objects including
            overlapping regions.</li>
            <li><i>%(OO_WITHOUT_OVERLAP)s:</i> Save only
            the portions of objects that do not overlap.</li>
            <li><i>%(OO_BOTH)s:</i> Save two versions: with and without overlap.</li>
            </ul>""" %
                                                                                                                                              globals())

        self.overlap_objects = cellprofiler.setting.ObjectNameProvider(
                "Name the output overlapping worm objects", "OverlappingWorms",
                provided_attributes={cellprofiler.worms.ATTR_WORM_MEASUREMENTS: True}, doc="""
            <i>(Used only if "%(MODE_UNTANGLE)s" mode and "%(OO_BOTH)s" or "%(OO_WITH_OVERLAP)s" overlap style are selected)</i> <br>
            This setting names the objects representing the overlapping
            worms. When worms cross, they overlap and pixels are shared by
            both of the overlapping worms. The overlapping worm objects share
            these pixels and measurements of both overlapping worms will include
            these pixels in the measurements of both worms.""" % globals())

        self.wants_overlapping_outlines = cellprofiler.setting.Binary(
            "Retain outlines of the overlapping objects?",
            False,
            doc="""
            <i>(Used only if "{untangle}" mode and "{both}" or "{with_overlap}" overlap style are selected)</i> <br>
            {retaining_outlines_help}""".format(**{
                'untangle': cellprofiler.worms.MODE_UNTANGLE,
                'both': cellprofiler.worms.OO_BOTH,
                'with_overlap': cellprofiler.worms.OO_WITH_OVERLAP,
                'retaining_outlines_help': cellprofiler.gui.help.RETAINING_OUTLINES_HELP
            })
        )

        self.overlapping_outlines_colormap = cellprofiler.setting.Colormap(
                "Outline colormap?", doc="""
            <i>(Used only if "%(MODE_UNTANGLE)s" mode, "%(OO_BOTH)s" or "%(OO_WITH_OVERLAP)s" overlap style and retaining outlines are selected )</i> <br>
            This setting controls the colormap used when drawing
            outlines. The outlines are drawn in color to highlight the
            shapes of each worm in a group of overlapping worms""" % globals())

        self.overlapping_outlines_name = cellprofiler.setting.OutlineNameProvider(
                "Name the overlapped outline image",
                "OverlappedWormOutlines", doc="""
            <i>(Used only if "%(MODE_UNTANGLE)s" mode and "%(OO_BOTH)s" or "%(OO_WITH_OVERLAP)s" overlap style are selected)</i> <br>
            This is the name of the outlines of the overlapped worms.""" % globals())

        self.nonoverlapping_objects = cellprofiler.setting.ObjectNameProvider(
                "Name the output non-overlapping worm objects", "NonOverlappingWorms",
                provided_attributes={cellprofiler.worms.ATTR_WORM_MEASUREMENTS: True}, doc="""
            <i>(Used only if "%(MODE_UNTANGLE)s" mode and "%(OO_BOTH)s" or "%(OO_WITH_OVERLAP)s" overlap style are selected)</i> <br>
            This setting names the objects representing the worms,
            excluding those regions where the worms overlap. When worms cross,
            there are pixels that cannot be unambiguously assigned to one
            worm or the other. These pixels are excluded from both worms
            in the non-overlapping objects and will not be a part of the
            measurements of either worm.""" % globals())

        self.wants_nonoverlapping_outlines = cellprofiler.setting.Binary(
                "Retain outlines of the non-overlapping worms?",
                False,
                doc="""<i>(Used only if "{untangle}" mode and "{both}" or "{with_overlap}" overlap style are selected)</i> <br>
                {retaining_outlines_help}""".format(**{
                'untangle': cellprofiler.worms.MODE_UNTANGLE,
                'both': cellprofiler.worms.OO_BOTH,
                'with_overlap': cellprofiler.worms.OO_WITH_OVERLAP,
                'retaining_outlines_help': cellprofiler.gui.help.RETAINING_OUTLINES_HELP
            }))

        self.nonoverlapping_outlines_name = cellprofiler.setting.OutlineNameProvider(
                "Name the non-overlapped outlines image",
                "NonoverlappedWormOutlines", doc="""
            <i>(Used only if "%(MODE_UNTANGLE)s" mode and "%(OO_BOTH)s" or "%(OO_WITH_OVERLAP)s" overlap style are selected)</i> <br>
            This is the name of the of the outlines of the worms
            with the overlapping sections removed.""" % globals())

        self.training_set_directory = cellprofiler.setting.DirectoryPath(
            "Training set file location",
            support_urls=True,
            allow_metadata=False,
            doc="""
            Select the folder containing the training set to be loaded.
            {}
            <p>An additional option is the following:
            <ul>
            <li><i>URL</i>: Use the path part of a URL. For instance, your
            training set might be hosted at
            <code>http://my_institution.edu/server/my_username/TrainingSet.xml</code>
            To access this file, you would choose <i>URL</i> and enter
            <code>http://my_institution.edu/server/my_username/</code>
            as the path location.</li>
            </ul></p>""".format(cellprofiler.preferences.IO_FOLDER_CHOICE_HELP_TEXT)
        )

        self.training_set_directory.dir_choice = cellprofiler.preferences.DEFAULT_OUTPUT_FOLDER_NAME

        def get_directory_fn():
            """Get the directory for the CSV file name"""
            return self.training_set_directory.get_absolute_path()

        def set_directory_fn(path):
            dir_choice, custom_path = self.training_set_directory.get_parts_from_path(path)
            self.training_set_directory.join_parts(dir_choice, custom_path)

        self.training_set_file_name = cellprofiler.setting.FilenameText(
                "Training set file name", "TrainingSet.xml",
                doc="This is the name of the training set file.",
                get_directory_fn=get_directory_fn,
                set_directory_fn=set_directory_fn,
                browse_msg="Choose training set",
                exts=[("Worm training set (*.xml)", "*.xml"),
                      ("All files (*.*)", "*.*")])

        self.wants_training_set_weights = cellprofiler.setting.Binary(
            "Use training set weights?",
            True,
            doc="""
            Select <i>{yes}</i> to use the overlap and leftover
            weights from the training set.
            <p>Select <i>{no}</i> to override
            these weights with user-specified values.</p>""".format(**{
                'yes': cellprofiler.setting.YES,
                'no': cellprofiler.setting.NO
            })
        )

        self.override_overlap_weight = cellprofiler.setting.Float(
                "Overlap weight", 5, 0, doc="""
            <i>(Used only if not using training set weights)</i> <br>
            This setting controls how much weight is given to overlaps
            between worms. <b>UntangleWorms</b> charges a penalty to a
            particular putative grouping of worms that overlap equal to the
            length of the overlapping region times the overlap weight.
            <ul>
            <li>Increase
            the overlap weight to make <b>UntangleWorms</b> avoid overlapping
            portions of worms.</li>
            <li>Decrease the overlap weight to make
            <b>UntangleWorms</b> ignore overlapping portions of worms.</li>
            </ul>""")

        self.override_leftover_weight = cellprofiler.setting.Float(
                "Leftover weight", 10, 0, doc="""
            <i>(Used only if not using training set weights)</i> <br>
            This setting controls how much weight is given to
            areas not covered by worms.
            <b>UntangleWorms</b> charges a penalty to a
            particular putative grouping of worms that fail to cover all
            of the foreground of a binary image. The penalty is equal to the
            length of the uncovered region times the leftover weight.
            <ul>
            <li> Increase the leftover weight to make <b>UntangleWorms</b>
            cover more foreground with worms.</li>
            <li>Decrease the overlap weight to make <b>UntangleWorms</b>
            ignore uncovered foreground.</li>
            </ul>""")

        self.min_area_percentile = cellprofiler.setting.Float(
                "Minimum area percentile", 1, 0, 100, doc="""
            <i>(Used only if "%(MODE_TRAIN)s" mode is selected)</i> <br>
            <b>UntangleWorms</b> will discard single worms whose area
            is less than a certain minimum. It ranks all worms in the training
            set according to area and then picks the worm at this percentile.
            It then computes the minimum area allowed as this worm's area
            times the minimum area factor.""" % globals())

        self.min_area_factor = cellprofiler.setting.Float(
                "Minimum area factor", .85, 0, doc="""
            <i>(Used only if "%(MODE_TRAIN)s" mode is selected)</i> <br>
            This setting is a multiplier that is applied to the
            area of the worm, selected as described in the documentation
            for <i>Minimum area percentile</i>.""" % globals())

        self.max_area_percentile = cellprofiler.setting.Float(
                "Maximum area percentile", 90, 0, 100, doc="""
            <i>(Used only if "%(MODE_TRAIN)s" mode is selected)</i><br>
            <b>UntangleWorms</b> uses a maximum area to distinguish
            between single worms and clumps of worms. Any blob whose area is
            less than the maximum area is considered to be a single worm
            whereas any blob whose area is greater is considered to be two
            or more worms. <b>UntangleWorms</b> orders all worms in the
            training set by area and picks the worm at the percentile
            given by this setting. It then multiplies this worm's area
            by the <i>Maximum area factor</i> (see below) to get the maximum
            area""" % globals())

        self.max_area_factor = cellprofiler.setting.Float(
                "Maximum area factor", 1.0, 0, doc="""
            <i>(Used only if "%(MODE_TRAIN)s" mode is selected)</i> <br>
            The <i>Maximum area factor</i> setting is used to
            compute the maximum area as decribed above in <i>Maximum area
            percentile</i>.""" % globals())

        self.min_length_percentile = cellprofiler.setting.Float(
                "Minimum length percentile", 1, 0, 100, doc="""
            <i>(Used only if "%(MODE_TRAIN)s" mode is selected)</i> <br>
            <b>UntangleWorms</b> uses the minimum length to restrict its
            search for worms in a clump to worms of at least the minimum length.
            <b>UntangleWorms</b> sorts all worms by length and picks the worm
            at the percentile indicated by this setting. It then multiplies the
            length of this worm by the <i>Mininmum length factor</i> (see below)
            to get the minimum length.""" % globals())

        self.min_length_factor = cellprofiler.setting.Float(
                "Minimum length factor", 0.9, 0, doc="""
            <i>(Used only if "%(MODE_TRAIN)s" mode is selected)</i> <br>
            <b>UntangleWorms</b> uses the <i>Minimum length factor</i>
            to compute the minimum length from the training set as described
            in the documentation above for <i>Minimum length percentile</i>""" % globals())

        self.max_length_percentile = cellprofiler.setting.Float(
                "Maximum length percentile", 99, 0, 100, doc="""
            <i>(Used only if "%(MODE_TRAIN)s" mode is selected)</i> <br>
            <b>UntangleWorms</b> uses the maximum length to restrict
            its search for worms in a clump to worms of at least the maximum
            length. It computes this length by sorting all of the training
            worms by length. It then selects the worm at the <i>Maximum
            length percentile</i> and multiplies that worm's length by
            the <i>Maximum length factor</i> to get the maximum length""" % globals())

        self.max_length_factor = cellprofiler.setting.Float(
                "Maximum length factor", 1.1, 0, doc="""
            <i>(Used only if "%(MODE_TRAIN)s" mode is selected)</i> <br>
            <b>UntangleWorms</b> uses this setting to compute the
            maximum length as described in <i>Maximum length percentile</i>
            above""" % globals())

        self.max_cost_percentile = cellprofiler.setting.Float(
                "Maximum cost percentile", 90, 0, 100, doc="""
            <i>(Used only if "%(MODE_TRAIN)s" mode is selected)</i><br>
            <b>UntangleWorms</b> computes a shape-based cost for
            each worm it considers. It will restrict the allowed cost to
            less than the cost threshold. During training, <b>UntangleWorms</b>
            computes the shape cost of every worm in the training set. It
            then orders them by cost and uses <i>Maximum cost percentile</i>
            to pick the worm at the given percentile. It them multiplies
            this worm's cost by the <i>Maximum cost factor</i> to compute
            the cost threshold.""" % globals())

        self.max_cost_factor = cellprofiler.setting.Float(
                "Maximum cost factor", 1.9, 0, doc="""
            <i>(Used only "%(MODE_TRAIN)s" mode is selected)</i> <br>
            <b>UntangleWorms</b> uses this setting to compute the
            cost threshold as described in <i>Maximum cost percentile</i>
            above.""" % globals())

        self.num_control_points = cellprofiler.setting.Integer(
                "Number of control points", 21, 3, 50, doc="""
            <i>(Used only if "%(MODE_TRAIN)s" mode is selected)</i> <br>
            This setting controls the number of control points that
            will be sampled when constructing a worm shape from its skeleton.""" % globals())

        self.max_radius_percentile = cellprofiler.setting.Float(
                "Maximum radius percentile", 90, 0, 100, doc="""
            <i>(Used only if "%(MODE_TRAIN)s" mode is selected)</i> <br>
            <b>UntangleWorms</b> uses the maximum worm radius during
            worm skeletonization. <b>UntangleWorms</b> sorts the radii of
            worms in increasing size and selects the worm at this percentile.
            It then multiplies this worm's radius by the <i>Maximum radius
            factor</i> (see below) to compute the maximum radius.""" % globals())

        self.max_radius_factor = cellprofiler.setting.Float(
                "Maximum radius factor", 1, 0, doc="""
            <i>(Used only if "%(MODE_TRAIN)s" mode is selected)</i> <br>
            <b>UntangleWorms</b> uses this setting to compute the
            maximum radius as described in <i>Maximum radius percentile</i>
            above.""" % globals())

        self.complexity = cellprofiler.setting.Choice(
                "Maximum complexity",
                [cellprofiler.worms.C_MEDIUM, cellprofiler.worms.C_HIGH, cellprofiler.worms.C_VERY_HIGH, cellprofiler.worms.C_ALL, cellprofiler.worms.C_CUSTOM],
                value=cellprofiler.worms.C_HIGH, doc="""
            <i>(Used only if "%(MODE_UNTANGLE)s" mode is selected)</i><br>
            This setting controls which clusters of worms are rejected as
            being too time-consuming to process. <b>UntangleWorms</b> judges
            complexity based on the number of segments in a cluster where
            a segment is the piece of a worm between crossing points or
            from the head or tail to the first or last crossing point.
            The choices are:<br>
            <ul><li><i>%(C_MEDIUM)s</i>: %(C_MEDIUM_VALUE)d segments
            (takes up to several minutes to process)</li>
            <li><i>%(C_HIGH)s</i>: %(C_HIGH_VALUE)d segments
            (takes up to a quarter-hour to process)</li>
            <li><i>%(C_VERY_HIGH)s</i>: %(C_VERY_HIGH_VALUE)d segments
            (can take hours to process)</li>
            <li><i>%(C_CUSTOM)s</i>: allows you to enter a custom number of
            segments.</li>
            <li><i>%(C_ALL)s</i>: Process all worms, regardless of complexity</li>
            </ul>""" % globals())

        self.custom_complexity = cellprofiler.setting.Integer(
                "Custom complexity", 400, 20, doc="""
            <i>(Used only if "%(MODE_UNTANGLE)s" mode and "%(C_CUSTOM)s" complexity are selected )</i>
            Enter the maximum number of segments of any cluster that should
            be processed.""" % globals())

    def settings(self):
        return [self.image_name, self.overlap, self.overlap_objects,
                self.nonoverlapping_objects, self.training_set_directory,
                self.training_set_file_name, self.wants_training_set_weights,
                self.override_overlap_weight, self.override_leftover_weight,
                self.wants_overlapping_outlines,
                self.overlapping_outlines_colormap,
                self.overlapping_outlines_name,
                self.wants_nonoverlapping_outlines,
                self.nonoverlapping_outlines_name,
                self.mode, self.min_area_percentile, self.min_area_factor,
                self.max_area_percentile, self.max_area_factor,
                self.min_length_percentile, self.min_length_factor,
                self.max_length_percentile, self.max_length_factor,
                self.max_cost_percentile, self.max_cost_factor,
                self.num_control_points, self.max_radius_percentile,
                self.max_radius_factor,
                self.complexity, self.custom_complexity]

    def help_settings(self):
        return [self.mode, self.image_name, self.overlap, self.overlap_objects,
                self.nonoverlapping_objects,
                self.complexity, self.custom_complexity,
                self.training_set_directory,
                self.training_set_file_name, self.wants_training_set_weights,
                self.override_overlap_weight, self.override_leftover_weight,
                self.wants_overlapping_outlines,
                self.overlapping_outlines_colormap,
                self.overlapping_outlines_name,
                self.wants_nonoverlapping_outlines,
                self.nonoverlapping_outlines_name,
                self.min_area_percentile, self.min_area_factor,
                self.max_area_percentile, self.max_area_factor,
                self.min_length_percentile, self.min_length_factor,
                self.max_length_percentile, self.max_length_factor,
                self.max_cost_percentile, self.max_cost_factor,
                self.num_control_points, self.max_radius_percentile,
                self.max_radius_factor]

    def visible_settings(self):
        result = [self.mode, self.image_name]
        if self.mode == cellprofiler.worms.MODE_UNTANGLE:
            result += [self.overlap]
            if self.overlap in (cellprofiler.worms.OO_WITH_OVERLAP, cellprofiler.worms.OO_BOTH):
                result += [self.overlap_objects, self.wants_overlapping_outlines]
                if self.wants_overlapping_outlines:
                    result += [self.overlapping_outlines_colormap,
                               self.overlapping_outlines_name]
            if self.overlap in (cellprofiler.worms.OO_WITHOUT_OVERLAP, cellprofiler.worms.OO_BOTH):
                result += [self.nonoverlapping_objects,
                           self.wants_nonoverlapping_outlines]
                if self.wants_nonoverlapping_outlines:
                    result += [self.nonoverlapping_outlines_name]
                result += [self.complexity]
                if self.complexity == cellprofiler.worms.C_CUSTOM:
                    result += [self.custom_complexity]
        result += [self.training_set_directory, self.training_set_file_name,
                   self.wants_training_set_weights]
        if not self.wants_training_set_weights:
            result += [self.override_overlap_weight,
                       self.override_leftover_weight]
            if self.mode == cellprofiler.worms.MODE_TRAIN:
                result += [
                    self.min_area_percentile, self.min_area_factor,
                    self.max_area_percentile, self.max_area_factor,
                    self.min_length_percentile, self.min_length_factor,
                    self.max_length_percentile, self.max_length_factor,
                    self.max_cost_percentile, self.max_cost_factor,
                    self.num_control_points, self.max_radius_percentile,
                    self.max_radius_factor]
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
        if self.mode == cellprofiler.worms.MODE_UNTANGLE:
            params = self.read_params()
            return params.num_control_points
        if not self.wants_training_set_weights:
            return 21
        else:
            return self.num_control_points.value

    @property
    def max_complexity(self):
        if self.complexity != cellprofiler.worms.C_CUSTOM:
            return cellprofiler.worms.complexity_limits[self.complexity.value]
        return self.custom_complexity.value

    def prepare_group(self, workspace, grouping, image_numbers):
        """Prepare to process a group of worms"""
        d = self.get_dictionary(workspace.image_set_list)
        d[cellprofiler.worms.TRAINING_DATA] = []

    def get_dictionary_for_worker(self):
        """Don't share the training data dictionary between workers"""
        return {cellprofiler.worms.TRAINING_DATA: []}

    def run(self, workspace):
        """Run the module on the current image set"""
        if self.mode == cellprofiler.worms.MODE_TRAIN:
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
        image = image_set.get_image(image_name,
                                    must_be_binary=True)
        num_control_points = self.ncontrol_points()
        labels, count = scipy.ndimage.label(image.pixel_data, centrosome.cpmorphology.eight_connect)
        skeleton = centrosome.cpmorphology.skeletonize(image.pixel_data)
        distances = scipy.ndimage.distance_transform_edt(image.pixel_data)
        worms = self.get_dictionary(workspace.image_set_list)[cellprofiler.worms.TRAINING_DATA]
        areas = numpy.bincount(labels.ravel())
        if self.show_window:
            dworms = workspace.display_data.worms = []
            workspace.display_data.input_image = image.pixel_data
        for i in range(1, count + 1):
            mask = labels == i
            graph = cellprofiler.worms.get_graph_from_binary(
                    image.pixel_data & mask, skeleton & mask)
            path_coords, path = cellprofiler.worms.get_longest_path_coords(
                    graph, numpy.iinfo(int).max)
            if len(path_coords) == 0:
                continue
            cumul_lengths = cellprofiler.worms.calculate_cumulative_lengths(path_coords)
            if cumul_lengths[-1] == 0:
                continue
            control_points = cellprofiler.worms.sample_control_points(path_coords, cumul_lengths,
                                                        num_control_points)
            angles = self.get_angles(control_points)
            #
            # Interpolate in 2-d when looking up the distances
            #
            fi, fj = (control_points - numpy.floor(control_points)).transpose()
            ci, cj = control_points.astype(int).transpose()
            ci1 = numpy.minimum(ci + 1, labels.shape[0] - 1)
            cj1 = numpy.minimum(cj + 1, labels.shape[1] - 1)
            radial_profile = numpy.zeros(num_control_points)
            for ii, jj, f in ((ci, cj, (1 - fi) * (1 - fj)),
                              (ci1, cj, fi * (1 - fj)),
                              (ci, cj1, (1 - fi) * fj),
                              (ci1, cj1, fi * fj)):
                radial_profile += distances[ii, jj] * f
            worms.append(self.TrainingData(areas[i], cumul_lengths[-1],
                                           angles, radial_profile))
            if self.show_window:
                dworms.append(control_points)

    def is_aggregation_module(self):
        """Building the model requires aggregation across image sets"""
        return self.mode == cellprofiler.worms.MODE_TRAIN

    def post_group(self, workspace, grouping):
        """Write the training data file as we finish grouping."""
        if self.mode == cellprofiler.worms.MODE_TRAIN:
            from cellprofiler.utilities.version import version_number
            worms = self.get_dictionary(workspace.image_set_list)[cellprofiler.worms.TRAINING_DATA]
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
            min_area = this.min_area_factor.value * matplotlib.mlab.prctile(
                    areas, this.min_area_percentile.value)
            max_area = this.max_area_factor.value * matplotlib.mlab.prctile(
                    areas, this.max_area_percentile.value)
            median_area = numpy.median(areas)
            min_length = this.min_length_factor.value * matplotlib.mlab.prctile(
                    lengths, this.min_length_percentile.value)
            max_length = this.max_length_factor.value * matplotlib.mlab.prctile(
                    lengths, this.max_length_percentile.value)
            max_skel_length = matplotlib.mlab.prctile(lengths, this.max_length_percentile.value)
            max_radius = this.max_radius_factor.value * matplotlib.mlab.prctile(
                    radial_profiles.flatten(), this.max_radius_percentile.value)
            mean_radial_profile = numpy.mean(radial_profiles, 1)
            #
            # Mirror the angles by negating them. Flip heads and tails
            # because they are arbitrary.
            #
            angles = numpy.hstack((
                angles,
                -angles,
                angles[::-1, :],
                -angles[::-1, :]))
            lengths = numpy.hstack([lengths] * 4)
            feat_vectors = numpy.vstack((angles, lengths[numpy.newaxis, :]))
            mean_angles_length = numpy.mean(feat_vectors, 1)
            fv_adjusted = feat_vectors - mean_angles_length[:, numpy.newaxis]
            angles_covariance_matrix = numpy.cov(fv_adjusted)
            inv_angles_covariance_matrix = numpy.linalg.inv(angles_covariance_matrix)
            angle_costs = [numpy.dot(numpy.dot(fv, inv_angles_covariance_matrix), fv)
                           for fv in fv_adjusted.transpose()]
            max_cost = this.max_cost_factor.value * matplotlib.mlab.prctile(
                    angle_costs, this.max_cost_percentile.value)
            #
            # Write it to disk
            #
            if workspace.pipeline.test_mode:
                return
            m = workspace.measurements
            assert isinstance(m, cellprofiler.measurement.Measurements)
            path = self.training_set_directory.get_absolute_path(m)
            file_name = m.apply_metadata(self.training_set_file_name.value)
            fd = open(os.path.join(path, file_name), "w")
            doc = xml.dom.minidom.getDOMImplementation().createDocument(
                cellprofiler.worms.T_NAMESPACE, cellprofiler.worms.T_TRAINING_DATA, None)
            top = doc.documentElement
            top.setAttribute("xmlns", cellprofiler.worms.T_NAMESPACE)
            for tag, value in (
                    (cellprofiler.worms.T_VERSION, version_number),
                    (cellprofiler.worms.T_MIN_AREA, min_area),
                    (cellprofiler.worms.T_MAX_AREA, max_area),
                    (cellprofiler.worms.T_COST_THRESHOLD, max_cost),
                    (cellprofiler.worms.T_NUM_CONTROL_POINTS, num_control_points),
                    (cellprofiler.worms.T_MAX_SKEL_LENGTH, max_skel_length),
                    (cellprofiler.worms.T_MIN_PATH_LENGTH, min_length),
                    (cellprofiler.worms.T_MAX_PATH_LENGTH, max_length),
                    (cellprofiler.worms.T_MEDIAN_WORM_AREA, median_area),
                    (cellprofiler.worms.T_MAX_RADIUS, max_radius),
                    (cellprofiler.worms.T_OVERLAP_WEIGHT, this.override_overlap_weight.value),
                    (cellprofiler.worms.T_LEFTOVER_WEIGHT, this.override_leftover_weight.value),
                    (cellprofiler.worms.T_TRAINING_SET_SIZE, nworms)):
                element = doc.createElement(tag)
                content = doc.createTextNode(str(value))
                element.appendChild(content)
                top.appendChild(element)
            for tag, values in ((cellprofiler.worms.T_MEAN_ANGLES, mean_angles_length),
                                (cellprofiler.worms.T_RADII_FROM_TRAINING, mean_radial_profile)):
                element = doc.createElement(tag)
                top.appendChild(element)
                for value in values:
                    value_element = doc.createElement(cellprofiler.worms.T_VALUE)
                    content = doc.createTextNode(str(value))
                    value_element.appendChild(content)
                    element.appendChild(value_element)
            element = doc.createElement(cellprofiler.worms.T_INV_ANGLES_COVARIANCE_MATRIX)
            top.appendChild(element)
            for row in inv_angles_covariance_matrix:
                values = doc.createElement(cellprofiler.worms.T_VALUES)
                element.appendChild(values)
                for col in row:
                    value = doc.createElement(cellprofiler.worms.T_VALUE)
                    content = doc.createTextNode(str(col))
                    value.appendChild(content)
                    values.appendChild(value)
            doc.writexml(fd, addindent="  ", newl="\n")
            fd.close()
            if self.show_window:
                workspace.display_data.angle_costs = angle_costs
                workspace.display_data.feat_vectors = feat_vectors
                workspace.display_data.angles_covariance_matrix = \
                    angles_covariance_matrix

    def run_untangle(self, workspace):
        """Untangle based on the current image set"""
        params = self.read_params()
        image_name = self.image_name.value
        image_set = workspace.image_set
        image = image_set.get_image(image_name,
                                    must_be_binary=True)
        labels, count = scipy.ndimage.label(image.pixel_data, centrosome.cpmorphology.eight_connect)
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
        eroded = scipy.ndimage.binary_erosion(image.pixel_data, centrosome.cpmorphology.eight_connect)
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
                if (areas[i] < params.min_worm_area or
                            i >= skeleton_areas.shape[0] or
                            skeleton_areas[i] == 0):
                    # Completely exclude the worm
                    continue
                elif areas[i] <= params.max_area:
                    path_coords, path_struct = self.single_worm_find_path(
                            workspace, labels, i, skeleton, params)
                    if len(path_coords) > 0 and self.single_worm_filter(
                            workspace, path_coords, params):
                        all_path_coords.append(path_coords)
                else:
                    graph = self.cluster_graph_building(
                            workspace, labels, i, skeleton, params)
                    if len(graph.segments) > self.max_complexity:
                        logger.warning(
                                "Warning: rejecting cluster of %d segments.\n" %
                                len(graph.segments))
                        continue
                    paths = cellprofiler.worms.get_all_paths(graph, params.min_path_length, params.max_path_length)
                    paths_selected = self.cluster_paths_selection(
                            graph, paths, labels, i, params)
                    del graph
                    del paths
                    all_path_coords += paths_selected
        ijv, all_lengths, all_angles, all_control_coords_x, all_control_coords_y = \
            self.worm_descriptor_building(all_path_coords, params,
                                          labels.shape)
        if self.show_window:
            workspace.display_data.input_image = image.pixel_data
        object_set = workspace.object_set
        assert isinstance(object_set, cellprofiler.region.Set)
        measurements = workspace.measurements
        assert isinstance(measurements, cellprofiler.measurement.Measurements)

        object_names = []
        if self.overlap in (cellprofiler.worms.OO_WITH_OVERLAP, cellprofiler.worms.OO_BOTH):
            o = cellprofiler.region.Region()
            o.ijv = ijv
            o.parent_image = image
            name = self.overlap_objects.value
            object_names.append(name)
            object_set.add_objects(o, name)
            cellprofiler.identify.add_object_count_measurements(measurements, name, o.count)
            if self.show_window:
                workspace.display_data.overlapping_labels = [
                    l for l, idx in o.labels()]

            if o.count == 0:
                center_x = numpy.zeros(0)
                center_y = numpy.zeros(0)
            else:
                center_x = numpy.bincount(ijv[:, 2], ijv[:, 1])[o.indices] / o.areas
                center_y = numpy.bincount(ijv[:, 2], ijv[:, 0])[o.indices] / o.areas
            measurements.add_measurement(name, cellprofiler.identify.M_LOCATION_CENTER_X, center_x)
            measurements.add_measurement(name, cellprofiler.identify.M_LOCATION_CENTER_Y, center_y)
            measurements.add_measurement(name, cellprofiler.identify.M_NUMBER_OBJECT_NUMBER, o.indices)
            #
            # Save outlines
            #
            if self.wants_overlapping_outlines:
                from matplotlib.cm import ScalarMappable
                colormap = self.overlapping_outlines_colormap.value
                if colormap == cellprofiler.setting.DEFAULT:
                    colormap = cellprofiler.preferences.get_default_colormap()
                if len(ijv) == 0:
                    ishape = image.pixel_data.shape
                    outline_pixels = numpy.zeros((ishape[0], ishape[1], 3))
                else:
                    my_map = ScalarMappable(cmap=colormap)
                    colors = my_map.to_rgba(numpy.unique(ijv[:, 2]))
                    outline_pixels = o.make_ijv_outlines(colors[:, :3])
                outline_image = cellprofiler.image.Image(outline_pixels, parent=image)
                image_set.add(self.overlapping_outlines_name.value,
                              outline_image)

        if self.overlap in (cellprofiler.worms.OO_WITHOUT_OVERLAP, cellprofiler.worms.OO_BOTH):
            #
            # Sum up the number of overlaps using a sparse matrix
            #
            overlap_hits = scipy.sparse.coo.coo_matrix(
                    (numpy.ones(len(ijv)), (ijv[:, 0], ijv[:, 1])),
                    image.pixel_data.shape)
            overlap_hits = overlap_hits.toarray()
            mask = overlap_hits == 1
            labels = scipy.sparse.coo.coo_matrix((ijv[:, 2], (ijv[:, 0], ijv[:, 1])), mask.shape)
            labels = labels.toarray()
            labels[~ mask] = 0
            o = cellprofiler.region.Region()
            o.segmented = labels
            o.parent_image = image
            name = self.nonoverlapping_objects.value
            object_names.append(name)
            object_set.add_objects(o, name)
            cellprofiler.identify.add_object_count_measurements(measurements, name, o.count)
            cellprofiler.identify.add_object_location_measurements(measurements, name, labels, o.count)
            if self.show_window:
                workspace.display_data.nonoverlapping_labels = [
                    l for l, idx in o.labels()]

            if self.wants_nonoverlapping_outlines:
                outline_pixels = centrosome.outline.outline(labels) > 0
                outline_image = cellprofiler.image.Image(outline_pixels, parent=image)
                image_set.add(self.nonoverlapping_outlines_name.value,
                              outline_image)
        for name in object_names:
            measurements.add_measurement(name, "_".join((cellprofiler.worms.C_WORM, cellprofiler.worms.F_LENGTH)),
                                         all_lengths)
            for values, ftr in ((all_angles, cellprofiler.worms.F_ANGLE),
                                (all_control_coords_x, cellprofiler.worms.F_CONTROL_POINT_X),
                                (all_control_coords_y, cellprofiler.worms.F_CONTROL_POINT_Y)):
                for i in range(values.shape[1]):
                    feature = "_".join((cellprofiler.worms.C_WORM, ftr, str(i + 1)))
                    measurements.add_measurement(name, feature, values[:, i])

    def display(self, workspace, figure):
        from cellprofiler.gui.figure import CPLDM_ALPHA
        if self.mode == cellprofiler.worms.MODE_UNTANGLE:
            figure.set_subplots((1, 1))
            cplabels = []
            if self.overlap in (cellprofiler.worms.OO_BOTH, cellprofiler.worms.OO_WITH_OVERLAP):
                title = self.overlap_objects.value
                cplabels.append(
                        dict(name=self.overlap_objects.value,
                             labels=workspace.display_data.overlapping_labels,
                             mode=CPLDM_ALPHA))
            else:
                title = self.nonoverlapping_objects.value
            if self.overlap in (cellprofiler.worms.OO_BOTH, cellprofiler.worms.OO_WITHOUT_OVERLAP):
                cplabels.append(
                        dict(name=self.nonoverlapping_objects.value,
                             labels=workspace.display_data.nonoverlapping_labels))
            image = workspace.display_data.input_image
            if image.ndim == 2:
                figure.subplot_imshow_grayscale(
                        0, 0, image, title=title, cplabels=cplabels)
        else:
            figure.set_subplots((1, 1))
            figure.subplot_imshow_bw(0, 0, workspace.display_data.input_image,
                                     title=self.image_name.value)
            axes = figure.subplot(0, 0)
            for control_points in workspace.display_data.worms:
                axes.plot(control_points[:, 1],
                          control_points[:, 0], "ro-",
                          markersize=4)

    def display_post_group(self, workspace, figure):
        """Display some statistical information about training, post-group

        workspace - holds the display data used to create the display

        figure - the module's figure.
        """
        if self.mode == cellprofiler.worms.MODE_TRAIN:
            from matplotlib.transforms import Bbox

            angle_costs = workspace.display_data.angle_costs
            feat_vectors = workspace.display_data.feat_vectors
            angles_covariance_matrix = workspace.display_data.angles_covariance_matrix
            figure = workspace.create_or_find_figure(
                    subplots=(4, 1),
                    window_name="UntangleWorms_PostGroup")
            f = figure.figure
            f.clf()
            a = f.add_subplot(1, 4, 1)
            a.set_position((Bbox([[.1, .1], [.15, .9]])))
            a.boxplot(angle_costs)
            a.set_title("Costs")
            a = f.add_subplot(1, 4, 2)
            a.set_position((Bbox([[.2, .1], [.25, .9]])))
            a.boxplot(feat_vectors[-1, :])
            a.set_title("Lengths")
            a = f.add_subplot(1, 4, 3)
            a.set_position((Bbox([[.30, .1], [.60, .9]])))
            a.boxplot(feat_vectors[:-1, :].transpose() * 180 / numpy.pi)
            a.set_title("Angles")
            a = f.add_subplot(1, 4, 4)
            a.set_position((Bbox([[.65, .1], [1, .45]])))
            a.imshow(angles_covariance_matrix[:-1, :-1],
                     interpolation="nearest")
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
        skeleton &= binary_im
        graph_struct = cellprofiler.worms.get_graph_from_binary(binary_im, skeleton)
        return cellprofiler.worms.get_longest_path_coords(graph_struct, params.max_path_length)

    def single_worm_filter(self, workspace, path_coords, params):
        """Given a path representing a single worm, caculates its shape cost, and
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
        cumul_lengths = cellprofiler.worms.calculate_cumulative_lengths(path_coords)
        total_length = cumul_lengths[-1]
        control_coords = cellprofiler.worms.sample_control_points(
                path_coords, cumul_lengths, params.num_control_points)
        cost = self.calculate_angle_shape_cost(
                control_coords, total_length, params.mean_angles,
                params.inv_angles_covariance_matrix)
        return cost < params.cost_threshold

    def calculate_angle_shape_cost(self, control_coords, total_length,
                                   mean_angles, inv_angles_covariance_matrix):
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
        points are sampled. (I.e. the distance along the path from the
        first control poin to the last. E.g. as returned by
        calculate_path_length().

        mean_angles: A (N-1) x 1 double array. The mu in the above formula,
        i.e. the mean of the feature vectors as calculated from the
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

        Note: All the angles in question here are direction angles,
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
        skeleton &= binary_im

        return cellprofiler.worms.get_graph_from_binary(
                binary_im, skeleton, params.max_radius,
                params.max_skel_length)

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
        overlaps, i.e. the a in the formula of the cost to be minimised.
        the unit is (shape cost unit)/(pixels as a unit of
        skeleton length).

        leftover_weight:  The
        weight factor assigned to leftover pieces, i.e. the b in the
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
            current_path_coords = cellprofiler.worms.path_to_pixel_coords(graph, path)
            cumul_lengths = cellprofiler.worms.calculate_cumulative_lengths(current_path_coords)
            total_length = cumul_lengths[-1]
            if total_length > max_path_length or total_length < min_path_length:
                continue
            control_coords = cellprofiler.worms.sample_control_points(
                    current_path_coords, cumul_lengths, num_control_points)
            #
            # Calculate the shape cost
            #
            current_shape_cost = self.calculate_angle_shape_cost(
                    control_coords, total_length, mean_angles,
                    inv_angles_covariance_matrix)
            if current_shape_cost < params.cost_threshold:
                paths_and_costs.append((path, current_shape_cost))

        if len(paths_and_costs) == 0:
            return []

        path_segment_matrix = numpy.zeros(
                (len(graph.segments), len(paths_and_costs)), bool)
        for i, (path, cost) in enumerate(paths_and_costs):
            path_segment_matrix[path.segments, i] = True
        overlap_weight = self.overlap_weight(params)
        leftover_weight = self.leftover_weight(params)
        #
        # Sort by increasing cost
        #
        costs = numpy.array([cost for path, cost in paths_and_costs])
        order = numpy.lexsort([costs])
        if len(order) > cellprofiler.worms.MAX_PATHS:
            order = order[:cellprofiler.worms.MAX_PATHS]
        costs = costs[order]
        path_segment_matrix = path_segment_matrix[:, order]

        current_best_subset, current_best_cost = self.fast_selection(
                costs, path_segment_matrix, graph.segment_lengths,
                overlap_weight, leftover_weight, max_num_worms)
        selected_paths = [paths_and_costs[order[i]][0]
                          for i in current_best_subset]
        path_coords_selected = [cellprofiler.worms.path_to_pixel_coords(graph, path)
                                for path in selected_paths]
        return path_coords_selected

    def fast_selection(self, costs, path_segment_matrix, segment_lengths,
                       overlap_weight, leftover_weight, max_num_worms):
        """Select the best subset of paths using a breadth-first search

        costs - the shape costs of every path

        path_segment_matrix - an N x M matrix where N are the segments
        and M are the paths. A cell is true if a path includes the segment

        segment_lengths - the length of each segment

        overlap_weight - the penalty per pixel of an overlap

        leftover_weight - the penalty per pixel of an unincluded segment

        max_num_worms - maximum # of worms allowed in returned match.
        """
        current_best_subset = []
        current_best_cost = numpy.sum(segment_lengths) * leftover_weight
        current_costs = costs
        current_path_segment_matrix = path_segment_matrix.astype(int)
        current_path_choices = numpy.eye(len(costs), dtype=bool)
        for i in range(min(max_num_worms, len(costs))):
            current_best_subset, current_best_cost, \
            current_path_segment_matrix, current_path_choices = \
                self.select_one_level(
                        costs, path_segment_matrix, segment_lengths,
                        current_best_subset, current_best_cost,
                        current_path_segment_matrix, current_path_choices,
                        overlap_weight, leftover_weight)
            if numpy.prod(current_path_choices.shape) == 0:
                break
        return current_best_subset, current_best_cost

    def select_one_level(self, costs, path_segment_matrix, segment_lengths,
                         current_best_subset, current_best_cost,
                         current_path_segment_matrix, current_path_choices,
                         overlap_weight, leftover_weight):
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
            numpy.sum(costs[:, numpy.newaxis] * current_path_choices, 0) +
            #
            # The sum of the multiply-covered segment lengths * penalty
            #
            numpy.sum(numpy.maximum(current_path_segment_matrix - 1, 0) *
                      segment_lengths[:, numpy.newaxis], 0) * overlap_weight)
        total_costs = (partial_costs +
                       #
                       # The sum of the uncovered segments * the penalty
                       #
                       numpy.sum((current_path_segment_matrix[:, :] == 0) *
                                 segment_lengths[:, numpy.newaxis], 0) * leftover_weight)

        order = numpy.lexsort([total_costs])
        if total_costs[order[0]] < current_best_cost:
            current_best_subset = numpy.argwhere(current_path_choices[:, order[0]]).flatten().tolist()
            current_best_cost = total_costs[order[0]]
        #
        # Weed out any that can't possibly be better
        #
        mask = partial_costs < current_best_cost
        if not numpy.any(mask):
            return current_best_subset, current_best_cost, \
                   numpy.zeros((len(costs), 0), int), numpy.zeros((len(costs), 0), bool)
        order = order[mask[order]]
        if len(order) * len(costs) > cellprofiler.worms.MAX_CONSIDERED:
            # Limit # to consider at next level
            order = order[:(1 + cellprofiler.worms.MAX_CONSIDERED / len(costs))]
        current_path_segment_matrix = current_path_segment_matrix[:, order]
        current_path_choices = current_path_choices[:, order]
        #
        # Create a matrix of disallowance - you can only add a path
        # that's higher than any existing path
        #
        i, j = numpy.mgrid[0:len(costs), 0:len(costs)]
        disallow = i >= j
        allowed = numpy.dot(disallow, current_path_choices) == 0
        if numpy.any(allowed):
            i, j = numpy.argwhere(allowed).transpose()
            current_path_choices = (numpy.eye(len(costs), dtype=bool)[:, i] |
                                    current_path_choices[:, j])
            current_path_segment_matrix = \
                path_segment_matrix[:, i] + current_path_segment_matrix[:, j]
            return current_best_subset, current_best_cost, \
                   current_path_segment_matrix, current_path_choices
        else:
            return current_best_subset, current_best_cost, \
                   numpy.zeros((len(costs), 0), int), numpy.zeros((len(costs), 0), bool)

    def search_recur(self, path_segment_matrix, segment_lengths,
                     path_raw_costs, overlap_weight, leftover_weight,
                     current_subset, last_chosen, current_cost,
                     current_segment_coverings, current_best_subset,
                     current_best_cost, branching_factors, current_level):
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
                segment_lengths[~ current_segment_coverings])
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
            path_raw_costs[last_chosen:] +
            numpy.sum(current_segment_coverings[:, numpy.newaxis] *
                      segment_lengths[:, numpy.newaxis] *
                   path_segment_matrix[:, last_chosen:], 0) * overlap_weight)
        order = numpy.lexsort([current_overlapped_costs])
        #
        # limit to number of branches allowed at this level
        #
        order = order[numpy.arange(len(order)) + 1 < this_branch_factor]
        for index in order:
            new_cost = current_cost + current_overlapped_costs[index]
            if new_cost >= current_best_cost:
                break  # No chance of subseequent better cost
            path_index = last_chosen + index
            current_best_subset, current_best_cost = self.search_recur(
                    path_segment_matrix, segment_lengths, path_raw_costs,
                    overlap_weight, leftover_weight,
                    current_subset + [path_index],
                    path_index,
                    new_cost,
                    current_segment_coverings | path_segment_matrix[:, path_index],
                    current_best_subset,
                    current_best_cost,
                    branching_factors,
                    current_level + 1)
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
            return (numpy.zeros((0, 3), int), numpy.zeros(0),
                    numpy.zeros((0, num_control_points - 2)),
                    numpy.zeros((0, num_control_points)),
                    numpy.zeros((0, num_control_points)))

        worm_radii = params.radii_from_training
        all_i = []
        all_j = []
        all_lengths = []
        all_angles = []
        all_control_coords_x = []
        all_control_coords_y = []
        for path in all_path_coords:
            cumul_lengths = cellprofiler.worms.calculate_cumulative_lengths(path)
            control_coords = cellprofiler.worms.sample_control_points(
                    path, cumul_lengths, num_control_points)
            ii, jj = self.rebuild_worm_from_control_points_approx(
                    control_coords, worm_radii, shape)
            all_i.append(ii)
            all_j.append(jj)
            all_lengths.append(cumul_lengths[-1])
            all_angles.append(self.get_angles(control_coords))
            all_control_coords_x.append(control_coords[:, 1])
            all_control_coords_y.append(control_coords[:, 0])
        ijv = numpy.column_stack((
            numpy.hstack(all_i),
            numpy.hstack(all_j),
            numpy.hstack([numpy.ones(len(ii), int) * (i + 1)
                          for i, ii in enumerate(all_i)])))
        all_lengths = numpy.array(all_lengths)
        all_angles = numpy.vstack(all_angles)
        all_control_coords_x = numpy.vstack(all_control_coords_x)
        all_control_coords_y = numpy.vstack(all_control_coords_y)
        return ijv, all_lengths, all_angles, all_control_coords_x, all_control_coords_y

    def rebuild_worm_from_control_points_approx(self, control_coords,
                                                worm_radii, shape):
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
        index, count, i, j = centrosome.cpmorphology.get_line_pts(control_coords[:-1, 0],
                                                control_coords[:-1, 1],
                                                control_coords[1:, 0],
                                                control_coords[1:, 1])
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
        radius = (worm_radii[label] * (1 - frac) +
                  worm_radii[label + 1] * frac)
        iworm_radius = int(numpy.max(numpy.ceil(radius)))
        #
        # Get dilation coordinates
        #
        ii, jj = numpy.mgrid[-iworm_radius:iworm_radius + 1,
                 -iworm_radius:iworm_radius + 1]
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
        return read_params(self.training_set_directory,
                           self.training_set_file_name,
                           self.training_params)

    def validate_module(self, pipeline):
        if self.mode == cellprofiler.worms.MODE_UNTANGLE:
            if self.training_set_directory.dir_choice != cellprofiler.preferences.URL_FOLDER_NAME:
                path = os.path.join(
                        self.training_set_directory.get_absolute_path(),
                        self.training_set_file_name.value)
                if not os.path.exists(path):
                    raise cellprofiler.setting.ValidationError(
                            "Can't find file %s" %
                            self.training_set_file_name.value,
                            self.training_set_file_name)

    def validate_module_warnings(self, pipeline):
        """Warn user re: Test mode """
        if pipeline.test_mode and self.mode == cellprofiler.worms.MODE_TRAIN:
            raise cellprofiler.setting.ValidationError("UntangleWorms will not produce training set output in Test Mode",
                                                       self.training_set_file_name)

    def get_measurement_columns(self, pipeline):
        """Return a column of information for each measurement feature"""
        result = []
        if self.mode == cellprofiler.worms.MODE_UNTANGLE:
            object_names = []
            if self.overlap in (cellprofiler.worms.OO_WITH_OVERLAP, cellprofiler.worms.OO_BOTH):
                object_names.append(self.overlap_objects.value)
            if self.overlap in (cellprofiler.worms.OO_WITHOUT_OVERLAP, cellprofiler.worms.OO_BOTH):
                object_names.append(self.nonoverlapping_objects.value)
            for object_name in object_names:
                result += cellprofiler.identify.get_object_measurement_columns(object_name)
                all_features = ([cellprofiler.worms.F_LENGTH] + self.angle_features() +
                                self.control_point_features(True) +
                                self.control_point_features(False))
                result += [
                    (object_name, "_".join((cellprofiler.worms.C_WORM, f)), cellprofiler.measurement.COLTYPE_FLOAT)
                    for f in all_features]
        return result

    def angle_features(self):
        """Return a list of angle feature names"""
        try:
            return ["_".join((cellprofiler.worms.F_ANGLE, str(n)))
                    for n in range(1, self.ncontrol_points() - 1)]
        except:
            logger.error("Failed to get # of control points from training file. Unknown number of angle measurements",
                         exc_info=True)
            return []

    def control_point_features(self, get_x):
        """Return a list of control point feature names

        get_x - return the X coordinate control point features if true, else y
        """
        try:
            return ["_".join((cellprofiler.worms.F_CONTROL_POINT_X if get_x else cellprofiler.worms.F_CONTROL_POINT_Y, str(n)))
                    for n in range(1, self.ncontrol_points() + 1)]
        except:
            logger.error(
                    "Failed to get # of control points from training file. Unknown number of control point features",
                    exc_info=True)
            return []

    def get_categories(self, pipeline, object_name):
        if object_name == cellprofiler.measurement.IMAGE:
            return [cellprofiler.identify.C_COUNT]
        if ((object_name == self.overlap_objects.value and
                     self.overlap in (cellprofiler.worms.OO_BOTH, cellprofiler.worms.OO_WITH_OVERLAP)) or
                (object_name == self.nonoverlapping_objects.value and
                         self.overlap in (cellprofiler.worms.OO_BOTH, cellprofiler.worms.OO_WITHOUT_OVERLAP))):
            return [cellprofiler.identify.C_LOCATION, cellprofiler.identify.C_NUMBER, cellprofiler.worms.C_WORM]
        return []

    def get_measurements(self, pipeline, object_name, category):
        wants_overlapping = self.overlap in (cellprofiler.worms.OO_BOTH, cellprofiler.worms.OO_WITH_OVERLAP)
        wants_nonoverlapping = self.overlap in (cellprofiler.worms.OO_BOTH, cellprofiler.worms.OO_WITHOUT_OVERLAP)
        result = []
        if object_name == cellprofiler.measurement.IMAGE and category == cellprofiler.identify.C_COUNT:
            if wants_overlapping:
                result += [self.overlap_objects.value]
            if wants_nonoverlapping:
                result += [self.nonoverlapping_objects.value]
        if ((wants_overlapping and object_name == self.overlap_objects) or
                (wants_nonoverlapping and object_name == self.nonoverlapping_objects)):
            if category == cellprofiler.identify.C_LOCATION:
                result += [cellprofiler.identify.FTR_CENTER_X, cellprofiler.identify.FTR_CENTER_Y]
            elif category == cellprofiler.identify.C_NUMBER:
                result += [cellprofiler.identify.FTR_OBJECT_NUMBER]
            elif category == cellprofiler.worms.C_WORM:
                result += [cellprofiler.worms.F_LENGTH, cellprofiler.worms.F_ANGLE, cellprofiler.worms.F_CONTROL_POINT_X, cellprofiler.worms.F_CONTROL_POINT_Y]
        return result

    def get_measurement_scales(self, pipeline, object_name, category,
                               measurement, image_name):
        wants_overlapping = self.overlap in (cellprofiler.worms.OO_BOTH, cellprofiler.worms.OO_WITH_OVERLAP)
        wants_nonoverlapping = self.overlap in (cellprofiler.worms.OO_BOTH, cellprofiler.worms.OO_WITHOUT_OVERLAP)
        scales = []
        if (((wants_overlapping and object_name == self.overlap_objects) or
                 (wants_nonoverlapping and object_name == self.nonoverlapping_objects)) and
                (category == cellprofiler.worms.C_WORM)):
            if measurement == cellprofiler.worms.F_ANGLE:
                scales += [str(n) for n in range(1, self.ncontrol_points() - 1)]
            elif measurement in [cellprofiler.worms.F_CONTROL_POINT_X, cellprofiler.worms.F_CONTROL_POINT_Y]:
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

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if variable_revision_number == 1:
            # Added complexity
            setting_values = setting_values + [cellprofiler.worms.C_ALL, "400"]
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab
