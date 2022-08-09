"""
MeasureObjectSkeleton
=====================

**MeasureObjectSkeleton** measures information for any branching structures,
such as neurons, root or branch systems, vasculature, or any skeletonized
system that originates from a single point (such as neurites branching from
a single nucleus/soma).

This module measures the number of trunks and branches for each branching system
in an image. The module takes a skeletonized image of the object plus
previously identified seed objects (for instance, each neuron's soma) and
finds the number of axon or dendrite trunks that emerge from the soma
and the number of branches along the axons and dendrites. Note that the
seed objects must be both smaller than the skeleton, and touching the
skeleton, in order to be counted.

The typical approach for this module is the following:

-  Identify a seed object. This object is typically a nucleus,
   identified with a module such as **IdentifyPrimaryObjects**.
-  Identify a larger object that touches or encloses this seed object.
   For example, the neuron cell can be grown outwards from the initial
   seed nuclei using **IdentifySecondaryObjects**.
-  Use the **Morph** module to skeletonize the secondary objects.
-  Finally, the primary objects and the skeleton objects are used as
   inputs to **MeasureObjectSkeleton**.

The module determines distances from the seed objects along the axons
and dendrites and assigns branchpoints based on distance to the closest
seed object when two seed objects appear to be attached to the same
dendrite or axon.

The module records *vertices* which include trunks, branchpoints, and endpoints.

Note that this module was referred to as MeasureNeurons in previous versions of CellProfiler.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

See also
^^^^^^^^

See also **MeasureImageSkeleton**.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *NumberTrunks:* The number of trunks. Trunks are branchpoints that
   lie within the seed objects
-  *NumberNonTrunkBranches:* The number of non-trunk branches. Branches
   are the branchpoints that lie outside the seed objects.
-  *NumberBranchEnds*: The number of branch end-points, i.e, termini.
-  *TotalObjectSkeletonLength*: The length of all skeleton segments per object.
"""

import os

import centrosome.cpmorphology
import centrosome.propagate as propagate
import numpy
import scipy.ndimage
from cellprofiler_core.constants.measurement import COLTYPE_FLOAT
from cellprofiler_core.constants.measurement import COLTYPE_INTEGER
from cellprofiler_core.image import Image
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.module import Module
from cellprofiler_core.preferences import ABSOLUTE_FOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_INPUT_FOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_INPUT_SUBFOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_OUTPUT_FOLDER_NAME
from cellprofiler_core.preferences import DEFAULT_OUTPUT_SUBFOLDER_NAME
from cellprofiler_core.preferences import get_default_colormap
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.subscriber import LabelSubscriber, ImageSubscriber
from cellprofiler_core.setting.text import ImageName, Directory
from cellprofiler_core.setting.text import Integer
from cellprofiler_core.setting.text import Text
from cellprofiler_core.utilities.core.object import size_similarly
from centrosome.cpmorphology import fixup_scipy_ndimage_result as fix
from scipy.ndimage import grey_dilation, grey_erosion

"""The measurement category"""
C_OBJSKELETON = "ObjectSkeleton"

"""The trunk count feature"""
F_NUMBER_TRUNKS = "NumberTrunks"

"""The branch feature"""
F_NUMBER_NON_TRUNK_BRANCHES = "NumberNonTrunkBranches"

"""The endpoint feature"""
F_NUMBER_BRANCH_ENDS = "NumberBranchEnds"

"""The neurite length feature"""
F_TOTAL_OBJSKELETON_LENGTH = "TotalObjectSkeletonLength"

F_ALL = [
    F_NUMBER_TRUNKS,
    F_NUMBER_NON_TRUNK_BRANCHES,
    F_NUMBER_BRANCH_ENDS,
    F_TOTAL_OBJSKELETON_LENGTH,
]


class MeasureObjectSkeleton(Module):
    module_name = "MeasureObjectSkeleton"
    category = "Measurement"
    variable_revision_number = 3

    def create_settings(self):
        """Create the UI settings for the module"""
        self.seed_objects_name = LabelSubscriber(
            "Select the seed objects",
            "None",
            doc="""\
Select the previously identified objects that you want to use as the
seeds for measuring branches and distances. Branches and trunks are assigned
per seed object. Seed objects are typically not single points/pixels but
instead are usually objects of varying sizes.""",
        )

        self.image_name = ImageSubscriber(
            "Select the skeletonized image",
            "None",
            doc="""\
Select the skeletonized image of the dendrites and/or axons as produced
by the **Morph** module’s *Skel* operation.""",
        )

        self.wants_branchpoint_image = Binary(
            "Retain the branchpoint image?",
            False,
            doc="""\
Select "*Yes*" if you want to save the color image of branchpoints and
trunks. This is the image that is displayed in the output window for
this module."""
            % globals(),
        )

        self.branchpoint_image_name = ImageName(
            "Name the branchpoint image",
            "BranchpointImage",
            doc="""\
*(Used only if a branchpoint image is to be retained)*

Enter a name for the branchpoint image here. You can then use this image
in a later module, such as **SaveImages**.""",
        )

        self.wants_to_fill_holes = Binary(
            "Fill small holes?",
            True,
            doc="""\
The algorithm reskeletonizes the image and this can leave artifacts
caused by small holes in the image prior to skeletonizing. These holes
result in false trunks and branchpoints. Select "*Yes*" to fill in
these small holes prior to skeletonizing."""
            % globals(),
        )

        self.maximum_hole_size = Integer(
            "Maximum hole size",
            10,
            minval=1,
            doc="""\
*(Used only when filling small holes)*

This is the area of the largest hole to fill, measured in pixels. The
algorithm will fill in any hole whose area is this size or smaller.""",
        )

        self.wants_objskeleton_graph = Binary(
            "Export the skeleton graph relationships?",
            False,
            doc="""\
Select "*Yes*" to produce an edge file and a vertex file that gives the
relationships between vertices (trunks, branchpoints and endpoints)."""
            % globals(),
        )

        self.intensity_image_name = ImageSubscriber(
            "Intensity image",
            "None",
            doc="""\
Select the image to be used to calculate
the total intensity along the edges between the vertices (trunks, branchpoints, and endpoints).""",
        )

        self.directory = Directory(
            "File output directory",
            doc="Select the directory you want to save the graph relationships to.",
            dir_choices=[
                DEFAULT_OUTPUT_FOLDER_NAME,
                DEFAULT_INPUT_FOLDER_NAME,
                ABSOLUTE_FOLDER_NAME,
                DEFAULT_OUTPUT_SUBFOLDER_NAME,
                DEFAULT_INPUT_SUBFOLDER_NAME,
            ],
        )
        self.directory.dir_choice = DEFAULT_OUTPUT_FOLDER_NAME

        self.vertex_file_name = Text(
            "Vertex file name",
            "vertices.csv",
            doc="""\
*(Used only when exporting graph relationships)*

Enter the name of the file that will hold the edge information. You can
use metadata tags in the file name.

Each line of the file is a row of comma-separated values. The first
row is the header; this names the file’s columns. Each subsequent row
represents a vertex in the skeleton graph: either a trunk, a
branchpoint or an endpoint. The file has the following columns:

-  *image\_number:* The image number of the associated image.
-  *vertex\_number:* The number of the vertex within the image.
-  *i:* The I coordinate of the vertex.
-  *j:* The J coordinate of the vertex.
-  *label:* The label of the seed object associated with the vertex.
-  *kind:* The vertex type, with the following choices:

   -  **T:** Trunk
   -  **B:** Branchpoint
   -  **E:** Endpoint
""",
        )

        self.edge_file_name = Text(
            "Edge file name",
            "edges.csv",
            doc="""\
*(Used only when exporting graph relationships)*

Enter the name of the file that will hold the edge information. You can
use metadata tags in the file name. Each line of the file is a row of
comma-separated values. The first row is the header; this names the
file’s columns. Each subsequent row represents an edge or connection
between two vertices (including between a vertex and itself for certain
loops). Note that vertices include trunks, branchpoints, and endpoints.

The file has the following columns:

-  *image\_number:* The image number of the associated image.
-  *v1:* The zero-based index into the vertex table of the first vertex
   in the edge.
-  *v2:* The zero-based index into the vertex table of the second vertex
   in the edge.
-  *length:* The number of pixels in the path connecting the two
   vertices, including both vertex pixels.
-  *total\_intensity:* The sum of the intensities of the pixels in the
   edge, including both vertex pixel intensities.
""",
        )

    def settings(self):
        """The settings, in the order that they are saved in the pipeline"""
        return [
            self.seed_objects_name,
            self.image_name,
            self.wants_branchpoint_image,
            self.branchpoint_image_name,
            self.wants_to_fill_holes,
            self.maximum_hole_size,
            self.wants_objskeleton_graph,
            self.intensity_image_name,
            self.directory,
            self.vertex_file_name,
            self.edge_file_name,
        ]

    def visible_settings(self):
        """The settings that are displayed in the GUI"""
        result = [self.seed_objects_name, self.image_name, self.wants_branchpoint_image]
        if self.wants_branchpoint_image:
            result += [self.branchpoint_image_name]
        result += [self.wants_to_fill_holes]
        if self.wants_to_fill_holes:
            result += [self.maximum_hole_size]
        result += [self.wants_objskeleton_graph]
        if self.wants_objskeleton_graph:
            result += [
                self.intensity_image_name,
                self.directory,
                self.vertex_file_name,
                self.edge_file_name,
            ]
        return result

    def get_graph_file_paths(self, m, image_number):
        """Get the paths to the graph files for the given image set

        Apply metadata tokens to the graph file names to get the graph files
        for the given image set.

        m - measurements for the run

        image_number - the image # for the current image set

        Returns the edge file's path and vertex file's path
        """
        path = self.directory.get_absolute_path(m)
        edge_file = m.apply_metadata(self.edge_file_name.value, image_number)
        edge_path = os.path.abspath(os.path.join(path, edge_file))
        vertex_file = m.apply_metadata(self.vertex_file_name.value, image_number)
        vertex_path = os.path.abspath(os.path.join(path, vertex_file))
        return edge_path, vertex_path

    VF_IMAGE_NUMBER = "image_number"
    VF_VERTEX_NUMBER = "vertex_number"
    VF_I = "i"
    VF_J = "j"
    VF_LABELS = "labels"
    VF_KIND = "kind"
    vertex_file_columns = (
        VF_IMAGE_NUMBER,
        VF_VERTEX_NUMBER,
        VF_I,
        VF_J,
        VF_LABELS,
        VF_KIND,
    )
    EF_IMAGE_NUMBER = "image_number"
    EF_V1 = "v1"
    EF_V2 = "v2"
    EF_LENGTH = "length"
    EF_TOTAL_INTENSITY = "total_intensity"
    edge_file_columns = (EF_IMAGE_NUMBER, EF_V1, EF_V2, EF_LENGTH, EF_TOTAL_INTENSITY)

    def prepare_run(self, workspace):
        """Initialize graph files"""
        if not self.wants_objskeleton_graph:
            return True
        edge_files = set()
        vertex_files = set()
        m = workspace.measurements
        assert isinstance(m, Measurements)
        for image_number in m.get_image_numbers():
            edge_path, vertex_path = self.get_graph_file_paths(m, image_number)
            edge_files.add(edge_path)
            vertex_files.add(vertex_path)

        for file_path, header in (
            (edge_path, self.edge_file_columns),
            (vertex_path, self.vertex_file_columns),
        ):
            if os.path.exists(file_path):
                import wx

                if (
                    wx.MessageBox(
                        "%s already exists. Do you want to overwrite it?" % file_path,
                        "Warning: overwriting file",
                        style=wx.YES_NO,
                        parent=workspace.frame,
                    )
                    != wx.YES
                ):
                    return False
                os.remove(file_path)
            with open(file_path, "wt") as fd:
                header = ",".join(header)
                fd.write(header + "\n")
        return True

    def run(self, workspace):
        """Run the module on the image set"""
        seed_objects_name = self.seed_objects_name.value
        skeleton_name = self.image_name.value
        seed_objects = workspace.object_set.get_objects(seed_objects_name)
        labels = seed_objects.segmented
        labels_count = numpy.max(labels)
        label_range = numpy.arange(labels_count, dtype=numpy.int32) + 1

        skeleton_image = workspace.image_set.get_image(
            skeleton_name, must_be_binary=True
        )
        skeleton = skeleton_image.pixel_data
        if skeleton_image.has_mask:
            skeleton = skeleton & skeleton_image.mask
        try:
            labels = skeleton_image.crop_image_similarly(labels)
        except:
            labels, m1 = size_similarly(skeleton, labels)
            labels[~m1] = 0
        #
        # The following code makes a ring around the seed objects with
        # the skeleton trunks sticking out of it.
        #
        # Create a new skeleton with holes at the seed objects
        # First combine the seed objects with the skeleton so
        # that the skeleton trunks come out of the seed objects.
        #
        # Erode the labels once so that all of the trunk branchpoints
        # will be within the labels
        #
        #
        # Dilate the objects, then subtract them to make a ring
        #
        my_disk = centrosome.cpmorphology.strel_disk(1.5).astype(int)
        dilated_labels = grey_dilation(labels, footprint=my_disk)
        seed_mask = dilated_labels > 0
        combined_skel = skeleton | seed_mask

        closed_labels = grey_erosion(dilated_labels, footprint=my_disk)
        seed_center = closed_labels > 0
        combined_skel = combined_skel & (~seed_center)
        #
        # Fill in single holes (but not a one-pixel hole made by
        # a one-pixel image)
        #
        if self.wants_to_fill_holes:

            def size_fn(area, is_object):
                return (~is_object) and (area <= self.maximum_hole_size.value)

            combined_skel = centrosome.cpmorphology.fill_labeled_holes(
                combined_skel, ~seed_center, size_fn
            )
        #
        # Reskeletonize to make true branchpoints at the ring boundaries
        #
        combined_skel = centrosome.cpmorphology.skeletonize(combined_skel)
        #
        # The skeleton outside of the labels
        #
        outside_skel = combined_skel & (dilated_labels == 0)
        #
        # Associate all skeleton points with seed objects
        #
        dlabels, distance_map = propagate.propagate(
            numpy.zeros(labels.shape), dilated_labels, combined_skel, 1
        )
        #
        # Get rid of any branchpoints not connected to seeds
        #
        combined_skel[dlabels == 0] = False
        #
        # Find the branchpoints
        #
        branch_points = centrosome.cpmorphology.branchpoints(combined_skel)
        #
        # Odd case: when four branches meet like this, branchpoints are not
        # assigned because they are arbitrary. So assign them.
        #
        # .  .
        #  B.
        #  .B
        # .  .
        #
        odd_case = (
            combined_skel[:-1, :-1]
            & combined_skel[1:, :-1]
            & combined_skel[:-1, 1:]
            & combined_skel[1, 1]
        )
        branch_points[:-1, :-1][odd_case] = True
        branch_points[1:, 1:][odd_case] = True
        #
        # Find the branching counts for the trunks (# of extra branches
        # emanating from a point other than the line it might be on).
        #
        branching_counts = centrosome.cpmorphology.branchings(combined_skel)
        branching_counts = numpy.array([0, 0, 0, 1, 2])[branching_counts]
        #
        # Only take branches within 1 of the outside skeleton
        #
        dilated_skel = scipy.ndimage.binary_dilation(
            outside_skel, centrosome.cpmorphology.eight_connect
        )
        branching_counts[~dilated_skel] = 0
        #
        # Find the endpoints
        #
        end_points = centrosome.cpmorphology.endpoints(combined_skel)
        #
        # We use two ranges for classification here:
        # * anything within one pixel of the dilated image is a trunk
        # * anything outside of that range is a branch
        #
        nearby_labels = dlabels.copy()
        nearby_labels[distance_map > 1.5] = 0

        outside_labels = dlabels.copy()
        outside_labels[nearby_labels > 0] = 0
        #
        # The trunks are the branchpoints that lie within one pixel of
        # the dilated image.
        #
        if labels_count > 0:
            trunk_counts = fix(
                scipy.ndimage.sum(branching_counts, nearby_labels, label_range)
            ).astype(int)
        else:
            trunk_counts = numpy.zeros((0,), int)
        #
        # The branches are the branchpoints that lie outside the seed objects
        #
        if labels_count > 0:
            branch_counts = fix(
                scipy.ndimage.sum(branch_points, outside_labels, label_range)
            )
        else:
            branch_counts = numpy.zeros((0,), int)
        #
        # Save the endpoints
        #
        if labels_count > 0:
            end_counts = fix(scipy.ndimage.sum(end_points, outside_labels, label_range))
        else:
            end_counts = numpy.zeros((0,), int)
        #
        # Calculate the distances
        #
        total_distance = centrosome.cpmorphology.skeleton_length(
            dlabels * outside_skel, label_range
        )
        #
        # Save measurements
        #
        m = workspace.measurements
        assert isinstance(m, Measurements)
        feature = "_".join((C_OBJSKELETON, F_NUMBER_TRUNKS, skeleton_name))
        m.add_measurement(seed_objects_name, feature, trunk_counts)
        feature = "_".join((C_OBJSKELETON, F_NUMBER_NON_TRUNK_BRANCHES, skeleton_name))
        m.add_measurement(seed_objects_name, feature, branch_counts)
        feature = "_".join((C_OBJSKELETON, F_NUMBER_BRANCH_ENDS, skeleton_name))
        m.add_measurement(seed_objects_name, feature, end_counts)
        feature = "_".join((C_OBJSKELETON, F_TOTAL_OBJSKELETON_LENGTH, skeleton_name))
        m[seed_objects_name, feature] = total_distance
        #
        # Collect the graph information
        #
        if self.wants_objskeleton_graph:
            trunk_mask = (branching_counts > 0) & (nearby_labels != 0)
            intensity_image = workspace.image_set.get_image(
                self.intensity_image_name.value
            )
            edge_graph, vertex_graph = self.make_objskeleton_graph(
                combined_skel,
                dlabels,
                trunk_mask,
                branch_points & ~trunk_mask,
                end_points,
                intensity_image.pixel_data,
            )

            image_number = workspace.measurements.image_set_number

            edge_path, vertex_path = self.get_graph_file_paths(m, m.image_number)
            workspace.interaction_request(
                self,
                m.image_number,
                edge_path,
                edge_graph,
                vertex_path,
                vertex_graph,
                headless_ok=True,
            )

            if self.show_window:
                workspace.display_data.edge_graph = edge_graph
                workspace.display_data.vertex_graph = vertex_graph
                workspace.display_data.intensity_image = intensity_image.pixel_data
        #
        # Make the display image
        #
        if self.show_window or self.wants_branchpoint_image:
            branchpoint_image = numpy.zeros((skeleton.shape[0], skeleton.shape[1], 3))
            trunk_mask = (branching_counts > 0) & (nearby_labels != 0)
            branch_mask = branch_points & (outside_labels != 0)
            end_mask = end_points & (outside_labels != 0)
            branchpoint_image[outside_skel, :] = 1
            branchpoint_image[trunk_mask | branch_mask | end_mask, :] = 0
            branchpoint_image[trunk_mask, 0] = 1
            branchpoint_image[branch_mask, 1] = 1
            branchpoint_image[end_mask, 2] = 1
            branchpoint_image[dilated_labels != 0, :] *= 0.875
            branchpoint_image[dilated_labels != 0, :] += 0.1
            if self.show_window:
                workspace.display_data.branchpoint_image = branchpoint_image
            if self.wants_branchpoint_image:
                bi = Image(branchpoint_image, parent_image=skeleton_image)
                workspace.image_set.add(self.branchpoint_image_name.value, bi)

    def handle_interaction(
        self, image_number, edge_path, edge_graph, vertex_path, vertex_graph
    ):
        columns = tuple(
            [vertex_graph[f].tolist() for f in self.vertex_file_columns[2:]]
        )
        with open(vertex_path, "at") as fd:
            for vertex_number, fields in enumerate(zip(*columns)):
                fd.write(
                    ("%d,%d," % (image_number, vertex_number + 1))
                    + ("%d,%d,%d,%s\n" % fields)
                )

        columns = tuple([edge_graph[f].tolist() for f in self.edge_file_columns[1:]])
        with open(edge_path, "at") as fd:
            line_format = "%d,%%d,%%d,%%d,%%.4f\n" % image_number
            for fields in zip(*columns):
                fd.write(line_format % fields)

    def display(self, workspace, figure):
        """Display a visualization of the results"""
        from matplotlib.axes import Axes
        from matplotlib.lines import Line2D
        import matplotlib.cm

        if self.wants_objskeleton_graph:
            figure.set_subplots((2, 1))
        else:
            figure.set_subplots((1, 1))
        title = (
            "Branchpoints of %s and %s\nTrunks are red\nBranches are green\nEndpoints are blue"
            % (self.seed_objects_name.value, self.image_name.value)
        )
        figure.subplot_imshow(0, 0, workspace.display_data.branchpoint_image, title)
        if self.wants_objskeleton_graph:
            image = workspace.display_data.intensity_image
            figure.subplot_imshow_grayscale(
                1, 0, image, title="ObjectSkeleton graph", sharexy=figure.subplot(0, 0)
            )
            axes = figure.subplot(1, 0)
            assert isinstance(axes, Axes)
            edge_graph = workspace.display_data.edge_graph
            vertex_graph = workspace.display_data.vertex_graph
            i = vertex_graph["i"]
            j = vertex_graph["j"]
            kind = vertex_graph["kind"]
            brightness = edge_graph["total_intensity"] / edge_graph["length"]
            brightness = (brightness - numpy.min(brightness)) / (
                numpy.max(brightness) - numpy.min(brightness) + 0.000001
            )
            cm = matplotlib.cm.get_cmap(get_default_colormap())
            cmap = matplotlib.cm.ScalarMappable(cmap=cm)
            edge_color = cmap.to_rgba(brightness)
            for idx in range(len(edge_graph["v1"])):
                v = numpy.array([edge_graph["v1"][idx] - 1, edge_graph["v2"][idx] - 1])
                line = Line2D(j[v], i[v], color=edge_color[idx])
                axes.add_line(line)

    def get_measurement_columns(self, pipeline):
        """Return database column definitions for measurements made here"""
        return [
            (
                self.seed_objects_name.value,
                "_".join((C_OBJSKELETON, feature, self.image_name.value)),
                COLTYPE_FLOAT
                if feature == F_TOTAL_OBJSKELETON_LENGTH
                else COLTYPE_INTEGER,
            )
            for feature in F_ALL
        ]

    def get_categories(self, pipeline, object_name):
        """Get the measurement categories generated by this module

        pipeline - pipeline being run
        object_name - name of seed object
        """
        if object_name == self.seed_objects_name:
            return [C_OBJSKELETON]
        else:
            return []

    def get_measurements(self, pipeline, object_name, category):
        """Return the measurement features generated by this module

        pipeline - pipeline being run
        object_name - object being measured (must be the seed object)
        category - category of measurement (must be C_OBJSKELETON)
        """
        if category == C_OBJSKELETON and object_name == self.seed_objects_name:
            return F_ALL
        else:
            return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        """Return the images measured by this module

        pipeline - pipeline being run
        object_name - object being measured (must be the seed object)
        category - category of measurement (must be C_OBJSKELETON)
        measurement - one of the object skeleton measurements
        """
        if measurement in self.get_measurements(pipeline, object_name, category):
            return [self.image_name.value]
        else:
            return []

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        """Provide backwards compatibility for old pipelines

        setting_values - the strings to be fed to settings
        variable_revision_number - the version number at time of saving
        module_name - name of original module
        """
        if variable_revision_number == 1:
            #
            # Added hole size questions
            #
            setting_values = setting_values + ["Yes", "10"]
            variable_revision_number = 2
        if variable_revision_number == 2:
            #
            # Added graph stuff
            #
            setting_values = setting_values + [
                "No",
                "None",
                Directory.static_join_string(DEFAULT_OUTPUT_FOLDER_NAME, "None"),
                "None",
                "None",
            ]
            variable_revision_number = 3
        return setting_values, variable_revision_number

    def make_objskeleton_graph(
        self, skeleton, skeleton_labels, trunks, branchpoints, endpoints, image
    ):
        """Make a table that captures the graph relationship of the skeleton

        skeleton - binary skeleton image + outline of seed objects
        skeleton_labels - labels matrix of skeleton
        trunks - binary image with trunk points as 1
        branchpoints - binary image with branchpoints as 1
        endpoints - binary image with endpoints as 1
        image - image for intensity measurement

        returns two tables.
        Table 1: edge table
        The edge table is a numpy record array with the following named
        columns in the following order:
        v1: index into vertex table of first vertex of edge
        v2: index into vertex table of second vertex of edge
        length: # of intermediate pixels + 2 (for two vertices)
        total_intensity: sum of intensities along the edge

        Table 2: vertex table
        The vertex table is a numpy record array:
        i: I coordinate of the vertex
        j: J coordinate of the vertex
        label: the vertex's label
        kind: kind of vertex = "T" for trunk, "B" for branchpoint or "E" for endpoint.
        """
        i, j = numpy.mgrid[0 : skeleton.shape[0], 0 : skeleton.shape[1]]
        #
        # Give each point of interest a unique number
        #
        points_of_interest = trunks | branchpoints | endpoints
        number_of_points = numpy.sum(points_of_interest)
        #
        # Make up the vertex table
        #
        tbe = numpy.zeros(points_of_interest.shape, "|S1")
        tbe[trunks] = "T"
        tbe[branchpoints] = "B"
        tbe[endpoints] = "E"
        i_idx = i[points_of_interest]
        j_idx = j[points_of_interest]
        poe_labels = skeleton_labels[points_of_interest]
        tbe = tbe[points_of_interest]
        vertex_table = {
            self.VF_I: i_idx,
            self.VF_J: j_idx,
            self.VF_LABELS: poe_labels,
            self.VF_KIND: tbe,
        }
        #
        # First, break the skeleton by removing the branchpoints, endpoints
        # and trunks
        #
        broken_skeleton = skeleton & (~points_of_interest)
        #
        # Label the broken skeleton: this labels each edge differently
        #
        edge_labels, nlabels = centrosome.cpmorphology.label_skeleton(skeleton)
        #
        # Reindex after removing the points of interest
        #
        edge_labels[points_of_interest] = 0
        if nlabels > 0:
            indexer = numpy.arange(nlabels + 1)
            unique_labels = numpy.sort(numpy.unique(edge_labels))
            nlabels = len(unique_labels) - 1
            indexer[unique_labels] = numpy.arange(len(unique_labels))
            edge_labels = indexer[edge_labels]
            #
            # find magnitudes and lengths for all edges
            #
            magnitudes = fix(
                scipy.ndimage.sum(
                    image, edge_labels, numpy.arange(1, nlabels + 1, dtype=numpy.int32)
                )
            )
            lengths = fix(
                scipy.ndimage.sum(
                    numpy.ones(edge_labels.shape),
                    edge_labels,
                    numpy.arange(1, nlabels + 1, dtype=numpy.int32),
                )
            ).astype(int)
        else:
            magnitudes = numpy.zeros(0)
            lengths = numpy.zeros(0, int)
        #
        # combine the edge labels and indexes of points of interest with padding
        #
        edge_mask = edge_labels != 0
        all_labels = numpy.zeros(numpy.array(edge_labels.shape) + 2, int)
        all_labels[1:-1, 1:-1][edge_mask] = edge_labels[edge_mask] + number_of_points
        all_labels[i_idx + 1, j_idx + 1] = numpy.arange(1, number_of_points + 1)
        #
        # Collect all 8 neighbors for each point of interest
        #
        p1 = numpy.zeros(0, int)
        p2 = numpy.zeros(0, int)
        for i_off, j_off in (
            (0, 0),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
            (2, 2),
        ):
            p1 = numpy.hstack((p1, numpy.arange(1, number_of_points + 1)))
            p2 = numpy.hstack((p2, all_labels[i_idx + i_off, j_idx + j_off]))
        #
        # Get rid of zeros which are background
        #
        p1 = p1[p2 != 0]
        p2 = p2[p2 != 0]
        #
        # Find point_of_interest -> point_of_interest connections.
        #
        p1_poi = p1[(p2 <= number_of_points) & (p1 < p2)]
        p2_poi = p2[(p2 <= number_of_points) & (p1 < p2)]
        #
        # Make sure matches are labeled the same
        #
        same_labels = (
            skeleton_labels[i_idx[p1_poi - 1], j_idx[p1_poi - 1]]
            == skeleton_labels[i_idx[p2_poi - 1], j_idx[p2_poi - 1]]
        )
        p1_poi = p1_poi[same_labels]
        p2_poi = p2_poi[same_labels]
        #
        # Find point_of_interest -> edge
        #
        p1_edge = p1[p2 > number_of_points]
        edge = p2[p2 > number_of_points]
        #
        # Now, each value that p2_edge takes forms a group and all
        # p1_edge whose p2_edge are connected together by the edge.
        # Possibly they touch each other without the edge, but we will
        # take the minimum distance connecting each pair to throw out
        # the edge.
        #
        edge, p1_edge, p2_edge = centrosome.cpmorphology.pairwise_permutations(
            edge, p1_edge
        )
        indexer = edge - number_of_points - 1
        lengths = lengths[indexer]
        magnitudes = magnitudes[indexer]
        #
        # OK, now we make the edge table. First poi<->poi. Length = 2,
        # magnitude = magnitude at each point
        #
        poi_length = numpy.ones(len(p1_poi)) * 2
        poi_magnitude = (
            image[i_idx[p1_poi - 1], j_idx[p1_poi - 1]]
            + image[i_idx[p2_poi - 1], j_idx[p2_poi - 1]]
        )
        #
        # Now the edges...
        #
        poi_edge_length = lengths + 2
        poi_edge_magnitude = (
            image[i_idx[p1_edge - 1], j_idx[p1_edge - 1]]
            + image[i_idx[p2_edge - 1], j_idx[p2_edge - 1]]
            + magnitudes
        )
        #
        # Put together the columns
        #
        v1 = numpy.hstack((p1_poi, p1_edge))
        v2 = numpy.hstack((p2_poi, p2_edge))
        lengths = numpy.hstack((poi_length, poi_edge_length))
        magnitudes = numpy.hstack((poi_magnitude, poi_edge_magnitude))
        #
        # Sort by p1, p2 and length in order to pick the shortest length
        #
        indexer = numpy.lexsort((lengths, v1, v2))
        v1 = v1[indexer]
        v2 = v2[indexer]
        lengths = lengths[indexer]
        magnitudes = magnitudes[indexer]
        if len(v1) > 0:
            to_keep = numpy.hstack(([True], (v1[1:] != v1[:-1]) | (v2[1:] != v2[:-1])))
            v1 = v1[to_keep]
            v2 = v2[to_keep]
            lengths = lengths[to_keep]
            magnitudes = magnitudes[to_keep]
        #
        # Put it all together into a table
        #
        edge_table = {
            self.EF_V1: v1,
            self.EF_V2: v2,
            self.EF_LENGTH: lengths,
            self.EF_TOTAL_INTENSITY: magnitudes,
        }
        return edge_table, vertex_table
