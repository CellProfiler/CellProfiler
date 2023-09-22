"""
IdentifyDeadWorms
=================

**IdentifyDeadWorms** identifies dead worms by their shape.

Dead *C. elegans* worms most often have a straight shape in an image
whereas live worms assume a sinusoidal shape. This module identifies
dead worms by fitting a straight shape to a binary image at many
different angles to identify the regions where the shape could fit. Each
placement point has a x and y location and an angle associated with the
fitted shape’s placement. Conceptually, these can be visualized in three
dimensions with the z direction being the angle (and with the angle, 0,
being adjacent to the largest angle as well as the smallest angle
greater than zero). The module labels the resulting 3-D volume. It
records the X, Y and angle of the centers of each of the found objects
and creates objects by collapsing the 3-D volume to 2-D. These objects
can then be used as seeds for **IdentifySecondaryObjects**.

**IdentifyDeadWorms** fits a diamond shape to the image. The shape is
defined by its width and length. The length is the distance in pixels
along the long axis of the diamond and should be less than the length of
the shortest dead worm to be detected. The width is the distance in
pixels along the short axis of the diamond and should be less than the
width of the worm.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

References
^^^^^^^^^^

-  Peng H, Long F, Liu X, Kim SK, Myers EW (2008) "Straightening
   *Caenorhabditis elegans* images." *Bioinformatics*,
   24(2):234-42. `(link) <https://doi.org/10.1093/bioinformatics/btm569>`__
-  Wählby C, Kamentsky L, Liu ZH, Riklin-Raviv T, Conery AL, O’Rourke
   EJ, Sokolnicki KL, Visvikis O, Ljosa V, Irazoqui JE, Golland P,
   Ruvkun G, Ausubel FM, Carpenter AE (2012). "An image analysis toolbox
   for high-throughput *C. elegans* assays." *Nature Methods* 9(7):
   714-716. `(link) <https://doi.org/10.1038/nmeth.1984>`__

See also
^^^^^^^^

See also: Our `Worm Toolbox`_ page for sample images and pipelines, as
well as video tutorials.

.. _Worm Toolbox: http://www.cellprofiler.org/wormtoolbox/
"""

import matplotlib.cm
import numpy
from cellprofiler_core.constants.measurement import (
    COLTYPE_INTEGER,
    M_LOCATION_CENTER_X,
    M_LOCATION_CENTER_Y,
    M_NUMBER_OBJECT_NUMBER,
    FF_COUNT,
    COLTYPE_FLOAT,
    IMAGE,
    C_COUNT,
    C_LOCATION,
    C_NUMBER,
    FTR_CENTER_X,
    FTR_CENTER_Y,
    FTR_OBJECT_NUMBER,
)
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.module import Module
from cellprofiler_core.object import Objects, ObjectSet
from cellprofiler_core.preferences import get_default_colormap
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import LabelName, Integer, Float
from centrosome.cpmorphology import all_connected_components
from centrosome.cpmorphology import fixup_scipy_ndimage_result
from centrosome.cpmorphology import get_line_pts
from scipy.ndimage import binary_erosion, binary_fill_holes
from scipy.ndimage import mean as mean_of_labels

C_WORMS = "Worm"
F_ANGLE = "Angle"
M_ANGLE = "_".join((C_WORMS, F_ANGLE))

"""Alpha value when drawing the binary mask"""
MASK_ALPHA = 0.1
"""Alpha value for labels"""
LABEL_ALPHA = 1.0
"""Alpha value for the worm shapes"""
WORM_ALPHA = 0.25


class IdentifyDeadWorms(Module):
    module_name = "IdentifyDeadWorms"
    variable_revision_number = 2
    category = ["Worm Toolbox"]

    def create_settings(self):
        """Create the settings for the module

        Create the settings for the module during initialization.
        """
        self.image_name = ImageSubscriber(
            "Select the input image",
            "None",
            doc="""\
The name of a binary image from a previous module. **IdentifyDeadWorms**
will use this image to establish the foreground and background for the
fitting operation. You can use **ApplyThreshold** to threshold a
grayscale image and create the binary mask. You can also use a module
such as **IdentifyPrimaryObjects** to label each worm and then use
**ConvertObjectsToImage** to make the result a mask.
""",
        )

        self.object_name = LabelName(
            "Name the dead worm objects to be identified",
            "DeadWorms",
            doc="""\
This is the name for the dead worm objects. You can refer
to this name in subsequent modules such as
**IdentifySecondaryObjects**""",
        )

        self.worm_width = Integer(
            "Worm width",
            10,
            minval=1,
            doc="""\
This is the width (the short axis), measured in pixels,
of the diamond used as a template when
matching against the worm. It should be less than the width
of a worm.""",
        )

        self.worm_length = Integer(
            "Worm length",
            100,
            minval=1,
            doc="""\
This is the length (the long axis), measured in pixels,
of the diamond used as a template when matching against the
worm. It should be less than the length of a worm""",
        )

        self.angle_count = Integer(
            "Number of angles",
            32,
            minval=1,
            doc="""\
This is the number of different angles at which the template will be
tried. For instance, if there are 12 angles, the template will be
rotated by 0°, 15°, 30°, 45° … 165°. The shape is bilaterally symmetric;
that is, you will get the same shape after rotating it by 180°.
""",
        )

        self.wants_automatic_distance = Binary(
            "Automatically calculate distance parameters?",
            True,
            doc="""\
This setting determines whether or not **IdentifyDeadWorms**
automatically calculates the parameters used to determine whether two
found-worm centers belong to the same worm.

Select "*Yes*" to have **IdentifyDeadWorms** automatically calculate
the distance from the worm length and width. Select "*No*" to set the
distances manually.
"""
            % globals(),
        )

        self.space_distance = Float(
            "Spatial distance",
            5,
            minval=1,
            doc="""\
*(Used only if not automatically calculating distance parameters)*

Enter the distance for calculating the worm centers, in units of pixels.
The worm centers must be at least many pixels apart for the centers to
be considered two separate worms.
""",
        )

        self.angular_distance = Float(
            "Angular distance",
            30,
            minval=1,
            doc="""\
*(Used only if automatically calculating distance parameters)*

**IdentifyDeadWorms** calculates the worm centers at different angles.
Two worm centers are considered to represent different worms if their
angular distance is larger than this number. The number is measured in
degrees.
""",
        )

    def settings(self):
        """The settings as they appear in the pipeline file"""
        return [
            self.image_name,
            self.object_name,
            self.worm_width,
            self.worm_length,
            self.angle_count,
            self.wants_automatic_distance,
            self.space_distance,
            self.angular_distance,
        ]

    def visible_settings(self):
        """The settings as they appear in the user interface"""
        result = [
            self.image_name,
            self.object_name,
            self.worm_width,
            self.worm_length,
            self.angle_count,
            self.wants_automatic_distance,
        ]
        if not self.wants_automatic_distance:
            result += [self.space_distance, self.angular_distance]
        return result

    def run(self, workspace):
        """Run the algorithm on one image set"""
        #
        # Get the image as a binary image
        #
        image_set = workspace.image_set
        image = image_set.get_image(self.image_name.value, must_be_binary=True)
        mask = image.pixel_data
        if image.has_mask:
            mask = mask & image.mask
        angle_count = self.angle_count.value
        #
        # We collect the i,j and angle of pairs of points that
        # are 3-d adjacent after erosion.
        #
        # i - the i coordinate of each point found after erosion
        # j - the j coordinate of each point found after erosion
        # a - the angle of the structuring element for each point found
        #
        i = numpy.zeros(0, int)
        j = numpy.zeros(0, int)
        a = numpy.zeros(0, int)

        ig, jg = numpy.mgrid[0 : mask.shape[0], 0 : mask.shape[1]]
        this_idx = 0
        for angle_number in range(angle_count):
            angle = float(angle_number) * numpy.pi / float(angle_count)
            strel = self.get_diamond(angle)
            erosion = binary_erosion(mask, strel)
            #
            # Accumulate the count, i, j and angle for all foreground points
            # in the erosion
            #
            this_count = numpy.sum(erosion)
            i = numpy.hstack((i, ig[erosion]))
            j = numpy.hstack((j, jg[erosion]))
            a = numpy.hstack((a, numpy.ones(this_count, float) * angle))
        #
        # Find connections based on distances, not adjacency
        #
        first, second = self.find_adjacent_by_distance(i, j, a)
        #
        # Do all connected components.
        #
        if len(first) > 0:
            ij_labels = all_connected_components(first, second) + 1
            nlabels = numpy.max(ij_labels)
            label_indexes = numpy.arange(1, nlabels + 1)
            #
            # Compute the measurements
            #
            center_x = fixup_scipy_ndimage_result(
                mean_of_labels(j, ij_labels, label_indexes)
            )
            center_y = fixup_scipy_ndimage_result(
                mean_of_labels(i, ij_labels, label_indexes)
            )
            #
            # The angles are wierdly complicated because of the wrap-around.
            # You can imagine some horrible cases, like a circular patch of
            # "worm" in which all angles are represented or a gentle "U"
            # curve.
            #
            # For now, I'm going to use the following heuristic:
            #
            # Compute two different "angles". The angles of one go
            # from 0 to 180 and the angles of the other go from -90 to 90.
            # Take the variance of these from the mean and
            # choose the representation with the lowest variance.
            #
            # An alternative would be to compute the variance at each possible
            # dividing point. Another alternative would be to actually trace through
            # the connected components - both overkill for such an inconsequential
            # measurement I hope.
            #
            angles = fixup_scipy_ndimage_result(
                mean_of_labels(a, ij_labels, label_indexes)
            )
            vangles = fixup_scipy_ndimage_result(
                mean_of_labels(
                    (a - angles[ij_labels - 1]) ** 2, ij_labels, label_indexes
                )
            )
            aa = a.copy()
            aa[a > numpy.pi / 2] -= numpy.pi
            aangles = fixup_scipy_ndimage_result(
                mean_of_labels(aa, ij_labels, label_indexes)
            )
            vaangles = fixup_scipy_ndimage_result(
                mean_of_labels(
                    (aa - aangles[ij_labels - 1]) ** 2, ij_labels, label_indexes
                )
            )
            aangles[aangles < 0] += numpy.pi
            angles[vaangles < vangles] = aangles[vaangles < vangles]
            #
            # Squish the labels to 2-d. The labels for overlaps are arbitrary.
            #
            labels = numpy.zeros(mask.shape, int)
            labels[i, j] = ij_labels
        else:
            center_x = numpy.zeros(0, int)
            center_y = numpy.zeros(0, int)
            angles = numpy.zeros(0)
            nlabels = 0
            label_indexes = numpy.zeros(0, int)
            labels = numpy.zeros(mask.shape, int)

        m = workspace.measurements
        assert isinstance(m, Measurements)
        object_name = self.object_name.value
        m.add_measurement(object_name, M_LOCATION_CENTER_X, center_x)
        m.add_measurement(object_name, M_LOCATION_CENTER_Y, center_y)
        m.add_measurement(object_name, M_ANGLE, angles * 180 / numpy.pi)
        m.add_measurement(
            object_name, M_NUMBER_OBJECT_NUMBER, label_indexes,
        )
        m.add_image_measurement(FF_COUNT % object_name, nlabels)
        #
        # Make the objects
        #
        object_set = workspace.object_set
        assert isinstance(object_set, ObjectSet)
        objects = Objects()
        objects.segmented = labels
        objects.parent_image = image
        object_set.add_objects(objects, object_name)
        if self.show_window:
            workspace.display_data.i = center_y
            workspace.display_data.j = center_x
            workspace.display_data.angle = angles
            workspace.display_data.mask = mask
            workspace.display_data.labels = labels
            workspace.display_data.count = nlabels

    def display(self, workspace, figure):
        """Show an informative display"""
        import matplotlib
        import cellprofiler.gui.figure

        figure.set_subplots((2, 1))
        assert isinstance(figure, cellprofiler.gui.figure.Figure)

        i = workspace.display_data.i
        j = workspace.display_data.j
        angles = workspace.display_data.angle
        mask = workspace.display_data.mask
        labels = workspace.display_data.labels
        count = workspace.display_data.count

        color_image = numpy.zeros((mask.shape[0], mask.shape[1], 4))
        #
        # We do the coloring using alpha values to let the different
        # things we draw meld together.
        #
        # The binary mask is white.
        #
        color_image[mask, :] = MASK_ALPHA
        if count > 0:
            mappable = matplotlib.cm.ScalarMappable(
                cmap=matplotlib.cm.get_cmap(get_default_colormap())
            )
            numpy.random.seed(0)
            colors = mappable.to_rgba(numpy.random.permutation(numpy.arange(count)))

            #
            # The labels
            #
            color_image[labels > 0, :] += (
                colors[labels[labels > 0] - 1, :] * LABEL_ALPHA
            )
            #
            # Do each diamond individually (because the angles are almost certainly
            # different for each
            #
            lcolors = colors * 0.5 + 0.5  # Wash the colors out a little
            for ii in range(count):
                diamond = self.get_diamond(angles[ii])
                hshape = ((numpy.array(diamond.shape) - 1) / 2).astype(int)
                iii = int(i[ii])
                jjj = int(j[ii])
                color_image[
                    iii - hshape[0] : iii + hshape[0] + 1,
                    jjj - hshape[1] : jjj + hshape[1] + 1,
                    :,
                ][diamond, :] += (lcolors[ii, :] * WORM_ALPHA)
        #
        # Do our own alpha-normalization
        #
        color_image[:, :, -1][color_image[:, :, -1] == 0] = 1
        color_image[:, :, :-1] = (
            color_image[:, :, :-1] / color_image[:, :, -1][:, :, numpy.newaxis]
        )
        plot00 = figure.subplot_imshow_bw(0, 0, mask, self.image_name.value)
        figure.subplot_imshow_color(
            1,
            0,
            color_image[:, :, :-1],
            title=self.object_name.value,
            normalize=False,
            sharexy=plot00,
        )

    def get_diamond(self, angle):
        """Get a diamond-shaped structuring element

        angle - angle at which to tilt the diamond

        returns a binary array that can be used as a footprint for
        the erosion
        """
        worm_width = self.worm_width.value
        worm_length = self.worm_length.value
        #
        # The shape:
        #
        #                   + x1,y1
        #
        # x0,y0 +                          + x2, y2
        #
        #                   + x3,y3
        #
        x0 = int(numpy.sin(angle) * worm_length / 2)
        x1 = int(numpy.cos(angle) * worm_width / 2)
        x2 = -x0
        x3 = -x1
        y2 = int(numpy.cos(angle) * worm_length / 2)
        y1 = int(numpy.sin(angle) * worm_width / 2)
        y0 = -y2
        y3 = -y1
        xmax = numpy.max(numpy.abs([x0, x1, x2, x3]))
        ymax = numpy.max(numpy.abs([y0, y1, y2, y3]))
        strel = numpy.zeros((ymax * 2 + 1, xmax * 2 + 1), bool)
        index, count, i, j = get_line_pts(
            numpy.array([y0, y1, y2, y3]) + ymax,
            numpy.array([x0, x1, x2, x3]) + xmax,
            numpy.array([y1, y2, y3, y0]) + ymax,
            numpy.array([x1, x2, x3, x0]) + xmax,
        )
        strel[i, j] = True
        strel = binary_fill_holes(strel)
        return strel

    @staticmethod
    def find_adjacent(img1, offset1, count1, img2, offset2, count2, first, second):
        """Find adjacent pairs of points between two masks

        img1, img2 - binary images to be 8-connected
        offset1 - number the foreground points in img1 starting at this offset
        count1 - number of foreground points in img1
        offset2 - number the foreground points in img2 starting at this offset
        count2 - number of foreground points in img2
        first, second - prior collection of points

        returns augmented collection of points
        """
        numbering1 = numpy.zeros(img1.shape, int)
        numbering1[img1] = numpy.arange(count1) + offset1
        numbering2 = numpy.zeros(img1.shape, int)
        numbering2[img2] = numpy.arange(count2) + offset2

        f = numpy.zeros(0, int)
        s = numpy.zeros(0, int)
        #
        # Do all 9
        #
        for oi in (-1, 0, 1):
            for oj in (-1, 0, 1):
                f1, s1 = IdentifyDeadWorms.find_adjacent_one(
                    img1, numbering1, img2, numbering2, oi, oj
                )
                f = numpy.hstack((f, f1))
                s = numpy.hstack((s, s1))
        return numpy.hstack((first, f)), numpy.hstack((second, s))

    @staticmethod
    def find_adjacent_same(img, offset, count, first, second):
        """Find adjacent pairs of points in the same mask
        img - binary image to be 8-connected
        offset - where to start numbering
        count - number of foreground points in image
        first, second - prior collection of points

        returns augmented collection of points
        """
        numbering = numpy.zeros(img.shape, int)
        numbering[img] = numpy.arange(count) + offset
        f = numpy.zeros(0, int)
        s = numpy.zeros(0, int)
        for oi in (0, 1):
            for oj in (0, 1):
                f1, s1 = IdentifyDeadWorms.find_adjacent_one(
                    img, numbering, img, numbering, oi, oj
                )
                f = numpy.hstack((f, f1))
                s = numpy.hstack((s, s1))
        return numpy.hstack((first, f)), numpy.hstack((second, s))

    @staticmethod
    def find_adjacent_one(img1, numbering1, img2, numbering2, oi, oj):
        """Find correlated pairs of foreground points at given offsets

        img1, img2 - binary images to be correlated
        numbering1, numbering2 - indexes to be returned for pairs
        oi, oj - offset for second image

        returns two vectors: index in first and index in second
        """
        i1, i2 = IdentifyDeadWorms.get_slices(oi)
        j1, j2 = IdentifyDeadWorms.get_slices(oj)
        match = img1[i1, j1] & img2[i2, j2]
        return numbering1[i1, j1][match], numbering2[i2, j2][match]

    def find_adjacent_by_distance(self, i, j, a):
        """Return pairs of worm centers that are deemed adjacent by distance

        i - i-centers of worms
        j - j-centers of worms
        a - angular orientation of worms

        Returns two vectors giving the indices of the first and second
        centers that are connected.
        """
        if len(i) < 2:
            return numpy.zeros(len(i), int), numpy.zeros(len(i), int)
        if self.wants_automatic_distance:
            space_distance = self.worm_width.value
            angle_distance = numpy.arctan2(
                self.worm_width.value, self.worm_length.value
            )
            angle_distance += numpy.pi / self.angle_count.value
        else:
            space_distance = self.space_distance.value
            angle_distance = self.angular_distance.value * numpy.pi / 180
        #
        # Sort by i and break the sorted vector into chunks where
        # consecutive locations are separated by more than space_distance
        #
        order = numpy.lexsort((a, j, i))
        i = i[order]
        j = j[order]
        a = a[order]
        breakpoint = numpy.hstack(([False], i[1:] - i[:-1] > space_distance))
        if numpy.all(~breakpoint):
            # No easy win - cross all with all
            first, second = numpy.mgrid[0 : len(i), 0 : len(i)]
        else:
            # The segment that each belongs to
            segment_number = numpy.cumsum(breakpoint)
            # The number of elements in each segment
            member_count = numpy.bincount(segment_number)
            # The index of the first element in the segment
            member_idx = numpy.hstack(([0], numpy.cumsum(member_count[:-1])))
            # The index of the first element, for every element in the segment
            segment_start = member_idx[segment_number]
            #
            # Develop the cross-products for each segment. Each segment has
            # member_count * member_count crosses.
            #
            # # of (first,second) pairs in each segment
            cross_size = member_count ** 2
            # Index in final array of first element of each segment
            segment_idx = numpy.cumsum(cross_size)
            # relative location of first "first"
            first_start_idx = numpy.cumsum(member_count[segment_number[:-1]])
            first = numpy.zeros(segment_idx[-1], int)
            first[first_start_idx] = 1
            # The "firsts" array
            first = numpy.cumsum(first)
            first_start_idx = numpy.hstack(([0], first_start_idx))
            second = (
                numpy.arange(len(first)) - first_start_idx[first] + segment_start[first]
            )
        mask = (
            numpy.abs((i[first] - i[second]) ** 2 + (j[first] - j[second]) ** 2)
            <= space_distance ** 2
        ) & (
            (numpy.abs(a[first] - a[second]) <= angle_distance)
            | (a[first] + numpy.pi - a[second] <= angle_distance)
            | (a[second] + numpy.pi - a[first] <= angle_distance)
        )
        return order[first[mask]], order[second[mask]]

    @staticmethod
    def get_slices(offset):
        """Get slices to use for a pair of arrays, given an offset

        offset - offset to be applied to the second array

        An offset imposes border conditions on an array, for instance,
        an offset of 1 means that the first array has a slice of :-1
        and the second has a slice of 1:. Return the slice to use
        for the first and second arrays.
        """
        if offset > 0:
            s0, s1 = slice(0, -offset), slice(offset, numpy.iinfo(int).max)
        elif offset < 0:
            s1, s0 = IdentifyDeadWorms.get_slices(-offset)
        else:
            s0 = s1 = slice(0, numpy.iinfo(int).max)
        return s0, s1

    def get_measurement_columns(self, pipeline):
        """Return column definitions for measurements made by this module"""
        object_name = self.object_name.value
        return [
            (object_name, M_LOCATION_CENTER_X, COLTYPE_INTEGER,),
            (object_name, M_LOCATION_CENTER_Y, COLTYPE_INTEGER,),
            (object_name, M_ANGLE, COLTYPE_FLOAT),
            (object_name, M_NUMBER_OBJECT_NUMBER, COLTYPE_INTEGER,),
            (IMAGE, FF_COUNT % object_name, COLTYPE_INTEGER,),
        ]

    def get_categories(self, pipeline, object_name):
        if object_name == IMAGE:
            return [C_COUNT]
        elif object_name == self.object_name:
            return [
                C_LOCATION,
                C_NUMBER,
                C_WORMS,
            ]
        else:
            return []

    def get_measurements(self, pipeline, object_name, category):
        if object_name == IMAGE and category == C_COUNT:
            return [self.object_name.value]
        elif object_name == self.object_name:
            if category == C_LOCATION:
                return [
                    FTR_CENTER_X,
                    FTR_CENTER_Y,
                ]
            elif category == C_NUMBER:
                return [FTR_OBJECT_NUMBER]
            elif category == C_WORMS:
                return [F_ANGLE]
        return []

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        """Upgrade the settings from a previous revison"""
        if variable_revision_number == 1:
            setting_values = setting_values + ["Yes", 5, 30]
            variable_revision_number = 2
        return setting_values, variable_revision_number
