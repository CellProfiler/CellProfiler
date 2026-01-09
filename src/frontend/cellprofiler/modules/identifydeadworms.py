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

from cellprofiler_library.modules._identifydeadworms import identify_dead_worms

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
        image_mask = image.mask if image.has_mask else None
        angle_count = self.angle_count.value
        #
        # Perform the identification
        #
        center_x, center_y, angles, nlabels, label_indexes, labels = identify_dead_worms(
            image.pixel_data,
            image_mask,
            self.wants_automatic_distance.value,
            self.worm_width.value,
            self.worm_length.value,
            self.angle_count.value,
            self.space_distance.value,
            self.angular_distance.value
        )

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
                diamond = self.get_diamond(angles[ii], self.worm_width.value, self.worm_length.value)
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
