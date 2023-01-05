"""
StraightenWorms
===============

**StraightenWorms** straightens untangled worms.

**StraightenWorms** uses the objects produced by **UntangleWorms** to
create images and objects of straight worms from the angles and control
points as computed by **UntangleWorms**. The resulting images can then
be uniformly analyzed to find features that correlate with position in
an ideal representation of the worm, such as the head or gut.
**StraightenWorms** works by calculating a transform on the image that
translates points in the image to points on the ideal worm.
**UntangleWorms** idealizes a worm as a series of control points that
define the worm’s shape and length. The training set contains
measurements of the width of an ideal worm at each control point.
Together, these can be used to reconstruct the worm’s shape and
correlate between the worm’s location and points on the body of an ideal
worm. **StraightenWorms** produces objects representing the straight
worms and images representing the intensity values of a source image
mapped onto the straight worms. The objects and images can then be used
to compute measurements using any of the object measurement modules, for
instance, **MeasureTexture**. The module can be configured to make
intensity measurements on parts of the worm, dividing the worm up into
pieces of equal width and/or height. Measurements are made longitudinally
in stripes from head to tail and transversely in segments across the
width of the worm. Longitudinal stripes are numbered from left to right
and transverse segments are numbered from top to bottom. The module will
divide the worm into a checkerboard of sections if configured to measure
more than one longitudinal stripe and transverse segment. These are
numbered by longitudinal stripe number, then transverse segment number.
For instance, “Worm\_MeanIntensity\_GFP\_L2of3\_T1of4”, is a measurement
of the mean GFP intensity of the center stripe (second of 3 stripes) of
the topmost band (first of four bands). Measurements of longitudinal
stripes are designated as “T1of1” indicating that the whole worm is one
transverse segment. Likewise measurements of transverse segments are
designated as “L1of1” indicating that there is only one longitudinal
stripe. Both mean intensity and standard deviation of intensity are
measured per worm sub-area. While **StraightenWorms** can straighten a
color image, the module needs a grayscale image to make its intensity
measurements. For a color image, the red, green and blue channels are
averaged to yield a grayscale image. The intensity measurements are then
made on that grayscale image.

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

**Object measurements:**

-  *Location\_X, Location\_Y:* The pixel (X,Y) coordinates of the
   primary object centroids. The centroid is calculated as the center of
   mass of the binary representation of the object.
-  *Worm\_MeanIntensity:* The average pixel intensity within a worm.
-  *Worm\_StdIntensity:* The standard deviation of the pixel intensities
   within a worm.

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

.. _Worm Toolbox: http://www.cellprofiler.org/wormtoolbox/
"""

import functools
import itertools
import os

import cellprofiler_core.utilities.legacy
import centrosome.index
import numpy
import scipy.ndimage
from cellprofiler_core.constants.measurement import (
    COLTYPE_FLOAT,
    IMAGE,
    C_COUNT,
    C_LOCATION,
    C_NUMBER,
    FTR_CENTER_X,
    FTR_CENTER_Y,
    FTR_OBJECT_NUMBER,
)
from cellprofiler_core.constants.module import IO_FOLDER_CHOICE_HELP_TEXT
from cellprofiler_core.image import Image
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.module import Module
from cellprofiler_core.object import ObjectSet
from cellprofiler_core.object import Objects
from cellprofiler_core.preferences import URL_FOLDER_NAME
from cellprofiler_core.preferences import get_primary_outline_color
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting import Divider
from cellprofiler_core.setting import HiddenCount
from cellprofiler_core.setting import SettingsGroup
from cellprofiler_core.setting import ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting.subscriber import LabelSubscriber, ImageSubscriber
from cellprofiler_core.setting.text import (
    Integer,
    Directory,
    LabelName,
    ImageName,
    Filename,
)
from cellprofiler_core.utilities.core.module.identify import (
    get_object_measurement_columns,
    add_object_location_measurements,
    add_object_count_measurements,
)
from scipy.interpolate import interp1d

from cellprofiler.modules.untangleworms import C_WORM
from cellprofiler.modules.untangleworms import F_CONTROL_POINT_X
from cellprofiler.modules.untangleworms import F_CONTROL_POINT_Y
from cellprofiler.modules.untangleworms import F_LENGTH
from cellprofiler.modules.untangleworms import read_params
from cellprofiler.modules.untangleworms import recalculate_single_worm_control_points

FTR_MEAN_INTENSITY = "MeanIntensity"
FTR_STD_INTENSITY = "StdIntensity"

"""The horizontal scale label - T = Transverse, a transverse strip"""
SCALE_HORIZONTAL = "T"

"""The vertical scale label - L = Longitudinal, a longitudinal strip"""
SCALE_VERTICAL = "L"

FLIP_NONE = "Do not align"
FLIP_TOP = "Top brightest"
FLIP_BOTTOM = "Bottom brightest"
FLIP_MANUAL = "Flip manually"

"""The index of the image count setting (# of images to process)"""
IDX_IMAGE_COUNT_V1 = 5
IDX_IMAGE_COUNT_V2 = 5
IDX_IMAGE_COUNT_V3 = 5
IDX_IMAGE_COUNT = 5
IDX_FLIP_WORMS_V2 = 8

FIXED_SETTINGS_COUNT_V1 = 6
VARIABLE_SETTINGS_COUNT_V1 = 2
FIXED_SETTINGS_COUNT_V2 = 10
VARIABLE_SETTINGS_COUNT_V2 = 2
FIXED_SETTINGS_COUNT_V3 = 11
VARIABLE_SETTINGS_COUNT_V3 = 2


class StraightenWorms(Module):
    variable_revision_number = 3
    category = ["Worm Toolbox"]
    module_name = "StraightenWorms"

    def create_settings(self):
        """Create the settings for the module"""
        self.images = []

        self.objects_name = LabelSubscriber(
            "Select the input untangled worm objects",
            "OverlappingWorms",
            doc="""\
This is the name of the objects produced by the **UntangleWorms**
module. **StraightenWorms** can use either the overlapping or
non-overlapping objects as input. It will use the control point
measurements associated with the objects to reconstruct the straight
worms. You can also use objects saved from a previous run and loaded via
the **Input** modules, objects edited using **EditObjectsManually** or
objects from one of the Identify modules. **StraightenWorms** will
recalculate the control points for these images.
""",
        )

        self.straightened_objects_name = LabelName(
            "Name the output straightened worm objects",
            "StraightenedWorms",
            doc="""\
This is the name that will be given to the straightened
worm objects. These objects can then be used in a subsequent
measurement module.""",
        )

        self.width = Integer(
            "Worm width",
            20,
            minval=3,
            doc="""\
This setting determines the width of the image of each
worm. The width should be set to at least the maximum width of
any untangled worm, but can be set to be larger to include the
worm's background in the straightened image.""",
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
   *http://my_institution.edu/server/my_username/TrainingSet.xml* To
   access this file, you would choose *URL* and enter
   *http://my_institution.edu/server/my_username/* as the path
   location.
""".format(
                folder_choice=IO_FOLDER_CHOICE_HELP_TEXT
            ),
        )

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
            doc="This is the name of the training set file.",
            get_directory_fn=get_directory_fn,
            set_directory_fn=set_directory_fn,
            browse_msg="Choose training set",
            exts=[("Worm training set (*.xml)", "*.xml"), ("All files (*.*)", "*.*")],
        )

        self.wants_measurements = Binary(
            "Measure intensity distribution?",
            True,
            doc="""\
Select *Yes* to divide a worm into sections and measure the
intensities of each section in each of the straightened images. These
measurements can help classify phenotypes if the staining pattern across
the segments differs between phenotypes.
"""
            % globals(),
        )

        self.number_of_segments = Integer(
            "Number of transverse segments",
            4,
            1,
            doc="""\
(*Only used if intensities are measured*)

This setting controls the number of segments measured, dividing the worm
longitudinally into transverse segments starting at the head and ending at
the tail. These measurements might be used to identify a phenotype in
which a stain is localized longitudinally, for instance, in the head. Set
the number of vertical segments to 1 to only measure intensity in the
horizontal direction.
""",
        )

        self.number_of_stripes = Integer(
            "Number of longitudinal stripes",
            3,
            1,
            doc="""\
(*Only used if intensities are measured*)

This setting controls the number of stripes measured, dividing the worm
transversely into areas that run longitudinally. These measurements might
be used to identify a phenotype in which a stain is localized
transversely, for instance in the gut of the worm. Set the number of
horizontal stripes to 1 to only measure intensity in the vertical
direction.
""",
        )

        self.flip_worms = Choice(
            "Align worms?",
            [FLIP_NONE, FLIP_TOP, FLIP_BOTTOM, FLIP_MANUAL],
            doc="""\
(*Only used if intensities are measured*)

**StraightenWorms** can align worms so that the brightest half of the
worm (the half with the highest mean intensity) is at the top of the
image or at the bottom of the image. This can be used to align all
worms similarly if some feature, such as the larynx, is stained and is
always at the same end of the worm.

-  *%(FLIP_TOP)s:* The brightest part of the worm should be at the top
   of the image.
-  *%(FLIP_BOTTOM)s:* The brightest part of the worm should be at the
   bottom.
-  *%(FLIP_NONE)s:* The worm should not be aligned.
-  *%(FLIP_MANUAL)s:* Bring up an editor for every cycle that allows
   you to choose the orientation of each worm.
"""
            % globals(),
        )

        def image_choices_fn(pipeline):
            """Return the image choices for the alignment image"""
            return [group.image_name.value for group in self.images]

        self.flip_image = Choice(
            "Alignment image",
            ["None"],
            choices_fn=image_choices_fn,
            doc="""
(*Only used if aligning worms*)

This is the image whose intensity will be used to align the worms.
You must use one of the straightened images below.""",
        )

        self.image_count = HiddenCount(self.images, "Image count")

        self.add_image(False)

        self.add_image_button = DoSomething(
            "",
            "Add another image",
            self.add_image,
            doc="""Press this button to add another image to be straightened""",
        )

    def add_image(self, can_delete=True):
        """Add an image to the list of images to be straightened"""

        group = SettingsGroup()
        group.append("divider", Divider())
        group.append(
            "image_name",
            ImageSubscriber(
                "Select an input image to straighten",
                "None",
                doc="""\
This is the name of an image that will be straightened
similarly to the worm. The straightened image and objects can
then be used in subsequent modules such as
**MeasureObjectIntensity**.""",
            ),
        )

        group.append(
            "straightened_image_name",
            ImageName(
                "Name the output straightened image",
                "StraightenedImage",
                doc="""
This is the name that will be given to the image
of the straightened worms.""",
            ),
        )

        if can_delete:
            group.append(
                "remover",
                RemoveSettingButton("", "Remove above image", self.images, group),
            )
        self.images.append(group)

    def settings(self):
        """Return the settings, in the order they appear in the pipeline"""
        result = [
            self.objects_name,
            self.straightened_objects_name,
            self.width,
            self.training_set_directory,
            self.training_set_file_name,
            self.image_count,
            self.wants_measurements,
            self.number_of_segments,
            self.number_of_stripes,
            self.flip_worms,
            self.flip_image,
        ] + sum([group.pipeline_settings() for group in self.images], [])
        return result

    def visible_settings(self):
        """Return the settings as displayed in the module view"""
        result = [
            self.objects_name,
            self.straightened_objects_name,
            self.width,
            self.training_set_directory,
            self.training_set_file_name,
            self.wants_measurements,
        ]
        if self.wants_measurements:
            result += [self.number_of_segments, self.number_of_stripes, self.flip_worms]
            if self.flip_worms in (FLIP_BOTTOM, FLIP_TOP):
                result += [self.flip_image]
        result += sum([group.visible_settings() for group in self.images], [])
        result += [self.add_image_button]
        return result

    def validate_module(self, pipeline):
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
        if (
            self.wants_measurements
            and self.number_of_segments == 1
            and self.number_of_stripes == 1
        ):
            raise ValidationError(
                "No measurements will be produced if the number of "
                "longitudinal stripes and the number of transverse segments "
                "are both equal to one. Please turn measurements off or change "
                "the number of stripes or segments.",
                self.wants_measurements,
            )

    def prepare_settings(self, setting_values):
        nimages = int(setting_values[IDX_IMAGE_COUNT])
        del self.images[1:]
        for i in range(1, nimages):
            self.add_image()

    K_PIXEL_DATA = "pixel_data"
    K_MASK = "mask"
    K_NAME = "name"
    K_PARENT_IMAGE = "__parent_image"
    K_PARENT_IMAGE_NAME = "__parent_image_name"

    class InteractionCancelledException(RuntimeError):
        def __init__(self, *args):
            if len(args) == 0:
                args = ["User cancelled StraightenWorms"]
            super(self.__class__, self).__init__(*args)

    def run(self, workspace):
        """Process one image set"""
        object_set = workspace.object_set
        assert isinstance(object_set, ObjectSet)

        image_set = workspace.image_set

        objects_name = self.objects_name.value
        orig_objects = object_set.get_objects(objects_name)
        assert isinstance(orig_objects, Objects)
        m = workspace.measurements
        assert isinstance(m, Measurements)
        #
        # Sort the features by control point number:
        # Worm_ControlPointX_2 < Worm_ControlPointX_10
        #
        features = m.get_feature_names(objects_name)
        cpx = [
            f for f in features if f.startswith("_".join((C_WORM, F_CONTROL_POINT_X)))
        ]
        cpy = [
            f for f in features if f.startswith("_".join((C_WORM, F_CONTROL_POINT_Y)))
        ]
        ncontrolpoints = len(cpx)
        if ncontrolpoints == 0:
            #
            # Recalculate control points.
            #
            params = self.read_params(workspace)
            ncontrolpoints = params.num_control_points
            all_labels = [l for l, idx in orig_objects.get_labels()]
            control_points, lengths = recalculate_single_worm_control_points(
                all_labels, ncontrolpoints
            )
            control_points = control_points.transpose(2, 1, 0)
        else:

            def sort_fn(a, b):
                """Sort by control point number"""
                acp = int(a.split("_")[-1])
                bcp = int(b.split("_")[-1])
                return cellprofiler_core.utilities.legacy.cmp(acp, bcp)

            cpx.sort(key=functools.cmp_to_key(sort_fn))
            cpy.sort(key=functools.cmp_to_key(sort_fn))

            control_points = numpy.array(
                [
                    [m.get_current_measurement(objects_name, f) for f in cp]
                    for cp in (cpy, cpx)
                ]
            )
            m_length = "_".join((C_WORM, F_LENGTH))
            lengths = numpy.ceil(m.get_current_measurement(objects_name, m_length))

        nworms = len(lengths)
        half_width = self.width.value // 2
        width = 2 * half_width + 1
        if nworms == 0:
            shape = (width, width)
        else:
            shape = (int(numpy.max(lengths)) + width, nworms * width)
        labels = numpy.zeros(shape, int)
        #
        # ix and jx are the coordinates of the straightened pixel in the
        # original space.
        #
        ix = numpy.zeros(shape)
        jx = numpy.zeros(shape)
        #
        # This is a list of tuples - first element in the tuples is
        # a labels matrix, second is a list of indexes in the matrix.
        # We need this for overlapping worms.
        #
        orig_labels_and_indexes = orig_objects.get_labels()
        #
        # Handle each of the worm splines separately
        #
        for i in range(nworms):
            if lengths[i] == 0:
                continue
            object_number = i + 1
            orig_labels = [
                x
                for x, y in orig_labels_and_indexes
                if object_number in y and object_number in x
            ]
            if len(orig_labels) == 0:
                continue
            orig_labels = orig_labels[0]

            ii = control_points[0, :, i]
            jj = control_points[1, :, i]

            si = interp1d(numpy.linspace(0, lengths[i], ncontrolpoints), ii)
            sj = interp1d(numpy.linspace(0, lengths[i], ncontrolpoints), jj)
            #
            # The coordinates of "length" points along the worm
            #
            ci = si(numpy.arange(0, int(lengths[i]) + 1))
            cj = sj(numpy.arange(0, int(lengths[i]) + 1))
            #
            # Find the normals at each point by taking the derivative,
            # and twisting by 90 degrees.
            #
            di = ci[1:] - ci[:-1]
            di = numpy.hstack([[di[0]], di])
            dj = cj[1:] - cj[:-1]
            dj = numpy.hstack([[dj[0]], dj])
            ni = -dj / numpy.sqrt(di ** 2 + dj ** 2)
            nj = di / numpy.sqrt(di ** 2 + dj ** 2)
            #
            # Extend the worm out from the head and tail by the width
            #
            ci = numpy.hstack(
                [
                    numpy.arange(-half_width, 0) * nj[0] + ci[0],
                    ci,
                    numpy.arange(1, half_width + 1) * nj[-1] + ci[-1],
                ]
            )
            cj = numpy.hstack(
                [
                    numpy.arange(-half_width, 0) * (-ni[0]) + cj[0],
                    cj,
                    numpy.arange(1, half_width + 1) * (-ni[-1]) + cj[-1],
                ]
            )
            ni = numpy.hstack([[ni[0]] * half_width, ni, [ni[-1]] * half_width])
            nj = numpy.hstack([[nj[0]] * half_width, nj, [nj[-1]] * half_width])
            iii, jjj = numpy.mgrid[0 : len(ci), -half_width : (half_width + 1)]

            #
            # Create a mapping of i an j in straightened space to
            # the coordinates in real space
            #
            islice = slice(0, len(ci))
            jslice = slice(width * i, width * (i + 1))
            ix[islice, jslice] = ci[iii] + ni[iii] * jjj
            jx[islice, jslice] = cj[iii] + nj[iii] * jjj
            #
            # We may need to flip the worm
            #
            if self.flip_worms in (FLIP_TOP, FLIP_BOTTOM):
                ixs = ix[islice, jslice]
                jxs = jx[islice, jslice]
                image_name = self.flip_image.value
                image = image_set.get_image(image_name, must_be_grayscale=True)
                simage = scipy.ndimage.map_coordinates(image.pixel_data, [ixs, jxs])
                halfway = int(len(ci)) / 2
                smask = scipy.ndimage.map_coordinates(orig_labels == i + 1, [ixs, jxs])
                if image.has_mask:
                    smask *= scipy.ndimage.map_coordinates(image.mask, [ixs, jxs])
                simage *= smask
                #
                # Compute the mean intensity of the top and bottom halves
                # of the worm.
                #
                area_top = numpy.sum(smask[: int(halfway), :])
                area_bottom = numpy.sum(smask[int(halfway) :, :])
                top_intensity = numpy.sum(simage[: int(halfway), :]) / area_top
                bottom_intensity = numpy.sum(simage[int(halfway) :, :]) / area_bottom
                if (top_intensity > bottom_intensity) != (self.flip_worms == FLIP_TOP):
                    # Flip worm if it doesn't match user expectations
                    iii = len(ci) - iii - 1
                    jjj = -jjj
                    ix[islice, jslice] = ci[iii] + ni[iii] * jjj
                    jx[islice, jslice] = cj[iii] + nj[iii] * jjj
            mask = (
                scipy.ndimage.map_coordinates(
                    (orig_labels == i + 1).astype(numpy.float32),
                    [ix[islice, jslice], jx[islice, jslice]],
                )
                > 0.5
            )
            labels[islice, jslice][mask] = object_number
        #
        # Now create one straightened image for each input image
        #
        straightened_images = []
        for group in self.images:
            image_name = group.image_name.value
            straightened_image_name = group.straightened_image_name.value
            image = image_set.get_image(image_name)
            if image.pixel_data.ndim == 2:
                straightened_pixel_data = scipy.ndimage.map_coordinates(
                    image.pixel_data, [ix, jx]
                )
            else:
                straightened_pixel_data = numpy.zeros(
                    (ix.shape[0], ix.shape[1], image.pixel_data.shape[2])
                )
                for d in range(image.pixel_data.shape[2]):
                    straightened_pixel_data[:, :, d] = scipy.ndimage.map_coordinates(
                        image.pixel_data[:, :, d], [ix, jx]
                    )
            straightened_mask = (
                scipy.ndimage.map_coordinates(image.mask, [ix, jx]) > 0.5
            )
            straightened_images.append(
                {
                    self.K_NAME: straightened_image_name,
                    self.K_PIXEL_DATA: straightened_pixel_data,
                    self.K_MASK: straightened_mask,
                    self.K_PARENT_IMAGE: image,
                    self.K_PARENT_IMAGE_NAME: image_name,
                }
            )
        if self.flip_worms == FLIP_MANUAL:
            result, labels = workspace.interaction_request(
                self, straightened_images, labels, m.image_set_number
            )
            for dorig, dedited in zip(straightened_images, result):
                dorig[self.K_PIXEL_DATA] = dedited[self.K_PIXEL_DATA]
                dorig[self.K_MASK] = dedited[self.K_MASK]

        if self.show_window:
            workspace.display_data.image_pairs = []
        for d in straightened_images:
            image = d[self.K_PARENT_IMAGE]
            image_name = d[self.K_PARENT_IMAGE_NAME]
            straightened_image_name = d[self.K_NAME]
            straightened_pixel_data = d[self.K_PIXEL_DATA]
            straightened_image = Image(
                d[self.K_PIXEL_DATA], d[self.K_MASK], parent_image=image
            )
            image_set.add(straightened_image_name, straightened_image)
            if self.show_window:
                workspace.display_data.image_pairs.append(
                    (
                        (image.pixel_data, image_name),
                        (straightened_pixel_data, straightened_image_name),
                    )
                )
        #
        # Measure the worms if appropriate
        #
        if self.wants_measurements:
            self.measure_worms(workspace, labels, nworms, width)
        #
        # Record the objects
        #
        self.make_objects(workspace, labels, nworms)

    def read_params(self, workspace):
        """Read the training params or use the cached value"""
        if not hasattr(self, "training_params"):
            self.training_params = {}
        params = read_params(
            self.training_set_directory,
            self.training_set_file_name,
            self.training_params,
        )
        return params

    def measure_worms(self, workspace, labels, nworms, width):
        m = workspace.measurements
        assert isinstance(m, Measurements)
        object_name = self.straightened_objects_name.value
        input_object_name = self.objects_name.value
        nbins_vertical = self.number_of_segments.value
        nbins_horizontal = self.number_of_stripes.value
        params = self.read_params(workspace)
        if nworms == 0:
            # # # # # # # # # # # # # # # # # # # # # #
            #
            # Record measurements if no worms
            #
            # # # # # # # # # # # # # # # # # # # # # #
            for ftr in (FTR_MEAN_INTENSITY, FTR_STD_INTENSITY):
                for group in self.images:
                    image_name = group.straightened_image_name.value
                    if nbins_vertical > 1:
                        for b in range(nbins_vertical):
                            measurement = "_".join(
                                (C_WORM, ftr, image_name, self.get_scale_name(None, b))
                            )
                            m.add_measurement(
                                input_object_name, measurement, numpy.zeros(0)
                            )
                    if nbins_horizontal > 1:
                        for b in range(nbins_horizontal):
                            measurement = "_".join(
                                (C_WORM, ftr, image_name, self.get_scale_name(b, None))
                            )
                            m.add_measurement(
                                input_object_name, measurement, numpy.zeros(0)
                            )
                        if nbins_vertical > 1:
                            for v in range(nbins_vertical):
                                for h in range(nbins_horizontal):
                                    measurement = "_".join(
                                        (
                                            C_WORM,
                                            ftr,
                                            image_name,
                                            self.get_scale_name(h, v),
                                        )
                                    )
                                    m.add_measurement(
                                        input_object_name, measurement, numpy.zeros(0)
                                    )

        else:
            #
            # Find the minimum and maximum i coordinate of each worm
            #
            object_set = workspace.object_set
            assert isinstance(object_set, ObjectSet)
            orig_objects = object_set.get_objects(input_object_name)

            i, j = numpy.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]
            min_i, max_i, _, _ = scipy.ndimage.extrema(i, labels, orig_objects.indices)
            min_i = numpy.hstack(([0], min_i))
            max_i = numpy.hstack(([labels.shape[0]], max_i)) + 1
            heights = max_i - min_i

            # # # # # # # # # # # # # # # # #
            #
            # Create up to 3 spaces which represent the gridding
            # of the worm and create a coordinate mapping into
            # this gridding for each straightened worm
            #
            # # # # # # # # # # # # # # # # #
            griddings = []
            if nbins_vertical > 1:
                scales = numpy.array(
                    [self.get_scale_name(None, b) for b in range(nbins_vertical)]
                )
                scales.shape = (nbins_vertical, 1)
                griddings += [(nbins_vertical, 1, scales)]
            if nbins_horizontal > 1:
                scales = numpy.array(
                    [self.get_scale_name(b, None) for b in range(nbins_horizontal)]
                )
                scales.shape = (1, nbins_horizontal)
                griddings += [(1, nbins_horizontal, scales)]
                if nbins_vertical > 1:
                    scales = numpy.array(
                        [
                            [self.get_scale_name(h, v) for h in range(nbins_horizontal)]
                            for v in range(nbins_vertical)
                        ]
                    )
                    griddings += [(nbins_vertical, nbins_horizontal, scales)]

            for i_dim, j_dim, scales in griddings:
                # # # # # # # # # # # # # # # # # # # # # #
                #
                # Start out mapping every point to a 1x1 space
                #
                # # # # # # # # # # # # # # # # # # # # # #
                labels1 = labels.copy()
                i, j = numpy.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]
                i_frac = (i - min_i[labels]).astype(float) / heights[labels]
                i_frac_end = i_frac + 1.0 / heights[labels].astype(float)
                i_radius_frac = (i - min_i[labels]).astype(float) / (
                    heights[labels] - 1
                )
                labels1[(i_frac >= 1) | (i_frac_end <= 0)] = 0
                # # # # # # # # # # # # # # # # # # # # # #
                #
                # Map the horizontal onto the grid.
                #
                # # # # # # # # # # # # # # # # # # # # # #
                radii = numpy.array(params.radii_from_training)
                #
                # For each pixel in the image, find the center of its worm
                # in the j direction (the width)
                #
                j_center = int(width / 2) + width * (labels - 1)
                #
                # Find which segment (from the training set) per pixel in
                # a fractional form
                #
                i_index = i_radius_frac * (len(radii) - 1)
                #
                # Interpolate
                #
                i_index_frac = i_index - numpy.floor(i_index)
                i_index_frac[i_index >= len(radii) - 1] = 1
                i_index = numpy.minimum(i_index.astype(int), len(radii) - 2)
                r = numpy.ceil(
                    (
                        radii[i_index] * (1 - i_index_frac)
                        + radii[i_index + 1] * i_index_frac
                    )
                )
                #
                # Map the worm width into the space 0-1
                #
                j_frac = (j - j_center + r) / (r * 2 + 1)
                j_frac_end = j_frac + 1.0 / (r * 2 + 1)
                labels1[(j_frac >= 1) | (j_frac_end <= 0)] = 0
                #
                # Map the worms onto the gridding.
                #
                i_mapping = numpy.maximum(i_frac * i_dim, 0)
                i_mapping_end = numpy.minimum(i_frac_end * i_dim, i_dim)
                j_mapping = numpy.maximum(j_frac * j_dim, 0)
                j_mapping_end = numpy.minimum(j_frac_end * j_dim, j_dim)
                i_mapping = i_mapping[labels1 > 0]
                i_mapping_end = i_mapping_end[labels1 > 0]
                j_mapping = j_mapping[labels1 > 0]
                j_mapping_end = j_mapping_end[labels1 > 0]
                labels_1d = labels1[labels1 > 0]
                i = i[labels1 > 0]
                j = j[labels1 > 0]

                #
                # There are easy cases and hard cases. The easy cases are
                # when a pixel in the input space wholly falls in the
                # output space.
                #
                easy = (i_mapping.astype(int) == i_mapping_end.astype(int)) & (
                    j_mapping.astype(int) == j_mapping_end.astype(int)
                )

                i_src = i[easy]
                j_src = j[easy]
                i_dest = i_mapping[easy].astype(int)
                j_dest = j_mapping[easy].astype(int)
                weight = numpy.ones(i_src.shape)
                labels_src = labels_1d[easy]
                #
                # The hard cases start in one pixel in the binning space,
                # possibly continue through one or more intermediate pixels
                # in horribly degenerate cases and end in a final
                # partial pixel.
                #
                # More horribly, a pixel in the straightened space
                # might span two or more in the binning space in the I
                # direction, the J direction or both.
                #
                if not numpy.all(easy):
                    i = i[~easy]
                    j = j[~easy]
                    i_mapping = i_mapping[~easy]
                    j_mapping = j_mapping[~easy]
                    i_mapping_end = i_mapping_end[~easy]
                    j_mapping_end = j_mapping_end[~easy]
                    labels_1d = labels_1d[~easy]
                    #
                    # A pixel in the straightened space can be wholly within
                    # a pixel in the bin space, it can straddle two pixels
                    # or straddle two and span one or more. It can do different
                    # things in the I and J direction.
                    #
                    # --- The number of pixels wholly spanned ---
                    #
                    i_span = numpy.maximum(
                        numpy.floor(i_mapping_end) - numpy.ceil(i_mapping), 0
                    )
                    j_span = numpy.maximum(
                        numpy.floor(j_mapping_end) - numpy.ceil(j_mapping), 0
                    )
                    #
                    # --- The fraction of a pixel covered by the lower straddle
                    #
                    i_low_straddle = i_mapping.astype(int) + 1 - i_mapping
                    j_low_straddle = j_mapping.astype(int) + 1 - j_mapping
                    #
                    # Segments that start at exact pixel boundaries and span
                    # whole pixels have low fractions that are 1. The span
                    # length needs to have these subtracted from it.
                    #
                    i_span[i_low_straddle == 1] -= 1
                    j_span[j_low_straddle == 1] -= 1
                    #
                    # --- the fraction covered by the upper straddle
                    #
                    i_high_straddle = i_mapping_end - i_mapping_end.astype(int)
                    j_high_straddle = j_mapping_end - j_mapping_end.astype(int)
                    #
                    # --- the total distance across the binning space
                    #
                    i_total = i_low_straddle + i_span + i_high_straddle
                    j_total = j_low_straddle + j_span + j_high_straddle
                    #
                    # --- The fraction in the lower straddle
                    #
                    i_low_frac = i_low_straddle / i_total
                    j_low_frac = j_low_straddle / j_total
                    #
                    # --- The fraction in the upper straddle
                    #
                    i_high_frac = i_high_straddle / i_total
                    j_high_frac = j_high_straddle / j_total
                    #
                    # later on, the high fraction will overwrite the low fraction
                    # for i and j hitting on a single pixel in the bin space
                    #
                    i_high_frac[
                        (i_mapping.astype(int) == i_mapping_end.astype(int))
                    ] = 1
                    j_high_frac[
                        (j_mapping.astype(int) == j_mapping_end.astype(int))
                    ] = 1
                    #
                    # --- The fraction in spans
                    #
                    i_span_frac = i_span / i_total
                    j_span_frac = j_span / j_total
                    #
                    # --- The number of bins touched by each pixel
                    #
                    i_count = (
                        numpy.ceil(i_mapping_end) - numpy.floor(i_mapping)
                    ).astype(int)
                    j_count = (
                        numpy.ceil(j_mapping_end) - numpy.floor(j_mapping)
                    ).astype(int)
                    #
                    # --- For I and J, calculate the weights for each pixel
                    #     along each axis.
                    #
                    i_idx = centrosome.index.Indexes([i_count])
                    j_idx = centrosome.index.Indexes([j_count])
                    i_weights = i_span_frac[i_idx.rev_idx]
                    j_weights = j_span_frac[j_idx.rev_idx]
                    i_weights[i_idx.fwd_idx] = i_low_frac
                    j_weights[j_idx.fwd_idx] = j_low_frac
                    mask = i_high_frac > 0
                    i_weights[i_idx.fwd_idx[mask] + i_count[mask] - 1] = i_high_frac[
                        mask
                    ]
                    mask = j_high_frac > 0
                    j_weights[j_idx.fwd_idx[mask] + j_count[mask] - 1] = j_high_frac[
                        mask
                    ]
                    #
                    # Get indexes for the 2-d array, i_count x j_count
                    #
                    idx = centrosome.index.Indexes([i_count, j_count])
                    #
                    # The coordinates in the straightened space
                    #
                    i_src_hard = i[idx.rev_idx]
                    j_src_hard = j[idx.rev_idx]
                    #
                    # The coordinates in the bin space
                    #
                    i_dest_hard = i_mapping[idx.rev_idx].astype(int) + idx.idx[0]
                    j_dest_hard = j_mapping[idx.rev_idx].astype(int) + idx.idx[1]
                    #
                    # The weights are the i-weight times the j-weight
                    #
                    # The i-weight can be found at the nth index of
                    # i_weights relative to the start of the i_weights
                    # for the pixel in the straightened space.
                    #
                    # The start is found at i_idx.fwd_idx[idx.rev_idx]
                    # the I offset is found at idx.idx[0]
                    #
                    # Similarly for J.
                    #
                    weight_hard = (
                        i_weights[i_idx.fwd_idx[idx.rev_idx] + idx.idx[0]]
                        * j_weights[j_idx.fwd_idx[idx.rev_idx] + idx.idx[1]]
                    )
                    i_src = numpy.hstack((i_src, i_src_hard))
                    j_src = numpy.hstack((j_src, j_src_hard))
                    i_dest = numpy.hstack((i_dest, i_dest_hard))
                    j_dest = numpy.hstack((j_dest, j_dest_hard))
                    weight = numpy.hstack((weight, weight_hard))
                    labels_src = numpy.hstack((labels_src, labels_1d[idx.rev_idx]))

                self.measure_bins(
                    workspace,
                    i_src,
                    j_src,
                    i_dest,
                    j_dest,
                    weight,
                    labels_src,
                    scales,
                    nworms,
                )

    def measure_bins(
        self,
        workspace,
        i_src,
        j_src,
        i_dest,
        j_dest,
        weight,
        labels_src,
        scales,
        nworms,
    ):
        """Measure the intensity in the worm by binning

        Consider a transformation from the space of images of straightened worms
        to the space of a grid (the worm gets stretched to fit into the grid).
        This function takes the coordinates of each labeled pixel in the
        straightened worm and computes per-grid-cell measurements on
        the pixels that fall into each grid cell for each straightened image.

        A pixel might span bins. In this case, it appears once per overlapped
        bin and it is given a weight proportional to the amount of it's area
        that falls in the bin.

        workspace - the workspace for the current image set
        i_src, j_src - the coordinates of the pixels in the straightened space
        i_dest, j_dest - the coordinates of the bins for those pixels
        weight - the fraction of the pixel that falls into the bin
        labels_src - the label for the pixel
        scales - the "scale" portion of the measurement for each of the bins
                 shaped the same as the i_dest, j_dest coordinates
        nworms - # of labels.
        """
        image_set = workspace.image_set
        m = workspace.measurements
        assert isinstance(m, Measurements)
        object_name = self.straightened_objects_name.value
        orig_name = self.objects_name.value
        nbins = len(scales)
        for group in self.images:
            image_name = group.straightened_image_name.value
            straightened_image = image_set.get_image(image_name).pixel_data
            if straightened_image.ndim == 3:
                straightened_image = numpy.mean(straightened_image, 2)
            straightened_image = straightened_image[i_src, j_src]
            bin_number = (
                labels_src - 1 + nworms * j_dest + nworms * scales.shape[1] * i_dest
            )
            bin_counts = numpy.bincount(bin_number)
            bin_weights = numpy.bincount(bin_number, weight)
            bin_means = (
                numpy.bincount(bin_number, weight * straightened_image) / bin_weights
            )
            deviances = straightened_image - bin_means[bin_number]
            #
            # Weighted variance =
            # sum(weight * (x - mean(x)) ** 2)
            # ---------------------------------
            #  N - 1
            #  ----- sum(weight)
            #    N
            #
            bin_vars = numpy.bincount(bin_number, weight * deviances * deviances) / (
                bin_weights * (bin_counts - 1) / bin_counts
            )
            bin_stds = numpy.sqrt(bin_vars)
            nexpected = numpy.prod(scales.shape) * nworms
            bin_means = numpy.hstack(
                (bin_means, [numpy.nan] * (nexpected - len(bin_means)))
            )
            bin_means.shape = (scales.shape[0], scales.shape[1], nworms)
            bin_stds = numpy.hstack(
                (bin_stds, [numpy.nan] * (nexpected - len(bin_stds)))
            )
            bin_stds.shape = (scales.shape[0], scales.shape[1], nworms)
            for i in range(scales.shape[0]):
                for j in range(scales.shape[1]):
                    for values, ftr in (
                        (bin_means, FTR_MEAN_INTENSITY),
                        (bin_stds, FTR_STD_INTENSITY),
                    ):
                        measurement = "_".join((C_WORM, ftr, image_name, scales[i][j]))
                        m.add_measurement(orig_name, measurement, values[i, j])

    def make_objects(self, workspace, labels, nworms):
        m = workspace.measurements
        assert isinstance(m, Measurements)
        object_set = workspace.object_set
        assert isinstance(object_set, ObjectSet)
        straightened_objects_name = self.straightened_objects_name.value
        straightened_objects = Objects()
        straightened_objects.segmented = labels
        object_set.add_objects(straightened_objects, straightened_objects_name)
        add_object_count_measurements(m, straightened_objects_name, nworms)
        add_object_location_measurements(m, straightened_objects_name, labels, nworms)

    def display(self, workspace, figure):
        """Display the results of the worm straightening"""
        image_pairs = workspace.display_data.image_pairs
        figure.set_subplots((2, len(image_pairs)))
        src_axis = None
        for i, ((src_pix, src_name), (dest_pix, dest_name)) in enumerate(image_pairs):
            if src_pix.ndim == 2:
                imshow = figure.subplot_imshow_grayscale
            else:
                imshow = figure.subplot_imshow_color
            axis = imshow(0, i, src_pix, title=src_name, sharexy=src_axis)
            if src_axis is None:
                src_axis = axis
            if dest_pix.ndim == 2:
                imshow = figure.subplot_imshow_grayscale
            else:
                imshow = figure.subplot_imshow_color
            imshow(1, i, dest_pix, title=dest_name)

    def get_scale_name(self, longitudinal, transverse):
        """Create a scale name, given a longitudinal and transverse band #

        longitudinal - band # (0 to # of stripes) or None for transverse-only
        transverse - band # (0 to # of stripes) or None  for longitudinal-only
        """
        if longitudinal is None:
            longitudinal = 0
            lcount = 1
        else:
            lcount = self.number_of_stripes.value
        if transverse is None:
            transverse = 0
            tcount = 1
        else:
            tcount = self.number_of_segments.value
        return "%s%dof%d_%s%dof%d" % (
            SCALE_HORIZONTAL,
            transverse + 1,
            tcount,
            SCALE_VERTICAL,
            longitudinal + 1,
            lcount,
        )

    def get_measurement_columns(self, pipeline):
        """Return columns that define the measurements produced by this module"""
        result = get_object_measurement_columns(self.straightened_objects_name.value)
        if self.wants_measurements:
            nsegments = self.number_of_segments.value
            nstripes = self.number_of_stripes.value
            worms_name = self.objects_name.value
            if nsegments > 1:
                result += [
                    (
                        worms_name,
                        "_".join(
                            (
                                C_WORM,
                                ftr,
                                group.straightened_image_name.value,
                                self.get_scale_name(None, segment),
                            )
                        ),
                        COLTYPE_FLOAT,
                    )
                    for ftr, group, segment in itertools.product(
                        (FTR_MEAN_INTENSITY, FTR_STD_INTENSITY),
                        self.images,
                        list(range(nsegments)),
                    )
                ]
            if nstripes > 1:
                result += [
                    (
                        worms_name,
                        "_".join(
                            (
                                C_WORM,
                                ftr,
                                group.straightened_image_name.value,
                                self.get_scale_name(stripe, None),
                            )
                        ),
                        COLTYPE_FLOAT,
                    )
                    for ftr, group, stripe in itertools.product(
                        (FTR_MEAN_INTENSITY, FTR_STD_INTENSITY),
                        self.images,
                        list(range(nstripes)),
                    )
                ]
            if nsegments > 1 and nstripes > 1:
                result += [
                    (
                        worms_name,
                        "_".join(
                            (
                                C_WORM,
                                ftr,
                                group.straightened_image_name.value,
                                self.get_scale_name(stripe, segment),
                            )
                        ),
                        COLTYPE_FLOAT,
                    )
                    for ftr, group, stripe, segment in itertools.product(
                        (FTR_MEAN_INTENSITY, FTR_STD_INTENSITY),
                        self.images,
                        list(range(nstripes)),
                        list(range(nsegments)),
                    )
                ]
        return result

    def get_categories(self, pipeline, object_name):
        result = []
        if object_name == IMAGE:
            result += [C_COUNT]
        elif object_name == self.straightened_objects_name:
            result += [C_LOCATION, C_NUMBER]
        elif object_name == self.objects_name and self.wants_measurements:
            result += [C_WORM]
        return result

    def get_measurements(self, pipeline, object_name, category):
        if object_name == IMAGE and category == C_COUNT:
            return [self.straightened_objects_name.value]
        elif object_name == self.straightened_objects_name:
            if category == C_LOCATION:
                return [FTR_CENTER_X, FTR_CENTER_Y]
            elif category == C_NUMBER:
                return [FTR_OBJECT_NUMBER]
        elif category == C_WORM and object_name == self.objects_name:
            return [FTR_MEAN_INTENSITY, FTR_STD_INTENSITY]
        return []

    def get_measurement_images(self, pipeline, object_name, category, measurement):
        if (
            object_name == self.objects_name
            and category == C_WORM
            and measurement in (FTR_MEAN_INTENSITY, FTR_STD_INTENSITY)
        ):
            return [group.straightened_image_name.value for group in self.images]
        return []

    def get_measurement_scales(
        self, pipeline, object_name, category, measurement, image_name
    ):
        result = []
        if image_name in self.get_measurement_images(
            pipeline, object_name, category, measurement
        ):
            nsegments = self.number_of_segments.value
            nstripes = self.number_of_stripes.value
            if nsegments > 1:
                result += [
                    self.get_scale_name(None, segment) for segment in range(nsegments)
                ]
            if nstripes > 1:
                result += [
                    self.get_scale_name(stripe, None) for stripe in range(nstripes)
                ]
            if nstripes > 1 and nsegments > 1:
                result += [
                    self.get_scale_name(h, v)
                    for h, v in itertools.product(
                        list(range(nstripes)), list(range(nsegments))
                    )
                ]
        return result

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        """Modify the settings to match the current version

        This method takes the settings from a previous revision of
        StraightenWorms and modifies them so that they match
        the settings that would be output by the current version.

        setting_values - setting value strings, possibly output by prev version

        variable_revision_number - revision of version of StraightenWorms that
        output the settings

        module_name - not used, see CPModule for use elsewhere.

        Overriding modules should return a tuple of setting_values,
        variable_revision_number and True if upgraded to CP 2.0, otherwise
        they should leave things as-is so that the caller can report
        an error.
        """

        if variable_revision_number == 1:
            #
            # Added worm measurement and flipping
            #
            setting_values = (
                setting_values[:FIXED_SETTINGS_COUNT_V1]
                + ["No", "4", "No", "None"]
                + setting_values[FIXED_SETTINGS_COUNT_V1:]
            )
            variable_revision_number = 2
        if variable_revision_number == 2:
            #
            # Added horizontal worm measurements
            #
            setting_values = (
                setting_values[:IDX_FLIP_WORMS_V2]
                + ["1"]
                + setting_values[IDX_FLIP_WORMS_V2:]
            )
            variable_revision_number = 3
        return setting_values, variable_revision_number

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

    def handle_interaction(self, straightened_images, labels, image_set_number):
        """Show a UI for flipping worms

        straightened_images - a tuple of dictionaries, one per image to be
                              straightened. The keys are "pixel_data",
                              "mask" and "name".

        labels - a labels matrix with one worm per label

        image_set_number - the cycle #

        returns a tuple of flipped worm images and the flipped labels matrix
        """
        import wx
        import matplotlib.backends.backend_wxagg
        import matplotlib.figure

        frame_size = wx.GetDisplaySize()
        frame_size = [max(frame_size[0], frame_size[1]) / 2] * 2
        style = wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX
        with wx.Dialog(
            None,
            -1,
            "Straighten worms: cycle #%d" % image_set_number,
            size=frame_size,
            style=style,
        ) as dlg:
            assert isinstance(dlg, wx.Dialog)
            dlg.Sizer = wx.BoxSizer(wx.VERTICAL)
            figure = matplotlib.figure.Figure()
            axes = figure.add_axes((0.05, 0.1, 0.9, 0.85))
            axes.set_title("Click on a worm to flip it.\n" "Hit OK when done")
            panel = matplotlib.backends.backend_wxagg.FigureCanvasWxAgg(dlg, -1, figure)
            toolbar = matplotlib.backends.backend_wxagg.NavigationToolbar2WxAgg(panel)
            dlg.Sizer.Add(toolbar, 0, wx.EXPAND)
            dlg.Sizer.Add(panel, 1, wx.EXPAND)

            ok_button = wx.Button(dlg, wx.ID_OK)
            cancel_button = wx.Button(dlg, wx.ID_CANCEL)
            button_sizer = wx.StdDialogButtonSizer()
            dlg.Sizer.Add(button_sizer, 0, wx.ALIGN_RIGHT)
            button_sizer.AddButton(ok_button)
            button_sizer.AddButton(cancel_button)
            button_sizer.Realize()

            big_labels = numpy.zeros(
                (labels.shape[0] + 2, labels.shape[1] + 2), dtype=labels.dtype
            )
            big_labels[1:-1, 1:-1] = labels
            outline_ij = numpy.argwhere(
                (labels != 0)
                & (
                    (big_labels[:-2, 1:-1] != big_labels[1:-1, 1:-1])
                    | (big_labels[2:, 1:-1] != big_labels[1:-1, 1:-1])
                    | (big_labels[1:-1, :-2] != big_labels[1:-1, 1:-1])
                    | (big_labels[1:-1, 2:] != big_labels[1:-1, 1:-1])
                )
            )
            outline_l = labels[outline_ij[:, 0], outline_ij[:, 1]]
            order = numpy.lexsort([outline_ij[:, 0], outline_ij[:, 1], outline_l])
            outline_ij = outline_ij[order, :]
            outline_l = outline_l[order].astype(int)
            outline_indexes = numpy.hstack(
                ([0], numpy.cumsum(numpy.bincount(outline_l)))
            )
            ii, jj = numpy.mgrid[0 : labels.shape[0], 0 : labels.shape[1]]
            half_width = self.width.value / 2
            width = 2 * half_width + 1

            active_worm = [None]
            needs_draw = [True]

            def refresh():
                object_number = active_worm[0]
                if len(straightened_images) == 1:
                    image = straightened_images[0][self.K_PIXEL_DATA]
                    imax = numpy.max(image)
                    imin = numpy.min(image)
                    if imax == imin:
                        image = numpy.zeros(image.shape)
                    else:
                        image = (image - imin) / (imax - imin)
                    image[labels == 0] = 1
                    if image.ndim == 2:
                        image = numpy.dstack([image] * 3)
                else:
                    shape = (labels.shape[0], labels.shape[1], 3)
                    image = numpy.zeros(shape)
                    image[labels == 0, :] = 1
                    for i, straightened_image in enumerate(straightened_images[:3]):
                        pixel_data = straightened_image[self.K_PIXEL_DATA]
                        if pixel_data.ndim == 3:
                            pixel_data = numpy.mean(pixel_data, 2)
                        imin, imax = [
                            fn(pixel_data[labels != 0]) for fn in (numpy.min, numpy.max)
                        ]
                        if imin == imax:
                            pixel_data = numpy.zeros(labels.shape)
                        else:
                            pixel_data = (pixel_data - imin) / imax
                        image[labels != 0, i] = pixel_data[labels != 0]
                if object_number is not None:
                    color = (
                        numpy.array(
                            get_primary_outline_color().asTuple(), dtype=float,
                        )
                        / 255
                    )
                    s = slice(
                        outline_indexes[object_number],
                        outline_indexes[object_number + 1],
                    )
                    image[outline_ij[s, 0], outline_ij[s, 1], :] = color[
                        numpy.newaxis, :
                    ]
                axes.imshow(image, origin="upper")
                needs_draw[0] = True
                panel.Refresh()

            def on_mouse_over(event):
                object_number = active_worm[0]
                new_object_number = None
                if event.inaxes == axes:
                    new_object_number = labels[
                        max(0, min(labels.shape[0] - 1, int(event.ydata + 0.5))),
                        max(0, min(labels.shape[1] - 1, int(event.xdata + 0.5))),
                    ]
                    if new_object_number == 0:
                        new_object_number = None
                    if object_number != new_object_number:
                        active_worm[0] = new_object_number
                        refresh()

            def on_mouse_click(event):
                object_number = active_worm[0]
                if (
                    event.inaxes == axes
                    and object_number is not None
                    and event.button == 1
                ):
                    imax = numpy.max(ii[labels == object_number]) + half_width
                    mask = (
                        (jj >= width * (object_number - 1))
                        & (jj < width * object_number)
                        & (ii <= imax)
                    )
                    isrc = ii[mask]
                    jsrc = jj[mask]
                    idest = imax - isrc
                    jdest = (object_number * 2 - 1) * width - jj[mask] - 1

                    for d in straightened_images:
                        for key in self.K_PIXEL_DATA, self.K_MASK:
                            src = d[key]
                            dest = src.copy()
                            ilim, jlim = src.shape[:2]
                            mm = (
                                (idest >= 0)
                                & (idest < ilim)
                                & (jdest >= 0)
                                & (jdest < jlim)
                                & (isrc >= 0)
                                & (isrc < ilim)
                                & (jsrc >= 0)
                                & (jsrc < jlim)
                            )
                            dest[idest[mm], jdest[mm]] = src[isrc[mm], jsrc[mm]]
                            d[key] = dest
                    ilim, jlim = labels.shape
                    mm = (
                        (idest >= 0)
                        & (idest < ilim)
                        & (jdest >= 0)
                        & (jdest < jlim)
                        & (isrc >= 0)
                        & (isrc < ilim)
                        & (jsrc >= 0)
                        & (jsrc < jlim)
                    )
                    labels[isrc[mm], jsrc[mm]] = labels[idest[mm], jdest[mm]]
                    s = slice(
                        outline_indexes[object_number],
                        outline_indexes[object_number + 1],
                    )
                    outline_ij[s, 0] = imax - outline_ij[s, 0]
                    outline_ij[s, 1] = (
                        (object_number * 2 - 1) * width - outline_ij[s, 1] - 1
                    )
                    refresh()

            def on_paint(event):
                dc = wx.PaintDC(panel)
                if needs_draw[0]:
                    panel.draw(dc)
                    needs_draw[0] = False
                else:
                    panel.gui_repaint(dc)
                dc.Destroy()
                event.Skip()

            def on_ok(event):
                dlg.EndModal(wx.OK)

            def on_cancel(event):
                dlg.EndModal(wx.CANCEL)

            dlg.Bind(wx.EVT_BUTTON, on_ok, ok_button)
            dlg.Bind(wx.EVT_BUTTON, on_cancel, cancel_button)

            refresh()
            panel.mpl_connect("button_press_event", on_mouse_click)
            panel.mpl_connect("motion_notify_event", on_mouse_over)
            panel.Bind(wx.EVT_PAINT, on_paint)
            result = dlg.ShowModal()
            if result != wx.OK:
                raise self.InteractionCancelledException()
            return straightened_images, labels
