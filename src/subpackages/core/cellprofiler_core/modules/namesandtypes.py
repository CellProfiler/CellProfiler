# coding=utf-8
import ast
import collections
import logging
import pickle
import re
import zlib
from collections import Counter

import numpy
import skimage.morphology

from ..constants.image import C_FRAME, CT_COLOR, CT_GRAYSCALE, CT_FUNCTION, CT_MASK, CT_OBJECTS
from ..constants.image import C_HEIGHT
from ..constants.image import C_MD5_DIGEST
from ..constants.image import C_SCALING
from ..constants.image import C_SERIES
from ..constants.image import C_WIDTH
from ..constants.image import NO_RESCALE
from ..constants.measurement import COLTYPE_FLOAT, C_OBJECTS_Z, C_OBJECTS_T, C_Z, C_T, C_OBJECTS_SERIES_NAME, \
    C_SERIES_NAME
from ..constants.measurement import FTR_CENTER_Z
from ..constants.measurement import M_LOCATION_CENTER_Z
from ..constants.measurement import COLTYPE_INTEGER
from ..constants.measurement import COLTYPE_VARCHAR
from ..constants.measurement import COLTYPE_VARCHAR_FILE_NAME
from ..constants.measurement import COLTYPE_VARCHAR_FORMAT
from ..constants.measurement import COLTYPE_VARCHAR_PATH_NAME
from ..constants.measurement import C_C
from ..constants.measurement import C_COUNT
from ..constants.measurement import C_FILE_NAME
from ..constants.measurement import C_LOCATION
from ..constants.measurement import C_METADATA
from ..constants.measurement import C_NUMBER
from ..constants.measurement import C_OBJECTS_CHANNEL
from ..constants.measurement import C_OBJECTS_FILE_NAME
from ..constants.measurement import C_OBJECTS_FRAME
from ..constants.measurement import C_OBJECTS_PATH_NAME
from ..constants.measurement import C_OBJECTS_SERIES
from ..constants.measurement import C_OBJECTS_URL
from ..constants.measurement import C_PATH_NAME
from ..constants.measurement import C_URL
from ..constants.measurement import EXPERIMENT
from ..constants.measurement import FTR_CENTER_X
from ..constants.measurement import FTR_CENTER_Y
from ..constants.measurement import FTR_OBJECT_NUMBER
from ..constants.measurement import IMAGE_NUMBER
from ..constants.module import FILTER_RULES_BUTTONS_HELP
from ..constants.module import PROTIP_RECOMMEND_ICON
from ..constants.module import USING_METADATA_HELP_REF
from ..constants.modules.namesandtypes import *
from ..image.abstract_image.file.url import ColorImage
from ..image.abstract_image.file.url import MaskImage
from ..image.abstract_image.file.url import MonochromeImage
from ..image.abstract_image.file.url import ObjectsImage
from ..measurement import Measurements
from ..module import Module
from ..object import Objects
from ..preferences import get_headless
from ..setting import Binary, FileCollectionDisplay
from ..setting import Divider
from ..setting import DoThings
from ..setting import HiddenCount
from ..setting import ImagePlane as ImagePlaneSetting
from ..setting import Joiner
from ..setting import SettingsGroup
from ..setting import ValidationError
from ..setting.choice import Choice
from ..setting.do_something import DoSomething
from ..setting.do_something import ImageSetDisplay
from ..setting.do_something import RemoveSettingButton
from ..setting.filter import (
    Filter,
    FilePredicate,
    DirectoryPredicate,
    ExtensionPredicate,
    ImagePredicate,
    FilterPredicate,
    DoesNotPredicate,
    DoesPredicate,
)
from ..setting.filter import MetadataPredicate
from ..setting.text import Float, FileImageName, LabelName
from ..utilities.channel_hasher import ChannelHasher
from ..utilities.core.module.identify import (
    add_object_location_measurements,
    add_object_location_measurements_ijv,
    add_object_count_measurements,
    get_object_measurement_columns,
)
from ..utilities.image import image_resource


LOGGER = logging.getLogger(__name__)

__doc__ = """\
NamesAndTypes
=============

The **NamesAndTypes** module gives images and/or channels a meaningful
name to a particular image or channel, as well as defining the
relationships between images to create an image set.  This module will also
let you define whether an image stack should be processed as sequential 2D slices
or as a whole 3D volume.

Once the relevant images have been identified using the **Images**
module (and/or has had metadata associated with the images using the
**Metadata** module), the **NamesAndTypes** module gives each image a
meaningful name by which modules in the analysis pipeline will refer to
it.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          YES
============ ============ ===============

What is an “image set”?
^^^^^^^^^^^^^^^^^^^^^^^

An *image set* is the collection of channels that represent a single
field of view. For example, a fluorescent assay may have samples using
DAPI and GFP to label separate cellular sub-compartments (see figure
below), and for each site imaged, one DAPI (left) and one GFP image
(right) is acquired by the microscope. Sometimes, the two channels are
combined into a single color images and other times they are stored as
two separate grayscale images, as in the figure.

+--------------+--------------+
| |NAT_image0| | |NAT_image1| |
+--------------+--------------+

For the purposes of analysis, you want the DAPI and GFP image for a
given site to be loaded and processed together. Therefore, the DAPI and
GFP image for a given site comprise an image set for that site.

What do I need as input?
^^^^^^^^^^^^^^^^^^^^^^^^

The **NamesAndTypes** module receives the file list produced by the
**Images** module. If you used the **Metadata** module to attach
metadata to the images, this information is also received by
**NamesAndTypes** and available for its use.

How do I configure the module?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the above example, the **NamesAndTypes** module allows you to assign
each of these channels a unique name, provided by you. All files of a
given channel will be referred to by the chosen name within the
pipeline, and the data exported by the pipeline will also be labeled
according to this name. This simplifies the bookkeeping of your pipeline
and results by making the input and output data more intuitive: a large
number of images are referred to by a small collection of names, which
are hopefully easier for you to recognize.

The most common way to perform this assignment is by specifying the
pattern in the filename which the channel(s) of interest have in common.
This is done using user-defined rules in a similar manner to that of the
**Images** module; other attributes of the file may also be used. If you
have multiple channels, you then assign the relationship between
channels. For example, in the case mentioned above, the DAPI and GFP
images are named in such a way that it is apparent to the researcher
which is which, e.g., “\_w2” is contained in the file for the DAPI
images, and “\_w1” in the file name for the GFP images.

You can also use **NamesAndTypes** to define the relationships between
images. For example, if you have acquired multiple wavelengths for your
assay, you will need to match the channels to each other for each field
of view so that they are loaded and processed together. This can be done
by using their associated metadata. If you would like to use the
metadata-specific settings, please see the **Metadata** module or *Help
> General help > Using Metadata in CellProfiler* for more details on
metadata usage and syntax.

What do I get as output?
^^^^^^^^^^^^^^^^^^^^^^^^

The **NamesAndTypes** module is the last of the required input modules.
After this module, you can choose any of the names you defined from a
drop-down list in any downstream analysis module which requires an image
as input. If you defined a set of objects using this module, those names
are also available for analysis modules that require an object as input.

In order to see whether the images are matched up correctly to form the
image sets you would expect, press the “Update” button below the divider
to display a table of results using the current settings. Each row
corresponds to a unique image set, and the columns correspond to the
name you specified for CellProfiler to identify the channel. You can
press this button as many times as needed to display the most current
image sets obtained. When you complete your pipeline and perform an
analysis run, CellProfiler will process the image sets in the order
shown.

|NAT_image2|

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *FileName, PathName:* The prefixes of the filename and location,
   respectively, of each image set written to the per-image table.
-  *ObjectFileName, ObjectPathName:* (For used for images loaded as
   objects) The prefixes of the filename and location, respectively, of
   each object set written to the per-image table.

.. |NAT_image0| image:: {DAPI}
.. |NAT_image1| image:: {GFP}
.. |NAT_image2| image:: {NAT_EXAMPLE_DISPLAY}
                :width: 100%
""".format(
    **{
        "DAPI": image_resource("dapi.png"),
        "GFP": image_resource("gfp.png"),
        "NAT_EXAMPLE_DISPLAY": image_resource("NamesAndTypes_ExampleDisplayTable.png"),
    }
)


class NamesAndTypes(Module):
    variable_revision_number = 9
    module_name = "NamesAndTypes"
    category = "File Processing"

    def create_settings(self):
        self.pipeline = None
        module_explanation = f"The {self.module_name} module allows you to assign a meaningful name to each image by " \
                             f"which other modules will refer to it.",

        self.set_notes(module_explanation)

        self.image_sets = []
        self.metadata_keys = []

        self.assignment_method = Choice(
            "Assign a name to",
            [ASSIGN_ALL, ASSIGN_RULES],
            doc="""\
This setting allows you to specify a name for types of images or subsets of
images so they can be treated separately by downstream modules. For
example, giving a different name to a GFP stain image and a brightfield
image of the same site allows each to be processed independently. In
other words, you are telling CellProfiler that your image set contains
pairs of images, one of which is GFP and the other brightfield.

The choices are:

-  *{ASSIGN_ALL}*: Give every image the same name. This is the simplest
   choice and the appropriate one if you have only one kind of image (or
   only one image). CellProfiler will give each image the same name and
   the pipeline will load only one of the images per iteration.
-  *{ASSIGN_RULES}*: Give images one of several names depending on the
   file name, directory and metadata. This is the appropriate choice if
   more than one image was acquired from each imaging site (ie if multiple
   channels were acquired at each site). You will be asked for distinctive
   criteria for each image and will be able to assign each category of image
   a name that can be referred to in downstream modules.
""".format(
                **{"ASSIGN_ALL": ASSIGN_ALL, "ASSIGN_RULES": ASSIGN_RULES}
            ),
        )

        self.single_load_as_choice = Choice(
            "Select the image type",
            [LOAD_AS_GRAYSCALE_IMAGE, LOAD_AS_COLOR_IMAGE, LOAD_AS_MASK],
            doc=LOAD_AS_CHOICE_HELP_TEXT,
        )

        self.process_as_3d = Binary(
            text="Process as 3D?",
            value=False,
            doc="""\
If you want to treat the data as three-dimensional, select "Yes" to
load files as volumes. Otherwise, select "No" to load files as separate,
two-dimensional images.
""",
            callback=lambda value: self.pipeline.set_volumetric(value),
        )

        self.x = Float(
            text="Relative pixel spacing in X",
            value=1.0,
            minval=0.0,
            doc="""\
*(Used only if "Process as 3D?" is "Yes")*

Enter the spacing between voxels in the X dimension, relative to Y and Z.
Normally, you will set one of these values to 1 and the others
relative to that. For example, most images have equal pixel spacing
in X and Y, but the distance between Z slices is often longer,
say two times the distance between pixels in X and Y. In such
a case, you would set X = 1, Y = 1, Z = 2. This calibration
affects modules that handle 3D images as volumes: for example,
**MeasureObjectAreaShape** will report different measurements for
volume depending on how far apart the Z-slices are from each other,
as set here in **NamesAndTypes**.
""",
        )

        self.y = Float(
            text="Relative pixel spacing in Y",
            value=1.0,
            minval=0.0,
            doc="""\
*(Used only if "Process as 3D?" is "Yes")*

Enter the spacing between voxels in the Y dimension, relative to X and Z.
See help for *Relative pixel spacing in X* for details.
""",
        )

        self.z = Float(
            text="Relative pixel spacing in Z",
            value=1.0,
            minval=0.0,
            doc="""\
*(Used only if "Process as 3D?" is "Yes")*

Enter the spacing between voxels in the Z dimension, relative to X and Y.
See help for *Relative pixel spacing in X* for details.
""",
        )

        self.single_image_provider = FileImageName(
            "Name to assign these images", IMAGE_NAMES[0]
        )

        self.single_rescale_method = Choice(
            "Set intensity range from",
            INTENSITY_ALL,
            value=INTENSITY_RESCALING_BY_DATATYPE,
            doc=RESCALING_HELP_TEXT,
        )

        self.manual_rescale = Float(
            MANUAL_INTENSITY_LABEL,
            DEFAULT_MANUAL_RESCALE,
            minval=numpy.finfo(numpy.float32).smallest_subnormal,
            doc=MANUAL_RESCALE_HELP_TEXT,
        )

        self.assignments = []

        self.single_images = []

        self.assignments_count = HiddenCount(self.assignments, "Assignments count")

        self.single_images_count = HiddenCount(
            self.single_images, "Single images count"
        )

        self.add_assignment(can_remove=False)

        self.add_assignment_divider = Divider()

        self.add_assignment_button = DoThings(
            "",
            (
                ("Add another image", self.add_assignment),
                ("Add a single image", self.add_single_image),
            ),
        )

        self.matching_choice = Choice(
            "Image set matching method",
            [MATCH_BY_ORDER, MATCH_BY_METADATA],
            doc="""\
Choose how you want to match the image from one channel with the images
from other channels.

This setting controls how CellProfiler picks which images should be
matched together when analyzing all of the images from one site.

You can match corresponding channels to each other in one of two ways:

-  *{MATCH_BY_ORDER}*: CellProfiler will order the images in each
   channel alphabetically by their file path name and, for movies or TIF
   stacks, will order the frames by their order in the file.
   CellProfiler will then match the first from one channel to the first
   from another channel.
   This approach is sufficient for most applications, but will match the
   wrong images if any of the files are missing or misnamed. The image
   set list will then get truncated according to the channel with the
   fewer number of files.
-  *{MATCH_BY_METADATA}*: CellProfiler will match files with the same
   metadata values. This option is more complex to use than
   *{MATCH_BY_ORDER}* but is more flexible and less prone to
   inadvertent errors.

   As an example, an experiment is run on a single multiwell plate with
   two image channels (OrigBlue, *w1* and OrigGreen, *w2*) containing
   well and site metadata extracted using the **Metadata** module. A set
   of images from two sites in well A01 might be described using the
   following:

   +----------------------------+------------+------------+------------------+
   | **File name**              | **Well**   | **Site**   | **Wavelength**   |
   +============================+============+============+==================+
   | P-12345\_A01\_s1\_w1.tif   | A01        | s1         | w1               |
   +----------------------------+------------+------------+------------------+
   | P-12345\_A01\_s1\_w2.tif   | A01        | s1         | w2               |
   +----------------------------+------------+------------+------------------+
   | P-12345\_A01\_s2\_w1.tif   | A01        | s2         | w1               |
   +----------------------------+------------+------------+------------------+
   | P-12345\_A01\_s2\_w2.tif   | A01        | s2         | w2               |
   +----------------------------+------------+------------+------------------+

   We want to match the channels so that each field of view in uniquely
   represented by the two channels. In this case, to match the *w1* and
   *w2* channels with their respective well and site metadata, you would
   select the *Well* metadata for both channels, followed by the *Site*
   metadata for both channels. In other words:

   +----------------+-----------------+
   | **OrigBlue**   | **OrigGreen**   |
   +================+=================+
   | Well           | Well            |
   +----------------+-----------------+
   | Site           | Site            |
   +----------------+-----------------+

   In this way, CellProfiler will match up files that have the same well
   and site metadata combination, so that the *w1* channel belonging to
   well A01 and site 1 will be paired with the *w2* channel belonging to
   well A01 and site 1. This will occur for all unique well and site
   pairings, to create an image set similar to the following:

   +----------------------+----------------+----------------------------+----------------------------+
   | **Image set tags**   | **Channels**   |                                                         |
   +======================+================+============================+============================+
   | **Well**             | **Site**       | **OrigBlue (w1)**          | **OrigGreen (w2)**         |
   +----------------------+----------------+----------------------------+----------------------------+
   | A01                  | s1             | P-12345\_A01\_s1\_w1.tif   | P-12345\_A01\_s1\_w2.tif   |
   +----------------------+----------------+----------------------------+----------------------------+
   | A01                  | s2             | P-12345\_A01\_s2\_w1.tif   | P-12345\_A01\_s2\_w2.tif   |
   +----------------------+----------------+----------------------------+----------------------------+

   Image sets for which a given metadata value combination (e.g., well,
   site) is either missing or duplicated for a given channel will simply
   be omitted.

   In addition, CellProfiler can match a single file for one channel
   against many files from another channel. This is useful, for
   instance, for applying an illumination correction file for an entire
   plate against every image file for that plate. In this instance, this
   would be done by selecting *Plate* as the common metadata tag and
   *(None)* for the rest:

   +----------------+-----------------+
   | **OrigBlue**   | **IllumBlue**   |
   +================+=================+
   | Plate          | Plate           |
   +----------------+-----------------+
   | Well           | (None)          |
   +----------------+-----------------+
   | Site           | (None)          |
   +----------------+-----------------+

   This sort of matching can also be useful in timelapse movies where you wish to
   measure the properties of a particular ROI over time:

   +----------------+----------------+-----------------+
   | **RFP**        | **GFP**        | **ROIMask**     |
   +================+================+=================+
   | MovieName      | MovieName      | MovieName       |
   +----------------+----------------+-----------------+
   | Timepoint      | Timepoint      | (None)          |
   +----------------+----------------+-----------------+



   The order of metadata matching is determined by the metadata data
   type (which is set in the **Metadata** module). The default is
   *text*, which means that the metadata is matched in alphabetical
   order. However, this can pose a problem if you need an image with a
   metadata value of “2” to be processed before one with “10”, since the
   latter is alphabetically first. To do this, you can set the metadata
   type to *integer* rather than *text*; the images will then be matched
   in numerical order.

   There are two special cases in metadata handling worth mentioning:

   -  *Missing metadata:* For a particular metadata tag, one image from
      a given image set has metadata values defined but another image
      does not. An example is when a microscope aborts acquisition
      prematurely in the middle of scanning two channels for a site, and
      captures one channel but not the other. In this case, plate, well
      and site metadata value exists for one image but not for the other
      since it was never acquired.
   -  *Duplicate metadata:* For a particular metadata tag, the same
      metadata values exist for multiple image sets such that they are
      not uniquely defined. An example is when a microscope re-scans a
      site in order to recover from a prior error. In this case, there
      may be one image from one channel but *two* images for the other
      channel, for the same site. Therefore, multiple instances of the
      same plate, well and site metadata values exist for the same image
      set.

   In both of these cases, the exact pairing between channels no longer
   exists. For missing metadata, the pairing is one-to-none, and for
   duplicate metadata, the pairing is one-to-two. In these instances
   where a match cannot be made, **NamesAndTypes** will simply omit the
   confounding metadata values from consideration. In the above example,
   an image set will not be created for the plate, well and site
   combination in question.

{USING_METADATA_HELP_REF}
""".format(
                **{
                    "MATCH_BY_METADATA": MATCH_BY_METADATA,
                    "MATCH_BY_ORDER": MATCH_BY_ORDER,
                    "USING_METADATA_HELP_REF": USING_METADATA_HELP_REF,
                }
            ),
        )

        self.join = Joiner(
            "Match metadata",
            doc="""
Select metadata keys which will be used to pair images.

Holding the *Shift* key while selecting a value will automatically fill any other
empty entries in that row with the same value.

See the help for "Image set matching method" for more information on how to match
keys across images.
            """,
        )

        self.imageset_setting = ImageSetDisplay("", "Update image set table")

    def add_assignment(self, can_remove=True):
        """Add a rules assignment"""
        unique_image_name = self.get_unique_image_name()
        unique_object_name = self.get_unique_object_name()
        group = SettingsGroup()
        self.assignments.append(group)

        if can_remove:
            group.append("divider", Divider())

        mp = MetadataPredicate(
            "Metadata",
            "Have %s matching",
            doc="Has metadata matching the value you enter",
        )

        mp.set_metadata_keys(self.metadata_keys)

        group.append(
            "rule_filter",
            Filter(
                "Select the rule criteria",
                [
                    FilePredicate(),
                    DirectoryPredicate(),
                    ExtensionPredicate(),
                    ImagePredicate(),
                    mp,
                ],
                'and (file does contain "")',
                doc="""\
Specify a filter using rules to narrow down the files to be analyzed.

{FILTER_RULES_BUTTONS_HELP}
""".format(
                    **{"FILTER_RULES_BUTTONS_HELP": FILTER_RULES_BUTTONS_HELP}
                ),
            ),
        )

        group.append(
            "image_name",
            FileImageName(
                "Name to assign these images",
                unique_image_name,
                doc="""\
Enter the name that you want to call this image.
After this point, this image will be referred to by this
name, and can be selected from any drop-down menu that
requests an image selection.

Names must start with a letter or underbar ("_") followed
by ASCII letters, underbars or digits.
""",
            ),
        )

        group.append(
            "object_name",
            LabelName(
                "Name to assign these objects",
                unique_object_name,
                doc="""\
Enter the name that you want to call this set of objects.
After this point, this object will be referred to by this
name, and can be selected from any drop-down menu that
requests an object selection.

Names must start with a letter or underbar ("_") followed
by ASCII letters, underbars or digits.
""",
            ),
        )

        group.append(
            "load_as_choice",
            Choice("Select the image type", LOAD_AS_ALL, doc=LOAD_AS_CHOICE_HELP_TEXT),
        )

        group.append(
            "rescale_method",
            Choice(
                "Set intensity range from",
                INTENSITY_ALL,
                value=INTENSITY_RESCALING_BY_METADATA,
                doc=RESCALING_HELP_TEXT,
            ),
        )

        group.append(
            "manual_rescale",
            Float(
                MANUAL_INTENSITY_LABEL,
                value=DEFAULT_MANUAL_RESCALE,
                minval=numpy.finfo(numpy.float32).eps,
                doc=MANUAL_RESCALE_HELP_TEXT,
            ),
        )

        def copy_assignment(group=group):
            self.copy_assignment(group, self.assignments, self.add_assignment)

        group.append(
            "copy_button",
            DoSomething(
                "",
                "Duplicate this image",
                copy_assignment,
                doc="""\
Duplicate the channel specification, creating a new image assignment
with the same settings as this one.

|NAT_CopyAssignment_image0| This button is useful if
you are specifying a long series of channels which differ by one or two
settings (e.g., an image stack with many frames). Using this button will
help avoid the tedium of having to select the same settings multiple
times.

.. |NAT_CopyAssignment_image0| image:: {PROTIP_RECOMMEND_ICON}
""".format(
                    **{"PROTIP_RECOMMEND_ICON": PROTIP_RECOMMEND_ICON}
                ),
            ),
        )

        group.can_remove = can_remove

        if can_remove:
            group.append(
                "remover",
                RemoveSettingButton("", "Remove this image", self.assignments, group),
            )

    @staticmethod
    def copy_assignment(assignment, assignment_list, add_assignment_fn):
        """Make a copy of an assignment

        Make a copy of the assignment and add it directly after the
        one being copied.

        assignment - assignment to copy
        assignment_list - add the assignment to this list
        add_assignment_fn - this appends a new assignment to the list
        """
        add_assignment_fn()
        new_assignment = assignment_list.pop()
        idx = assignment_list.index(assignment) + 1
        assignment_list.insert(idx, new_assignment)
        for old_setting, new_setting in zip(
            assignment.pipeline_settings(), new_assignment.pipeline_settings()
        ):
            new_setting.set_value_text(old_setting.get_value_text())

    def get_unique_image_name(self):
        """Return an unused name for naming images"""
        all_image_names = [
            other_group.image_name
            for other_group in self.assignments + self.single_images
        ]
        for image_name in IMAGE_NAMES:
            if image_name not in all_image_names:
                return image_name
        else:
            for i in range(1, 1000):
                image_name = "Channel%d" % i
                if image_name not in all_image_names:
                    return image_name

    def get_unique_object_name(self):
        """Return an unused name for naming objects"""
        all_object_names = [
            other_group.object_name
            for other_group in self.assignments + self.single_images
        ]
        for object_name in OBJECT_NAMES:
            if object_name not in all_object_names:
                return object_name
        else:
            for i in range(1, 1000):
                object_name = "Object%d" % i
                if object_name not in all_object_names:
                    return object_name

    def add_single_image(self):
        """Add another single image group to the settings"""
        unique_image_name = self.get_unique_image_name()
        unique_object_name = self.get_unique_object_name()
        group = SettingsGroup()
        self.single_images.append(group)

        group.append("divider", Divider())

        group.append(
            "image_plane",
            ImagePlaneSetting(
                "Single image location",
                doc="""\
Choose the single image to add to all image sets. You can
either drag an image onto the setting to select it and add it
to the image file list or you can press the "Browse" button to
select an existing image from the file list.
""",
            ),
        )
        group.append(
            "image_name",
            FileImageName(
                "Name to assign this image",
                unique_image_name,
                doc="""\
Enter the name that you want to call this image.
After this point, this image will be referred to by this
name, and can be selected from any drop-down menu that
requests an image selection.
""",
            ),
        )

        group.append(
            "object_name",
            LabelName(
                "Name to assign these objects",
                unique_object_name,
                doc="""\
Enter the name that you want to call this set of objects.
After this point, this object will be referred to by this
name, and can be selected from any drop-down menu that
requests an object selection.
""",
            ),
        )

        group.append(
            "load_as_choice",
            Choice("Select the image type", LOAD_AS_ALL, doc=LOAD_AS_CHOICE_HELP_TEXT),
        )

        group.append(
            "rescale_method",
            Choice(
                "Set intensity range from",
                INTENSITY_ALL,
                value=INTENSITY_RESCALING_BY_METADATA,
                doc=RESCALING_HELP_TEXT,
            ),
        )

        group.append(
            "manual_rescale",
            Float(
                MANUAL_INTENSITY_LABEL,
                value=DEFAULT_MANUAL_RESCALE,
                minval=numpy.finfo(numpy.float32).eps,
                doc=MANUAL_RESCALE_HELP_TEXT,
            ),
        )

        def copy_assignment(group=group):
            self.copy_assignment(group, self.single_images, self.add_single_image)

        group.append(
            "copy_button",
            DoSomething(
                "",
                "Copy",
                copy_assignment,
                doc="Make a copy of this channel specification",
            ),
        )

        group.can_remove = True
        group.append(
            "remover",
            RemoveSettingButton("", "Remove this image", self.single_images, group),
        )

    def settings(self):
        result = [
            self.assignment_method,
            self.single_load_as_choice,
            self.single_image_provider,
            self.join,
            self.matching_choice,
            self.single_rescale_method,
            self.assignments_count,
            self.single_images_count,
            self.manual_rescale,
            self.process_as_3d,
            self.x,
            self.y,
            self.z,
        ]

        for assignment in self.assignments:
            result += [
                assignment.rule_filter,
                assignment.image_name,
                assignment.object_name,
                assignment.load_as_choice,
                assignment.rescale_method,
                assignment.manual_rescale,
            ]

        for single_image in self.single_images:
            result += [
                single_image.image_plane,
                single_image.image_name,
                single_image.object_name,
                single_image.load_as_choice,
                single_image.rescale_method,
                single_image.manual_rescale,
            ]

        return result

    def help_settings(self):
        result = [
            self.assignment_method,
            self.single_load_as_choice,
            self.matching_choice,
            self.single_rescale_method,
            self.assignments_count,
            self.single_images_count,
            self.manual_rescale,
            self.process_as_3d,
            self.x,
            self.y,
            self.z,
        ]
        assignment = self.assignments[0]
        result += [
            assignment.rule_filter,
            assignment.image_name,
            assignment.object_name,
            assignment.load_as_choice,
            assignment.rescale_method,
            assignment.manual_rescale,
        ]
        return result

    def visible_settings(self):
        result = [self.assignment_method, self.process_as_3d]

        if self.process_as_3d.value:
            result += [self.x, self.y, self.z]

        if self.assignment_method == ASSIGN_ALL:
            result += [self.single_load_as_choice, self.single_image_provider]
            if self.single_load_as_choice in (
                LOAD_AS_COLOR_IMAGE,
                LOAD_AS_GRAYSCALE_IMAGE,
            ):
                if not self.process_as_3d.value:
                    result += [self.single_rescale_method]
                    if self.single_rescale_method == INTENSITY_MANUAL:
                        result += [self.manual_rescale]
        elif self.assignment_method == ASSIGN_RULES:
            for assignment in self.assignments:
                if assignment.can_remove:
                    result += [assignment.divider]
                result += [assignment.rule_filter]
                if assignment.load_as_choice == LOAD_AS_OBJECTS:
                    result += [assignment.object_name]
                else:
                    result += [assignment.image_name]
                result += [assignment.load_as_choice]
                if assignment.load_as_choice in (
                    LOAD_AS_COLOR_IMAGE,
                    LOAD_AS_GRAYSCALE_IMAGE,
                ):
                    if not self.process_as_3d.value:
                        result += [assignment.rescale_method]
                        if assignment.rescale_method == INTENSITY_MANUAL:
                            result += [self.manual_rescale]
                result += [assignment.copy_button]
                if assignment.can_remove:
                    result += [assignment.remover]
            for single_image in self.single_images:
                result += [single_image.divider, single_image.image_plane]
                if single_image.load_as_choice == LOAD_AS_OBJECTS:
                    result += [single_image.object_name]
                else:
                    result += [single_image.image_name]
                result += [single_image.load_as_choice]
                if single_image.load_as_choice in (
                    LOAD_AS_COLOR_IMAGE,
                    LOAD_AS_GRAYSCALE_IMAGE,
                ):
                    if not self.process_as_3d.value:
                        result += [single_image.rescale_method]
                        if single_image.rescale_method == INTENSITY_MANUAL:
                            result += [single_image.manual_rescale]
                result += [single_image.copy_button, single_image.remover]
            result += [self.add_assignment_divider, self.add_assignment_button]
            if len(self.assignments) > 1:
                result += [self.matching_choice]
                if self.matching_method == MATCH_BY_METADATA:
                    result += [self.join]
        result += [self.imageset_setting]
        return result

    def prepare_settings(self, setting_values):
        n_assignments = int(setting_values[IDX_ASSIGNMENTS_COUNT])
        if len(self.assignments) > n_assignments:
            del self.assignments[n_assignments:]
        while len(self.assignments) < n_assignments:
            self.add_assignment()
        n_single_images = int(setting_values[IDX_SINGLE_IMAGES_COUNT])
        if len(self.single_images) > n_single_images:
            del self.single_images[n_single_images:]
        while len(self.single_images) < n_single_images:
            self.add_single_image()

    def post_pipeline_load(self, pipeline):
        """Fix up metadata predicates after the pipeline loads"""
        if self.assignment_method == ASSIGN_RULES:
            filters = []
            self.metadata_keys = []
            for group in self.assignments:
                rules_filter = group.rule_filter
                filters.append(rules_filter)
                assert isinstance(rules_filter, Filter)
                #
                # The problem here is that the metadata predicates don't
                # know what possible metadata keys are allowable and
                # that computation could be (insanely) expensive. The
                # hack is to scan for the string we expect in the
                # raw text.
                #
                # The following looks for "(metadata does <kwd>" or
                # "(metadata doesnot <kwd>"
                #
                # This isn't perfect, of course, but it is enough to get
                # the filter's text to parse if the text is valid.
                #
                pattern = r"\(%s (?:%s|%s) ((?:\\.|[^ )])+)" % (
                    MetadataPredicate.SYMBOL,
                    DoesNotPredicate.SYMBOL,
                    DoesPredicate.SYMBOL,
                )
                text = rules_filter.value_text
                while True:
                    match = re.search(pattern, text)
                    if match is None:
                        break
                    key = FilterPredicate.decode_symbol(match.groups()[0])
                    self.metadata_keys.append(key)
                    text = text[match.end() :]
            self.metadata_keys = list(set(self.metadata_keys))
            for rules_filter in filters:
                for predicate in rules_filter.predicates:
                    if isinstance(predicate, MetadataPredicate):
                        predicate.set_metadata_keys(self.metadata_keys)

    def is_load_module(self):
        return True

    def change_causes_prepare_run(self, setting):
        """Return True if changing the setting passed changes the image sets

        setting - the setting that was changed
        """
        if setting is self.add_assignment_button:
            return True
        if isinstance(setting, RemoveSettingButton,):
            return True
        return setting in self.settings()

    def get_metadata_features(self):
        """Get the names of the metadata features used during metadata matching

        Unfortunately, these are the only predictable metadata keys that
        we can harvest in a reasonable amount of time.
        """
        result = set()
        if self.matching_method == MATCH_BY_METADATA:
            md_keys = self.join.parse()
            for md_dict in md_keys:
                for name, val in md_dict.items():
                    if val is None:
                        continue
                    elif val in (C_FRAME, C_SERIES,):
                        result.add(f"{val}_{name}")
                    else:
                        result.add(f"{C_METADATA}_{val}")
        return list(result)

    def prepare_run(self, workspace):
        """Write the image sets to the measurements"""
        if workspace.pipeline.in_batch_mode():
            return True
        column_names = self.get_column_names()
        image_sets = self.make_image_sets(workspace)
        if image_sets is None:
            return False
        if len(image_sets) == 0:
            return True

        m = workspace.measurements
        assert isinstance(m, Measurements)

        image_numbers = list(range(1, len(image_sets) + 1))
        if len(image_numbers) == 0:
            return False
        m.add_all_measurements(
            "Image", IMAGE_NUMBER, image_numbers,
        )

        # Alongside image sets we also need to store
        # instructions on how to load each channel.
        # These are called channel descriptors.
        if self.assignment_method == ASSIGN_ALL:
            load_choices = [self.single_load_as_choice.value]
        elif self.assignment_method == ASSIGN_RULES:
            load_choices = [
                group.load_as_choice.value
                for group in self.assignments + self.single_images
            ]
            if self.matching_method == MATCH_BY_METADATA:
                m.set_metadata_tags(self.get_metadata_features())
            else:
                m.set_metadata_tags([IMAGE_NUMBER])
        else:
            raise NotImplementedError(f"Unsupported assignment method {self.assignment_method.value}")

        descriptor_map = {
            LOAD_AS_COLOR_IMAGE: CT_COLOR,
            LOAD_AS_GRAYSCALE_IMAGE: CT_GRAYSCALE,
            LOAD_AS_ILLUMINATION_FUNCTION: CT_FUNCTION,
            LOAD_AS_MASK: CT_MASK,
            LOAD_AS_OBJECTS: CT_OBJECTS,
        }
        channel_descriptors = {column_name: descriptor_map[load_choice]
                               for column_name, load_choice in zip(column_names, load_choices)}

        m.set_channel_descriptors(channel_descriptors)

        # With the descriptors done, let's store the image sets.
        self.compress_imagesets(workspace, image_sets)

        # Now we store the metadata for each image set.
        for channel_name, channel_type in channel_descriptors.items():
            if channel_type == CT_OBJECTS:
                url_category = C_OBJECTS_URL
                path_name_category = C_OBJECTS_PATH_NAME
                file_name_category = C_OBJECTS_FILE_NAME
                series_category = C_OBJECTS_SERIES
                series_name_category = C_OBJECTS_SERIES_NAME
                frame_category = C_OBJECTS_FRAME
                channel_category = C_OBJECTS_CHANNEL
                z_category = C_OBJECTS_Z
                t_category = C_OBJECTS_T
            else:
                url_category = C_URL
                path_name_category = C_PATH_NAME
                file_name_category = C_FILE_NAME
                series_category = C_SERIES
                series_name_category = C_SERIES_NAME
                frame_category = C_FRAME
                channel_category = C_C
                z_category = C_Z
                t_category = C_T

            urls = []
            path_names = []
            file_names = []
            series = []
            series_names = []
            frames = []
            channels = []
            z_planes = []
            timepoints = []
            for image_set in image_sets:
                plane = image_set.get(channel_name, None)
                if plane is None:
                    raise ValueError(f"No images are assigned to channel {channel_name}, cannot create image sets")
                urls.append(plane.file.url)
                path_names.append(plane.file.dirname)
                file_names.append(plane.file.filename)
                series.append(plane.series)
                series_names.append(plane.series_name)
                frames.append(plane.index)
                channels.append(plane.channel)
                z_planes.append(plane.z)
                timepoints.append(plane.t)
            for feature_name, feature_values in (
                    (f"{url_category}_{channel_name}", urls),
                    (f"{path_name_category}_{channel_name}", path_names),
                    (f"{file_name_category}_{channel_name}", file_names),
                    (f"{series_category}_{channel_name}", series),
                    (f"{series_name_category}_{channel_name}", series_names),
                    (f"{frame_category}_{channel_name}", frames),
                    (f"{channel_category}_{channel_name}", channels),
                    (f"{z_category}_{channel_name}", z_planes),
                    (f"{t_category}_{channel_name}", timepoints),
            ):
                m.add_all_measurements("Image", feature_name, feature_values)

        # Now we aggregate the metadata to create a summary stat per image.
        self.aggregate_metadata(workspace, image_sets)
        # Todo: Consider metadata measurements per channel, not per image?

        return True

    def aggregate_metadata(self, workspace, image_sets):
        """
        This function finds Metadata measurements and aggregates them
        into a single value per image.
        If all images in a set have the same key, that key becomes the
        imageset-level Metadata key. If they differ, we use "None" as
        this can't be aggregated.
        """
        m = workspace.measurements
        measurement_columns = workspace.pipeline.get_measurement_columns(self) + self.get_measurement_columns(workspace.pipeline)
        required = dict([(x[1], x[2]) for x in measurement_columns if x[1].startswith(C_METADATA)])
        aggregated = {key: [] for key in required.keys()}
        offset = len(C_METADATA) + 1
        for image_set in image_sets:
            for key, key_type in required.items():
                plane_key = key[offset:]
                # We need to ignore missing keys ('None')
                orig_values = [plane.get_metadata(plane_key) for plane in image_set.values()
                               if plane.get_metadata(plane_key) is not None]
                if not orig_values:
                    # No data or everything was None.
                    aggregated[key].append(None)
                    continue
                if key_type.startswith('varchar'):
                    # String data, we may need to be case-insensitive
                    compare_values = list(map(str, orig_values))
                    if workspace.pipeline.use_case_insensitive_metadata_matching(key):
                        compare_values = list(map(str.lower, compare_values))
                else:
                    compare_values = orig_values
                if all(val == compare_values[0] for val in compare_values):
                    # All metadata values identical, so aggregate as the first entry.
                    aggregated[key].append(orig_values[0])
                else:
                    # Ambiguous metadata can't be aggregated.
                    aggregated[key].append(None)
        # Add Image-level metadata measurements
        for feature, data_type in required.items():
            values = aggregated[feature]
            if data_type == COLTYPE_INTEGER:
                values = [int(v) for v in values]
            elif data_type == COLTYPE_FLOAT:
                values = [float(v) for v in values]
            m.add_all_measurements("Image", feature, values)

    @staticmethod
    def compress_imagesets(workspace, image_sets):
        """Pickles imagesets and applies zlib compression.
        We use the first imageset pickle as a compression dictionary template.
        Most pickled imagesets are very similar and match most
        of the template, so this improves compression substantially.
        There are probably ways to do this more effectively, but
        for our purposes it should be good enough.

        Compressed image sets are stored in the measurements dict.

        The compression template is stored in measurements.
        We'll need it to decompress the data.
        """
        pickles = [pickle.dumps(image_set) for image_set in image_sets]

        compression_dict = pickles[0]
        m = workspace.measurements
        m.add_experiment_measurement(M_IMAGE_SET_ZIP_DICTIONARY, compression_dict)
        compressor = zlib.compressobj(zdict=compression_dict)
        compressed_image_sets = []
        for pickled in pickles:
            comp = compressor.copy()
            compressed_image_sets.append(comp.compress(pickled) + comp.flush())
        m.add_all_measurements(
            "Image",
            M_IMAGE_SET,
            compressed_image_sets,
        )

    def decompress_imageset(self, workspace, compressed_image_set):
        compression_dict = self.get_compression_dictionary(workspace)
        decompressor = zlib.decompressobj(zdict=compression_dict)
        data = decompressor.decompress(compressed_image_set)
        return pickle.loads(data)

    @property
    def matching_method(self):
        """Get the method used to match the files in different channels together

        returns either MATCH_BY_ORDER or MATCH_BY_METADATA
        """
        if self.assignment_method == ASSIGN_ALL:
            # A single column, match in the simplest way
            return MATCH_BY_ORDER
        elif len(self.assignments) == 1:
            return MATCH_BY_ORDER
        return self.matching_choice.value

    def make_image_sets(self, workspace):
        pipeline = workspace.pipeline
        image_planes = pipeline.image_plane_list

        if self.assignment_method == ASSIGN_ALL:
            image_sets = self.make_image_sets_assign_all(image_planes)
        elif self.matching_method == MATCH_BY_ORDER:
            image_sets = self.make_image_sets_by_order(image_planes)
        else:
            image_sets = self.make_image_sets_by_metadata(image_planes)

        return image_sets

    def make_image_sets_assign_all(self, image_planes):
        name = self.single_image_provider.value
        return [{name: plane} for plane in image_planes]

    def make_image_sets_by_order(self, image_planes):
        if not image_planes:
            return []
        groups = collections.defaultdict(list)
        filters = [(group.rule_filter, name) for group, name in
                   zip(self.assignments, self.get_column_names(want_singles=False))]
        for plane in image_planes:
            plane_comparator = (FileCollectionDisplay.NODE_IMAGE_PLANE, plane.modpath, plane)
            for rule_filter, name in filters:
                if rule_filter.evaluate(plane_comparator):
                    groups[name].append(plane)
                    break
        errors = []
        if not groups:
            LOGGER.warning("No images passed group filters")
            return []
        desired_length = max([len(grp) for grp in groups.values()])
        for name in self.get_column_names(want_singles=False):
            if name not in groups:
                errors.append((E_WRONG_LENGTH, name, desired_length))
            elif len(groups[name]) < desired_length:
                errors.append((E_WRONG_LENGTH, name, desired_length - len(groups[name])))
        group_names = list(groups.keys())
        image_sets = []
        for pack in zip(*groups.values()):
            image_sets.append({group_names[i]: planes for i, planes in enumerate(pack)})
        if len(errors) > 0:
            if not self.handle_error_messages(errors):
                return None
        self.append_single_images(image_sets)
        return image_sets

    def make_image_sets_by_metadata(self, image_planes):
        joins = self.join.parse()
        #
        # Find the anchor channel - it's the first one which has metadata
        # definitions for all joins
        #
        anchor_channel = None
        channel_names = self.get_column_names(want_singles=False)
        for name in channel_names:
            anchor_keys = []
            for join in joins:
                if join.get(name) is None:
                    break
                anchor_keys.append(join[name])
            else:
                anchor_channel = name
                break
        if anchor_channel is None:
            raise ValueError(
                "Please choose valid metadata keys for at least one channel in the metadata matcher"
            )
        channel_names.remove(anchor_channel)
        channel_names.insert(0, anchor_channel)

        """
        Now to filter the ImagePlanes into each channel.
        Unlike the other matching methods, for each channel 
        we're going to create a dictionary which maps tuples of
        metadata values to a list of ImagePlanes with that set
        of values. The ChannelHasher class handles channels 
        which don't use all possible keys.
        """
        groups = {name: ChannelHasher(name, [join[name] for join in joins])
                  for name in channel_names}
        filters = [(group.rule_filter, name) for group, name in
                   zip(self.assignments, self.get_column_names(want_singles=False))]
        for plane in image_planes:
            plane_comparator = (FileCollectionDisplay.NODE_IMAGE_PLANE, plane.modpath, plane)
            for rule_filter, name in filters:
                if rule_filter.evaluate(plane_comparator):
                    groups[name].add(plane)
                    break

        # Planes should now be assigned to hasher objects.
        # Let's make some image sets.
        image_sets = []
        errors = []
        for key in groups[anchor_channel].keys():
            image_set = {}
            for channel_name in channel_names:
                match = groups[channel_name][key]
                if len(match) < 1:
                    # This is a bad image set.
                    errors.append((E_MISSING, channel_name, key))
                    break
                elif len(match) > 1:
                    errors.append((E_TOO_MANY, channel_name, key))
                    break
                else:
                    # One match, yay!
                    image_set[channel_name] = match[0]
            else:
                image_sets.append(image_set)

        if len(errors) > 0:
            if not self.handle_error_messages(errors):
                return None

        self.append_single_images(image_sets)
        return image_sets

    def append_single_images(self, image_sets):
        """Append the single image channels to every image set

        image_sets - a list of image sets
        """
        to_add = {}
        for group in self.single_images:
            plane = group.image_plane.get_plane()
            if group.load_as_choice == LOAD_AS_OBJECTS:
                name = group.object_name.value
            else:
                name = group.image_name.value
            to_add[name] = plane
        for image_set in image_sets:
            image_set.update(to_add)

    @staticmethod
    def handle_error_messages(errors):
        # Errors should be a list of tuples.
        # Each tuple is (error type, channel name, optional extra data)
        if len(errors) == 0:
            return True
        error_types = Counter()
        for error_type, error_chan, error_info in errors:
            if error_info is not None:
                if error_type == E_WRONG_LENGTH:
                    text = f"Channel {error_chan} had {error_info} missing images"
                else:
                    text = f"Metadata {error_info} for channel {error_chan} had {error_type}"
            else:
                text = f"Channel {error_chan} had {error_type}"
            LOGGER.warning(text)
            error_types[(error_type, error_chan)] += 1
        if not get_headless():
            msg = f"Warning: found {len(errors)} image set errors (see log for details)\n \n"
            msg += "\nOf these:\n"
            for (error_type, error_chan), count in sorted(error_types.items()):
                if error_type == E_WRONG_LENGTH:
                    msg += f"Channel {error_chan} had {error_type}.\n"
                else:
                    msg += f"Channel {error_chan} had {error_type} in {count} sets.\n"
            msg += "\n \nDo you want to continue?"

            import wx

            result = wx.MessageBox(
                msg,
                caption="NamesAndTypes: image set matching error",
                style=wx.YES_NO | wx.ICON_QUESTION,
            )
            if result == wx.NO:
                return False
        return True

    @staticmethod
    def get_compression_dictionary(workspace):
        """Returns the imageset compression byte array"""
        m = workspace.measurements
        if m.has_feature(EXPERIMENT, M_IMAGE_SET_ZIP_DICTIONARY,):
            d = m[EXPERIMENT, M_IMAGE_SET_ZIP_DICTIONARY]
            # This comes out of hdf5 as str. Make bytes.
            return ast.literal_eval(d)
        return None

    def get_imageset(self, workspace):
        m = workspace.measurements
        compressed_imageset = m["Image", M_IMAGE_SET]
        # The HDF5 Dict is currently set up to store bytes as a string and return
        # a stringified object. We need to consider it as bytes again.
        compressed_imageset = ast.literal_eval(compressed_imageset)
        return self.decompress_imageset(workspace, compressed_imageset)

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        """Alter pathnames in preparation for batch processing

        workspace - workspace containing pipeline & image measurements
        fn_alter_path - call this function to alter any path to target
                        operating environment
        """
        if self.assignment_method == ASSIGN_ALL:
            names = [self.single_image_provider.value]
            is_image = [True]
        else:
            names = []
            is_image = []
            for group in self.assignments + self.single_images:
                if group.load_as_choice == LOAD_AS_OBJECTS:
                    names.append(group.object_name.value)
                    is_image.append(False)
                else:
                    names.append(group.image_name.value)
                    is_image.append(True)
        for name, iz_image in zip(names, is_image):
            workspace.measurements.alter_path_for_create_batch(
                name, iz_image, fn_alter_path
            )

    @classmethod
    def is_input_module(cls):
        return True

    def run(self, workspace):
        image_set = self.get_imageset(workspace)
        if self.assignment_method == ASSIGN_ALL:
            name = self.single_image_provider.value
            load_choice = self.single_load_as_choice.value
            rescale_method = self.single_rescale_method.value
            if rescale_method == INTENSITY_MANUAL:
                rescale_method = (0.0, self.manual_rescale.value)
            self.add_image_provider(workspace, name, load_choice, rescale_method, image_set)
        else:
            for group in self.assignments + self.single_images:
                if group.load_as_choice == LOAD_AS_OBJECTS:
                    self.add_objects(workspace, group.object_name.value, image_set)
                else:
                    rescale_method = group.rescale_method.value
                    if rescale_method == INTENSITY_MANUAL:
                        rescale_method = (0.0, group.manual_rescale.value)
                    self.add_image_provider(
                        workspace,
                        group.image_name.value,
                        group.load_as_choice.value,
                        rescale_method,
                        image_set,
                    )

    def add_image_provider(self, workspace, name, load_choice, rescale_method, image_set):
        """Put an image provider into the image set

        workspace - current workspace
        name - name of the image
        load_choice - one of the LOAD_AS_... choices
        rescale_method - whether to rescale the image intensity (ignored
                  for mask and illumination function). Either
                  INTENSITY_RESCALING_BY_METADATA, INTENSITY_RESCALING_BY_DATATYPE
                  or a 2-tuple of manual floating point values.
        stack - the ImagePlaneDetailsStack that describes the image's planes
        """
        if rescale_method == INTENSITY_RESCALING_BY_DATATYPE:
            rescale_range = None
            metadata_rescale = False
        elif rescale_method == INTENSITY_RESCALING_BY_METADATA:
            rescale_range = None
            metadata_rescale = True
        elif type(rescale_method) == tuple and len(rescale_method) == 2:
            rescale_range = rescale_method
            metadata_rescale = False

        image_plane = image_set[name]

        url = image_plane.url
        series = image_plane.series
        index = image_plane.index
        channel = image_plane.channel
        z = image_plane.z
        t = image_plane.t
        reader_name = image_plane.reader_name
        self.add_simple_image(
            workspace, name, load_choice, rescale_range, metadata_rescale, url, series, index, channel, z, t, reader_name
        )

    def add_simple_image(
        self, workspace, name, load_choice, rescale_range, metadata_rescale, url, series, index, channel, z=None, t=None, reader_name=None,
    ):
        m = workspace.measurements

        url = m.alter_url_post_create_batch(url)

        volume = self.process_as_3d.value

        spacing = (self.z.value, self.x.value, self.y.value) if volume else None

        if load_choice == LOAD_AS_COLOR_IMAGE:
            provider = ColorImage(
                name,
                url,
                series,
                index,
                rescale_range=rescale_range,
                metadata_rescale=metadata_rescale,
                volume=volume,
                spacing=spacing,
                z=z,
                t=t
            )
        elif load_choice == LOAD_AS_GRAYSCALE_IMAGE:
            provider = MonochromeImage(
                name,
                url,
                series,
                index,
                channel,
                rescale_range=rescale_range,
                metadata_rescale=metadata_rescale,
                volume=volume,
                spacing=spacing,
                z=z,
                t=t
            )
        elif load_choice == LOAD_AS_ILLUMINATION_FUNCTION:
            provider = MonochromeImage(
                name, url, series, index, channel, rescale_range=NO_RESCALE, metadata_rescale=False, volume=volume, spacing=spacing, z=z, t=t
            )
        elif load_choice == LOAD_AS_MASK:
            provider = MaskImage(
                name, url, series, index, channel, volume=volume, spacing=spacing, z=z, t=t
            )
        else:
            raise NotImplementedError(f"Unknown load choice: {load_choice}")
        provider.reader_name = reader_name
        workspace.image_set.add_provider(provider)

        self.add_provider_measurements(provider, m, "Image")

    @staticmethod
    def add_provider_measurements(provider, m, image_or_objects):
        """Add image measurements using the provider image and file

        provider - an image provider: get the height and width of the image
                   from the image pixel data and the MD5 hash from the file
                   itself.
        m - measurements structure
        image_or_objects - cpmeas.IMAGE if the provider is an image provider
                           otherwise cpmeas.OBJECT if it provides objects
        """
        name = provider.get_name()
        img = provider.provide_image(m)
        m["Image", C_MD5_DIGEST + "_" + name,] = NamesAndTypes.get_file_hash(
            provider, m
        )
        m["Image", C_WIDTH + "_" + name,] = img.pixel_data.shape[1]
        m["Image", C_HEIGHT + "_" + name,] = img.pixel_data.shape[0]
        if image_or_objects == "Image":
            m["Image", C_SCALING + "_" + name,] = provider.scale

    @staticmethod
    def get_file_hash(provider, measurements):
        """Get an md5 checksum from the (cached) file courtesy of the provider"""
        return provider.get_md5_hash(measurements)

    def add_objects(self, workspace, name, image_set):
        """Add objects loaded from a file to the object set

        workspace - the workspace for the analysis
        name - the objects' name in the pipeline
        stack - the ImagePlaneDetailsStack representing the planes to be loaded
        """

        image_plane = image_set[name]

        url = image_plane.url
        series = image_plane.series
        index = image_plane.index
        channel = image_plane.channel
        z = image_plane.z
        t = image_plane.t
        url = workspace.measurements.alter_url_post_create_batch(url)
        volume = self.process_as_3d.value
        spacing = (self.z.value, self.x.value, self.y.value) if volume else None
        provider = ObjectsImage(
            name, url, series, index, volume=volume, spacing=spacing, z=z, t=t,
        )
        self.add_provider_measurements(
            provider, workspace.measurements, "Object",
        )
        image = provider.provide_image(workspace.image_set)
        o = Objects()
        shape = image.pixel_data.shape
        #image.set_image(skimage.morphology.label(image.pixel_data), convert=False)
        #renumber if non-continuous label matrix: 
        labels=image.pixel_data
        nobjects = numpy.max(labels)
        unique_labels = numpy.unique(labels[labels != 0])
        contig_labels=numpy.arange(1, len(unique_labels) + 1)
        if all(unique_labels==contig_labels):
            image.set_image(labels, convert=False)
        else:
            indexer = numpy.zeros(nobjects + 1, int)
            indexer[unique_labels] =contig_labels
            image.set_image(skimage.morphology.label(indexer[labels]), convert=False)
        if shape[2] == 1:
            o.segmented = image.pixel_data[:, :, 0]
            add_object_location_measurements(
                workspace.measurements, name, o.segmented, o.count
            )
        elif volume:
            if len(shape) == 3:
                o.segmented = image.pixel_data
            elif len(shape) == 4 and shape[-1] == 1:
                o.segmented = image.pixel_data[:, :, :, 0]
            else:
                raise NotImplementedError("ijv volumes not yet supported")
            add_object_location_measurements(
                workspace.measurements, name, o.segmented, o.count
            )
        else:
            ijv = numpy.zeros((0, 3), int)
            for i in range(image.pixel_data.shape[2]):
                plane = image.pixel_data[:, :, i]
                shape = plane.shape
                i, j = numpy.mgrid[0 : shape[0], 0 : shape[1]]
                ijv = numpy.vstack(
                    (ijv, numpy.column_stack([x[plane != 0] for x in (i, j, plane)]))
                )
            o.set_ijv(ijv, shape)
            add_object_location_measurements_ijv(
                workspace.measurements, name, o.ijv, o.count
            )
        add_object_count_measurements(workspace.measurements, name, o.count)
        workspace.object_set.add_objects(o, name)

    def on_activated(self, workspace):
        self.pipeline = workspace.pipeline
        self.metadata_keys = sorted(self.pipeline.get_available_metadata_keys().keys())
        self.update_all_metadata_predicates()
        self.update_joiner()

    def on_deactivated(self):
        self.pipeline = None

    def on_setting_changed(self, setting, pipeline):
        """Handle updates to all settings"""
        self.update_joiner()
        self.update_all_metadata_predicates()

    def update_all_metadata_predicates(self):
        if self.assignment_method == ASSIGN_RULES:
            for group in self.assignments:
                rules_filter = group.rule_filter
                for predicate in rules_filter.predicates:
                    if isinstance(predicate, MetadataPredicate):
                        predicate.set_metadata_keys(self.metadata_keys)

    def get_image_names(self):
        """Return the names of all images produced by this module"""
        if self.assignment_method == ASSIGN_ALL:
            return [self.single_image_provider.value]
        elif self.assignment_method == ASSIGN_RULES:
            return [
                group.image_name.value
                for group in self.assignments + self.single_images
                if group.load_as_choice != LOAD_AS_OBJECTS
            ]
        return []

    def get_object_names(self):
        """Return the names of all objects produced by this module"""
        if self.assignment_method == ASSIGN_RULES:
            return [
                group.object_name.value
                for group in self.assignments + self.single_images
                if group.load_as_choice == LOAD_AS_OBJECTS
            ]
        return []

    def get_column_names(self, want_singles=True):
        if self.assignment_method == ASSIGN_ALL:
            return self.get_image_names()
        column_names = []
        if want_singles:
            groups = self.assignments + self.single_images
        else:
            groups = self.assignments
        for group in groups:
            if group.load_as_choice == LOAD_AS_OBJECTS:
                column_names.append(group.object_name.value)
            else:
                column_names.append(group.image_name.value)
        return column_names

    def get_measurement_columns(self, pipeline):
        """Create a list of measurements produced by this module

        For NamesAndTypes, we anticipate that the pipeline will create
        the text measurements for the images.
        """
        image_names = self.get_image_names()
        object_names = self.get_object_names()
        result = []
        for image_name in image_names:
            result += [
                ("Image", "_".join([category, image_name]), coltype,)
                for category, coltype in (
                    (C_FILE_NAME, COLTYPE_VARCHAR_FILE_NAME,),
                    (C_PATH_NAME, COLTYPE_VARCHAR_PATH_NAME,),
                    (C_URL, COLTYPE_VARCHAR_PATH_NAME,),
                    (C_MD5_DIGEST, COLTYPE_VARCHAR_FORMAT % 32,),
                    (C_SERIES_NAME, COLTYPE_VARCHAR,),
                    (C_SCALING, COLTYPE_FLOAT,),
                    (C_WIDTH, COLTYPE_INTEGER,),
                    (C_HEIGHT, COLTYPE_INTEGER,),
                    (C_SERIES, COLTYPE_INTEGER,),
                    (C_FRAME, COLTYPE_INTEGER,),
                )
            ]
        for object_name in object_names:
            result += [
                ("Image", "_".join([category, object_name]), coltype,)
                for category, coltype in (
                    (C_OBJECTS_FILE_NAME, COLTYPE_VARCHAR_FILE_NAME,),
                    (C_OBJECTS_PATH_NAME, COLTYPE_VARCHAR_PATH_NAME,),
                    (C_OBJECTS_URL, COLTYPE_VARCHAR_PATH_NAME,),
                    (C_COUNT, COLTYPE_INTEGER,),
                    (C_MD5_DIGEST, COLTYPE_VARCHAR_FORMAT % 32,),
                    (C_OBJECTS_SERIES_NAME, COLTYPE_VARCHAR,),
                    (C_WIDTH, COLTYPE_INTEGER,),
                    (C_HEIGHT, COLTYPE_INTEGER,),
                    (C_OBJECTS_SERIES, COLTYPE_INTEGER,),
                    (C_OBJECTS_FRAME, COLTYPE_INTEGER,),
                )
            ]
            result += get_object_measurement_columns(object_name)
            if self.process_as_3d.value:
                result += ((object_name, M_LOCATION_CENTER_Z, COLTYPE_FLOAT,),)
        result += [
            ("Image", ftr, COLTYPE_VARCHAR,) for ftr in self.get_metadata_features()
        ]
        return result

    def get_categories(self, pipeline, object_name):
        result = []
        if object_name == "Image":
            has_images = any(self.get_image_names())
            has_objects = any(self.get_object_names())
            if has_images:
                result += [
                    C_FILE_NAME,
                    C_PATH_NAME,
                    C_URL,
                ]
            if has_objects:
                result += [
                    C_OBJECTS_FILE_NAME,
                    C_OBJECTS_PATH_NAME,
                    C_OBJECTS_SERIES_NAME,
                    C_OBJECTS_URL,
                    C_COUNT,
                ]
            result += [
                C_MD5_DIGEST,
                C_SCALING,
                C_HEIGHT,
                C_WIDTH,
                C_SERIES,
                C_SERIES_NAME,
                C_FRAME,
            ]
        elif object_name in self.get_object_names():
            result += [
                C_LOCATION,
                C_NUMBER,
            ]
        return result

    def get_measurements(self, pipeline, object_name, category):
        image_names = self.get_image_names()
        object_names = self.get_object_names()
        if object_name == "Image":
            if category in (C_FILE_NAME, C_PATH_NAME, C_URL,):
                return image_names
            elif category in (C_OBJECTS_FILE_NAME, C_OBJECTS_PATH_NAME, C_OBJECTS_URL, C_OBJECTS_SERIES_NAME, C_COUNT):
                return object_names
            elif category in (
                C_MD5_DIGEST,
                C_SCALING,
                C_HEIGHT,
                C_WIDTH,
                C_SERIES,
                C_SERIES_NAME,
                C_FRAME,
            ):
                return list(image_names) + list(object_names)
        elif object_name in self.get_object_names():
            if category == C_NUMBER:
                return [FTR_OBJECT_NUMBER]
            elif category == C_LOCATION:
                result = [
                    FTR_CENTER_X,
                    FTR_CENTER_Y,
                ]
                if self.process_as_3d.value:
                    result += [FTR_CENTER_Z]
                return result
        return []

    def validate_module(self, pipeline):
        """Validate the settings for the NamesAndTypes module

        Make sure the metadata matcher has at least one completely
        specified channel.
        """
        if (
            self.assignment_method == ASSIGN_RULES
            and self.matching_choice == MATCH_BY_METADATA
            and len(self.assignments) > 1
        ):
            joins = self.join.parse()
            for name in self.get_column_names():
                for join in joins:
                    if join.get(name) is None:
                        break
                else:
                    return
            raise ValidationError(
                "At least one channel must have all metadata keys specified. "
                "All channels have at least one metadata key of (None).",
                self.join,
            )

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            # Changed naming of assignment methods
            setting_values[0] = (
                ASSIGN_ALL if setting_values[0] == "Assign all images" else ASSIGN_RULES
            )
            variable_revision_number = 2

        if variable_revision_number == 2:
            # Added single rescale and assignment method rescale
            n_assignments = int(setting_values[IDX_ASSIGNMENTS_COUNT_V2])
            new_setting_values = setting_values[:IDX_ASSIGNMENTS_COUNT_V2] + [
                "Yes",
                setting_values[IDX_ASSIGNMENTS_COUNT_V2],
            ]
            idx = IDX_ASSIGNMENTS_COUNT_V2 + 1
            for i in range(n_assignments):
                next_idx = idx + NUM_ASSIGNMENT_SETTINGS_V2
                new_setting_values += setting_values[idx:next_idx]
                new_setting_values.append(INTENSITY_RESCALING_BY_METADATA)
                idx = next_idx
            setting_values = new_setting_values
            variable_revision_number = 3
        if variable_revision_number == 3:
            # Added object outlines
            n_assignments = int(setting_values[IDX_ASSIGNMENTS_COUNT_V3])
            new_setting_values = setting_values[:IDX_FIRST_ASSIGNMENT_V3]
            for i in range(n_assignments):
                idx = IDX_FIRST_ASSIGNMENT_V3 + NUM_ASSIGNMENT_SETTINGS_V3 * i
                new_setting_values += setting_values[
                    idx : (idx + NUM_ASSIGNMENT_SETTINGS_V3)
                ]
                new_setting_values += ["No", "LoadedObjects"]
            setting_values = new_setting_values
            variable_revision_number = 4

        if variable_revision_number == 4:
            # Added single images (+ single image count)
            setting_values = (
                setting_values[:IDX_SINGLE_IMAGES_COUNT_V5]
                + ["0"]
                + setting_values[IDX_SINGLE_IMAGES_COUNT_V5:]
            )
            variable_revision_number = 5
        if variable_revision_number == 5:
            #
            # Convert LOAD_AS_MASK_V5A to LOAD_AS_MASK if present
            #
            #
            # Added manual_rescale
            #
            new_setting_values = setting_values[:IDX_FIRST_ASSIGNMENT_V5] + [
                DEFAULT_MANUAL_RESCALE
            ]
            n_assignments = int(setting_values[IDX_ASSIGNMENTS_COUNT_V5])
            n_single_images = int(setting_values[IDX_SINGLE_IMAGES_COUNT_V5])
            for i in range(n_assignments):
                offset = IDX_FIRST_ASSIGNMENT_V5 + NUM_ASSIGNMENT_SETTINGS_V5 * i
                new_setting_values += setting_values[
                    offset : offset + OFF_LOAD_AS_CHOICE_V5
                ]
                load_as = setting_values[offset + OFF_LOAD_AS_CHOICE_V5]
                if load_as == LOAD_AS_MASK_V5A:
                    load_as = LOAD_AS_MASK
                new_setting_values += (
                    [load_as]
                    + setting_values[
                        offset
                        + OFF_LOAD_AS_CHOICE_V5
                        + 1 : offset
                        + NUM_ASSIGNMENT_SETTINGS_V5
                    ]
                    + [DEFAULT_MANUAL_RESCALE]
                )
            for i in range(n_single_images):
                offset = (
                    IDX_FIRST_ASSIGNMENT_V5
                    + NUM_ASSIGNMENT_SETTINGS_V5 * n_assignments
                    + NUM_SINGLE_IMAGE_SETTINGS_V5 * i
                )
                new_setting_values += setting_values[
                    offset : offset + OFF_SI_LOAD_AS_CHOICE_V5
                ]
                load_as = setting_values[offset + OFF_SI_LOAD_AS_CHOICE_V5]
                if load_as == LOAD_AS_MASK_V5A:
                    load_as = LOAD_AS_MASK
                new_setting_values += (
                    [load_as]
                    + setting_values[
                        offset
                        + OFF_SI_LOAD_AS_CHOICE_V5
                        + 1 : offset
                        + NUM_ASSIGNMENT_SETTINGS_V5
                    ]
                    + [DEFAULT_MANUAL_RESCALE]
                )
            setting_values = new_setting_values
            variable_revision_number = 6

        if variable_revision_number == 6:
            new_setting_values = (
                setting_values[:9] + [False, 1.0, 1.0, 1.0] + setting_values[9:]
            )
            setting_values = new_setting_values
            variable_revision_number = 7

        if variable_revision_number == 7:
            offset = IDX_FIRST_ASSIGNMENT_V7
            n_settings = NUM_ASSIGNMENT_SETTINGS_V7
            n_assignments = int(setting_values[IDX_ASSIGNMENTS_COUNT_V7])

            assignment_rule_filter = setting_values[offset::n_settings][:n_assignments]
            assignment_image_name = setting_values[offset + 1 :: n_settings][
                :n_assignments
            ]
            assignment_object_name = setting_values[offset + 2 :: n_settings][
                :n_assignments
            ]
            assignment_load_as_choice = setting_values[offset + 3 :: n_settings][
                :n_assignments
            ]
            assignment_rescale_method = setting_values[offset + 4 :: n_settings][
                :n_assignments
            ]
            assignment_manual_rescale = setting_values[offset + 7 :: n_settings][
                :n_assignments
            ]

            assignment_settings = sum(
                [
                    list(settings)
                    for settings in zip(
                        assignment_rule_filter,
                        assignment_image_name,
                        assignment_object_name,
                        assignment_load_as_choice,
                        assignment_rescale_method,
                        assignment_manual_rescale,
                    )
                ],
                [],
            )

            offset = IDX_FIRST_ASSIGNMENT_V7 + (
                n_assignments * NUM_ASSIGNMENT_SETTINGS_V7
            )
            n_settings = NUM_SINGLE_IMAGE_SETTINGS_V7
            n_single_images = int(setting_values[IDX_SINGLE_IMAGES_COUNT_V7])

            single_image_image_plane = setting_values[offset::n_settings][
                :n_single_images
            ]
            single_image_image_name = setting_values[offset + 1 :: n_settings][
                :n_single_images
            ]
            single_image_object_name = setting_values[offset + 2 :: n_settings][
                :n_single_images
            ]
            single_image_load_as_choice = setting_values[offset + 3 :: n_settings][
                :n_single_images
            ]
            single_image_rescale_method = setting_values[offset + 4 :: n_settings][
                :n_single_images
            ]
            single_image_manual_rescale = setting_values[offset + 7 :: n_settings][
                :n_single_images
            ]

            single_image_settings = sum(
                [
                    list(settings)
                    for settings in zip(
                        single_image_image_plane,
                        single_image_image_name,
                        single_image_object_name,
                        single_image_load_as_choice,
                        single_image_rescale_method,
                        single_image_manual_rescale,
                    )
                ],
                [],
            )

            setting_values = (
                setting_values[:IDX_FIRST_ASSIGNMENT_V7]
                + assignment_settings
                + single_image_settings
            )

            variable_revision_number = 8

        return setting_values, variable_revision_number

    def volumetric(self):
        return True

    def update_joiner(self):
        """Update the joiner setting's entities"""
        if self.assignment_method == ASSIGN_RULES:
            self.join.entities = dict(
                [
                    (column_name, self.metadata_keys)
                    for column_name in self.get_column_names()
                ]
            )
            try:
                joins = self.join.parse()
                if len(joins) > 0:
                    for join in joins:
                        best_value = None
                        for key in list(join.keys()):
                            if key not in self.get_column_names():
                                del join[key]
                            elif join[key] is not None and best_value is None:
                                best_value = join[key]
                        for i, column_name in enumerate(self.get_column_names()):
                            if column_name not in join:
                                if best_value in self.metadata_keys:
                                    join[column_name] = best_value
                                else:
                                    join[column_name] = None
                self.join.build(repr(joins))
            except:
                pass  # bad field value

    def get_metadata_column_names(self):
        if self.matching_method == MATCH_BY_METADATA:
            joins = self.join.parse()
            metadata_columns = [
                " / ".join(set([k for k in list(join.values()) if k is not None]))
                for join in joins
            ]
        else:
            metadata_columns = [IMAGE_NUMBER]
        return metadata_columns
