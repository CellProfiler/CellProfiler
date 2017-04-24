import cellprofiler.icons
from cellprofiler.gui.help import PROTIP_RECOMEND_ICON, PROTIP_AVOID_ICON, TECH_NOTE_ICON, NAMESANDTYPES_DISPLAY_TABLE, \
    EXAMPLE_DAPI_PIC, EXAMPLE_GFP_PIC

__doc__ = """
The <b>NamesAndTypes</b> module gives images and/or channels a meaningful name to a particular image or channel,
as well as defining the relationships between images to create an image set.
<hr>
Once the relevant images have been identified using the <b>Images</b> module (and/or has
had metadata associated with the images using the <b>Metadata</b> module), the <b>NamesAndTypes</b> module
gives each image a meaningful name by which modules in the analysis pipeline will refer to it.

<h4>What is an "image set"?</h4>
An <i>image set</i> is the collection of channels that represent a single
field of view. For example, a fluorescent assay may have samples using DAPI and GFP to label separate
cellular sub-compartments (see figure below), and for each site imaged, one DAPI (left) and one GFP image (right)
is acquired by the microscope. Sometimes, the two channels are combined into a single color images and other
times they are stored as two separate grayscale images, as in the figure.
<table border="0" cellpadding="10" cellspacing="4" width="100%%">
<tr>
<td align="right"><img src="memory:%(EXAMPLE_DAPI_PIC)s"></td>
<td align="left"><img src="memory:%(EXAMPLE_GFP_PIC)s"></td>
</tr>
</table>
For the purposes of analysis, you want the DAPI and GFP image for a given site to be loaded and processed
together. Therefore, the DAPI and GFP image for a given site comprise an image set for that site.

<h4>What do I need as input?</h4>
The <b>NamesAndTypes</b> module receives the file list produced by the <b>Images</b> module. If you
used the <b>Metadata</b> module to attach metadata to the images, this information is also received by
<b>NamesAndTypes</b> and available for its use.

<h4>What do the settings mean?</h4>
In the above example, the <b>NamesAndTypes</b> module allows you to assign each of these channels a unique name,
provided by you. All files of a given channel will be referred to by the chosen name within the pipeline,
and the data exported by the pipeline will also be labeled according to this name. This simplifies the
bookkeeping of your pipeline and results by making the input and output data more intuitive: a large
number of images are referred to by a small collection of names, which are hopefully easier for you
to recognize.

<p>The most common way to perform this assignment is by specifying the pattern in the filename which
the channel(s) of interest have in common. This is done using user-defined rules in a similar manner
to that of the <b>Images</b> module; other attributes of the file may also be used. If you have multiple
channels, you then assign the relationship between channels.
For example, in the case mentioned above, the DAPI and GFP images are named in such a way that it
is apparent to the researcher which is which, e.g., "_w1" is contained in the file for the DAPI images,
and "_w1" in the file name for the GFP images.</p>

<p>You can also use <b>NamesAndTypes</b> to define the relationships between images. For example,
if you have acquired multiple wavelengths for your assay, you will need to
match the channels to each other for each field of view so that they are loaded and processed together.
This can be done by using their
associated metadata. If you would like to use the metadata-specific settings, please see the <b>Metadata</b> module
or <i>Help > General help > Using Metadata in CellProfiler</i> for more details on metadata usage
and syntax. </p>

<h4>What do I get as output?</h4>
The <b>NamesAndTypes</b> module is the last of the required input modules. After this module, you can choose
any of the names you defined from a drop-down list in any downstream analysis module which requires an
image as input. If you defined a set of objects using this module, those names are also available for analysis
modules that require an object as input.

<p>In order to see whether the images are matched up correctly to form the image sets you would expect, press
the "Update" button below the divider to display a table of results using the current settings. Each row
corresponds to a unique image set, and the columns correspond to the name you specified for CellProfiler
to identify the channel. You can press this button as many times as needed to display the most current image
sets obtained. When you complete your pipeline and perform an analysis run, CellProfiler will process the
image sets in the order shown.</p>
<table cellpadding="0" width="100%%">
<tr align="center"><td><img src="memory:%(NAMESANDTYPES_DISPLAY_TABLE)s"></td></tr>
</table>

<h4>Available measurements</h4>
<ul>
<li><i>FileName, PathName:</i> The prefixes of the filename and location, respectively, of each image set
written to the per-image table.</li>
<li><i>ObjectFileName, ObjectPathName:</i> (For used for images loaded as objects) The prefixes of the
filename and location, respectively, of each object set written to the per-image table.</li>
</ul>
""" % globals()

import logging

logger = logging.getLogger(__name__)

import hashlib
import numpy as np
import os
import re
import traceback

import cellprofiler.module as cpm
import cellprofiler.object as cpo
import cellprofiler.image as cpi
import cellprofiler.measurement as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.setting as cps
import centrosome.outline
import cellprofiler.preferences as cpprefs
from cellprofiler.modules.images import FilePredicate
from cellprofiler.modules.images import ExtensionPredicate
from cellprofiler.modules.images import ImagePredicate
from cellprofiler.modules.images import DirectoryPredicate
from cellprofiler.modules.loadimages import LoadImagesImageProviderURL
from cellprofiler.modules.loadimages import convert_image_to_objects
from cellprofiler.gui.help import FILTER_RULES_BUTTONS_HELP, USING_METADATA_HELP_REF
from cellprofiler.gui.help import RETAINING_OUTLINES_HELP, NAMING_OUTLINES_HELP
from bioformats import get_omexml_metadata, load_image
import bioformats.omexml as OME
import javabridge as J
import skimage.color

ASSIGN_ALL = "All images"
ASSIGN_GUESS = "Try to guess image assignment"
ASSIGN_RULES = "Images matching rules"

LOAD_AS_GRAYSCALE_IMAGE = "Grayscale image"
LOAD_AS_COLOR_IMAGE = "Color image"
LOAD_AS_MASK = "Binary mask"
LOAD_AS_MASK_V5A = "Mask"
LOAD_AS_ILLUMINATION_FUNCTION = "Illumination function"
LOAD_AS_OBJECTS = "Objects"
LOAD_AS_ALL = [LOAD_AS_GRAYSCALE_IMAGE,
               LOAD_AS_COLOR_IMAGE,
               LOAD_AS_MASK,
               LOAD_AS_ILLUMINATION_FUNCTION,
               LOAD_AS_OBJECTS]

INTENSITY_RESCALING_BY_METADATA = "Image metadata"
INTENSITY_RESCALING_BY_DATATYPE = "Image bit-depth"
INTENSITY_MANUAL = "Manual"
INTENSITY_ALL = [INTENSITY_RESCALING_BY_METADATA,
                 INTENSITY_RESCALING_BY_DATATYPE,
                 INTENSITY_MANUAL]
MANUAL_INTENSITY_LABEL = "Maximum intensity"

RESCALING_HELP_TEXT = """
This option determines how the image intensity should be
rescaled from 0.0 &ndash; 1.0.
<ul>
<li><i>%(INTENSITY_RESCALING_BY_METADATA)s:</i> Rescale the image
intensity so that saturated values are rescaled to 1.0 by dividing
all pixels in the image by the maximum possible intensity value
allowed by the imaging hardware. Some image formats save the maximum
possible intensity value along with the pixel data.
For instance, a microscope might acquire images using a 12-bit
A/D converter which outputs intensity values between zero and 4095,
but stores the values in a field that can take values up to 65535.
Choosing this setting ensures that the intensity scaling value is
the maximum allowed by the hardware, and not the maximum allowable
by the file format. </li>
<li><i>%(INTENSITY_RESCALING_BY_DATATYPE)s:</i> Ignore the image
metadata and rescale the image to 0 &ndash; 1 by dividing by 255
or 65535, depending on the number of bits used to store the image.</li>
<li><i>%(INTENSITY_MANUAL)s:</i> Divide each pixel value by the value entered
in the <i>%(MANUAL_INTENSITY_LABEL)s</i> setting. <i>%(INTENSITY_MANUAL)s</i> can be
used to rescale an image whose maximum intensity metadata value is absent or
incorrect, but is less than the value that would be supplied if
<i>%(INTENSITY_RESCALING_BY_DATATYPE)s</i> were specified.</li>
</ul>
Please note that CellProfiler does not provide the option of loading
the image as the raw, unscaled values. If you wish to make measurements
on the unscaled image, use the <b>ImageMath</b> module to multiply the
scaled image by the actual image bit-depth.""" % globals()

MANUAL_RESCALE_HELP_TEXT = """
<i>(Used only if %(INTENSITY_MANUAL)s is chosen)</i><br>
<b>NamesAndTypes</b> divides the pixel value, as read from the image file, by
this value to get the loaded image's per-pixel intensity.
""" % globals()

LOAD_AS_CHOICE_HELP_TEXT = """
    You can specify how these images should be treated:
    <ul>
    <li><i>%(LOAD_AS_GRAYSCALE_IMAGE)s:</i> An image in which each pixel
    represents a single intensity value. Most of the modules in CellProfiler
    operate on images of this type. <br>
    If this option is applied to a color image, the red, green and blue
    pixel intensities will be averaged to produce a single intensity value.</li>
    <li><i>%(LOAD_AS_COLOR_IMAGE)s:</i> An image in which each pixel
    repesents a red, green and blue (RGB) triplet of intensity values.
    Please note that the object detection modules such as <b>IdentifyPrimaryObjects</b>
    expect a grayscale image, so if you want to identify objects, you
    should use the <b>ColorToGray</b> module in the analysis pipeline
    to split the color image into its component channels.<br>
    You can use the <i>%(LOAD_AS_GRAYSCALE_IMAGE)s</i> option to collapse the
    color channels to a single grayscale value if you don't need CellProfiler
    to treat the image as color.</li>
    <li><i>%(LOAD_AS_MASK)s:</i> A <i>mask</i> is an image where some of the
    pixel intensity values are zero, and others are non-zero. The most common
    use for a mask is to exclude particular image regions from consideration. By
    applying a mask to another image, the portion of the image that overlaps with
    the non-zero regions of the mask are included. Those that overlap with the
    zeroed region are "hidden" and not included in downstream calculations.
    For this option, the input image should be a binary image, i.e, foreground is
    white, background is black. The module will convert any nonzero values
    to 1, if needed. You can use this option to load a foreground/background
    segmentation produced by one of the <b>Identify</b> modules.</li>
    <li><i>%(LOAD_AS_ILLUMINATION_FUNCTION)s:</i> An <i>illumination correction function</i>
    is an image which has been generated for the purpose of correcting uneven
    illumination/lighting/shading or to reduce uneven background in images. Typically,
    is a file in the MATLAB .mat format. See <b>CorrectIlluminationCalculate</b> and
    <b>CorrectIlluminationApply</b> for more details. </li>
    <li><i>%(LOAD_AS_OBJECTS)s:</i> Use this option if the input image
    is a label matrix and you want to obtain the objects that it defines.
    A label matrix is a grayscale or color image in which the connected
    regions share the same label, which defines how objects are represented
    in CellProfiler. The labels are integer values greater than or equal
    to 0. The elements equal to 0 are the background, whereas the elements
    equal to 1 make up one object, the elements equal to 2 make up a second
    object, and so on. This option allows you to use the objects
    immediately without needing to insert an <b>Identify</b> module to
    extract them first. See <b>IdentifyPrimaryObjects</b> for more details. <br>
    This option can load objects created by the <b>SaveImages</b> module. These objects
    can take two forms, with different considerations for each:
    <ul>
    <li><i>Non-overalapping</i> objects are stored as a label matrix. This matrix should be
    saved as grayscale, rather than color.</li>
    <li><i>Overlapping objects</i> are stored in a multi-frame TIF, each frame of which consists of a
    grayscale label matrix. The frames are constructed so that objects that overlap are placed
    in different frames.</li>
    </ul></li>
    </ul>
    """ % globals()

IDX_ASSIGNMENTS_COUNT_V2 = 5
IDX_ASSIGNMENTS_COUNT_V3 = 6
IDX_ASSIGNMENTS_COUNT_V5 = 6
IDX_ASSIGNMENTS_COUNT_V6 = 6
IDX_ASSIGNMENTS_COUNT = 6

IDX_SINGLE_IMAGES_COUNT_V5 = 7
IDX_SINGLE_IMAGES_COUNT_V6 = 7
IDX_SINGLE_IMAGES_COUNT = 7

IDX_FIRST_ASSIGNMENT_V3 = 7
IDX_FIRST_ASSIGNMENT_V4 = 7
IDX_FIRST_ASSIGNMENT_V5 = 8
IDX_FIRST_ASSIGNMENT_V6 = 9
IDX_FIRST_ASSIGNMENT = 9

NUM_ASSIGNMENT_SETTINGS_V2 = 4
NUM_ASSIGNMENT_SETTINGS_V3 = 5
NUM_ASSIGNMENT_SETTINGS_V5 = 7
NUM_ASSIGNMENT_SETTINGS_V6 = 8
NUM_ASSIGNMENT_SETTINGS = 8

NUM_SINGLE_IMAGE_SETTINGS_V5 = 7
NUM_SINGLE_IMAGE_SETTINGS_V6 = 8
NUM_SINGLE_IMAGE_SETTINGS = 8

OFF_LOAD_AS_CHOICE_V5 = 3
OFF_LOAD_AS_CHOICE = 3

OFF_SI_LOAD_AS_CHOICE_V5 = 3
OFF_SI_LOAD_AS_CHOICE = 3

MATCH_BY_ORDER = "Order"
MATCH_BY_METADATA = "Metadata"

IMAGE_NAMES = ["DNA", "GFP", "Actin"]
OBJECT_NAMES = ["Cell", "Nucleus", "Cytoplasm", "Speckle"]

DEFAULT_MANUAL_RESCALE = 255

'''The experiment measurement that holds the ZLIB compression dictionary for image sets'''
M_IMAGE_SET_ZIP_DICTIONARY = "ImageSet_Zip_Dictionary"
'''The image measurement that holds the compressed image set'''
M_IMAGE_SET = "ImageSet_ImageSet"


class NamesAndTypes(cpm.Module):
    variable_revision_number = 7
    module_name = "NamesAndTypes"
    category = "File Processing"

    def create_settings(self):
        self.pipeline = None
        module_explanation = [
            "The %s module allows you to assign a meaningful name to each image" %
            self.module_name,
            "by which other modules will refer to it."]
        self.set_notes([" ".join(module_explanation)])

        self.image_sets = []
        self.metadata_keys = []

        self.assignment_method = cps.Choice(
                "Assign a name to", [ASSIGN_ALL, ASSIGN_RULES], doc="""
            This setting allows the user to specify a name to images or subsets
            of images so they can be treated separately by downstream modules.
            For example, giving a different name to a GFP stain image and a
            brightfield image of the same site allows each to be processed
            independently.
            <p>The choices are:<br>
            <ul><li><i>%(ASSIGN_ALL)s</i>: Give every image the same name.
            This is the simplest choice and the appropriate one if you have
            only one kind of image (or only one image). CellProfiler will
            give each image the same name and the pipeline will load only
            one of the images per iteration.</li>
            <li><i>%(ASSIGN_RULES)s</i>: Give images one of several names
            depending on the file name, directory and metadata. This is the
            appropriate choice if more than one image was acquired from each
            imaging site. You will be asked for distinctive criteria for
            each image and will be able to assign each category of image
            a name that can be referred to in downstream modules.</li>
            </ul></p>
            """ % globals())

        self.single_load_as_choice = cps.Choice(
                "Select the image type", [LOAD_AS_GRAYSCALE_IMAGE,
                                          LOAD_AS_COLOR_IMAGE,
                                          LOAD_AS_MASK],
                doc=LOAD_AS_CHOICE_HELP_TEXT)

        self.process_as_3d = cps.Binary(
            text="Process as 3D",
            value=False,
            doc="""
            If you want to treat the data as three-dimensional, select "Yes" to
            load files as volumes. Otherwise, select "No" to load files as separate,
            two-dimensional images.
            """,
            callback=lambda value: self.pipeline.set_volumetric(value)
        )

        self.x = cps.Float(
            text="x",
            value=1.0,
            minval=0.0,
            doc="Spacing between voxels in the x-dimension."
        )

        self.y = cps.Float(
            text="y",
            value=1.0,
            minval=0.0,
            doc="Spacing between voxels in the y-dimension."
        )

        self.z = cps.Float(
            text="z",
            value=1.0,
            minval=0.0,
            doc="Spacing between voxels in the z-dimension."
        )

        self.single_image_provider = cps.FileImageNameProvider(
                "Name to assign these images", IMAGE_NAMES[0])

        self.single_rescale = cps.Choice(
                "Set intensity range from",
                INTENSITY_ALL,
                value=INTENSITY_RESCALING_BY_METADATA,
                doc=RESCALING_HELP_TEXT)

        self.manual_rescale = cps.Float(
                MANUAL_INTENSITY_LABEL, DEFAULT_MANUAL_RESCALE,
                minval=np.finfo(np.float32).eps,
                doc=MANUAL_RESCALE_HELP_TEXT)

        self.assignments = []
        self.single_images = []

        self.assignments_count = cps.HiddenCount(self.assignments,
                                                 "Assignments count")
        self.single_images_count = cps.HiddenCount(
                self.single_images, "Single images count")
        self.add_assignment(can_remove=False)

        self.add_assignment_divider = cps.Divider()
        self.add_assignment_button = cps.DoThings(
                "", (("Add another image", self.add_assignment),
                     ("Add a single image", self.add_single_image)))

        self.matching_choice = cps.Choice(
                "Image set matching method",
                [MATCH_BY_ORDER, MATCH_BY_METADATA], doc="""
            Select how you want to match the image from one channel with
            the images from other channels.
            <p>This setting controls how CellProfiler picks which images
            should be matched together when analyzing all of the images
            from one site. </p>
            <p>You can match corresponding channels to each other in one of two ways:
            <ul>
            <li><i>%(MATCH_BY_ORDER)s</i>: CellProfiler will order the images in
            each channel alphabetically by their file path name and, for movies
            or TIF stacks, will order the frames by their order in the file.
            CellProfiler will then match the first from one channel to the
            first from another channel. <br>
            This approach is sufficient for most applications, but
            will match the wrong images if any of the files are missing or misnamed.
            The image set list will then get truncated according to the channel with
            the fewer number of files.</li>

            <li><i>%(MATCH_BY_METADATA)s</i>: CellProfiler will match files with
            the same metadata values. This option is more complex to use than
            <i>%(MATCH_BY_ORDER)s</i> but is more flexible and less prone to inadvertent
            errors. %(USING_METADATA_HELP_REF)s.
            <p>As an example, an experiment is run on a single multiwell plate with two
            image channels (OrigBlue, <i>w1</i> and OrigGreen, <i>w2</i>) containing
            well and site metadata extracted using the <b>Metadata</b> module. A set of
            images from two sites in well A01 might be described using the following:
            <table border="1" align="center">
            <tr><th><b>File name</b></th><th><b>Well</b></th><th><b>Site</b></th><th><b>Wavelength</b></th></tr>
            <tr><td>P-12345_<font color="#ce5f33">A01</font>_<font color="#3dce33">s1</font>_<font color="#33bbce">w1</font>.tif</td><td><font color="#ce5f33">A01</font></td><td><font color="#3dce33">s1</font></td><td><font color="#33bbce">w1</font></td></tr>
            <tr><td>P-12345_<font color="#ce5f33">A01</font>_<font color="#3dce33">s1</font>_<font color="#33bbce">w2</font>.tif</td><td><font color="#ce5f33">A01</font></td><td><font color="#3dce33">s1</font></td><td><font color="#33bbce">w2</font></td></tr>
            <tr><td>P-12345_<font color="#ce5f33">A01</font>_<font color="#3dce33">s2</font>_<font color="#33bbce">w1</font>.tif</td><td><font color="#ce5f33">A01</font></td><td><font color="#3dce33">s2</font></td><td><font color="#33bbce">w1</font></td></tr>
            <tr><td>P-12345_<font color="#ce5f33">A01</font>_<font color="#3dce33">s2</font>_<font color="#33bbce">w2</font>.tif</td><td><font color="#ce5f33">A01</font></td><td><font color="#3dce33">s2</font></td><td><font color="#33bbce">w2</font></td></tr>
            </table>
            </p>
            <p>We want to match the channels so that each field of view in uniquely represented by the two channels. In this case,
            to match the <i>w1</i> and <i>w2</i> channels with their respective well and site metadata,
            you would select the <i>Well</i> metadata for both channels, followed by the <i>Site</i> metadata
            for both channels. In other words:
            <table border="1" align="center">
            <tr><th><b>OrigBlue</b></th><th><b>OrigGreen</b></th></tr>
            <tr><td>Well</td><td>Well</td></tr>
            <tr><td>Site</td><td>Site</td></tr>
            </table>
            In this way, CellProfiler will match up files that have the same
            well and site metadata combination, so that the <i>w1</i> channel belonging to well A01 and site 1
            will be paired with the <i>w2</i> channel belonging to well A01 and site 1. This will occur for all
            unique well and site pairings, to create an image set similar to the following:
            <table border="1" align="center">
            <tr><th colspan="2"><b>Image set tags</b></th><th colspan="2"><b>Channels</b></th></tr>
            <tr><td><b>Well</b></td><td><b>Site</b></td><td><b>OrigBlue (w1)</b></td><td><b>OrigGreen (w2)</b></td></tr>
            <tr><td><font color="#ce5f33">A01</font></td><td><font color="#3dce33">s1</font></td><td>P-12345_<font color="#ce5f33">A01</font>_<font color="#3dce33">s1</font>_<font color="#33bbce">w1</font>.tif</td><td>P-12345_<font color="#ce5f33">A01</font>_<font color="#3dce33">s1</font>_<font color="#33bbce">w2</font>.tif</td></tr>
            <tr><td><font color="#ce5f33">A01</font></td><td><font color="#3dce33">s2</font></td><td>P-12345_<font color="#ce5f33">A01</font>_<font color="#3dce33">s2</font>_<font color="#33bbce">w1</font>.tif</td><td>P-12345_<font color="#ce5f33">A01</font>_<font color="#3dce33">s2</font>_<font color="#33bbce">w2</font>.tif</td></tr>
            </table>
            Image sets for which a given metadata value combination (e.g., well, site) is either
            missing or duplicated for a given channel will simply be omitted.</p>
            <p>In addition, CellProfiler can match a single file for one channel against many files from
            another channel. This is useful, for instance, for applying an illumination correction file
            for an entire plate against every image file for that plate. In this instance, this would be
            done by selecting <i>Plate</i> as the common metadata tag and <i>(None)</i> for the rest:
            <table border="1" align="center">
            <tr><th><b>OrigBlue</b></th><th><b>IllumBlue</b></th></tr>
            <tr><td>Plate</td><td>Plate</td></tr>
            <tr><td>Well</td><td>(None)</td></tr>
            <tr><td>Site</td><td>(None)</td></tr>
            </table>
            </p>
            <p>The order of metadata matching is determined by the metadata data type (which is set in the
            <b>Metadata</b> module). The default is <i>text</i>, which means that the metadata is matched
            in alphabetical order. However, this can pose a problem if you need an image with a metadata value
            of "2" to be processed before one with "10", since the latter is alphabetically first. To do
            this, you can set the metadata type to <i>integer</i> rather than <i>text</i>; the images will
            then be matched in numerical order.</p>
            <p>There are two special cases in metadata handling worth mentioning:
            <ul>
            <li><i>Missing metadata:</i> For a particular metadata tag, one image from a given
            image set has metadata values defined but another image does not. An example is when a microscope
            aborts acquisition prematurely in the middle of scanning two channels for a site, and captures
            one channel but not the other. In this case, plate, well and site metadata value exists for one
            image but not for the other since it was never acquired. </li>
            <li><i>Duplicate metadata:</i> For a particular metadata tag, the same metadata values exist
            for multiple image sets such that they are not uniquely defined. An example is when a microscope
            re-scans a site in order to recover from a prior error. In this case, there may be one image from
            one channel but <i>two</i> images for the other channel, for the same site. Therefore, multiple instances
            of the same plate, well and site metadata values exist for the same image set.</li>
            </ul>
            In both of these cases, the exact pairing between channels no longer exists. For missing metadata, the pairing is one-to-none,
            and for duplicate metadata, the pairing is one-to-two. In these instances where a match cannot be
            made, <b>NamesAndTypes</b> will simply omit the confounding metadata values from consideration. In the above
            example, an image set will not be created for the plate, well and site combination in question.
            </p>
            </li>
            </ul>""" % globals())
        self.join = cps.Joiner("Match metadata")
        self.imageset_setting = cps.ImageSetDisplay("", "Update image set table")

    def add_assignment(self, can_remove=True):
        '''Add a rules assignment'''
        unique_image_name = self.get_unique_image_name()
        unique_object_name = self.get_unique_object_name()
        group = cps.SettingsGroup()
        self.assignments.append(group)

        if can_remove:
            group.append("divider", cps.Divider())

        mp = MetadataPredicate("Metadata", "Have %s matching",
                               doc="Has metadata matching the value you enter")
        mp.set_metadata_keys(self.metadata_keys)

        group.append("rule_filter", cps.Filter(
                "Select the rule criteria",
                [FilePredicate(),
                 DirectoryPredicate(),
                 ExtensionPredicate(),
                 ImagePredicate(),
                 mp],
                'and (file does contain "")', doc="""
            Specify a filter using rules to narrow down the files to be analyzed.
            <p>%(FILTER_RULES_BUTTONS_HELP)s</p>""" % globals()))

        group.append("image_name", cps.FileImageNameProvider(
                "Name to assign these images", unique_image_name, doc="""
            Enter the name that you want to call this image.
            After this point, this image will be referred to by this
            name, and can be selected from any drop-down menu that
            requests an image selection."""))

        group.append("object_name", cps.ObjectNameProvider(
                "Name to assign these objects", unique_object_name, doc="""
            Enter the name that you want to call this set of objects.
            After this point, this object will be referred to by this
            name, and can be selected from any drop-down menu that
            requests an object selection."""))

        group.append("load_as_choice", cps.Choice(
                "Select the image type", LOAD_AS_ALL,
                doc=LOAD_AS_CHOICE_HELP_TEXT))

        group.append("rescale", cps.Choice(
                "Set intensity range from",
                INTENSITY_ALL,
                value=INTENSITY_RESCALING_BY_METADATA,
                doc=RESCALING_HELP_TEXT))

        group.append("manual_rescale", cps.Float(
                MANUAL_INTENSITY_LABEL, value=DEFAULT_MANUAL_RESCALE,
                minval=np.finfo(np.float32).eps,
                doc=MANUAL_RESCALE_HELP_TEXT))

        group.append("should_save_outlines", cps.Binary(
                "Retain outlines of loaded objects?", False, doc="""
            %(RETAINING_OUTLINES_HELP)s""" % globals()))

        group.append("save_outlines", cps.OutlineNameProvider(
                "Name the outline image", "LoadedOutlines", doc=
                """%(NAMING_OUTLINES_HELP)s""" % globals()))

        def copy_assignment(group=group):
            self.copy_assignment(group, self.assignments, self.add_assignment)

        group.append("copy_button", cps.DoSomething(
                "", "Duplicate this image", copy_assignment, doc="""
            Duplicate the channel specification, creating a new image assignment
            with the same settings as this one.
            <dl>
            <dd><img src="memory:%(PROTIP_RECOMEND_ICON)s">&nbsp; This button is
            useful if you are specifying a long series of channels which differ
            by one or two settings (e.g., an image stack with many frames). Using
            this button will help avoid the tedium of having to select the same settings
            multiple times.</dd>
            </dl>""" % globals()))

        group.can_remove = can_remove
        if can_remove:
            group.append(
                    "remover",
                    cps.RemoveSettingButton(
                            '', "Remove this image", self.assignments, group))

    def copy_assignment(self, assignment, assignment_list, add_assignment_fn):
        '''Make a copy of an assignment

        Make a copy of the assignment and add it directly after the
        one being copied.

        assignment - assignment to copy
        assignment_list - add the assignment to this list
        add_assignment_fn - this appends a new assignment to the list
        '''
        add_assignment_fn()
        new_assignment = assignment_list.pop()
        idx = assignment_list.index(assignment) + 1
        assignment_list.insert(idx, new_assignment)
        for old_setting, new_setting in zip(
                assignment.pipeline_settings(),
                new_assignment.pipeline_settings()):
            new_setting.set_value_text(old_setting.get_value_text())

    def get_unique_image_name(self):
        '''Return an unused name for naming images'''
        all_image_names = [
            other_group.image_name for other_group in
            self.assignments + self.single_images]
        for image_name in IMAGE_NAMES:
            if image_name not in all_image_names:
                return image_name
        else:
            for i in xrange(1, 1000):
                image_name = "Channel%d" % i
                if image_name not in all_image_names:
                    return image_name

    def get_unique_object_name(self):
        '''Return an unused name for naming objects'''
        all_object_names = [
            other_group.object_name for other_group in
            self.assignments + self.single_images]
        for object_name in OBJECT_NAMES:
            if object_name not in all_object_names:
                return object_name
        else:
            for i in xrange(1, 1000):
                object_name = "Object%d" % i
                if object_name not in all_object_names:
                    return object_name

    def add_single_image(self):
        '''Add another single image group to the settings'''
        unique_image_name = self.get_unique_image_name()
        unique_object_name = self.get_unique_object_name()
        group = cps.SettingsGroup()
        self.single_images.append(group)

        group.append("divider", cps.Divider())
        group.append("image_plane", cps.ImagePlane(
                "Single image location",
                doc="""Choose the single image to add to all image sets. You can
            either drag an image onto the setting to select it and add it
            to the image file list or you can press the "Browse" button to
            select an existing image from the file list."""))
        group.append("image_name", cps.FileImageNameProvider(
                "Name to assign this image", unique_image_name, doc="""
            Enter the name that you want to call this image.
            After this point, this image will be referred to by this
            name, and can be selected from any drop-down menu that
            requests an image selection."""))

        group.append("object_name", cps.ObjectNameProvider(
                "Name to assign these objects", unique_object_name, doc="""
            Enter the name that you want to call this set of objects.
            After this point, this object will be referred to by this
            name, and can be selected from any drop-down menu that
            requests an object selection."""))

        group.append("load_as_choice", cps.Choice(
                "Select the image type", LOAD_AS_ALL,
                doc=LOAD_AS_CHOICE_HELP_TEXT))

        group.append("rescale", cps.Choice(
                "Set intensity range from",
                INTENSITY_ALL,
                value=INTENSITY_RESCALING_BY_METADATA, doc=RESCALING_HELP_TEXT))

        group.append("manual_rescale", cps.Float(
                MANUAL_INTENSITY_LABEL, value=DEFAULT_MANUAL_RESCALE,
                minval=np.finfo(np.float32).eps,
                doc=MANUAL_RESCALE_HELP_TEXT))

        group.append("should_save_outlines", cps.Binary(
                "Retain object outlines?", False, doc=RETAINING_OUTLINES_HELP))

        group.append("save_outlines", cps.OutlineNameProvider(
                "Name the outline image", "LoadedOutlines",
                doc=NAMING_OUTLINES_HELP))

        def copy_assignment(group=group):
            self.copy_assignment(
                    group, self.single_images, self.add_single_image)

        group.append("copy_button", cps.DoSomething(
                "", "Copy", copy_assignment,
                doc="Make a copy of this channel specification"))

        group.can_remove = True
        group.append(
                "remover",
                cps.RemoveSettingButton(
                        '', "Remove this image", self.single_images, group))

    def settings(self):
        result = [
            self.assignment_method,
            self.single_load_as_choice,
            self.single_image_provider,
            self.join,
            self.matching_choice,
            self.single_rescale,
            self.assignments_count,
            self.single_images_count,
            self.manual_rescale,
            self.process_as_3d,
            self.x,
            self.y,
            self.z
        ]

        for assignment in self.assignments:
            result += [assignment.rule_filter, assignment.image_name,
                       assignment.object_name, assignment.load_as_choice,
                       assignment.rescale, assignment.should_save_outlines,
                       assignment.save_outlines, assignment.manual_rescale]
        for single_image in self.single_images:
            result += [
                single_image.image_plane, single_image.image_name,
                single_image.object_name, single_image.load_as_choice,
                single_image.rescale, single_image.should_save_outlines,
                single_image.save_outlines, single_image.manual_rescale]

        return result

    def visible_settings(self):
        result = [self.assignment_method, self.process_as_3d]

        if self.process_as_3d.value:
            result += [
                self.x,
                self.y,
                self.z
            ]

        if self.assignment_method == ASSIGN_ALL:
            result += [
                self.single_load_as_choice,
                self.single_image_provider
            ]
            if self.single_load_as_choice in (LOAD_AS_COLOR_IMAGE,
                                              LOAD_AS_GRAYSCALE_IMAGE):
                result += [self.single_rescale]
                if self.single_rescale == INTENSITY_MANUAL:
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
                if assignment.load_as_choice in (LOAD_AS_COLOR_IMAGE,
                                                 LOAD_AS_GRAYSCALE_IMAGE):
                    result += [assignment.rescale]
                    if assignment.rescale == INTENSITY_MANUAL:
                        result += [self.manual_rescale]
                elif assignment.load_as_choice == LOAD_AS_OBJECTS:
                    result += [assignment.should_save_outlines]
                    if assignment.should_save_outlines.value:
                        result += [assignment.save_outlines]
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
                        LOAD_AS_COLOR_IMAGE, LOAD_AS_GRAYSCALE_IMAGE):
                    result += [single_image.rescale]
                    if single_image.rescale == INTENSITY_MANUAL:
                        result += [single_image.manual_rescale]
                elif single_image.load_as_choice == LOAD_AS_OBJECTS:
                    result += [single_image.should_save_outlines]
                    if single_image.should_save_outlines.value:
                        result += [single_image.save_outlines]
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
        '''Fix up metadata predicates after the pipeline loads'''
        if self.assignment_method == ASSIGN_RULES:
            filters = []
            self.metadata_keys = []
            for group in self.assignments:
                rules_filter = group.rule_filter
                filters.append(rules_filter)
                assert isinstance(rules_filter, cps.Filter)
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
                pattern = r"\(%s (?:%s|%s) ((?:\\.|[^ )])+)" % \
                          (MetadataPredicate.SYMBOL,
                           cps.Filter.DoesNotPredicate.SYMBOL,
                           cps.Filter.DoesPredicate.SYMBOL)
                text = rules_filter.value_text
                while True:
                    match = re.search(pattern, text)
                    if match is None:
                        break
                    key = cps.Filter.FilterPredicate.decode_symbol(
                            match.groups()[0])
                    self.metadata_keys.append(key)
                    text = text[match.end():]
            self.metadata_keys = list(set(self.metadata_keys))
            for rules_filter in filters:
                for predicate in rules_filter.predicates:
                    if isinstance(predicate, MetadataPredicate):
                        predicate.set_metadata_keys(self.metadata_keys)

    def is_load_module(self):
        return True

    def change_causes_prepare_run(self, setting):
        '''Return True if changing the setting passed changes the image sets

        setting - the setting that was changed
        '''
        if setting is self.add_assignment_button:
            return True
        if isinstance(setting, cps.RemoveSettingButton):
            return True
        return setting in self.settings()

    def get_metadata_features(self):
        '''Get the names of the metadata features used during metadata matching

        Unfortunately, these are the only predictable metadata keys that
        we can harvest in a reasonable amount of time.
        '''
        column_names = self.get_column_names()
        result = []
        if self.matching_method == MATCH_BY_METADATA:
            md_keys = self.join.parse()
            for column_name in column_names:
                if all([k[column_name] is not None for k in md_keys]):
                    for k in md_keys:
                        if k[column_name] in (cpmeas.C_FRAME, cpmeas.C_SERIES):
                            result.append(
                                    '_'.join((k[column_name], column_name)))
                        else:
                            result.append(
                                    '_'.join((cpmeas.C_METADATA, k[column_name])))
                    break
        return result

    def prepare_run(self, workspace):
        '''Write the image set to the measurements'''
        if workspace.pipeline.in_batch_mode():
            return True
        column_names = self.get_column_names()
        image_sets, channel_map = self.java_make_image_sets(workspace)
        if image_sets is None:
            return False
        if len(image_sets) == 0:
            return True

        image_set_channel_names = [None] * len(column_names)
        for name, idx in channel_map.iteritems():
            image_set_channel_names[idx] = name

        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)

        image_numbers = range(1, len(image_sets) + 1)
        if len(image_numbers) == 0:
            return False
        m.add_all_measurements(cpmeas.IMAGE, cpmeas.IMAGE_NUMBER,
                               image_numbers)

        if self.assignment_method == ASSIGN_ALL:
            load_choices = [self.single_load_as_choice.value]
        elif self.assignment_method == ASSIGN_RULES:
            load_choices = [group.load_as_choice.value
                            for group in self.assignments + self.single_images]
            if self.matching_method == MATCH_BY_METADATA:
                m.set_metadata_tags(self.get_metadata_features())
            else:
                m.set_metadata_tags([cpmeas.IMAGE_NUMBER])

        ImageSetChannelDescriptor = workspace.pipeline.ImageSetChannelDescriptor
        d = {
            LOAD_AS_COLOR_IMAGE: ImageSetChannelDescriptor.CT_COLOR,
            LOAD_AS_GRAYSCALE_IMAGE: ImageSetChannelDescriptor.CT_GRAYSCALE,
            LOAD_AS_ILLUMINATION_FUNCTION: ImageSetChannelDescriptor.CT_FUNCTION,
            LOAD_AS_MASK: ImageSetChannelDescriptor.CT_MASK,
            LOAD_AS_OBJECTS: ImageSetChannelDescriptor.CT_OBJECTS}
        iscds = [ImageSetChannelDescriptor(column_name, d[load_choice])
                 for column_name, load_choice in zip(column_names, load_choices)]
        m.set_channel_descriptors(iscds)

        zip_dict = self.create_imageset_dictionary(
                workspace, image_sets, image_set_channel_names)
        env = J.get_env()
        intcls = env.find_class("[I")
        strcls = env.find_class("[Ljava/lang/String;")
        urls, path_names, file_names, series, index, channel = [
            env.make_object_array(len(image_set_channel_names), cls)
            for cls in (strcls, strcls, strcls, intcls, intcls, intcls)]
        image_set_blobs = J.run_script("""
        importPackage(Packages.org.cellprofiler.imageset);
        ImageSet.convertToColumns(imageSets, channelNames, urls, pathNames,
            fileNames, series, index, channel, dict);
        """, dict(imageSets=image_sets.o,
                  channelNames=J.make_list(image_set_channel_names).o,
                  urls=urls,
                  pathNames=path_names,
                  fileNames=file_names,
                  series=series,
                  index=index,
                  channel=channel,
                  dict=zip_dict))
        m.add_all_measurements(
                cpmeas.IMAGE, M_IMAGE_SET,
                [env.get_byte_array_elements(x)
                 for x in env.get_object_array_elements(image_set_blobs)],
                data_type=np.uint8)

        urls, path_names, file_names, series, index, channel = [
            env.get_object_array_elements(x) for x in
            urls, path_names, file_names, series, index, channel]
        for i, iscd in enumerate(iscds):
            image_set_column_idx = channel_map[column_names[i]]
            if iscd.channel_type == ImageSetChannelDescriptor.CT_OBJECTS:
                url_category = cpmeas.C_OBJECTS_URL
                path_name_category = cpmeas.C_OBJECTS_PATH_NAME
                file_name_category = cpmeas.C_OBJECTS_FILE_NAME
                series_category = cpmeas.C_OBJECTS_SERIES
                frame_category = cpmeas.C_OBJECTS_FRAME
                channel_category = cpmeas.C_OBJECTS_CHANNEL
            else:
                url_category = cpmeas.C_URL
                path_name_category = cpmeas.C_PATH_NAME
                file_name_category = cpmeas.C_FILE_NAME
                series_category = cpmeas.C_SERIES
                frame_category = cpmeas.C_FRAME
                channel_category = cpmeas.C_CHANNEL
            url_feature, path_name_feature, file_name_feature, \
            series_feature, frame_feature, channel_feature = [
                "%s_%s" % (category, iscd.name) for category in (
                    url_category, path_name_category, file_name_category,
                    series_category, frame_category, channel_category)]
            for ftr, jarray in ((url_feature, urls),
                                (path_name_feature, path_names),
                                (file_name_feature, file_names)):
                col_values = [
                    env.get_string(x)
                    for x in env.get_object_array_elements(
                            jarray[image_set_column_idx])]
                m.add_all_measurements(cpmeas.IMAGE, ftr, col_values)
                del col_values

            for ftr, jarray in ((series_feature, series),
                                (frame_feature, index),
                                (channel_feature, channel)):
                col_values = list(env.get_int_array_elements(
                        jarray[image_set_column_idx]))
                m.add_all_measurements(cpmeas.IMAGE, ftr, col_values)

        #
        # Make a Java map of metadata key to column for matching metadata.
        # This is used to pick out the preferred column for must-have
        # metadata items (see issue #971).
        #
        must_have = J.make_map()
        if self.matching_method == MATCH_BY_METADATA:
            md_keys = self.join.parse()
            for column_name in column_names:
                for k in md_keys:
                    ck = k.get(column_name)
                    if ck is not None and not must_have.containsKey(ck):
                        must_have.put(ck, channel_map[column_name])
        #
        # Make a Java map of metadata key to metadata comparator
        #
        comparators = J.make_map(**dict(
                [(key, self.get_metadata_comparator(workspace, key))
                 for key in workspace.pipeline.get_available_metadata_keys()]))
        #
        # Do the giant collation in Java
        #
        md_dict = J.get_map_wrapper(J.static_call(
                "org/cellprofiler/imageset/MetadataUtils",
                "getImageSetMetadata",
                "(Ljava/util/List;Ljava/util/Map;Ljava/util/Map;)Ljava/util/Map;",
                image_sets.o, must_have.o, comparators.o))
        #
        # Populate the metadata measurements
        #
        env = J.get_env()
        mc = workspace.pipeline.get_measurement_columns(self)
        type_dict = dict([(c[1], c[2]) for c in mc if c[0] == cpmeas.IMAGE])

        def get_string_utf(x):
            return None if x is None else env.get_string_utf(x)

        promised = dict([(x[1], x[2]) for x in mc
                         if x[1].startswith(cpmeas.C_METADATA)])
        for name in J.iterate_collection(md_dict.keySet(), get_string_utf):
            feature_name = "_".join((cpmeas.C_METADATA, name))
            values = J.iterate_collection(md_dict[name], get_string_utf)
            data_type = type_dict.get(feature_name, cpmeas.COLTYPE_VARCHAR_FILE_NAME)
            if data_type == cpmeas.COLTYPE_INTEGER:
                values = [int(v) for v in values]
            elif data_type == cpmeas.COLTYPE_FLOAT:
                values = [float(v) for v in values]
            m.add_all_measurements(cpmeas.IMAGE,
                                   feature_name,
                                   values)
            if feature_name in promised:
                del promised[feature_name]
        #
        # Sadness - at this late date, we discover we promised something
        #           we could not deliver...
        #
        if len(promised) > 0:
            values = [None] * len(image_sets)
            for feature_name in promised:
                coltype = promised[feature_name]
                if coltype == cpmeas.COLTYPE_INTEGER:
                    data_type = int
                elif coltype == cpmeas.COLTYPE_FLOAT:
                    data_type = float
                else:
                    data_type = None
                m.add_all_measurements(cpmeas.IMAGE,
                                       feature_name,
                                       values,
                                       data_type=data_type)
        return True

    @property
    def matching_method(self):
        '''Get the method used to match the files in different channels together

        returns either MATCH_BY_ORDER or MATCH_BY_METADATA
        '''
        if self.assignment_method == ASSIGN_ALL:
            # A single column, match in the simplest way
            return MATCH_BY_ORDER
        elif len(self.assignments) == 1:
            return MATCH_BY_ORDER
        return self.matching_choice.value

    def java_make_image_sets(self, workspace):
        '''Make image sets using the Java framework

        workspace - the current workspace
        '''
        pipeline = workspace.pipeline
        ipds = pipeline.get_image_plane_details(workspace)
        #
        # Put the IPDs into a list
        #
        ipd_list = J.make_list([ipd.jipd for ipd in ipds])

        if self.assignment_method == ASSIGN_ALL:
            image_sets = self.java_make_image_sets_assign_all(
                    workspace, ipd_list)
            channels = {self.single_image_provider.value: 0}
        elif self.matching_method == MATCH_BY_ORDER:
            image_sets = self.java_make_image_sets_by_order(
                    workspace, ipd_list)
            channels = {}
            for i, group in enumerate(self.assignments + self.single_images):
                if group.load_as_choice == LOAD_AS_OBJECTS:
                    channels[group.object_name.value] = i
                else:
                    channels[group.image_name.value] = i
        else:
            image_sets, channels = \
                self.java_make_image_sets_by_metadata(workspace, ipd_list)
        if image_sets is not None:
            image_sets = J.get_collection_wrapper(image_sets)
        return image_sets, channels

    @staticmethod
    def get_axes_for_load_as_choice(load_as_choice):
        '''Get the appropriate set of axes for a given way of loading an image

        load_as_choice - one of the LOAD_AS_ constants

        returns the CellProfiler java PlaneStack prebuilt axes list that
        is the appropriate shape for the channel's image stack, e.g. XYCAxes
        for color.
        '''
        script = "Packages.org.cellprofiler.imageset.PlaneStack.%s;"
        if load_as_choice == LOAD_AS_COLOR_IMAGE:
            return J.run_script(script % "XYCAxes")
        elif load_as_choice == LOAD_AS_OBJECTS:
            return J.run_script(script % "XYOAxes")
        else:
            return J.run_script(script % "XYAxes")

    def make_channel_filter(self, group, name):
        '''Make a channel filter to get images for this group'''
        script = """
        importPackage(Packages.org.cellprofiler.imageset);
        importPackage(Packages.org.cellprofiler.imageset.filter);
        var ipdscls = java.lang.Class.forName(
            "org.cellprofiler.imageset.ImagePlaneDetailsStack");
        var filter = new Filter(expr, ipdscls);
        new ChannelFilter(name, filter, axes);
        """
        axes = self.get_axes_for_load_as_choice(group.load_as_choice.value)
        return J.run_script(
                script, dict(expr=group.rule_filter.value, name=name, axes=axes))

    def get_metadata_comparator(self, workspace, key):
        '''Get a Java Comparator<String> for a metadata key'''
        pipeline = workspace.pipeline
        if pipeline.get_available_metadata_keys().get(key) in (
                cpmeas.COLTYPE_FLOAT, cpmeas.COLTYPE_INTEGER):
            script = \
                """importPackage(Packages.org.cellprofiler.imageset);
                MetadataKeyPair.getNumericComparator();
                """
        elif pipeline.use_case_insensitive_metadata_matching(key):
            script = \
                """importPackage(Packages.org.cellprofiler.imageset);
                MetadataKeyPair.getCaseInsensitiveComparator();
                """
        else:
            script = \
                """importPackage(Packages.org.cellprofiler.imageset);
                MetadataKeyPair.getCaseSensitiveComparator();
                """
        return J.run_script(script)

    def make_metadata_key_pair(self, workspace, left_key, right_key):
        c = self.get_metadata_comparator(workspace, left_key)
        return J.run_script("""
        importPackage(Packages.org.cellprofiler.imageset);
        new MetadataKeyPair(left_key, right_key, c);
        """, dict(left_key=left_key, right_key=right_key, c=c))

    def java_make_image_sets_by_metadata(self, workspace, ipd_list):
        '''Make image sets by matching images by metadata

        workspace - current workspace
        ipd_list - a wrapped Java List<ImagePlaneDetails> containing
                   the IPDs to be composed into channels.

        returns a Java list of ImageSet objects and a dictionary of
        channel name to index in the image set.
        '''
        metadata_types = workspace.pipeline.get_available_metadata_keys()
        #
        # Find the anchor channel - it's the first one which has metadata
        # definitions for all joins
        #
        joins = self.join.parse()
        anchor_channel = None
        channel_names = self.get_column_names()
        anchor_cf = None
        for i, group in enumerate(self.assignments):
            name = channel_names[i]
            if anchor_channel is None:
                anchor_keys = []
                for join in joins:
                    if join.get(name) is None:
                        break
                    anchor_keys.append(join[name])
                else:
                    anchor_channel = i
                    anchor_cf = self.make_channel_filter(group, name)
        if anchor_cf is None:
            raise ValueError(
                    "Please choose valid metadata keys for at least one channel in the metadata matcher")
        channels = dict([(c, 0 if i == anchor_channel
        else i + 1 if i < anchor_channel
        else i) for i, c in enumerate(channel_names)])
        #
        # Make the joiner
        #
        jkeys = J.make_list(anchor_keys)
        jcomparators = J.make_list([
                                       self.get_metadata_comparator(workspace, key)
                                       for key in anchor_keys])

        script = """
        importPackage(Packages.org.cellprofiler.imageset);
        new Joiner(anchor_cf, keys, comparators)
        """
        joiner = J.run_script(
                script, dict(anchor_cf=anchor_cf, keys=jkeys,
                             comparators=jcomparators))
        #
        # Make the column filters and joins for the others
        #
        for i, (group, name) in enumerate(zip(self.assignments, channel_names)):
            if i == anchor_channel:
                continue
            cf = self.make_channel_filter(group, name)
            joining_keys = J.make_list()
            for j, join in enumerate(joins):
                if join.get(name) is not None:
                    joining_keys.add(self.make_metadata_key_pair(
                            workspace, anchor_keys[j], join[name]))
            J.run_script("""
            joiner.addChannel(cf, joiningKeys);
            """, dict(joiner=joiner, cf=cf, joiningKeys=joining_keys))
        errors = J.make_list()
        image_sets = J.run_script("""
        joiner.join(ipds, errors);
        """, dict(joiner=joiner, ipds=ipd_list.o, errors=errors.o))
        if len(errors) > 0:
            if not self.handle_errors(errors):
                return None, None

        self.append_single_images(image_sets)
        return image_sets, channels

    def java_make_image_sets_assign_all(self, workspace, ipd_list):
        '''Group all IPDs into stacks and assign to a single channel

        workspace - workspace for the analysis
        ipd_list - a wrapped Java List<ImagePlaneDetails> containing
                   the IPDs to be composed into channels.
        '''
        axes = self.get_axes_for_load_as_choice(
                self.single_load_as_choice.value)
        name = self.single_image_provider.value
        errors = J.make_list()
        image_sets = J.run_script("""
        importPackage(Packages.org.cellprofiler.imageset);
        var cf = new ChannelFilter(name, axes);
        var cfs = java.util.Collections.singletonList(cf);
        ChannelFilter.makeImageSets(cfs, ipds, errors);
        """, dict(axes=axes, name=name, ipds=ipd_list.o, errors=errors))
        if len(errors) > 0:
            if not self.handle_errors(errors):
                return None
        return image_sets

    def java_make_image_sets_by_order(self, workspace, ipd_list):
        '''Make image sets by coallating channels of image plane stacks

        workspace - workspace for the analysis
        ipd_list - a wrapped Java List<ImagePlaneDetails> containing
                   the IPDs to be composed into channels.
        '''
        channel_filters = J.make_list(
                [self.make_channel_filter(group, name)
                 for group, name in zip(self.assignments, self.get_column_names())])
        errors = J.make_list()
        image_sets = J.run_script("""
        importPackage(Packages.org.cellprofiler.imageset);
        ChannelFilter.makeImageSets(cfs, ipds, errors);
        """, dict(cfs=channel_filters.o, ipds=ipd_list.o, errors=errors))
        if len(errors) > 0:
            if not self.handle_errors(errors):
                return None
        self.append_single_images(image_sets)
        return image_sets

    def append_single_images(self, image_sets):
        '''Append the single image channels to every image set

        image_sets - a java list of image sets
        '''
        for group in self.single_images:
            url = group.image_plane.url
            series = group.image_plane.series or 0
            index = group.image_plane.index or 0
            axes = self.get_axes_for_load_as_choice(group.load_as_choice.value)
            if group.load_as_choice == LOAD_AS_COLOR_IMAGE:
                field_name = "INTERLEAVED"
            elif group.load_as_choice == LOAD_AS_OBJECTS:
                field_name = "OBJECT_PLANES"
            else:
                field_name = "ALWAYS_MONOCHROME"
            channel = J.get_static_field(
                    "org/cellprofiler/imageset/ImagePlane", field_name, "I")
            stack = J.make_instance(
                    "org/cellprofiler/imageset/ImagePlaneDetailsStack",
                    "([Lnet/imglib2/meta/TypedAxis;)V", axes)
            J.run_script("""
            importPackage(Packages.org.cellprofiler.imageset);
            importClass(java.net.URI);
            var imageFile = new ImageFile(new URI(url));
            var imageFileDetails = new ImageFileDetails(imageFile);
            var imageSeries = new ImageSeries(imageFile, series);
            var imageSeriesDetails =
                new ImageSeriesDetails(imageSeries, imageFileDetails);
            var imagePlane = new ImagePlane(imageSeries, index, channel);
            var imagePlaneDetails = new ImagePlaneDetails(imagePlane, imageSeriesDetails);
            if (stack.numDimensions() == 2) {
                stack.add(imagePlaneDetails, 0, 0);
            } else {
                stack.add(imagePlaneDetails, 0, 0, 0);
            }
            for (var i=0; i<image_sets.size(); i++) {
                image_sets.get(i).add(stack);
            }
            """,
                         dict(url=url, series=series, index=index, stack=stack,
                              channel=channel, image_sets=image_sets))

    def handle_errors(self, errors):
        '''Handle UI presentation of errors and user's response

        errors - a wrapped Java list of ImageSetError objects

        returns True if no errors or if user is OK with them
                False if user wants to abort.
        '''
        if len(errors) == 0:
            return True

        for error in errors:
            key = " / ".join(J.get_collection_wrapper(
                    J.call(error, "getKey", "()Ljava/util/List;"), J.to_string))
            echannel = J.call(error, "getChannelName", "()Ljava/lang/String;")
            message = J.call(error, "getMessage", "()Ljava/lang/String;")
            logger.warning(
                    "Error for image set, channel=%s, metadata=%s: %s" %
                    (str(key), echannel, message))
        if not cpprefs.get_headless():
            msg = (
                      "Warning: %d image set errors found (see log for details)\n"
                      "Do you want to continue?") % (errors.size())
            import wx
            result = wx.MessageBox(
                    msg,
                    caption="NamesAndTypes: matching by order error",
                    style=wx.YES_NO | wx.ICON_QUESTION)
            if result == wx.NO:
                return False
        return True

    def create_imageset_dictionary(self, workspace, image_sets, channel_names):
        '''Create a compression dictionary for OME-encoded image sets

        Image sets are serialized as OME-XML which is bulky and repetitive.
        ZLIB has a facility for using an input dictionary for priming
        the deflation and inflation process.

        This writes the dictionary to the experiment measurements.
        '''
        if len(image_sets) < 4:
            dlist = image_sets
        else:
            # Pick somewhere between four and 8 image sets from the whole
            dlist = J.make_list(image_sets[::int(len(image_sets) / 4)])
        cd = J.run_script(
                """importPackage(Packages.org.cellprofiler.imageset);
                   ImageSet.createCompressionDictionary(image_sets, channel_names);
                """,
                dict(image_sets=dlist,
                     channel_names=J.make_list(channel_names).o))
        m = workspace.measurements
        np_d = J.get_env().get_byte_array_elements(cd)
        m[cpmeas.EXPERIMENT, M_IMAGE_SET_ZIP_DICTIONARY, 0, np.uint8] = np_d
        return cd

    def get_imageset_dictionary(self, workspace):
        '''Returns the imageset dictionary as a Java byte array'''
        m = workspace.measurements
        if m.has_feature(cpmeas.EXPERIMENT, M_IMAGE_SET_ZIP_DICTIONARY):
            d = m[cpmeas.EXPERIMENT, M_IMAGE_SET_ZIP_DICTIONARY]
            return J.get_env().make_byte_array(d.astype(np.uint8))
        return None

    def get_imageset(self, workspace):
        '''Get the Java ImageSet for the current image number'''
        compression_dictionary = self.get_imageset_dictionary(workspace)
        m = workspace.measurements
        blob = m[cpmeas.IMAGE, M_IMAGE_SET]
        if blob is None:
            return None
        jblob = J.get_env().make_byte_array(blob)
        column_names = J.make_list(self.get_column_names())
        if self.assignment_method == ASSIGN_ALL:
            load_choices = [self.single_load_as_choice.value]
        elif self.assignment_method == ASSIGN_RULES:
            load_choices = [group.load_as_choice.value
                            for group in self.assignments + self.single_images]
        axes = J.make_list([self.get_axes_for_load_as_choice(load_as_choice)
                            for load_as_choice in load_choices])
        image_set = J.run_script("""
        importPackage(Packages.org.cellprofiler.imageset);
        ImageSet.decompress(blob, column_names, axes, dictionary);
        """, dict(blob=jblob, column_names=column_names.o,
                  axes=axes.o, dictionary=compression_dictionary))
        return J.get_collection_wrapper(image_set)

    def append_single_image_columns(self, columns, ipds):
        max_len = np.max([len(x) for x in columns])
        for single_image in self.single_images:
            ipd = self.get_single_image_ipd(single_image, ipds)
            columns.append([ipd] * max_len)

    def get_single_image_ipd(self, single_image, ipds):
        '''Get an image plane descriptor for this single_image group'''
        if single_image.image_plane.url is None:
            raise ValueError("Single image is not yet specified")
        ipd = cpp.find_image_plane_details(cpp.ImagePlaneDetails(
                single_image.image_plane.url,
                single_image.image_plane.series,
                single_image.image_plane.index,
                single_image.image_plane.channel), ipds)
        if ipd is None:
            raise ValueError("Could not find single image %s in file list",
                             single_image.image_plane.url)
        return ipd

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        '''Alter pathnames in preparation for batch processing

        workspace - workspace containing pipeline & image measurements
        fn_alter_path - call this function to alter any path to target
                        operating environment
        '''
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
                    name, iz_image, fn_alter_path)

    @classmethod
    def is_input_module(self):
        return True

    def run(self, workspace):
        image_set = self.get_imageset(workspace)
        if self.assignment_method == ASSIGN_ALL:
            name = self.single_image_provider.value
            load_choice = self.single_load_as_choice.value
            rescale = self.single_rescale.value
            if rescale == INTENSITY_MANUAL:
                rescale = self.manual_rescale.value
            self.add_image_provider(workspace, name, load_choice,
                                    rescale, image_set[0])
        else:
            for group, stack in zip(self.assignments + self.single_images,
                                    image_set):
                if group.load_as_choice == LOAD_AS_OBJECTS:
                    self.add_objects(workspace,
                                     group.object_name.value,
                                     group.should_save_outlines.value,
                                     group.save_outlines.value,
                                     stack)
                else:
                    rescale = group.rescale.value
                    if rescale == INTENSITY_MANUAL:
                        rescale = group.manual_rescale.value
                    self.add_image_provider(workspace,
                                            group.image_name.value,
                                            group.load_as_choice.value,
                                            rescale,
                                            stack)

    def add_image_provider(self, workspace, name, load_choice, rescale, stack):
        '''Put an image provider into the image set

        workspace - current workspace
        name - name of the image
        load_choice - one of the LOAD_AS_... choices
        rescale - whether or not to rescale the image intensity (ignored
                  for mask and illumination function). Either
                  INTENSITY_RESCALING_BY_METADATA, INTENSITY_RESCALING_BY_DATATYPE
                  or a floating point manual value.
        stack - the ImagePlaneDetailsStack that describes the image's planes
        '''
        if rescale == INTENSITY_RESCALING_BY_METADATA:
            rescale = True
        elif rescale == INTENSITY_RESCALING_BY_DATATYPE:
            rescale = False
        # else it's a manual rescale.
        num_dimensions = J.call(stack, "numDimensions", "()I")
        if num_dimensions == 2:
            coords = J.get_env().make_int_array(np.zeros(2, np.int32))
            ipds = [
                cpp.ImagePlaneDetails(
                        J.call(stack, "get", "([I)Ljava/lang/Object;", coords))]
        else:
            coords = np.zeros(num_dimensions, np.int32)
            ipds = []
            for i in range(J.call(stack, "size", "(I)I", 2)):
                coords[2] = i
                jcoords = J.get_env().make_int_array(coords)
                ipds.append(cpp.ImagePlaneDetails(
                        J.call(stack, "get", "([I)Ljava/lang/Object;", coords)))

        if len(ipds) == 1:
            interleaved = J.get_static_field(
                    "org/cellprofiler/imageset/ImagePlane", "INTERLEAVED", "I")
            monochrome = J.get_static_field(
                    "org/cellprofiler/imageset/ImagePlane", "ALWAYS_MONOCHROME", "I")
            ipd = ipds[0]
            url = ipd.url
            series = ipd.series
            index = ipd.index
            channel = ipd.channel
            if channel == monochrome:
                channel = None
            elif channel == interleaved:
                channel = None
                if index == 0:
                    index = None
            self.add_simple_image(
                    workspace, name, load_choice, rescale, url,
                    series, index, channel)
        elif all([ipd.url == ipds[0].url for ipd in ipds[1:]]):
            # Can load a simple image with a vector of series/index/channel
            url = ipds[0].url
            series = [ipd.series for ipd in ipds]
            index = [ipd.index for ipd in ipds]
            channel = [None if ipd.channel < 0 else ipd.channel for ipd in ipds]
            self.add_simple_image(
                    workspace, name, load_choice, rescale, url,
                    series, index, channel)
        else:
            # Different URLs - someone is a clever sadist
            # At this point, I believe there's no way to do this using
            # NamesAndTypes. When implemented, pay attention to
            # cacheing multiple readers for the same channel.
            #
            raise NotImplementedError("To do: support assembling image files into a stack")

    def add_simple_image(self, workspace, name, load_choice, rescale, url, series, index, channel):
        m = workspace.measurements

        url = m.alter_url_post_create_batch(url)

        volume = self.process_as_3d.value

        spacing = (self.z.value, self.x.value, self.y.value) if volume else None

        if load_choice == LOAD_AS_COLOR_IMAGE:
            provider = ColorImageProvider(name, url, series, index, rescale, volume=volume, spacing=spacing)
        elif load_choice == LOAD_AS_GRAYSCALE_IMAGE:
            provider = MonochromeImageProvider(name, url, series, index, channel, rescale, volume=volume, spacing=spacing)
        elif load_choice == LOAD_AS_ILLUMINATION_FUNCTION:
            provider = MonochromeImageProvider(name, url, series, index, channel, False, volume=volume, spacing=spacing)
        elif load_choice == LOAD_AS_MASK:
            provider = MaskImageProvider(name, url, series, index, channel, volume=volume, spacing=spacing)

        workspace.image_set.providers.append(provider)

        self.add_provider_measurements(provider, m, cpmeas.IMAGE)

    @staticmethod
    def add_provider_measurements(provider, m, image_or_objects):
        '''Add image measurements using the provider image and file

        provider - an image provider: get the height and width of the image
                   from the image pixel data and the MD5 hash from the file
                   itself.
        m - measurements structure
        image_or_objects - cpmeas.IMAGE if the provider is an image provider
                           otherwise cpmeas.OBJECT if it provides objects
        '''
        from cellprofiler.modules.loadimages import \
            C_MD5_DIGEST, C_SCALING, C_HEIGHT, C_WIDTH

        name = provider.get_name()
        if name in m.get_names():
            # Get the image with cacheing.
            img = m.get_image(name)
        else:
            img = provider.provide_image(m)
        m[cpmeas.IMAGE, C_MD5_DIGEST + "_" + name] = \
            NamesAndTypes.get_file_hash(provider, m)
        m[cpmeas.IMAGE, C_WIDTH + "_" + name] = img.pixel_data.shape[1]
        m[cpmeas.IMAGE, C_HEIGHT + "_" + name] = img.pixel_data.shape[0]
        if image_or_objects == cpmeas.IMAGE:
            m[cpmeas.IMAGE, C_SCALING + "_" + name] = provider.scale

    @staticmethod
    def get_file_hash(provider, measurements):
        '''Get an md5 checksum from the (cached) file courtesy of the provider'''
        return provider.get_md5_hash(measurements)

    def add_objects(self, workspace, name, should_save_outlines,
                    outlines_name, stack):
        '''Add objects loaded from a file to the object set

        workspace - the workspace for the analysis
        name - the objects' name in the pipeline
        should_save_outlines - True if the user wants to save outlines as an image
        outlines_name - the name of the outlines image in the pipeline
        stack - the ImagePlaneDetailsStack representing the planes to be loaded
        '''
        from cellprofiler.modules.identify import add_object_count_measurements
        from cellprofiler.modules.identify import add_object_location_measurements
        from cellprofiler.modules.identify import add_object_location_measurements_ijv

        num_dimensions = J.call(stack, "numDimensions", "()I")
        if num_dimensions == 2:
            # Should never reach here - should be 3D, but we defensively code
            num_frames = 1
            index = None  # signal that we haven't read the metadata
            series = None
            coords = J.get_env().make_int_array(np.zeros(2, int))
            ipd = cpp.ImagePlaneDetails(
                    J.call(stack, "get", "([I)Ljava/lang/Object;", coords))
            url = ipd.url
        else:
            coords = np.zeros(num_dimensions, np.int32)
            ipds = []
            for i in range(J.call(stack, "size", "(I)I", 2)):
                coords[2] = i
                jcoords = J.get_env().make_int_array(coords)
                ipds.append(cpp.ImagePlaneDetails(
                        J.call(stack, "get", "([I)Ljava/lang/Object;", coords)))
            OBJECTS_CHANNEL = J.get_static_field(
                    "org/cellprofiler/imageset/ImagePlane",
                    "OBJECT_PLANES", "I")
            if len(ipds) == 1 and ipds[0].channel == OBJECTS_CHANNEL and \
                            ipds[0].series == 0 and ipds[0].index == 0:
                # Most likely metadata has not been read.
                # Not much harm in rereading
                index = None
                series = None
                url = ipds[0].url
            else:
                if any([ipd.url != ipds[0].url for ipd in ipds[1:]]):
                    raise NotImplementedError(
                            "Planes from different files not yet supported.")
                index = [ipd.index for ipd in ipds]
                series = [ipd.series for ipd in ipds]
                url = ipds[0].url

        url = workspace.measurements.alter_url_post_create_batch(url)
        provider = ObjectsImageProvider(name, url, series, index)
        self.add_provider_measurements(provider, workspace.measurements,
                                       cpmeas.OBJECT)
        image = provider.provide_image(workspace.image_set)
        o = cpo.Objects()
        if image.pixel_data.shape[2] == 1:
            o.segmented = image.pixel_data[:, :, 0]
            add_object_location_measurements(workspace.measurements,
                                             name,
                                             o.segmented,
                                             o.count)
        else:
            ijv = np.zeros((0, 3), int)
            for i in range(image.pixel_data.shape[2]):
                plane = image.pixel_data[:, :, i]
                shape = plane.shape
                i, j = np.mgrid[0:shape[0], 0:shape[1]]
                ijv = np.vstack(
                        (ijv,
                         np.column_stack([x[plane != 0] for x in (i, j, plane)])))
            o.set_ijv(ijv, shape)
            add_object_location_measurements_ijv(workspace.measurements,
                                                 name, o.ijv, o.count)
        add_object_count_measurements(workspace.measurements, name, o.count)
        workspace.object_set.add_objects(o, name)
        if should_save_outlines:
            outline_image = np.zeros(image.pixel_data.shape[:2], bool)
            for labeled_image, indices in o.get_labels():
                plane = centrosome.outline.outline(labeled_image)
                outline_image |= plane.astype(outline_image.dtype)
            out_img = cpi.Image(outline_image)
            workspace.image_set.add(outlines_name, out_img)

    def on_activated(self, workspace):
        self.pipeline = workspace.pipeline
        self.metadata_keys = sorted(self.pipeline.get_available_metadata_keys().keys())
        self.update_all_metadata_predicates()
        self.update_joiner()

    def on_deactivated(self):
        self.pipeline = None

    def on_setting_changed(self, setting, pipeline):
        '''Handle updates to all settings'''
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
        '''Return the names of all images produced by this module'''
        if self.assignment_method == ASSIGN_ALL:
            return [self.single_image_provider.value]
        elif self.assignment_method == ASSIGN_RULES:
            return [group.image_name.value
                    for group in self.assignments + self.single_images
                    if group.load_as_choice != LOAD_AS_OBJECTS]
        return []

    def get_object_names(self):
        '''Return the names of all objects produced by this module'''
        if self.assignment_method == ASSIGN_RULES:
            return [group.object_name.value
                    for group in self.assignments + self.single_images
                    if group.load_as_choice == LOAD_AS_OBJECTS]
        return []

    def get_column_names(self):
        if self.assignment_method == ASSIGN_ALL:
            return self.get_image_names()
        column_names = []
        for group in self.assignments + self.single_images:
            if group.load_as_choice == LOAD_AS_OBJECTS:
                column_names.append(group.object_name.value)
            else:
                column_names.append(group.image_name.value)
        return column_names

    def get_measurement_columns(self, pipeline):
        '''Create a list of measurements produced by this module

        For NamesAndTypes, we anticipate that the pipeline will create
        the text measurements for the images.
        '''
        from cellprofiler.modules.loadimages import \
            C_FILE_NAME, C_PATH_NAME, C_URL, C_MD5_DIGEST, C_SCALING, \
            C_HEIGHT, C_WIDTH, C_SERIES, C_FRAME, \
            C_OBJECTS_FILE_NAME, C_OBJECTS_PATH_NAME, C_OBJECTS_URL
        from cellprofiler.measurement import \
            C_OBJECTS_SERIES, C_OBJECTS_FRAME
        from cellprofiler.modules.identify import get_object_measurement_columns
        from cellprofiler.measurement import FTR_OBJECT_NUMBER
        from cellprofiler.measurement import FTR_CENTER_Y
        from cellprofiler.measurement import FTR_CENTER_X
        from cellprofiler.measurement import C_COUNT
        from cellprofiler.measurement import C_NUMBER
        from cellprofiler.measurement import C_LOCATION

        image_names = self.get_image_names()
        object_names = self.get_object_names()
        result = []
        for image_name in image_names:
            result += [(cpmeas.IMAGE,
                        "_".join([category, image_name]),
                        coltype)
                       for category, coltype in (
                           (C_FILE_NAME, cpmeas.COLTYPE_VARCHAR_FILE_NAME),
                           (C_PATH_NAME, cpmeas.COLTYPE_VARCHAR_PATH_NAME),
                           (C_URL, cpmeas.COLTYPE_VARCHAR_PATH_NAME),
                           (C_MD5_DIGEST, cpmeas.COLTYPE_VARCHAR_FORMAT % 32),
                           (C_SCALING, cpmeas.COLTYPE_FLOAT),
                           (C_WIDTH, cpmeas.COLTYPE_INTEGER),
                           (C_HEIGHT, cpmeas.COLTYPE_INTEGER),
                           (C_SERIES, cpmeas.COLTYPE_INTEGER),
                           (C_FRAME, cpmeas.COLTYPE_INTEGER)
                       )]
        for object_name in object_names:
            result += [(cpmeas.IMAGE,
                        "_".join([category, object_name]),
                        coltype)
                       for category, coltype in (
                           (C_OBJECTS_FILE_NAME, cpmeas.COLTYPE_VARCHAR_FILE_NAME),
                           (C_OBJECTS_PATH_NAME, cpmeas.COLTYPE_VARCHAR_PATH_NAME),
                           (C_OBJECTS_URL, cpmeas.COLTYPE_VARCHAR_PATH_NAME),
                           (C_COUNT, cpmeas.COLTYPE_INTEGER),
                           (C_MD5_DIGEST, cpmeas.COLTYPE_VARCHAR_FORMAT % 32),
                           (C_WIDTH, cpmeas.COLTYPE_INTEGER),
                           (C_HEIGHT, cpmeas.COLTYPE_INTEGER),
                           (C_OBJECTS_SERIES, cpmeas.COLTYPE_INTEGER),
                           (C_OBJECTS_FRAME, cpmeas.COLTYPE_INTEGER)
                       )]
            result += get_object_measurement_columns(object_name)
        result += [(cpmeas.IMAGE, ftr, cpmeas.COLTYPE_VARCHAR)
                   for ftr in self.get_metadata_features()]

        return result

    def get_categories(self, pipeline, object_name):
        from cellprofiler.modules.loadimages import \
            C_FILE_NAME, C_PATH_NAME, C_URL, C_MD5_DIGEST, C_SCALING, \
            C_HEIGHT, C_WIDTH, C_SERIES, C_FRAME, \
            C_OBJECTS_FILE_NAME, C_OBJECTS_PATH_NAME, C_OBJECTS_URL
        from cellprofiler.measurement import C_COUNT
        from cellprofiler.measurement import C_NUMBER
        from cellprofiler.measurement import C_LOCATION
        result = []
        if object_name == cpmeas.IMAGE:
            has_images = any(self.get_image_names())
            has_objects = any(self.get_object_names())
            if has_images:
                result += [C_FILE_NAME, C_PATH_NAME, C_URL]
            if has_objects:
                result += [C_OBJECTS_FILE_NAME, C_OBJECTS_PATH_NAME,
                           C_OBJECTS_URL, C_COUNT]
            result += [C_MD5_DIGEST, C_SCALING, C_HEIGHT, C_WIDTH, C_SERIES,
                       C_FRAME]
        elif object_name in self.get_object_names():
            result += [C_LOCATION, C_NUMBER]
        return result

    def get_measurements(self, pipeline, object_name, category):
        from cellprofiler.modules.loadimages import \
            C_FILE_NAME, C_PATH_NAME, C_URL, C_MD5_DIGEST, C_SCALING, \
            C_HEIGHT, C_WIDTH, C_SERIES, C_FRAME, \
            C_OBJECTS_FILE_NAME, C_OBJECTS_PATH_NAME, C_OBJECTS_URL
        from cellprofiler.measurement import FTR_OBJECT_NUMBER
        from cellprofiler.measurement import FTR_CENTER_Y
        from cellprofiler.measurement import FTR_CENTER_X
        from cellprofiler.measurement import C_COUNT
        from cellprofiler.measurement import C_NUMBER
        from cellprofiler.measurement import C_LOCATION

        image_names = self.get_image_names()
        object_names = self.get_object_names()
        if object_name == cpmeas.IMAGE:
            if category in (C_FILE_NAME, C_PATH_NAME, C_URL):
                return image_names
            elif category in (C_OBJECTS_FILE_NAME, C_OBJECTS_PATH_NAME,
                              C_OBJECTS_URL):
                return object_names
            elif category == C_COUNT:
                return object_names
            elif category in (C_MD5_DIGEST, C_SCALING, C_HEIGHT, C_WIDTH,
                              C_SERIES, C_FRAME):
                return list(image_names) + list(object_names)
        elif object_name in self.get_object_names():
            if category == C_NUMBER:
                return [FTR_OBJECT_NUMBER]
            elif category == C_LOCATION:
                return [FTR_CENTER_X, FTR_CENTER_Y]
        return []

    def validate_module(self, pipeline):
        '''Validate the settings for the NamesAndTypes module

        Make sure the metadata matcher has at least one completely
        specified channel.
        '''
        if self.assignment_method == ASSIGN_RULES \
                and self.matching_choice == MATCH_BY_METADATA \
                and len(self.assignments) > 1:
            joins = self.join.parse()
            for name in self.get_column_names():
                for join in joins:
                    if join.get(name) is None:
                        break
                else:
                    return
            raise cps.ValidationError(
                    "At least one channel must have all metadata keys specified. "
                    "All channels have at least one metadata key of (None).", self.join)

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if variable_revision_number == 1:
            # Changed naming of assignment methods
            setting_values[0] = ASSIGN_ALL if setting_values[0] == "Assign all images" else ASSIGN_RULES
            variable_revision_number = 2

        if variable_revision_number == 2:
            # Added single rescale and assignment method rescale
            n_assignments = int(setting_values[IDX_ASSIGNMENTS_COUNT_V2])
            new_setting_values = setting_values[:IDX_ASSIGNMENTS_COUNT_V2] + [
                "Yes", setting_values[IDX_ASSIGNMENTS_COUNT_V2]]
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
                                      idx:(idx + NUM_ASSIGNMENT_SETTINGS_V3)]
                new_setting_values += [cps.NO, "LoadedObjects"]
            setting_values = new_setting_values
            variable_revision_number = 4

        if variable_revision_number == 4:
            # Added single images (+ single image count)
            setting_values = setting_values[:IDX_SINGLE_IMAGES_COUNT_V5] + \
                             ["0"] + setting_values[IDX_SINGLE_IMAGES_COUNT_V5:]
            variable_revision_number = 5
        if variable_revision_number == 5:
            #
            # Convert LOAD_AS_MASK_V5A to LOAD_AS_MASK if present
            #
            #
            # Added manual_rescale
            #
            new_setting_values = setting_values[
                                 :IDX_FIRST_ASSIGNMENT_V5] + [DEFAULT_MANUAL_RESCALE]
            n_assignments = int(setting_values[IDX_ASSIGNMENTS_COUNT_V5])
            n_single_images = int(setting_values[IDX_SINGLE_IMAGES_COUNT_V5])
            for i in range(n_assignments):
                offset = IDX_FIRST_ASSIGNMENT_V5 + \
                         NUM_ASSIGNMENT_SETTINGS_V5 * i
                new_setting_values += \
                    setting_values[offset: offset + OFF_LOAD_AS_CHOICE_V5]
                load_as = setting_values[offset + OFF_LOAD_AS_CHOICE_V5]
                if load_as == LOAD_AS_MASK_V5A:
                    load_as = LOAD_AS_MASK
                new_setting_values += [load_as] + \
                                      setting_values[offset + OFF_LOAD_AS_CHOICE_V5 + 1:
                                      offset + NUM_ASSIGNMENT_SETTINGS_V5] + \
                                      [DEFAULT_MANUAL_RESCALE]
            for i in range(n_single_images):
                offset = IDX_FIRST_ASSIGNMENT_V5 + \
                         NUM_ASSIGNMENT_SETTINGS_V5 * n_assignments + \
                         NUM_SINGLE_IMAGE_SETTINGS_V5 * i
                new_setting_values += \
                    setting_values[offset: offset + OFF_SI_LOAD_AS_CHOICE_V5]
                load_as = setting_values[offset + OFF_SI_LOAD_AS_CHOICE_V5]
                if load_as == LOAD_AS_MASK_V5A:
                    load_as = LOAD_AS_MASK
                new_setting_values += [load_as] + \
                                      setting_values[offset + OFF_SI_LOAD_AS_CHOICE_V5 + 1:
                                      offset + NUM_ASSIGNMENT_SETTINGS_V5] + \
                                      [DEFAULT_MANUAL_RESCALE]
            setting_values = new_setting_values
            variable_revision_number = 6

        if variable_revision_number == 6:
            new_setting_values = setting_values[:9] + [False, 1.0, 1.0, 1.0] + setting_values[9:]
            setting_values = new_setting_values
            variable_revision_number = 7

        return setting_values, variable_revision_number, from_matlab

    def volumetric(self):
        return True

    class FakeModpathResolver(object):
        '''Resolve one modpath to one ipd'''

        def __init__(self, modpath, ipd):
            self.modpath = modpath
            self.ipd = ipd

        def get_image_plane_details(self, modpath):
            assert len(modpath) == len(self.modpath)
            assert all([m1 == m2 for m1, m2 in zip(self.modpath, modpath)])
            return self.ipd

    def update_joiner(self):
        '''Update the joiner setting's entities'''
        if self.assignment_method == ASSIGN_RULES:
            self.join.entities = dict([
                                          (column_name, self.metadata_keys)
                                          for column_name in self.get_column_names()])
            try:
                joins = self.join.parse()
                if len(joins) > 0:
                    for join in joins:
                        best_value = None
                        for key in join.keys():
                            if key not in self.get_column_names():
                                del join[key]
                            elif join[key] is not None and best_value is None:
                                best_value = join[key]
                        for i, column_name in enumerate(self.get_column_names()):
                            if not join.has_key(column_name):
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
                " / ".join(set([k for k in join.values() if k is not None]))
                for join in joins]
        else:
            metadata_columns = [cpmeas.IMAGE_NUMBER]
        return metadata_columns


class MetadataPredicate(cps.Filter.FilterPredicate):
    '''A predicate that compares an ifd against a metadata key and value'''

    SYMBOL = "metadata"

    def __init__(self, display_name, display_fmt="%s", **kwargs):
        subpredicates = [cps.Filter.DoesPredicate([]),
                         cps.Filter.DoesNotPredicate([])]

        super(self.__class__, self).__init__(
                self.SYMBOL, display_name, MetadataPredicate.do_filter,
                subpredicates, **kwargs)
        self.display_fmt = display_fmt

    def set_metadata_keys(self, keys):
        '''Define the possible metadata keys to be matched against literal values

        keys - a list of keys
        '''
        sub_subpredicates = [
            cps.Filter.FilterPredicate(
                    key,
                    self.display_fmt % key,
                    lambda ipd, match, key=key:
                    ipd.metadata.has_key(key) and
                    ipd.metadata[key] == match,
                    [cps.Filter.LITERAL_PREDICATE])
            for key in keys]
        #
        # The subpredicates are "Does" and "Does not", so we add one level
        # below that.
        #
        for subpredicate in self.subpredicates:
            subpredicate.subpredicates = sub_subpredicates

    @classmethod
    def do_filter(cls, arg, *vargs):
        '''Perform the metadata predicate's filter function

        The metadata predicate has subpredicates that look up their
        metadata key in the ipd and compare it against a literal.
        '''
        node_type, modpath, resolver = arg
        ipd = resolver.get_image_plane_details(modpath)
        return vargs[0](ipd, *vargs[1:])

    def test_valid(self, pipeline, *args):
        modpath = ["imaging", "image.png"]
        ipd = cpp.ImagePlaneDetails("/imaging/image.png", None, None, None)
        self((cps.FileCollectionDisplay.NODE_IMAGE_PLANE, modpath,
              NamesAndTypes.FakeModpathResolver(modpath, ipd)), *args)


class ColorImageProvider(LoadImagesImageProviderURL):
    '''Provide a color image, tripling a monochrome plane if needed'''

    def __init__(self, name, url, series, index, rescale=True, volume=False, spacing=None):
        LoadImagesImageProviderURL.__init__(self, name, url,
                                            rescale=rescale,
                                            series=series,
                                            index=index,
                                            volume=volume,
                                            spacing=spacing)

    def provide_image(self, image_set):
        image = LoadImagesImageProviderURL.provide_image(self, image_set)

        if image.pixel_data.ndim == image.dimensions:
            image.pixel_data = skimage.color.gray2rgb(image.pixel_data, alpha=False)

        return image


class MonochromeImageProvider(LoadImagesImageProviderURL):
    '''Provide a monochrome image, combining RGB if needed'''

    def __init__(self, name, url, series, index, channel, rescale=True, volume=False, spacing=None):
        LoadImagesImageProviderURL.__init__(self, name, url,
                                            rescale=rescale,
                                            series=series,
                                            index=index,
                                            channel=channel,
                                            volume=volume,
                                            spacing=spacing)

    def provide_image(self, image_set):
        image = LoadImagesImageProviderURL.provide_image(self, image_set)

        if image.pixel_data.ndim == image.dimensions + 1:
            image.pixel_data = skimage.color.rgb2gray(image.pixel_data)

        return image


class MaskImageProvider(MonochromeImageProvider):
    '''Provide a boolean image, converting nonzero to True, zero to False if needed'''

    def __init__(self, name, url, series, index, channel, volume=False, spacing=None):
        MonochromeImageProvider.__init__(self, name, url,
                                         rescale=True,
                                         series=series,
                                         index=index,
                                         channel=channel,
                                         volume=volume,
                                         spacing=spacing)

    def provide_image(self, image_set):
        image = MonochromeImageProvider.provide_image(self, image_set)
        if image.pixel_data.dtype.kind != 'b':
            image.pixel_data = image.pixel_data != 0
        return image


class ObjectsImageProvider(LoadImagesImageProviderURL):
    '''Provide a multi-plane integer image, interpreting an image file as objects'''

    def __init__(self, name, url, series, index):
        LoadImagesImageProviderURL.__init__(self, name, url,
                                            rescale=False,
                                            series=series,
                                            index=index,
                                            volume=False)

    def provide_image(self, image_set):
        """Load an image from a pathname
        """
        self.cache_file()
        filename = self.get_filename()
        channel_names = []
        url = self.get_url()
        properties = {}
        if self.index is None:
            metadata = get_omexml_metadata(self.get_full_name())

            ometadata = OME.OMEXML(metadata)
            pixel_metadata = ometadata.image(0 if self.series is None
                                             else self.series).Pixels
            nplanes = (pixel_metadata.SizeC * pixel_metadata.SizeZ *
                       pixel_metadata.SizeT)
            indexes = range(nplanes)
        elif np.isscalar(self.index):
            indexes = [self.index]
        else:
            indexes = self.index
        planes = []
        offset = 0
        for i, index in enumerate(indexes):
            properties["index"] = str(index)
            if self.series is not None:
                if np.isscalar(self.series):
                    properties["series"] = self.series
                else:
                    properties["series"] = self.series[i]
            img = load_image(
                    self.get_full_name(),
                    rescale=False, **properties).astype(int)
            img = convert_image_to_objects(img).astype(np.int32)
            img[img != 0] += offset
            offset += np.max(img)
            planes.append(img)

        image = cpi.Image(np.dstack(planes),
                          path_name=self.get_pathname(),
                          file_name=self.get_filename(),
                          convert=False)
        return image
