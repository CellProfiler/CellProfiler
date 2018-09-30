# coding=utf-8

"""
LoadImages
==========

**LoadImages** allows you to specify which images or movies are to be
loaded and in which order.

This module tells CellProfiler where to retrieve images and gives each
image a meaningful name by which other modules can access it. You can
also use **LoadImages** to extract or define the relationships between
images and their associated metadata. For example, you could load a
group of images (such as three channels that represent the same field of
view) together for processing in a single CellProfiler cycle. Finally,
you can use this module to retrieve a label matrix and give the
collection of objects a meaningful name.

*Disclaimer:* Please note that the Input modules (i.e., **Images**,
**Metadata**, **NamesAndTypes** and **Groups**) largely supersedes this
module. However, old pipelines loaded into CellProfiler that contain
this module will provide the option of preserving them; these pipelines
will operate exactly as before.

When used in combination with a **SaveImages** module, you can load
images in one file format and save them in another, using CellProfiler
as a file format converter.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

See also
^^^^^^^^

See also the **Input** modules (**Images**, **NamesAndTypes**,
**MetaData**, **Groups**), **LoadData**, **LoadSingleImage**,
and **SaveImages**.

Using metadata in LoadImages
''''''''''''''''''''''''''''

If you would like to use the metadata-specific settings, please see
*Help > General help > Using metadata in CellProfiler* for more details
on metadata usage and syntax. Briefly, **LoadImages** can extract
metadata from the image filename using pattern-matching strings, for
grouping similar images together for the analysis run and for
metadata-specific options in other modules; see the settings help for
*Where to extract metadata*, and if an option for that setting is
selected, *Regular expression that finds metadata in the file name*
for the necessary syntax.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *Pathname, Filename:* The full path and the filename of each image.
-  *Metadata:* The metadata information extracted from the path and/or
   filename, if requested.
-  *Scaling:* The maximum possible intensity value for the image format.
-  *Height, Width:* The height and width of the current image.

"""
from __future__ import print_function

import hashlib
import logging
import os
import os.path
import re
import sys
import tempfile
import urllib
import urlparse
import shutil

import six

from cellprofiler.modules import _help
import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.misc
import cellprofiler.module
import cellprofiler.object
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.setting
import centrosome.outline
from cellprofiler.modules import identify, images
import numpy
import scipy.io.matlab.mio
import skimage.external.tifffile

logger = logging.getLogger(__name__)
cached_file_lists = {}

'''STK TIFF Tag UIC1 - for MetaMorph internal use'''
UIC1_TAG = 33628
'''STK TIFF Tag UIC2 - stack z distance, creation time...'''
UIC2_TAG = 33629
'''STK TIFF TAG UIC3 - wavelength'''
UIC3_TAG = 33630
'''STK TIFF TAG UIC4 - internal'''
UIC4_TAG = 33631

'''The MD5 digest measurement category'''
C_MD5_DIGEST = "MD5Digest"

'''The intensity scaling metadata for this file'''
C_SCALING = "Scaling"

'''The dimension metadata for the image'''
C_HEIGHT = "Height"
C_WIDTH = "Width"

# strings for choice variables
MS_EXACT_MATCH = 'Text-Exact match'
MS_REGEXP = 'Text-Regular expressions'
MS_ORDER = 'Order'

FF_INDIVIDUAL_IMAGES = 'individual images'
FF_STK_MOVIES = 'stk movies'
FF_AVI_MOVIES = 'avi,mov movies'
FF_AVI_MOVIES_OLD = ['avi movies']
FF_OTHER_MOVIES = 'tif,tiff,flex,zvi movies'
FF_OTHER_MOVIES_OLD = ['tif,tiff,flex movies', 'tif,tiff,flex movies, zvi movies']

'''Tag for loading images as images'''
IO_IMAGES = "Images"
'''Tag for loading images as segmentation results'''
IO_OBJECTS = "Objects"
IO_ALL = (IO_IMAGES, IO_OBJECTS)

'''The format string for naming the image for some objects'''
IMAGE_FOR_OBJECTS_F = "IMAGE_FOR_%s"

# The following is a list of extensions supported by PIL 1.1.7
SUPPORTED_IMAGE_EXTENSIONS = set([
    '.ppm', '.grib', '.im', '.rgba', '.rgb', '.pcd', '.h5', '.jpe', '.jfif',
    '.jpg', '.fli', '.sgi', '.gbr', '.pcx', '.mpeg', '.jpeg', '.ps', '.flc',
    '.tif', '.hdf', '.icns', '.gif', '.palm', '.mpg', '.fits', '.pgm', '.mic',
    '.fit', '.xbm', '.eps', '.emf', '.dcx', '.bmp', '.bw', '.pbm', '.dib',
    '.ras', '.cur', '.fpx', '.png', '.msp', '.iim', '.wmf', '.tga', '.bufr',
    '.ico', '.psd', '.xpm', '.arg', '.pdf', '.tiff'])
SUPPORTED_IMAGE_EXTENSIONS.add(".mat")
SUPPORTED_IMAGE_EXTENSIONS.add(".npy")
# The following is a list of the extensions as gathered from Bio-formats
# Missing are .cfg, .csv, .html, .htm, .log, .txt, .xml and .zip which are likely
# not to be images but you are welcome to add if needed
#
SUPPORTED_IMAGE_EXTENSIONS.update(
        [".1sc", ".2fl", ".acff", ".afi", ".afm", ".aiix", ".aim", ".aisf",
         ".al3d", ".ali", ".am", ".amiramesh", ".ano", ".apl", ".arf", ".atsf",
         ".avi", ".bip", ".bmp", ".btf", ".c01", ".cr2", ".crw",
         ".cxd", ".czi", ".dat", ".dcm", ".df3", ".dib", ".dic", ".dicom", ".dm2",
         ".dm3", ".dm4", ".dti", ".dv", ".dv.log", ".eps", ".epsi", ".ets",
         ".exp", ".fake", ".fdf", ".fff", ".ffr", ".fits", ".flex", ".fli",
         ".frm", ".fts", ".gel", ".gif", ".grey", ".hdr", ".hed", ".his", ".htd",
         ".hx", ".ics", ".ids", ".ima", ".img", ".ims", ".inf", ".inr", ".ipl",
         ".ipm", ".ipw", ".j2k", ".j2ki", ".j2kr", ".jp2", ".jpe", ".jpeg",
         ".jpf", ".jpg", ".jpk", ".jpx", ".l2d", ".labels", ".lei", ".lif",
         ".liff", ".lim", ".lsm", ".lut", ".map", ".mdb", ".mea",
         ".mnc", ".mng", ".mod", ".mrc", ".mrw", ".msr", ".mtb",
         ".mvd2", ".naf", ".nd", ".nd2", ".ndpi", ".ndpis", ".nef", ".nhdr",
         ".nii", ".nrrd", ".obf", ".oib", ".oif", ".ome", ".ome.tif",
         ".ome.tiff", ".par", ".pcoraw", ".pct", ".pcx", ".pgm", ".pic",
         ".pict", ".png", ".pnl", ".pr3", ".ps", ".psd", ".pst", ".pty",
         ".r3d", ".r3d.log", ".r3d_d3d", ".raw", ".rec", ".res", ".scn",
         ".sdt", ".seq", ".sif", ".sld", ".sm2", ".sm3", ".spi", ".spl",
         ".st", ".stk", ".stp", ".svs", ".sxm", ".tf2", ".tf8", ".tfr",
         ".tga", ".thm", ".tif", ".tiff", ".tim", ".tnb", ".top",
         ".v", ".vms", ".vsi", ".vws", ".wat", ".wav", ".wlz", ".xdce",
         ".xlog", ".xqd", ".xqf", ".xv", ".xys", ".zfp", ".zfr",
         ".zpo", ".zvi"])
SUPPORTED_MOVIE_EXTENSIONS = set(['.avi', '.mpeg', '.stk', '.flex', '.mov', '.tif',
                                  '.tiff', '.zvi'])

FF = [FF_INDIVIDUAL_IMAGES, FF_STK_MOVIES, FF_AVI_MOVIES, FF_OTHER_MOVIES]
SUPPORTED_IMAGE_EXTENSIONS.update([
    ".1sc", ".2fl", ".afm", ".aim", ".avi", ".co1", ".flex", ".fli", ".gel",
    ".ics", ".ids", ".im", ".img", ".j2k", ".lif", ".lsm", ".mpeg", ".pic",
    ".pict", ".ps", ".raw", ".svs", ".stk", ".tga", ".zvi", ".c01", ".xdce"])
SUPPORTED_MOVIE_EXTENSIONS.update(['mng'])

# The metadata choices:
# M_NONE - don't extract metadata
# M_FILE_NAME - extract metadata from the file name
# M_PATH_NAME - extract metadata from the subdirectory path
# M_BOTH      - extract metadata from both the file name and path
M_NONE = "None"
M_FILE_NAME = "File name"
M_PATH = "Path"
M_BOTH = "Both"

#
# FLEX metadata
#
M_Z = "Z"
M_T = "T"
'''For formats like TIF, the series is the image stack index'''
C_SERIES = "Series"
'''The frame is the frame number in a movie or in a TIF image stack'''
C_FRAME = "Frame"

'''The provider name for the image file image provider'''
P_IMAGES = "LoadImagesImageProvider"
'''The version number for the __init__ method of the image file image provider'''
V_IMAGES = 1

'''The provider name for the movie file image provider'''
P_MOVIES = "LoadImagesMovieProvider"
'''The version number for the __init__ method of the movie file image provider'''
V_MOVIES = 2

'''The provider name for the flex file image provider'''
P_FLEX = 'LoadImagesFlexFrameProvider'
'''The version number for the __init__ method of the flex file image provider'''
V_FLEX = 1

'''Interleaved movies'''
I_INTERLEAVED = "Interleaved"

'''Separated movies'''
I_SEPARATED = "Separated"

'''Subfolder choosing options'''
SUB_NONE = "None"
SUB_ALL = "All"
SUB_SOME = "Some"


def default_cpimage_name(index):
    # the usual suspects
    names = ['DNA', 'Actin', 'Protein']
    if index < len(names):
        return names[index]
    return 'Channel%d' % (index + 1)


class LoadImages(cellprofiler.module.Module):
    module_name = "LoadImages"
    variable_revision_number = 11
    category = "File Processing"

    def create_settings(self):
        # Settings
        self.file_types = cellprofiler.setting.Choice(
            'File type to be loaded',
            FF,
            doc="""\
CellProfiler accepts the following image file types. For movie file
formats, the files are opened as a stack of images and each image is
processed individually, although **TrackObjects** can be used to relate
objects across timepoints.

-  *{FF_INDIVIDUAL_IMAGES}:* Each file represents a single image.
   Some methods of file compression sacrifice image quality (“lossy”)
   and should be avoided for automated image analysis if at all possible
   (e.g., .jpg). Other file compression formats retain exactly the
   original image information but in a smaller file (“lossless”) so they
   are perfectly acceptable for image analysis (e.g., .png, .tif, .gif).
   Uncompressed file formats are also fine for image analysis (e.g.,
   .bmp).
-  *{FF_AVI_MOVIES}:* AVIs (Audio Video Interleave) and MOVs
   (QuicktTime) files are types of movie files. Only uncompressed AVIs
   are supported; supported MOVs are listed `here`_. Note that .mov
   files are not supported on 64-bit systems.
-  *{FF_STK_MOVIES}:* STKs are a proprietary image format used by
   MetaMorph (Molecular Devices). It is typically used to encode 3D
   image data, e.g., from confocal microscopy, and is a special version
   of the TIF format.
-  *{FF_OTHER_MOVIES}:* A TIF/TIFF movie is a file that contains a
   series of images as individual frames. The same is true for the FLEX
   file format (used by Evotec Opera automated microscopes). ZVIs are a
   proprietary image format used by Zeiss. It is typically used to
   encode 3D image data, e.g., from fluorescence microscopy.

.. _here: http://www.openmicroscopy.org/site/support/bio-formats5/formats/quicktime-movie.html
""".format(**{
                "FF_INDIVIDUAL_IMAGES": FF_INDIVIDUAL_IMAGES,
                "FF_AVI_MOVIES": FF_AVI_MOVIES,
                "FF_STK_MOVIES": FF_STK_MOVIES,
                "FF_OTHER_MOVIES": FF_OTHER_MOVIES
            })
        )

        self.match_method = cellprofiler.setting.Choice(
            'File selection method',
            [
                MS_EXACT_MATCH,
                MS_REGEXP,
                MS_ORDER
            ],
            doc="""\
Three options are available:

-  *{MS_EXACT_MATCH}:* Used to load image (or movie) files that have
   a particular piece of text in the name. The specific text that is
   entered will be searched for in the filenames and the files that
   contain that text exactly will be loaded and given the name you
   specify. The search for the text is case-sensitive.
-  *{MS_REGEXP}:* Used to load image (or movie) files that match a
   pattern of regular expressions.
-  *{MS_ORDER}:* Used when image (or movie) files are present in a
   repeating order, like “DAPI, FITC, Red; DAPI, FITC, Red;” and so on.
   Images are loaded based on the order of their location on the hard
   disk, and they are assigned an identity based on how many images are
   in each group and what position within each group the file is located
   (e.g., three images per group; DAPI is always first).

{REGEXP_HELP_REF}
""".format(**{
                "MS_EXACT_MATCH": MS_EXACT_MATCH,
                "MS_REGEXP": MS_REGEXP,
                "MS_ORDER": MS_ORDER,
                "REGEXP_HELP_REF": _help.REGEXP_HELP_REF
            })
        )

        self.exclude = cellprofiler.setting.Binary(
            'Exclude certain files?',
            False,
            doc="""\
*(Used only if “{MS_EXACT_MATCH}” for loading files is selected)*

The image/movie files specified with the *Text* options may also include
files that you want to exclude from analysis (such as thumbnails created
by an imaging system). Select *{YES}* to enter text to match against
such files for exclusion.
""".format(**{
                "MS_EXACT_MATCH": MS_EXACT_MATCH,
                "YES": cellprofiler.setting.YES
            })
        )

        self.match_exclude = cellprofiler.setting.Text(
            'Type the text that the excluded images have in common',
            cellprofiler.setting.DO_NOT_USE,
            doc="""\
*(Used only if file exclusion is selected)*

Specify text that marks files for exclusion. **LoadImages** looks for
this text as an exact match within the filename and not as a regular
expression.
"""
        )

        self.order_group_size = cellprofiler.setting.Integer(
            'Number of images in each group?',
            3,
            doc="""\
*(Used only when Order is selected for file loading)*

Enter the number of images that comprise a group. For example, for
images given in the order: *DAPI, FITC, Red; DAPI, FITC, Red* and so on,
the number of images that in each group would be 3.
"""
        )

        self.descend_subdirectories = cellprofiler.setting.Choice(
            'Analyze all subfolders within the selected folder?',
            [
                SUB_NONE,
                SUB_ALL,
                SUB_SOME
            ],
            doc="""\
This setting determines whether **LoadImages** analyzes just the images
in the specified folder or whether it analyzes images in subfolders as
well:

-  *{SUB_ALL}:* Analyze all matching image files in subfolders under
   your specified image folder location.
-  *{SUB_NONE}:* Only analyze files in the specified location.
-  *{SUB_SOME}:* Select which subfolders to analyze.
""".format(**{
                "SUB_ALL": SUB_ALL,
                "SUB_NONE": SUB_NONE,
                "SUB_SOME": SUB_SOME
            })
        )

        # Location settings
        self.location = cellprofiler.setting.DirectoryPath(
            "Input image file location",
            dir_choices=[
                cellprofiler.preferences.ABSOLUTE_FOLDER_NAME,
                cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME,
                cellprofiler.preferences.DEFAULT_OUTPUT_FOLDER_NAME,
                cellprofiler.preferences.DEFAULT_INPUT_SUBFOLDER_NAME,
                cellprofiler.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME
            ],
            allow_metadata=False,
            doc="Select the folder containing the images to be loaded. {IO_FOLDER_CHOICE_HELP_TEXT}".format(**{
                "IO_FOLDER_CHOICE_HELP_TEXT": _help.IO_FOLDER_CHOICE_HELP_TEXT
            })
        )

        self.subdirectory_filter = cellprofiler.setting.SubdirectoryFilter(
            "Select subfolders to analyze",
            directory_path=self.location,
            doc="""\
Use this control to select some subfolders and exclude
others from analysis. Press the button to see the folder tree
and check or uncheck the checkboxes to enable or disable analysis
of the associated folders.
"""
        )

        self.check_images = cellprofiler.setting.Binary(
            'Check image sets for unmatched or duplicate files?',
            True,
            doc="""\
*(Used only if metadata is extracted from the image file and not loading by order)*

Select *{YES}* to examine the filenames for unmatched or duplicate
files based on extracted metadata. This is useful for images generated
by HCS systems where acquisition may produce a corrupted image and
create a duplicate as a correction or may miss an image entirely. See
the *Extract metadata from where?* setting for more details on
obtaining, extracting, and using metadata tags.
""".format(**{
                "YES": cellprofiler.setting.YES
            })
        )

        self.group_by_metadata = cellprofiler.setting.Binary(
            'Group images by metadata?',
            False,
            doc="""\
*(Used only if metadata is extracted from the image file or if movies are used)*

Select *{YES}* to process those images that share a particular
metadata tag as a group. For example, if you are performing per-plate
illumination correction and the plate metadata is part of the image file
name, image grouping will enable you to process those images that have
the same plate field together (the alternative would be to place the
images from each plate in a separate folder). The next setting allows
you to select the metadata tags by which to
group.{USING_METADATA_GROUPING_HELP_REF}

Please note that if you are loading a movie file(e.g., TIFs, FLEX, STKs,
AVIs, ZVIs), each movie is already treated as a group of images, so
there is no need to enable here.
""".format(**{
                "YES": cellprofiler.setting.YES,
                "USING_METADATA_GROUPING_HELP_REF": _help.USING_METADATA_GROUPING_HELP_REF
            })
        )

        self.metadata_fields = cellprofiler.setting.MultiChoice(
                'Specify metadata fields to group by', [], doc="""\
*(Used only if grouping images by metadata)*

Select the fields by which you want group the image files. You can
select multiple tags. For example, if a set of images had metadata for
“Run”, “Plate”, “Well”, and “Site”, selecting *Run* and *Plate* will
create groups containing images that share the same [*Run*,\ *Plate*]
pair of fields.""")

        # Add the first image to the images list
        self.images = []
        self.add_imagecb(False)
        self.image_count = cellprofiler.setting.HiddenCount(self.images,
                                                            text="Image count")

        # Add another image
        self.add_image = cellprofiler.setting.DoSomething("", "Add another image", self.add_imagecb)

    def add_imagecb(self, can_remove=True):
        'Adds another image to the settings'
        group = cellprofiler.setting.SettingsGroup()

        def example_file_fn(path=None):
            '''Get an example file for use in the file metadata regexp editor'''
            if path is None:
                path = self.image_directory()
                default = "plateA-2008-08-06_A12_s1_w1_[89A882DE-E675-4C12-9F8E-46C9976C4ABE].tif"
            else:
                default = None
            #
            # Find out the index we expect from filter_filename
            #
            for i, test_group in enumerate(self.images):
                if id(test_group) == id(group):
                    break

            filenames = [x for x in os.listdir(path)
                         if x.find('.') != -1 and
                         os.path.splitext(x)[1].upper() in
                         ('.TIF', '.JPG', '.PNG', '.BMP')]
            filtered_filenames = [x for x in filenames
                                  if self.filter_filename(x) == i]
            if len(filtered_filenames) > 0:
                return filtered_filenames[0]
            if len(filenames) > 0:
                return filenames[0]
            if self.analyze_sub_dirs():
                d = [x for x in [os.path.abspath(os.path.join(path, x))
                                 for x in os.listdir(path)
                                 if not x.startswith('.')]
                     if os.path.isdir(x)]
                for subdir in d:
                    result = example_file_fn(subdir)
                    if result is not None:
                        return result
            return default

        def example_path_fn():
            '''Get an example path for use in the path metadata regexp editor'''
            root = self.image_directory()
            d = [x for x in [os.path.abspath(os.path.join(root, x))
                             for x in os.listdir(root)
                             if not x.startswith('.')]
                 if os.path.isdir(x)]
            if len(d) > 0:
                return d[0]
            return root

        img_index = len(self.images)
        self.images.append(group)
        group.append("divider", cellprofiler.setting.Divider(line=True))
        group.append(
            "common_text",
            cellprofiler.setting.Text(
                'Text that these images have in common (case-sensitive)',
                '',
                doc="""\

*(Used only for the image-loading Text options)*

For *Text-Exact match*, type the text string that all the images have
in common. For example, if all the images for the given channel end
with the text “D.TIF”, type ``D.TIF`` here.

For *Text-Regular expression*, type the regular expression that would
capture all the images for this channel. See the module help for more
information on regular expressions.
"""
            )
        )

        group.append(
            "order_position",
            cellprofiler.setting.Integer(
                'Position of this image in each group', img_index + 1,
                minval=1,
                doc="""\
*(Used only for the image-loading Order option)*

Enter the number in the image order that this image channel occupies.
For example, if the order is “DAPI, FITC, Red; DAPI, FITC, Red” and so
on, the DAPI channel would occupy position 1.
"""
            )
        )

        group.append(
            "metadata_choice",
            cellprofiler.setting.Choice(
                'Extract metadata from where?',
                [
                    M_NONE,
                    M_FILE_NAME,
                    M_PATH,
                    M_BOTH
                ],
                doc="""\
Metadata fields can be specified from the image filename, the image path
(including subfolders), or both. The metadata entered here can be used
for image grouping (see the *Group images by metadata?* setting) or
simply used as additional columns in the exported measurements (see the
**ExportToSpreadsheet** module).
"""
            )
        )

        group.append(
            "file_metadata",
            cellprofiler.setting.RegexpText(
                'Regular expression that finds metadata in the file name',
                '^(?P.*)_(?P[A-P][0-9]{2})_s(?P[0-9])',
                get_example_fn=example_file_fn,
                doc="""\
*(Used only if you want to extract metadata from the file name)*

The regular expression to extract the metadata from the file name is
entered here. Note that this field is available whether you have
selected *Text-Regular expressions* to load the files or not. Please see
the general module help for more information on construction of a
regular expression.

Clicking the magnifying glass icon to the right will bring up a tool for
checking the accuracy of your regular expression. The regular expression
syntax can be used to name different parts of your expression. The
syntax *(?P<fieldname>expr)* will extract whatever matches *expr* and
assign it to the measurement,\ *fieldname* for the image.

For instance, a researcher uses plate names composed of a string of
letters and numbers, followed by an underscore, then the well,
followed by another underscore, followed by an “s” and a digit
representing the site taken within the well (e.g.,
*TE12345_A05_s1.tif*). The following regular expression will capture
the plate, well, and site in the fields “Plate”, “Well”, and “Site”:

+----------------------------------------------------------------------------------------------------------------------+
| ^(?P<Plate>.*)_(?P<Well>[A-P][0-9]{1,2})_s(?P<Site>[0-9])                                                            |
+-----------------------------------------------------------+----------------------------------------------------------+
| ^                                                         | Start only at beginning of the file name                 |
+-----------------------------------------------------------+----------------------------------------------------------+
| ?P<Plate>                                                 | Name the captured field *Plate*                          |
+-----------------------------------------------------------+----------------------------------------------------------+
| .*                                                        | Capture as many characters as follow                     |
+-----------------------------------------------------------+----------------------------------------------------------+
| _                                                         | Discard the underbar separating plate from well          |
+-----------------------------------------------------------+----------------------------------------------------------+
| ?P<Well>                                                  | Name the captured field *Well*                           |
+-----------------------------------------------------------+----------------------------------------------------------+
| [A-P]                                                     | Capture exactly one letter between A and P               |
+-----------------------------------------------------------+----------------------------------------------------------+
| [0-9]{1,2}                                                | Capture one or two digits that follow                    |
+-----------------------------------------------------------+----------------------------------------------------------+
| _s                                                        | Discard the underbar followed by *s*                     |
|                                                           | separating well from site                                |
+-----------------------------------------------------------+----------------------------------------------------------+
| ?P<Site>                                                  | Name the captured field *Site*                           |
+-----------------------------------------------------------+----------------------------------------------------------+
| [0-9]                                                     | Capture one digit following                              |
+-----------------------------------------------------------+----------------------------------------------------------+

The regular expression can be typed in the upper text box, with a sample
file name given in the lower text box. Provided the syntax is correct,
the corresponding fields will be highlighted in the same color in the
two boxes. Press *Submit* to enter the typed regular expression.

You can create metadata tags for any portion of the filename or path,
but if you are specifying metadata for multiple images in a single
**LoadImages** module, an image cycle can only have one set of values
for each metadata tag. This means that you can only specify the metadata
tags which have the same value across all images listed in the module.
For example, in the example above, you might load two wavelengths of
data, one named *TE12345_A05_s1_w1.tif* and the other
*TE12345_A05_s1_w2.tif*, where the number following the *w* is the
wavelength. In this case, a “Wavelength” tag *should not* be included in
the regular expression because while the “Plate”, “Well” and “Site”
metadata is identical for both images, the wavelength metadata is not.

Note that if you use the special fieldnames *<WellColumn>* and
*<WellRow>* together, LoadImages will automatically create a *<Well>*
metadata field by joining the two fieldname values together. For
example, if *<WellRow>* is “A” and *<WellColumn>* is “01”, a field
*<Well>* will be “A01”. This is useful if your well row and column names
are separated from each other in the filename, but you want to retain
the standard well nomenclature.
"""
            )
        )

        group.append("path_metadata", cellprofiler.setting.RegexpText(
                'Type the regular expression that finds metadata in the subfolder path',
                '.\*[\\\/](?P.\*)[\\\/](?P.\*)$',
                get_example_fn=example_path_fn,
                guess=cellprofiler.setting.RegexpText.GUESS_FOLDER,
                doc="""\
*(Used only if you want to extract metadata from the path)*

Enter the regular expression for extracting the metadata from the
path. Note that this field is available whether you have selected
*Text-Regular expressions* to load the files or not.

Clicking the magnifying glass icon to the right will bring up a tool
that will allow you to check the accuracy of your regular expression.
The regular expression syntax can be used to name different parts of
your expression. The syntax *(?<fieldname>expr)* will extract whatever
matches *expr* and assign it to the image’s *fieldname* measurement.

For instance, a researcher uses folder names with the date and
subfolders containing the images with the run ID (e.g.,
*.2009_10_02/1234*) The following regular expression will capture
the plate, well, and site in the fields *Date* and *Run*:

+---------------------------------------------------------------------------------------------------------------------+
| .\*[\\\/](?P<Date>.\*)[\\\/](?P<Run>.\*)$                                                                             |
+---------------------------------------------------+-----------------------------------------------------------------+
| .\*[\\\/]                                          | Skip characters at the beginning of the pathname until either a |
|                                                   | slash (/) or backslash (\\\) is encountered (depending on the    |
|                                                   | operating system)                                               |
+---------------------------------------------------+-----------------------------------------------------------------+
| ?P<Date>                                          | Name the captured field *Date*                                  |
+---------------------------------------------------+-----------------------------------------------------------------+
| .\*                                               | Capture as many characters that follow                          |
+---------------------------------------------------+-----------------------------------------------------------------+
| [\\\/]                                             | Discard the slash/backslash character                           |
+---------------------------------------------------+-----------------------------------------------------------------+
| ?P<Run>                                           | Name the captured field *Run*                                   |
+---------------------------------------------------+-----------------------------------------------------------------+
| .\*                                               | Capture as many characters as follow                            |
+---------------------------------------------------+-----------------------------------------------------------------+
| $                                                 | The *Run* field must be at the end of the path string, i.e.,    |
|                                                   | the last folder on the path. This also means that the Date      |
|                                                   | field contains the parent folder of the Date folder.            |
+---------------------------------------------------+-----------------------------------------------------------------+
"""))

        group.append("wants_movie_frame_grouping", cellprofiler.setting.Binary(
                "Group the movie frames?", False,
                doc="""\
*(Used only if a movie image format is selected as file type)*

**LoadImages** can load several frames from a movie into different
images within the same cycle. For example, a movie’s first frame might
be an image of the red fluorescence channel at time zero, the second
might be the green channel at time zero, the third might be the red
channel at time one, etc. Select *{YES}* to extract both channels
for this movie as separate images within the same cycle.

**LoadImages** refers to the individual images in a group as *channels*.
Channels are numbered consecutively, starting at channel 1. To set up
grouping, first specify how the channels are grouped (interleaving and
number of channels per group), then assign image names to each of the
channels individually.
""".format(**{
                    "YES": cellprofiler.setting.YES
                })))

        group.append("interleaving", cellprofiler.setting.Choice(
                "Grouping method", [I_INTERLEAVED, I_SEPARATED],
                doc="""\
*(Used only if a movie image format is selected as file type and movie
frame grouping are selected)*

Channels in a movie can be interleaved or separated.

In an interleaved movie, the first frame is channel 1, the second is
channel 2 and so on up to the number of channels per group for a given
image cycle. In a separated movie, all of the frames for channel 1 are
processed as the first image cycle, then the frames for channel 2 for
the second image cycle, and so on.

For example, a movie may consist of 6 frames and we would like to
process the movie as two channels per group. An interleaved movie would
be processed like this:

+-----------+-------------+-----------------+
| Frame #   | Channel #   | Image cycle #   |
+===========+=============+=================+
| 1         | 1           | 1               |
+-----------+-------------+-----------------+
| 2         | 2           | 1               |
+-----------+-------------+-----------------+
| 3         | 1           | 2               |
+-----------+-------------+-----------------+
| 4         | 2           | 2               |
+-----------+-------------+-----------------+
| 5         | 1           | 3               |
+-----------+-------------+-----------------+
| 6         | 2           | 3               |
+-----------+-------------+-----------------+

For a separated movie, the channels would be processed like this:

+-----------+-------------+-----------------+
| Frame #   | Channel #   | Image cycle #   |
+===========+=============+=================+
| 1         | 1           | 1               |
+-----------+-------------+-----------------+
| 2         | 1           | 2               |
+-----------+-------------+-----------------+
| 3         | 1           | 3               |
+-----------+-------------+-----------------+
| 4         | 2           | 1               |
+-----------+-------------+-----------------+
| 5         | 2           | 2               |
+-----------+-------------+-----------------+
| 6         | 2           | 3               |
+-----------+-------------+-----------------+

Note the difference in which frames are processed in which image cycle
between the two methods."""))

        group.append("channels_per_group", cellprofiler.setting.Integer(
                "Number of channels per group", 3, minval=2,
                reset_view=True, doc="""\
*(Used only if a movie image format is selected as file type and movie
frame grouping is selected)*

This setting controls the number of frames to be grouped together. As an
example, for an interleaved movie with 12 frames and three channels per
group, the first, fourth, seventh and tenth frame will be assigned to
channel 1, the 2\ :sup:`nd`, 5,\ :sup:`th` 8\ :sup:`th` and
11\ :sup:`th` frame will be assigned to channel 2 and the 3\ :sup:`rd`,
6\ :sup:`th`, 9\ :sup:`th`, and 12\ :sup:`th` will be assigned to
channel 3. For a separated movie, frames 1 through 4 will be assigned to
channel 1, 5 through 8 to channel 2 and 9 through 12 to channel 3."""))
        #
        # Flex files (and arguably others like color images and multichannel
        # TIF files) can have more than one channel. So, within each image,
        # we have a list of channels.
        #
        group.channels = []
        group.append("channel_count", cellprofiler.setting.HiddenCount(group.channels, "Channel count"))

        def add_channel(can_remove=True):
            self.add_channel(group, can_remove)

        add_channel(False)

        group.append("add_channel_button", cellprofiler.setting.DoSomething(
                "Add another channel", "Add channel", add_channel))

        group.can_remove = can_remove
        if can_remove:
            group.append("remover", cellprofiler.setting.RemoveSettingButton(
                    '', 'Remove this image', self.images, group))

    def add_channel(self, image_settings, can_remove=True):
        '''Add another channel to an image

        image_settings - the image's settings group
        can_remove - true if we are allowed to remove this channel
        '''

        group = cellprofiler.setting.SettingsGroup()
        image_settings.channels.append(group)
        img_index = 0
        for ii in self.images:
            for jj in ii.channels:
                if id(jj) == id(group):
                    break
                img_index += 1

        group.append(
            "image_object_choice",
            cellprofiler.setting.Choice(
                'Load the input as images or objects?',
                IO_ALL,
                doc="""\
This setting determines whether you load an image as image data or as
segmentation results (i.e., objects):

-  *{IO_IMAGES}:* The input image will be given the name you specify,
   by which it will be referred downstream. This is the most common usage
   for this module.
-  *{IO_OBJECTS}:* Use this option if the input image is a label
   matrix and you want to obtain the objects that it defines. A *label
   matrix* is a grayscale or color image in which the connected regions
   share the same label, and defines how objects are represented in
   CellProfiler. The labels are integer values greater than or equal to
   0. The elements equal to 0 are the background, whereas the elements
   equal to 1 make up one object, the elements equal to 2 make up a
   second object, and so on. This option allows you to use the objects
   without needing to insert an **Identify** module to extract them
   first. See **IdentifyPrimaryObjects** for more details.
""".format(**{
                    "IO_IMAGES": IO_IMAGES,
                    "IO_OBJECTS": IO_OBJECTS
                })
            )
        )

        group.append(
            "image_name",
            cellprofiler.setting.FileImageNameProvider(
                'Name this loaded image',
                default_cpimage_name(img_index),
                doc="""\
What do you want to call the images you are loading for use downstream
in the pipeline? Give your images a meaningful name that you can use to
refer to these images in later modules. Keep the following points in
mind:

-  Image names can consist of any combination of characters (e.g.,
   letters, digits, and other non-alphanumeric characters). However, if
   you are using **ExportToDatabase**, these names will become part of
   the measurement column name, and some characters are not permitted in
   MySQL (e.g., slashes).
-  Names are not case sensitive. Therefore, *OrigBlue*, *origblue*, and
   *ORIGBLUE* will all correspond to the same name, and unexpected
   results may ensue.
-  Although CellProfiler can accept names of any length, you may want to
   avoid making the name too long, especially if you are uploading to a
   database. The name is used to generate the column header for a given
   measurement, and in MySQL the total bytes used for all column headers
   cannot exceed 64K. A warning will be generated later if this limit
   has been exceeded.
   """
            )
        )

        group.append(
            "object_name",
            cellprofiler.setting.ObjectNameProvider(
                'Name this loaded object',
                "Nuclei",
                doc="""\
*(Used only if objects are output)*

This is the name for the objects loaded from your image
"""
            )
        )

        group.append(
            "wants_outlines",
            cellprofiler.setting.Binary(
                'Retain outlines of loaded objects?',
                False,
                doc="""\
*(Used only if objects are output)*

Select *{YES}* if you want to create an image of the outlines of the
loaded objects.
""".format(**{
                    "YES": cellprofiler.setting.YES
                })
            )
        )

        group.append("outlines_name", cellprofiler.setting.OutlineNameProvider(
                'Name the outline image', 'LoadedImageOutlines', doc='''\
*(Used only if objects are output and outlines are saved)*

Enter a name that will allow the outlines to be selected later in the pipeline.

*Special note on saving images:* You can use the settings in this module
to pass object outlines along to the module **OverlayOutlines**, and
then save them with the **SaveImages** module.'''))

        group.get_image_name = lambda: (
            group.image_name.value if self.channel_wants_images(group) else IMAGE_FOR_OBJECTS_F % group.object_name.value
        )

        channels = [
            str(x) for x in range(1, max(10, len(image_settings.channels) + 2))]

        group.append(
            "channel_number",
            cellprofiler.setting.Choice(
                "Channel number",
                channels,
                channels[len(image_settings.channels) - 1],
                doc="""\
*(Used only if a movie image format is selected as file type and movie
frame grouping is selected)*

The channels of a multichannel image are numbered starting from 1. Each
channel is a greyscale image, acquired using different illumination
sources and/or optics. Use this setting to pick the channel to associate
with the above image name.
"""
            )
        )

        group.append(
            "rescale",
            cellprofiler.setting.Binary(
                "Rescale intensities?",
                True,
                doc="""\
This option determines whether image metadata should be used to rescale
the image’s intensities. Some image formats save the maximum possible
intensity value along with the pixel data. For instance, a microscope
might acquire images using a 12-bit A/D converter which outputs
intensity values between zero and 4095, but stores the values in a field
that can take values up to 65535.

Select *{YES}* to rescale the image intensity so that saturated values
are rescaled to 1.0 by dividing all pixels in the image by the maximum
possible intensity value.

Select *{NO}* to ignore the image metadata and rescale the image to 0
– 1.0 by dividing by 255 or 65535, depending on the number of bits used
to store the image.
""".format(**{
                    "YES": cellprofiler.setting.YES,
                    "NO": cellprofiler.setting.NO
                })
            )
        )

        group.can_remove = can_remove
        if can_remove:
            group.append("remover", cellprofiler.setting.RemoveSettingButton(
                    "Remove this channel", "Remove channel", image_settings.channels,
                    group))

    def channel_wants_images(self, channel):
        '''True if the channel produces images, false if it produces objects'''
        return channel.image_object_choice == IO_IMAGES

    def help_settings(self):
        result = [self.file_types,
                  self.match_method,
                  self.order_group_size,
                  self.exclude,
                  self.match_exclude,
                  self.descend_subdirectories,
                  self.subdirectory_filter,
                  self.check_images,
                  self.group_by_metadata,
                  self.metadata_fields]
        image_group = self.images[0]
        result += [
            image_group.common_text,
            image_group.order_position,
            image_group.channels[0].image_object_choice,
            image_group.channels[0].image_name,
            image_group.channels[0].object_name,
            image_group.channels[0].wants_outlines,
            image_group.channels[0].outlines_name,
            image_group.channels[0].channel_number,
            image_group.channels[0].rescale,
            image_group.metadata_choice,
            image_group.file_metadata,
            image_group.path_metadata,
            image_group.wants_movie_frame_grouping,
            image_group.interleaving,
            image_group.channels_per_group
        ]

        result += [self.location]
        return result

    def visible_settings(self):
        varlist = [self.file_types, self.match_method]

        if self.match_method == MS_EXACT_MATCH:
            varlist += [self.exclude]
            if self.exclude.value:
                varlist += [self.match_exclude]
        elif self.match_method == MS_ORDER:
            varlist += [self.order_group_size]
        varlist += [self.descend_subdirectories]
        if self.descend_subdirectories == SUB_SOME:
            varlist += [self.subdirectory_filter]
        if self.has_metadata and not self.match_method == MS_ORDER:
            varlist += [self.check_images]
        if self.has_metadata:
            varlist += [self.group_by_metadata]
        if self.do_group_by_metadata:
            varlist += [self.metadata_fields]
            choices = set()
            for fd in self.images:
                for setting, tag in (
                        (fd.file_metadata, M_FILE_NAME),
                        (fd.path_metadata, M_PATH)):
                    if fd.metadata_choice in (tag, M_BOTH):
                        choices.update(
                                cellprofiler.measurement.find_metadata_tokens(setting.value))
            if (any([cellprofiler.measurement.is_well_column_token(x) for x in choices]) and
                    any([cellprofiler.measurement.is_well_row_token(x) for x in choices]) and not
                    any([x.lower() == cellprofiler.measurement.FTR_WELL.lower() for x in choices])):
                choices.add(cellprofiler.measurement.FTR_WELL)
            if self.file_types == FF_OTHER_MOVIES:
                choices.update([M_Z, M_T, C_SERIES])
            elif self.file_types in (FF_AVI_MOVIES, FF_STK_MOVIES):
                choices.add(M_T)
            choices = list(choices)
            choices.sort()
            self.metadata_fields.choices = choices

        # per image settings
        for i, fd in enumerate(self.images):
            is_multichannel = (self.is_multichannel or fd.wants_movie_frame_grouping)
            varlist += [fd.divider]
            if self.match_method != MS_ORDER:
                varlist += [fd.common_text]
            else:
                varlist += [fd.order_position]
            if not is_multichannel:
                varlist += [fd.channels[0].image_object_choice]
                if self.channel_wants_images(fd.channels[0]):
                    varlist += [fd.channels[0].image_name, fd.channels[0].rescale]
                else:
                    varlist += [fd.channels[0].object_name, fd.channels[0].wants_outlines]
                    if fd.channels[0].wants_outlines.value:
                        varlist += [fd.channels[0].outlines_name]
            varlist += [fd.metadata_choice]
            if self.has_file_metadata(fd):
                varlist += [fd.file_metadata]
            if self.has_path_metadata(fd):
                varlist += [fd.path_metadata]
            max_channels = 9
            if self.file_types in (FF_AVI_MOVIES, FF_STK_MOVIES, FF_OTHER_MOVIES):
                varlist += [fd.wants_movie_frame_grouping]
                if fd.wants_movie_frame_grouping:
                    varlist += [fd.interleaving, fd.channels_per_group]
                    is_multichannel = True
                    max_channels = fd.channels_per_group.value
            if is_multichannel:
                for channel in fd.channels:
                    varlist += [channel.image_object_choice]
                    if self.channel_wants_images(channel):
                        varlist += [channel.image_name]
                    else:
                        varlist += [channel.object_name, channel.wants_outlines]
                        if channel.wants_outlines.value:
                            varlist += [channel.outlines_name]
                    varlist += [channel.channel_number]
                    if self.channel_wants_images(channel):
                        varlist += [channel.rescale]
                    choices = channel.channel_number.choices
                    del choices[:]
                    choices += [str(x + 1) for x in range(max_channels)]
                    if channel.can_remove:
                        varlist += [channel.remover]
                varlist += [fd.add_channel_button]
            if fd.can_remove:
                varlist += [fd.remover]

        varlist += [self.add_image]
        varlist += [self.location]
        return varlist

    def validate_module(self, pipeline):
        '''Validate a module's settings

        LoadImages marks the common_text as invalid if it's blank.
        '''
        if self.match_method == MS_EXACT_MATCH:
            for image_group in self.images:
                if len(image_group.common_text.value) == 0:
                    raise cellprofiler.setting.ValidationError(
                            "The matching text is blank. This would match all images.\n"
                            "Use regular expressions to match with a matching\n"
                            'expression of ".*" if this is the desired behavior.',
                            image_group.common_text)

    def validate_module_warnings(self, pipeline):
        '''Check for potentially dangerous settings'''

        # Check that user has selected fields for grouping if grouping is turned on
        if self.group_by_metadata.value and (len(self.metadata_fields.selections) == 0):
            raise cellprofiler.setting.ValidationError(
                "Group images by metadata is True, but no metadata " 
                "fields have been chosen for grouping.",
                self.metadata_fields
            )

        # Check that user-specified names don't have bad characters
        invalid_chars_pattern = "^[A-Za-z][A-Za-z0-9_]+$"
        warning_text = "The image name has questionable characters. The pipeline can use this name " \
                       "and produce results, but downstream programs that use this data (e.g, MATLAB, MySQL) may error."
        for i, fd in enumerate(self.images):
            is_multichannel = (self.is_multichannel or fd.wants_movie_frame_grouping)
            if not is_multichannel:
                if self.channel_wants_images(fd.channels[0]):
                    if not re.match(invalid_chars_pattern, fd.channels[0].image_name.value):
                        raise cellprofiler.setting.ValidationError(warning_text, fd.channels[0].image_name)
            else:
                for channel in fd.channels:
                    if self.channel_wants_images(channel):
                        if not re.match(invalid_chars_pattern, fd.channels[0].image_name.value):
                            raise cellprofiler.setting.ValidationError(warning_text, channel.image_name)

        # The best practice is to have a single LoadImages or LoadData module.
        from cellprofiler.modules.loaddata import LoadData
        for module in pipeline.modules():
            if id(module) == id(self):
                return
            if isinstance(module, LoadData):
                raise cellprofiler.setting.ValidationError(
                        "Your pipeline has a LoadImages and LoadData module.\n"
                        "The best practice is to have only a single LoadImages\n"
                        "or LoadData module. This LoadImages module will match its\n"
                        "metadata against that of the previous LoadData module\n"
                        "in an attempt to reconcile the two modules' image\n"
                        "set lists and this can result in image sets with\n"
                        "missing images or metadata.", self.add_image)
            if isinstance(module, LoadImages):
                raise cellprofiler.setting.ValidationError(
                        "Your pipeline has two or more LoadImages modules.\n"
                        "The best practice is to have only one LoadImages module.\n"
                        "Consider loading all of your images using a single\n"
                        "LoadImages module. You can add additional images using\n"
                        "the Add button", self.add_image)

    #
    # Slots for storing settings in the array
    #
    SLOT_FILE_TYPE = 0
    SLOT_MATCH_METHOD = 1
    SLOT_ORDER_GROUP_SIZE = 2
    SLOT_MATCH_EXCLUDE = 3
    SLOT_DESCEND_SUBDIRECTORIES = 4
    SLOT_LOCATION = 5
    SLOT_CHECK_IMAGES = 6
    SLOT_GROUP_BY_METADATA = 7
    SLOT_EXCLUDE = 8
    SLOT_GROUP_FIELDS = 9
    SLOT_FIRST_IMAGE_V1 = 8
    SLOT_FIRST_IMAGE_V2 = 9
    SLOT_FIRST_IMAGE_V3 = 10
    SLOT_FIRST_IMAGE_V4 = 11
    SLOT_FIRST_IMAGE_V5 = 10
    SLOT_FIRST_IMAGE_V6 = 11
    SLOT_FIRST_IMAGE_V7 = 11
    SLOT_FIRST_IMAGE_V8 = 11
    SLOT_FIRST_IMAGE_V9 = 11
    SLOT_FIRST_IMAGE_V10 = 11
    SLOT_FIRST_IMAGE = 12
    SLOT_IMAGE_COUNT_V6 = 10
    SLOT_IMAGE_COUNT_V7 = 10
    SLOT_IMAGE_COUNT_V8 = 10
    SLOT_IMAGE_COUNT_V9 = 10
    SLOT_IMAGE_COUNT_V10 = 10
    SLOT_IMAGE_COUNT = 11

    SLOT_OFFSET_COMMON_TEXT = 0
    SLOT_OFFSET_IMAGE_NAME_V5 = 1
    SLOT_OFFSET_ORDER_POSITION_V5 = 2
    SLOT_OFFSET_METADATA_CHOICE_V5 = 3
    SLOT_OFFSET_FILE_METADATA_V5 = 4
    SLOT_OFFSET_PATH_METADATA_V5 = 5
    SLOT_IMAGE_FIELD_COUNT_V1 = 3
    SLOT_IMAGE_FIELD_COUNT_V5 = 6
    SLOT_IMAGE_FIELD_COUNT_V7 = 9
    SLOT_IMAGE_FIELD_COUNT_V8 = 9
    SLOT_IMAGE_FIELD_COUNT_V9 = 9
    SLOT_IMAGE_FIELD_COUNT = 9

    SLOT_OFFSET_ORDER_POSITION = 1
    SLOT_OFFSET_METADATA_CHOICE = 2
    SLOT_OFFSET_FILE_METADATA = 3
    SLOT_OFFSET_PATH_METADATA = 4
    SLOT_OFFSET_CHANNEL_COUNT = 5
    SLOT_OFFSET_CHANNEL_COUNT_V6 = 5
    SLOT_OFFSET_CHANNEL_COUNT_V7 = 5
    SLOT_OFFSET_CHANNEL_COUNT_V8 = 5
    SLOT_OFFSET_CHANNEL_COUNT_V9 = 5
    SLOT_OFFSET_WANTS_MOVIE_FRAME_GROUPING = 6
    SLOT_OFFSET_INTERLEAVING = 7
    SLOT_OFFSET_CHANNELS_PER_GROUP = 8

    SLOT_OFFSET_IO_CHOICE = 0
    SLOT_OFFSET_IMAGE_NAME_V8 = 0
    SLOT_OFFSET_IMAGE_NAME_V9 = 1
    SLOT_OFFSET_IMAGE_NAME = 1
    SLOT_OFFSET_OBJECT_NAME_V9 = 3
    SLOT_OFFSET_OBJECT_NAME = 2
    SLOT_OFFSET_CHANNEL_NUMBER_V8 = 1
    SLOT_OFFSET_CHANNEL_NUMBER_V9 = 3
    SLOT_OFFSET_CHANNEL_NUMBER = 3
    SLOT_OFFSET_RESCALE_V8 = 2
    SLOT_OFFSET_RESCALE_V9 = 4
    SLOT_OFFSET_RESCALE = 4
    SLOT_CHANNEL_FIELD_COUNT = 7
    SLOT_CHANNEL_FIELD_COUNT_V6 = 2
    SLOT_CHANNEL_FIELD_COUNT_V7 = 2
    SLOT_CHANNEL_FIELD_COUNT_V8 = 3
    SLOT_CHANNEL_FIELD_COUNT_V9 = 5

    def settings(self):
        """Return the settings array in a consistent order"""
        setting_values = [
            self.file_types, self.match_method, self.order_group_size,
            self.match_exclude, self.descend_subdirectories, self.location,
            self.check_images, self.group_by_metadata, self.exclude,
            self.metadata_fields, self.subdirectory_filter, self.image_count]
        for image_group in self.images:
            setting_values += [
                image_group.common_text, image_group.order_position,
                image_group.metadata_choice, image_group.file_metadata,
                image_group.path_metadata, image_group.channel_count,
                image_group.wants_movie_frame_grouping, image_group.interleaving,
                image_group.channels_per_group]
            for channel in image_group.channels:
                setting_values += [
                    channel.image_object_choice, channel.image_name,
                    channel.object_name, channel.wants_outlines,
                    channel.outlines_name, channel.channel_number,
                    channel.rescale]
        return setting_values

    def prepare_settings(self, setting_values):
        #
        # Figure out how many images are in the saved settings - make sure
        # the array size matches the incoming #
        #
        image_count = int(setting_values[self.SLOT_IMAGE_COUNT])
        setting_values = setting_values[self.SLOT_FIRST_IMAGE:]
        del self.images[:]
        for i in range(image_count):
            self.add_imagecb(i > 0)
            image_settings = self.images[-1]
            channel_count = int(setting_values[self.SLOT_OFFSET_CHANNEL_COUNT])
            setting_values = setting_values[self.SLOT_IMAGE_FIELD_COUNT:]
            for j in range(channel_count):
                if j > 0:
                    self.add_channel(image_settings)
                setting_values = setting_values[self.SLOT_CHANNEL_FIELD_COUNT:]

    @property
    def is_multichannel(self):
        '''True if the image is one of the multichannel types and needs to be split'''
        #
        # Currently, only Flex are handled this way
        #
        return self.file_types == FF_OTHER_MOVIES

    @property
    def has_metadata(self):
        if self.file_types in (FF_AVI_MOVIES, FF_STK_MOVIES, FF_OTHER_MOVIES):
            return True
        return any([self.has_file_metadata(fd) or self.has_path_metadata(fd)
                    for fd in self.images])

    @property
    def do_group_by_metadata(self):
        '''Return true if we should group by metadata

        The group-by-metadata checkbox won't show unless there are
        metadata groupings to group by - so go by the checkbox
        and the presence of metadata to group by
        '''
        if not self.group_by_metadata:
            return False
        return self.has_metadata

    def get_channel_for_image_name(self, image_name):
        '''Given an image name, return the channel that holds its settings'''
        for image_settings in self.images:
            for channel in image_settings.channels:
                if channel.get_image_name() == image_name:
                    return channel

        return None

    def upgrade_settings(self, setting_values, variable_revision_number, module_name, from_matlab):

        #
        # historic rewrites from CP1.0
        #
        def upgrade_1_to_2(setting_values):
            """Upgrade rev 1 LoadImages to rev 2

            Handle movie formats new to rev 2
            """
            new_values = list(setting_values[:10])
            image_or_movie = setting_values[10]
            if image_or_movie == 'Image':
                new_values.append('individual images')
            elif setting_values[11] == 'avi':
                new_values.append('avi movies')
            elif setting_values[11] == 'stk':
                new_values.append('stk movies')
            else:
                raise ValueError('Unhandled movie type: %s' % (setting_values[11]))
            new_values.extend(setting_values[12:])
            return new_values, 2

        def upgrade_2_to_3(setting_values):
            """Added binary/grayscale question"""
            new_values = list(setting_values)
            new_values.append('grayscale')
            new_values.append('')
            return new_values, 3

        def upgrade_3_to_4(setting_values):
            """Added text exclusion at slot # 10"""
            new_values = list(setting_values)
            new_values.insert(10, cellprofiler.setting.DO_NOT_USE)
            return new_values, 4

        def upgrade_4_to_5(setting_values):
            new_values = list(setting_values)
            new_values.append(cellprofiler.setting.NO)
            return new_values, 5

        def upgrade_5_to_new_1(setting_values):
            """Take the old LoadImages values and put them in the correct slots"""
            loc = setting_values[13]
            custom_path = loc
            if loc == '.':
                dir_choice = cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME
            elif loc == '&':
                dir_choice = cellprofiler.preferences.DEFAULT_OUTPUT_FOLDER_NAME
            elif loc.startswith('.'):
                dir_choice = cellprofiler.preferences.DEFAULT_INPUT_SUBFOLDER_NAME
            elif loc.startswith('&'):
                dir_choice = cellprofiler.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME
                custom_path = '.' + loc[1:]
            else:
                dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME

            new_values = [
                setting_values[11],  # file_types
                setting_values[0],  # match_method
                setting_values[9],  # order_group_size
                setting_values[10],  # match_exclude
                setting_values[12],  # descend_subdirectories
                dir_choice,  # was location
                custom_path,  # was location_other
                setting_values[16]]  # check_images

            for i in range(0, 4):
                text_to_find = setting_values[i * 2 + 1]
                image_name = setting_values[i * 2 + 2]
                if text_to_find == cellprofiler.setting.DO_NOT_USE or image_name == cellprofiler.setting.DO_NOT_USE or \
                        text_to_find == '/' or image_name == '/' or text_to_find == '\\' or image_name == '\\':
                    break
                new_values.extend([text_to_find, image_name, text_to_find])
            return new_values, 1

        #
        # New revisions in CP2.0
        #
        def upgrade_new_1_to_2(setting_values):
            """Add the metadata slots to the images"""
            new_values = list(setting_values[:self.SLOT_FIRST_IMAGE_V1])
            new_values.append(cellprofiler.setting.NO)  # Group by metadata is off
            for i in range((len(setting_values) - self.SLOT_FIRST_IMAGE_V1) / self.SLOT_IMAGE_FIELD_COUNT_V1):
                off = self.SLOT_FIRST_IMAGE_V1 + i * self.SLOT_IMAGE_FIELD_COUNT_V1
                new_values.extend([setting_values[off],
                                   setting_values[off + 1],
                                   setting_values[off + 2],
                                   M_NONE,
                                   cellprofiler.setting.NONE,
                                   cellprofiler.setting.NONE])
            return new_values, 2

        def upgrade_new_2_to_3(setting_values):
            """Add the checkbox for excluding certain files"""
            new_values = list(setting_values[:self.SLOT_FIRST_IMAGE_V2])
            if setting_values[self.SLOT_MATCH_EXCLUDE] == cellprofiler.setting.DO_NOT_USE:
                new_values += [cellprofiler.setting.NO]
            else:
                new_values += [cellprofiler.setting.YES]
            new_values += setting_values[self.SLOT_FIRST_IMAGE_V2:]
            return new_values, 3

        def upgrade_new_3_to_4(setting_values):
            """Add the metadata_fields setting"""
            new_values = list(setting_values[:self.SLOT_FIRST_IMAGE_V3])
            new_values.append('')
            new_values += setting_values[self.SLOT_FIRST_IMAGE_V3:]
            return new_values, 4

        def upgrade_new_4_to_5(setting_values):
            """Combine the location and custom location values"""
            setting_values = cellprofiler.setting.standardize_default_folder_names(
                    setting_values, self.SLOT_LOCATION)
            custom_location = setting_values[self.SLOT_LOCATION + 1]
            location = setting_values[self.SLOT_LOCATION]
            if location == cellprofiler.preferences.ABSOLUTE_FOLDER_NAME:
                if custom_location.startswith('.'):
                    location = cellprofiler.setting.DEFAULT_INPUT_SUBFOLDER_NAME
                elif custom_location.startswith('&'):
                    location = cellprofiler.setting.DEFAULT_OUTPUT_SUBFOLDER_NAME
                    custom_location = "." + custom_location[1:]
            location = cellprofiler.setting.DirectoryPath.static_join_string(
                    location, custom_location)
            setting_values = (setting_values[:self.SLOT_LOCATION] +
                              [location] +
                              setting_values[self.SLOT_LOCATION + 2:])
            return setting_values, 5

        def upgrade_new_5_to_6(setting_values):
            '''Added separate channels for flex images'''
            new_values = list(setting_values[:self.SLOT_FIRST_IMAGE_V5])
            setting_values = setting_values[self.SLOT_FIRST_IMAGE_V5:]
            image_count = (len(setting_values) / self.SLOT_IMAGE_FIELD_COUNT_V5)
            #
            # Add the image count to the settings
            #
            new_values += [str(image_count)]
            for i in range(image_count):
                new_values += setting_values[:self.SLOT_OFFSET_IMAGE_NAME_V5]
                new_values += setting_values[(self.SLOT_OFFSET_IMAGE_NAME_V5 + 1):self.SLOT_IMAGE_FIELD_COUNT_V5]
                #
                # Add a channel count of 1, the image name and a channel
                # number of 1
                #
                new_values += [
                    "1", setting_values[self.SLOT_OFFSET_IMAGE_NAME_V5], "1"]
                setting_values = setting_values[self.SLOT_IMAGE_FIELD_COUNT_V5:]
            return new_values, 6

        def upgrade_new_6_to_7(setting_values):
            '''Added movie frame grouping'''
            new_values = list(setting_values[:self.SLOT_FIRST_IMAGE_V6])
            image_count = int(setting_values[self.SLOT_IMAGE_COUNT_V6])
            setting_values = setting_values[self.SLOT_FIRST_IMAGE_V6:]
            for i in range(image_count):
                new_values += setting_values[:self.SLOT_IMAGE_FIELD_COUNT_V5]
                new_values += [cellprofiler.setting.NO, I_INTERLEAVED, "2"]
                channel_count = int(setting_values[self.SLOT_OFFSET_CHANNEL_COUNT_V6])
                setting_values = setting_values[self.SLOT_IMAGE_FIELD_COUNT_V5:]
                channel_field_count = self.SLOT_CHANNEL_FIELD_COUNT_V6 * channel_count
                new_values += setting_values[:channel_field_count]
                setting_values = setting_values[channel_field_count:]
            return new_values, 7

        def upgrade_new_7_to_8(setting_values):
            '''Added rescale control'''
            new_values = list(setting_values[:self.SLOT_FIRST_IMAGE_V7])
            image_count = int(setting_values[self.SLOT_IMAGE_COUNT_V7])
            setting_values = setting_values[self.SLOT_FIRST_IMAGE_V7:]
            for i in range(image_count):
                new_values += setting_values[:self.SLOT_IMAGE_FIELD_COUNT_V7]
                channel_count = int(setting_values[self.SLOT_OFFSET_CHANNEL_COUNT_V7])
                setting_values = setting_values[self.SLOT_IMAGE_FIELD_COUNT_V7:]
                for j in range(channel_count):
                    new_values += setting_values[:self.SLOT_CHANNEL_FIELD_COUNT_V7] + [cellprofiler.setting.YES]
                    setting_values = setting_values[self.SLOT_CHANNEL_FIELD_COUNT_V7:]
            return new_values, 8

        def upgrade_new_8_to_9(setting_values):
            '''Added object loading'''
            new_values = list(setting_values[:self.SLOT_FIRST_IMAGE_V8])
            image_count = int(setting_values[self.SLOT_IMAGE_COUNT_V8])
            setting_values = setting_values[self.SLOT_FIRST_IMAGE_V8:]
            for i in range(image_count):
                new_values += setting_values[:self.SLOT_IMAGE_FIELD_COUNT_V8]
                channel_count = int(setting_values[self.SLOT_OFFSET_CHANNEL_COUNT_V8])
                setting_values = setting_values[self.SLOT_IMAGE_FIELD_COUNT_V8:]
                for j in range(channel_count):
                    new_values += [
                        IO_IMAGES,
                        setting_values[self.SLOT_OFFSET_IMAGE_NAME_V8],
                        "Nuclei"]
                    new_values += setting_values[(self.SLOT_OFFSET_IMAGE_NAME_V8 + 1):self.SLOT_CHANNEL_FIELD_COUNT_V8]
                    setting_values = setting_values[self.SLOT_CHANNEL_FIELD_COUNT_V8:]
            return new_values, 9

        def upgrade_new_9_to_10(setting_values):
            '''Added outlines to object loading'''
            new_values = list(setting_values[:self.SLOT_FIRST_IMAGE_V9])
            image_count = int(setting_values[self.SLOT_IMAGE_COUNT_V9])
            setting_values = setting_values[self.SLOT_FIRST_IMAGE_V9:]
            for i in range(image_count):
                new_values += setting_values[:self.SLOT_IMAGE_FIELD_COUNT_V9]
                channel_count = int(setting_values[self.SLOT_OFFSET_CHANNEL_COUNT_V9])
                setting_values = setting_values[self.SLOT_IMAGE_FIELD_COUNT_V9:]
                for j in range(channel_count):
                    new_values += setting_values[:self.SLOT_OFFSET_OBJECT_NAME_V9] + \
                                  [cellprofiler.setting.NO, "NucleiOutlines"] + \
                                  setting_values[self.SLOT_OFFSET_OBJECT_NAME_V9:self.SLOT_CHANNEL_FIELD_COUNT_V9]
                    setting_values = setting_values[self.SLOT_CHANNEL_FIELD_COUNT_V9:]
            return new_values, 10

        def upgrade_new_10_to_11(setting_values):
            '''Added subdirectory filter'''
            new_values = (setting_values[:self.SLOT_IMAGE_COUNT_V10] +
                          [""] + setting_values[self.SLOT_IMAGE_COUNT_V10:])
            if new_values[self.SLOT_DESCEND_SUBDIRECTORIES] == cellprofiler.setting.YES:
                new_values[self.SLOT_DESCEND_SUBDIRECTORIES] = SUB_ALL
            else:
                new_values[self.SLOT_DESCEND_SUBDIRECTORIES] = SUB_NONE
            return new_values, 11

        if from_matlab:
            if variable_revision_number == 1:
                setting_values, variable_revision_number = upgrade_1_to_2(setting_values)
            if variable_revision_number == 2:
                setting_values, variable_revision_number = upgrade_2_to_3(setting_values)
            if variable_revision_number == 3:
                setting_values, variable_revision_number = upgrade_3_to_4(setting_values)
            if variable_revision_number == 4:
                setting_values, variable_revision_number = upgrade_4_to_5(setting_values)
            if variable_revision_number == 5:
                setting_values, variable_revision_number = upgrade_5_to_new_1(setting_values)
            from_matlab = False

        assert not from_matlab
        if variable_revision_number == 1:
            setting_values, variable_revision_number = upgrade_new_1_to_2(setting_values)
        if variable_revision_number == 2:
            setting_values, variable_revision_number = upgrade_new_2_to_3(setting_values)
        if variable_revision_number == 3:
            setting_values, variable_revision_number = upgrade_new_3_to_4(setting_values)
        if variable_revision_number == 4:
            setting_values, variable_revision_number = upgrade_new_4_to_5(setting_values)
        if variable_revision_number == 5:
            setting_values, variable_revision_number = upgrade_new_5_to_6(setting_values)
        if variable_revision_number == 6:
            setting_values, variable_revision_number = upgrade_new_6_to_7(setting_values)
        if variable_revision_number == 7:
            setting_values, variable_revision_number = upgrade_new_7_to_8(setting_values)
        if variable_revision_number == 8:
            setting_values, variable_revision_number = upgrade_new_8_to_9(setting_values)
        if variable_revision_number == 9:
            setting_values, variable_revision_number = upgrade_new_9_to_10(setting_values)
        if variable_revision_number == 10:
            setting_values, variable_revision_number = upgrade_new_10_to_11(setting_values)

        # Standardize input/output directory name references
        setting_values[self.SLOT_LOCATION] = \
            cellprofiler.setting.DirectoryPath.upgrade_setting(setting_values[self.SLOT_LOCATION])
        # Upgrade the file type slot
        if setting_values[self.SLOT_FILE_TYPE] in FF_OTHER_MOVIES_OLD:
            setting_values[self.SLOT_FILE_TYPE] = FF_OTHER_MOVIES
        if setting_values[self.SLOT_FILE_TYPE] in FF_AVI_MOVIES_OLD:
            setting_values[self.SLOT_FILE_TYPE] = FF_AVI_MOVIES

        assert variable_revision_number == self.variable_revision_number, "Cannot read version %d of %s" % (
            variable_revision_number, self.module_name)

        return setting_values, variable_revision_number, from_matlab

    def is_load_module(self):
        '''LoadImages creates image sets so it is a load module'''
        return True

    def prepare_run(self, workspace):
        """Set up all of the image providers inside the image_set_list
        """
        if workspace.pipeline.in_batch_mode():
            # Don't set up if we're going to retrieve the image set list
            # from batch mode
            return True
        if self.file_types == FF_OTHER_MOVIES:
            return self.prepare_run_of_flex(workspace)
        elif self.load_movies():
            return self.prepare_run_of_movies(workspace)
        else:
            return self.prepare_run_of_images(workspace)

    def prepare_run_of_images(self, workspace):
        """Set up image providers for image files"""
        pipeline = workspace.pipeline
        # OK to use workspace.frame, since we're in prepare_run
        frame = workspace.frame
        files = self.collect_files(workspace)
        if len(files) == 0:
            return False

        if self.do_group_by_metadata and len(self.get_metadata_tags()):
            result = self.organize_by_metadata(workspace, files)
        else:
            result = self.organize_by_order(workspace, files)
        if not result:
            return result
        return True

    def organize_by_order(self, workspace, files):
        """Organize each kind of file by their lexical order

        """
        pipeline = workspace.pipeline
        # OK to use workspace.frame, since we're in prepare_run
        frame = workspace.frame
        m = workspace.measurements
        assert isinstance(m, cellprofiler.measurement.Measurements)
        image_names = self.image_name_vars()
        list_of_lists = [[] for x in image_names]
        for pathname, image_index in files:
            list_of_lists[image_index].append(pathname)

        image_set_count = len(list_of_lists[0])

        for x, name in zip(list_of_lists[1:], image_names[1:]):
            if len(x) != image_set_count:
                message = "Image %s has %d files, but image %s has %d files" % \
                          (image_names[0], image_set_count, name, len(x))
                pipeline.report_prepare_run_error(self, message)
                if frame is not None:
                    images = [(tuple(),
                               [os.path.split(list_of_lists[i][j])
                                if len(list_of_lists[i]) > j else None
                                for i in range(len(list_of_lists))])
                              for j in range(image_set_count)]
                    self.report_errors([], images, frame)
                return False
        list_of_lists = numpy.array(list_of_lists)
        root = self.image_directory()
        for i in range(image_set_count):
            for j, image_name in enumerate(image_names):
                image_number = i + 1
                full_path = os.path.join(root, list_of_lists[j, i])
                self.write_measurements(
                        m, image_number, self.images[j], full_path)

        return True

    def organize_by_metadata(self, workspace, files):
        """Organize each kind of file by metadata

        """
        pipeline = workspace.pipeline
        # OK to use workspace.frame, since we're in prepare_run
        frame = workspace.frame
        #
        # Distribute files according to metadata tags. Each image_name
        # potentially has a subset of the total list of tags. For images
        # without the complete list of tags, we give the image a wildcard
        # for the metadata item that matches everything.
        #
        tags = self.get_metadata_tags()
        #
        # files_by_image_name holds one list of files for each image name index
        #
        files_by_image_name = [[] for i in range(len(self.images))]
        for file, i in files:
            files_by_image_name[i].append(os.path.split(file))
        #
        # Create a monster dictionary tree for each image describing the
        # metadata for each tag. Give a file a metadata value of None
        # if it has no value for a given metadata tag
        #
        d = [{} for i in range(len(self.images))]
        conflicts = []
        for i in range(len(self.images)):
            fd = self.images[i]
            for path, filename in files_by_image_name[i]:
                metadata = self.get_filename_metadata(fd, filename, path)
                parent = d[i]
                for tag in tags[:-1]:
                    value = metadata.get(tag)
                    if value in parent:
                        child = parent[value]
                    else:
                        child = {}
                        parent[value] = child
                    parent = child
                    last_value = value
                tag = tags[-1]
                value = metadata.get(tag)
                if value in parent:
                    # There's already a match to this metadata
                    conflict = [fd, parent[value], (path, filename)]
                    conflicts.append(conflict)
                    if not self.check_images:
                        #
                        # Disambiguate conflicts by picking the most recently
                        # modified file.
                        #
                        old_path, old_filename = parent[value]
                        old_stat = os.stat(os.path.join(self.image_directory(),
                                                        old_path, old_filename))
                        old_time = old_stat.st_mtime
                        new_stat = os.stat(os.path.join(self.image_directory(),
                                                        path, filename))
                        new_time = new_stat.st_mtime
                        if new_time > old_time:
                            parent[value] = (path, filename)
                else:
                    parent[value] = (path, filename)
        image_sets = self.get_image_sets(d)
        missing_images = [image_set for image_set in image_sets
                          if None in image_set[1]]
        image_sets = [image_set for image_set in image_sets
                      if None not in image_set[1]]
        #
        # Handle errors, raising an exception if the user wants to check images
        #
        if len(conflicts) or len(missing_images):
            if frame:
                self.report_errors(conflicts, missing_images, frame)
                if self.check_images:
                    return False
            if self.check_images.value:
                message = ""
                if len(conflicts):
                    message += "Conflicts found:\n"
                    for conflict in conflicts:
                        metadata = self.get_filename_metadata(conflict[0],
                                                              conflict[1][1],
                                                              conflict[1][0])
                        message += (
                            "Metadata: %s, First path: %s, First filename: %s, Second path: %s, Second filename: %s\n" %
                            (str(metadata), conflict[1][0], conflict[1][1],
                             conflict[2][0], conflict[2][1]))
                if len(missing_images):
                    message += "Missing images:\n"
                    for mi in missing_images:
                        for i in range(len(self.images)):
                            fd = self.images[i]
                            if mi[1][i] is None:
                                message += ("%s: missing " %
                                            fd.channels[0].image_name.value)
                            else:
                                message += ("%s: path=%s, file=%s" %
                                            (fd.channels[0].image_name.value,
                                             mi[1][i][0], mi[1][i][1]))
                pipeline.report_prepare_run_error(self, message)
                return False

        root = self.image_directory()
        features = {}
        #
        # Two cases - the measurements already have image sets or the
        #             measurements have no image sets.
        # 1: measurements must have an image measurement corresponding to each
        #    tag. Create a dictionary of tag values to a list of image numbers
        #    with those tag values. Populate our images to that list.
        # 2: write fresh image set records to measurements.
        #
        measurements = workspace.measurements
        assert isinstance(measurements, cellprofiler.measurement.Measurements)
        if measurements.image_set_count > 0:
            match_metadata = True
            md_dict = self.get_image_numbers_by_tags(workspace, tags)
            if md_dict is None:
                return False
            n_image_sets = measurements.image_set_count
        else:
            match_metadata = False
            n_image_sets = len(image_sets)
        for idx, image_set in enumerate(image_sets):
            keys = {}
            for tag, value in zip(tags, image_set[0]):
                keys[tag] = value
            if match_metadata:
                image_numbers = md_dict[keys]
            else:
                image_numbers = [idx + 1]
            for i in range(len(self.images)):
                path = os.path.join(image_set[1][i][0], image_set[1][i][1])
                full_path = os.path.join(root, path)
                for image_number in image_numbers:
                    d = self.write_measurements(
                            None,
                            image_number,
                            self.images[i], full_path)
                    for k, v in d.iteritems():
                        if k not in features:
                            features[k] = [None] * n_image_sets
                        features[k][image_number - 1] = v
        for k, v in features.iteritems():
            workspace.measurements.add_all_measurements(cellprofiler.measurement.IMAGE, k, v)

        return True

    def get_image_sets(self, d):
        """Get image sets from a dictionary tree

        d - one dictionary tree per image.
        returns a list of tuples. The first element contains a tuple of
        metadata values and the second element contains a list of
        (path,filename) tuples for each image (or are None for errors where
        there is no metadata match).
        """

        if not any([isinstance(dd, dict) for dd in d]):
            # This is a leaf if no elements are dictionaries
            return [(tuple(), d)]
        #
        # Get each represented value for this slot
        #
        values = set()
        for dd in d:
            if dd is not None:
                values.update([x for x in dd.keys() if x is not None])
        result = []
        for value in sorted(values):
            subgroup = tuple((dd and None in dd and dd[None]) or  # wildcard metadata
                             (dd and dd.get(value)) or  # fetch subvalue or None if missing
                             None  # metadata is missing
                             for dd in d)
            subsets = self.get_image_sets(subgroup)
            for subset in subsets:
                # prepend the current key to the tuple of keys and
                # duplicate the filename tuple
                subset = ((value,) + subset[0], subset[1])
                result.append(subset)
        return result

    def get_image_numbers_by_tags(self, workspace, tags):
        '''Make a dictionary of tag values to image numbers

        Look up the metadata values for each of the tags in "tags" and
        create a dictionary of tag values to the list of image numbers
        that have those tag values. This dictionary can then be used to
        populate all image sets with a particular tag value.

        workspace.measurements - the measurements populated with metadata values
                                 for each tag

        tags - the metadata tags to be used for matching
        '''
        measurements = workspace.measurements
        metadata_features = [
            x for x in measurements.get_feature_names(cellprofiler.measurement.IMAGE)
            if x.startswith(cellprofiler.measurement.C_METADATA)]
        for tag in tags:
            if "_".join((cellprofiler.measurement.C_METADATA, tag)) not in metadata_features:
                message = (
                    "LoadImages needs the prior Load modules to define "
                    'the metadata tag, "%s", in order to match metadata' %
                    tag)
                workspace.pipeline.report_prepare_run_error(self, message)
                return None
        md_dict = {}
        for i in measurements.get_image_numbers():
            keys = tuple([measurements.get_measurement(
                    cellprofiler.measurement.IMAGE, "_".join((cellprofiler.measurement.C_METADATA, tag)), i)
                          for tag in tags])
            if keys not in md_dict:
                md_dict.append(i)
            else:
                md_dict[keys] = [i]
        return md_dict

    def report_errors(self, conflicts, missing_images, frame):
        """Create an error report window

        conflicts: 3-tuple of file dictionary, first path/file and second path/file
                   for two images with the same metadata
        missing_images: two-tuple of metadata keys and the file tuples
                        where at least one of the file tuples is None indicating
                        a missing image
        frame: the parent for the error report
        """
        import wx.html
        my_frame = wx.Frame(frame, title="Load images: Error report",
                            size=(600, 800),
                            pos=(frame.Position[0],
                                 frame.Position[1] + frame.Size[1]))
        my_frame.Icon = frame.Icon
        panel = wx.Panel(my_frame)
        uber_sizer = wx.BoxSizer()
        my_frame.SetSizer(uber_sizer)
        uber_sizer.Add(panel, 1, wx.EXPAND)
        sizer = wx.BoxSizer(wx.VERTICAL)
        panel.SetSizer(sizer)
        font = wx.Font(16, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL,
                       wx.FONTWEIGHT_BOLD)
        tags = self.get_metadata_tags()
        tag_ct = len(tags)
        tables = []
        if len(conflicts):
            title = wx.StaticText(panel,
                                  label="Conflicts (two files with same metadata)")
            title.Font = font
            sizer.Add(title, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 3)
            table = wx.ListCtrl(panel, style=wx.LC_REPORT)
            tables.append(table)
            sizer.Add(table, 1, wx.EXPAND | wx.ALL, 3)
            for tag, index in zip(tags, range(tag_ct)):
                table.InsertColumn(index, tag)
            table.InsertColumn(index + 1, "First path")
            table.InsertColumn(index + 2, "First file name")
            table.InsertColumn(index + 3, "Second path")
            table.InsertColumn(index + 4, "Second file name")
            for conflict in conflicts:
                metadata = self.get_filename_metadata(conflict[0],
                                                      conflict[1][1],
                                                      conflict[1][0])
                row = [metadata.get(tag) or "-" for tag in tags]
                row.extend([conflict[1][0], conflict[1][1],
                            conflict[2][0], conflict[2][1]])
                table.Append(row)

        if len(missing_images):
            title = wx.StaticText(panel,
                                  label="Missing images")
            title.Font = font
            sizer.Add(title, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL | wx.ALL, 3)
            table = wx.ListCtrl(panel, style=wx.LC_REPORT)
            tables.append(table)
            sizer.Add(table, 1, wx.EXPAND | wx.ALL, 3)
            if self.do_group_by_metadata:
                for tag, index in zip(tags, range(tag_ct)):
                    table.InsertColumn(index, tag)
            for fd, index in zip(self.images, range(len(self.images))):
                table.InsertColumn(index * 2 + tag_ct,
                                   "%s path" % fd.channels[0].image_name.value)
                table.InsertColumn(index * 2 + 1 + tag_ct,
                                   "%s filename" % fd.channels[0].image_name.value)
            for metadata, files_and_paths in missing_images:
                row = list(metadata)
                for file_and_path in files_and_paths:
                    if file_and_path is None:
                        row.extend(["missing", "missing"])
                    else:
                        row.extend(file_and_path)
                table.Append(row)
        best_total_width = 0
        for table in tables:
            total_width = 0
            dc = wx.ClientDC(table)
            dc.Font = table.Font
            try:
                for col in range(table.ColumnCount):
                    text = table.GetColumn(col).Text
                    width = dc.GetTextExtent(text)[0]
                    for row in range(table.ItemCount):
                        text = table.GetItem(row, col).Text
                        width = max(width, dc.GetTextExtent(text)[0])
                    width += 16
                    table.SetColumnWidth(col, width)
                    total_width += width + 4
            finally:
                dc.Destroy()
            best_total_width = max(best_total_width, total_width)
            table.SetMinSize((best_total_width, table.GetMinHeight()))
        my_frame.Fit()
        my_frame.Show()

    def prepare_run_of_flex(self, workspace):
        '''Set up image providers for flex files'''
        import bioformats.omexml
        from bioformats.formatreader import get_omexml_metadata

        pipeline = workspace.pipeline
        # OK to use workspace.frame, since we're in prepare_run
        frame = workspace.frame
        m = workspace.measurements
        assert isinstance(m, cellprofiler.measurement.Measurements)
        if m.image_set_count > 0 and self.do_group_by_metadata:
            match_metadata = True
            tags = list(self.get_metadata_tags()) + [M_Z, M_T, C_SERIES]
            md_dict = self.get_image_numbers_by_tags(workspace, tags)
            if md_dict is None:
                return False
        else:
            match_metadata = False
        files = self.collect_files(workspace)
        if len(files) == 0:
            return False
        #
        # Organize the files into one column per image index.
        #
        index_count = numpy.bincount([x[1] for x in files])
        if numpy.any(index_count != index_count[0]):
            bad = numpy.argwhere(index_count != index_count[0]).flatten()
            raise RuntimeError("Image %s has %d files, but image %s has %d files" %
                               (self.images[0].channels[0].image_name.value,
                                index_count[0],
                                self.images[bad].channels[0].image_name.value,
                                index_count[bad]))
        file_list = [[] for _ in range(len(index_count))]
        for file_pathname, image_index in files:
            file_list[image_index].append(file_pathname)
        #
        # Rearrange to a row-oriented list
        #
        file_list = [[column[i] for column in file_list]
                     for i in range(len(file_list[0]))]
        root = self.image_directory()

        image_set_count = 0
        for image_set_files in file_list:
            starting_image_index = image_set_count
            d = []
            for image_settings, file_pathname in \
                    zip(self.images, image_set_files):
                pathname = os.path.join(self.image_directory(), file_pathname)
                path, filename = os.path.split(pathname)
                url = pathname2url(os.path.abspath(pathname))
                metadata = self.get_filename_metadata(image_settings, filename,
                                                      path)
                image_set_count = starting_image_index
                xml = get_omexml_metadata(pathname)
                omemetadata = bioformats.omexml.OMEXML(xml)
                image_count = omemetadata.image_count
                if len(d) == 0:
                    d = [{} for _ in range(image_count)]
                elif len(d) != image_count:
                    message = (
                        "File %s has %d series, but file %s has %d series." %
                        (image_set_files[0], len(d),
                         file_pathname, image_count))
                    pipeline.report_prepare_run_error(self, message)
                    return False
                for i in range(image_count):
                    pixels = omemetadata.image(i).Pixels
                    channel_count = pixels.SizeC
                    stack_count = pixels.SizeZ
                    if "Z" not in d[i]:
                        d[i]["Z"] = stack_count
                    elif stack_count != d[i]["Z"]:
                        message = (
                            ("File %s, series %d has %d Z stacks, "
                             "but file %s has %d") %
                            (image_set_files[0], i,
                             d[i]["Z"], file_pathname, stack_count))
                        pipeline.report_prepare_run_error(self, message)
                        return False
                    timepoint_count = pixels.SizeT
                    if "T" not in d[i]:
                        d[i]["T"] = timepoint_count
                    elif timepoint_count != d[i]["T"]:
                        message = (
                            ("File %s, series %d has %d "
                             "timepoints, but file %s has %d") %
                            (image_set_files[0], i,
                             d[i]["T"], file_pathname, timepoint_count))
                        pipeline.report_prepare_run_error(self, message)
                        return False
                    if image_settings.wants_movie_frame_grouping:
                        #
                        # For movie frame grouping, assume that all of
                        # the images are to be processed in a consecutive
                        # series taking T as the outside, then Z, then C
                        # and divvied up among the channels.
                        #
                        # Metadata Z = 1
                        # Metadata T is the cycle #
                        #
                        # Really, whoever set up the microscope should
                        # have done it right so that the file saved
                        # the channels into the TIF correctly.
                        #
                        nframes = timepoint_count * stack_count * channel_count
                        nchannels = image_settings.channels_per_group.value
                        if nframes % nchannels != 0:
                            logger.warning(
                                    ("Warning: the movie, %s, has %d frames divided into "
                                     "%d channels per group.\n"
                                     "%d frames will be discarded.\n") %
                                    (pathname, nframes, nchannels, nframes % nchannels))
                            nframes -= nframes % nchannels
                        nsets = int(nframes / nchannels)
                        for idx in range(nsets):
                            frame_metadata = metadata.copy()
                            frame_metadata[M_Z] = 0  # so sorry, real Z obliterated
                            frame_metadata[M_T] = idx
                            frame_metadata[C_SERIES] = i
                            image_numbers = [image_set_count + 1]
                            if match_metadata:
                                key = dict([(k, str(v))
                                            for k, v in frame_metadata.items()])
                                if key not in md_dict:
                                    message = (
                                        "Could not find a matching image set for " %
                                        ", ".join(["%s=%s%" % kv for kv in frame_metadata.items()]))
                                    pipeline.report_prepare_run_error(self, message)
                                    return False
                                image_numbers = md_dict[key]
                            for image_number in image_numbers:
                                for channel_settings in image_settings.channels:
                                    image_name = channel_settings.get_image_name()
                                    channel = int(channel_settings.channel_number.value) - 1
                                    if image_settings.interleaving == I_INTERLEAVED:
                                        cidx = idx * nchannels + channel
                                    else:
                                        cidx = channel * nsets + idx
                                    c = cidx % channel_count
                                    z = int(cidx / channel_count) % stack_count
                                    t = int(cidx / channel_count / stack_count) % timepoint_count
                                    m.add_measurement(
                                            cellprofiler.measurement.IMAGE,
                                            "_".join((cellprofiler.measurement.C_FILE_NAME, image_name)),
                                            filename,
                                            image_set_number=image_number)
                                    m.add_measurement(
                                            cellprofiler.measurement.IMAGE,
                                            "_".join((cellprofiler.measurement.C_PATH_NAME, image_name)),
                                            path,
                                            image_set_number=image_number)
                                    m.add_measurement(
                                            cellprofiler.measurement.IMAGE,
                                            "_".join((cellprofiler.measurement.C_URL, image_name)),
                                            url,
                                            image_set_number=image_number)
                                    m.add_measurement(
                                            cellprofiler.measurement.IMAGE,
                                            "_".join((C_SERIES, image_name)), i,
                                            image_set_number=image_number)
                                    m.add_measurement(
                                            cellprofiler.measurement.IMAGE,
                                            "_".join((C_FRAME, image_name)), cidx,
                                            image_set_number=image_number)
                                for k in frame_metadata.keys():
                                    m.add_measurement(
                                            cellprofiler.measurement.IMAGE,
                                            "_".join((cellprofiler.measurement.C_METADATA, k)),
                                            frame_metadata[k],
                                            image_set_number=image_number)
                                image_set_count += 1
                    else:
                        distance = 1
                        for dimension in pixels.DimensionOrder[2:]:
                            if dimension == "C":
                                stride_c = distance
                                distance *= pixels.SizeC
                            elif dimension == "Z":
                                stride_z = distance
                                distance *= pixels.SizeZ
                            elif dimension == "T":
                                stride_t = distance
                                distance *= pixels.SizeT
                        for z in range(pixels.SizeZ):
                            for t in range(pixels.SizeT):
                                frame_metadata = metadata.copy()
                                frame_metadata[M_Z] = z
                                frame_metadata[M_T] = t
                                frame_metadata[C_SERIES] = i
                                image_numbers = [image_set_count + 1]
                                if match_metadata:
                                    key = dict([(k, str(v))
                                                for k, v in frame_metadata.items()])
                                    if key not in md_dict:
                                        message = (
                                            "Could not find a matching image set for " %
                                            ", ".join(["%s=%s%" % kv for kv in frame_metadata.items()]))
                                        pipeline.report_prepare_run_error(
                                                self, message)
                                        return False
                                    image_numbers = md_dict[key]
                                for image_number in image_numbers:
                                    for channel_settings in image_settings.channels:
                                        c = int(channel_settings.channel_number.value) - 1
                                        image_name = channel_settings.get_image_name()
                                        if c >= channel_count:
                                            message = \
                                                ("The flex file, ""%s"", series # %d, has only %d channels. "
                                                 "%s is assigned to channel % d") % (file_pathname, i, channel_count,
                                                                                     image_name, c + 1)
                                            pipeline.report_prepare_run_error(
                                                    self, message)
                                            return False
                                        index = c * stride_c + t * stride_t + z * stride_z
                                        m.add_measurement(
                                                cellprofiler.measurement.IMAGE,
                                                "_".join((cellprofiler.measurement.C_FILE_NAME, image_name)),
                                                filename,
                                                image_set_number=image_number)
                                        m.add_measurement(
                                                cellprofiler.measurement.IMAGE,
                                                "_".join((cellprofiler.measurement.C_PATH_NAME, image_name)),
                                                path,
                                                image_set_number=image_number)
                                        m.add_measurement(
                                                cellprofiler.measurement.IMAGE,
                                                "_".join((cellprofiler.measurement.C_URL, image_name)),
                                                url,
                                                image_set_number=image_number)
                                        m.add_measurement(
                                                cellprofiler.measurement.IMAGE,
                                                "_".join((C_SERIES, image_name)), i,
                                                image_set_number=image_number)
                                        m.add_measurement(
                                                cellprofiler.measurement.IMAGE,
                                                "_".join((C_FRAME, image_name)),
                                                index,
                                                image_set_number=image_number)
                                    for k in frame_metadata.keys():
                                        m.add_measurement(
                                                cellprofiler.measurement.IMAGE,
                                                "_".join((cellprofiler.measurement.C_METADATA, k)),
                                                frame_metadata[k],
                                                image_set_number=image_number)
                                    image_set_count += 1

        return True

    def prepare_run_of_movies(self, workspace):
        """Set up image providers for movie files"""
        pipeline = workspace.pipeline
        # OK to use workspace.frame, since we're in prepare_run
        frame = workspace.frame
        m = workspace.measurements
        files = self.collect_files(workspace)
        if len(files) == 0:
            return False
        root = self.image_directory()
        image_names = self.image_name_vars()
        #
        # The list of lists has one list per image type. Each per-image type
        # list is composed of tuples of pathname and frame #
        #
        list_of_lists = [[] for x in image_names]
        image_index = 0
        for pathname, image_group_index in files:
            pathname = os.path.join(self.image_directory(), pathname)
            frame_count = self.get_frame_count(pathname)
            if frame_count == 0:
                print("Warning - no frame count detected")
                frame_count = 256
            #
            # 3 choices here:
            #
            # No grouping by movie frames: one channel
            # Interleaved grouping
            # Sequential grouping
            #
            image = self.images[image_group_index]
            if image.wants_movie_frame_grouping:
                group_size = image.channels_per_group.value
                remainder = frame_count % group_size
                if remainder > 0:
                    logger.warning(
                            ("Warning: the movie, %s, has %d frames divided into "
                             "%d channels per group.\n"
                             "%d frames will be discarded.\n") %
                            (pathname, frame_count, group_size, remainder))
                group_count = int(frame_count / group_size)
                for group_number in range(group_count):
                    for i, channel in enumerate(image.channels):
                        channel_idx = int(channel.channel_number.value) - 1
                        if image.interleaving == I_INTERLEAVED:
                            frame_number = \
                                group_number * group_size + channel_idx
                        else:
                            frame_number = \
                                group_count * channel_idx + group_number
                        list_of_lists[image_index + i] += [
                            (pathname, frame_number, group_number,
                             image_group_index)]
                image_index += len(image.channels)
            else:
                for i in range(frame_count):
                    list_of_lists[image_group_index].append(
                            (pathname, i, i, image_group_index))
        image_set_count = len(list_of_lists[0])
        for x, name in zip(list_of_lists[1:], image_names):
            if len(x) != image_set_count:
                message = (
                    "Image %s has %d frames, but image %s has %d frames" %
                    (image_names[0], image_set_count, name, len(x)))
                pipeline.report_prepare_run_error(self, message)
                return False
        list_of_lists = numpy.array(list_of_lists, dtype=object)
        for i in range(0, image_set_count):
            for name, (file, frame, t, image_group_index) \
                    in zip(image_names, list_of_lists[:, i]):
                image_group = self.images[image_group_index]
                self.write_measurements(m, i + 1, image_group, file,
                                        frame=frame,
                                        channel_name=name)
                m.add_measurement(cellprofiler.measurement.IMAGE,
                                  "_".join((cellprofiler.measurement.C_METADATA, M_T)), t,
                                  image_set_number=i + 1)

        return True

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        '''Prepare to create a batch file

        This function is called when CellProfiler is about to create a
        file for batch processing. The function adjusts any paths in settings
        or paths in measurements using the function "fn_alter_path" to
        map between file systems.

        pipeline - the pipeline to be saved
        image_set_list - the image set list to be saved
        fn_alter_path - this is a function that takes a pathname on the local
                        host and returns a pathname on the remote host. It
                        handles issues such as replacing backslashes and
                        mapping mountpoints. It should be called for every
                        pathname stored in the settings or legacy fields.
        '''
        image_set_list = workspace.image_set_list
        m = workspace.measurements
        for image_group in self.images:
            for channel in image_group.channels:
                m.alter_path_for_create_batch(
                        channel.get_image_name(),
                        self.channel_wants_images(channel),
                        fn_alter_path)

        self.location.alter_for_create_batch_files(fn_alter_path)
        return True

    def run(self, workspace):
        """Run the module - add the measurements

        """
        header = ["Image name", "Path", "Filename"]
        ratio = [1.0, 2.5, 2.0]
        tags = self.get_metadata_tags()
        ratio += [1.0 for tag in tags]
        ratio = [x / sum(ratio) for x in ratio]
        header += tags
        statistics = []
        m = workspace.measurements
        image_set = workspace.image_set
        image_set_metadata = {}
        image_size = None
        first_image_filename = None
        for fd in self.images:
            for channel in fd.channels:
                wants_images = self.channel_wants_images(channel)
                image_name = channel.get_image_name()
                if wants_images:
                    feature = cellprofiler.measurement.C_URL + "_" + image_name
                    path_feature = cellprofiler.measurement.C_PATH_NAME + "_" + image_name
                    file_feature = cellprofiler.measurement.C_FILE_NAME + "_" + image_name
                else:
                    feature = cellprofiler.measurement.C_OBJECTS_URL + "_" + image_name
                    path_feature = cellprofiler.measurement.C_OBJECTS_PATH_NAME + "_" + image_name
                    file_feature = cellprofiler.measurement.C_OBJECTS_FILE_NAME + "_" + image_name
                row = [image_name,
                       m.get_current_image_measurement(path_feature),
                       m.get_current_image_measurement(file_feature)]
                url = m.get_measurement(cellprofiler.measurement.IMAGE, feature)
                full_name = url2pathname(url.encode('utf-8'))
                path, filename = os.path.split(full_name)
                rescale = channel.rescale.value
                metadata = self.get_filename_metadata(fd, filename, path)
                if self.file_types == FF_STK_MOVIES:
                    index = m.get_measurement(cellprofiler.measurement.IMAGE,
                                              "_".join((C_FRAME, image_name)))
                    provider = LoadImagesSTKFrameProvider(
                            image_name, path, filename, index, rescale)
                elif self.file_types == FF_OTHER_MOVIES:
                    series = m.get_measurement(cellprofiler.measurement.IMAGE,
                                               "_".join((C_SERIES, image_name)))
                    index = m.get_measurement(cellprofiler.measurement.IMAGE,
                                              "_".join((C_FRAME, image_name)))
                    metadata[C_SERIES] = series
                    for f in (M_Z, M_T):
                        feature = cellprofiler.measurement.C_METADATA + "_" + f
                        metadata[f] = m.get_measurement(cellprofiler.measurement.IMAGE, feature)
                    provider = LoadImagesFlexFrameProvider(
                            image_name, path, filename, series, index, rescale)
                elif self.file_types == FF_AVI_MOVIES:
                    index = m.get_measurement(cellprofiler.measurement.IMAGE,
                                              "_".join((C_FRAME, image_name)))
                    metadata[M_T] = m.get_measurement(
                            cellprofiler.measurement.IMAGE, cellprofiler.measurement.C_METADATA + "_" + M_T)
                    provider = LoadImagesMovieFrameProvider(
                            image_name, path, filename, index, rescale)
                else:
                    provider = LoadImagesImageProvider(
                            image_name, path, filename, rescale)
                if wants_images:
                    image = provider.provide_image(workspace.image_set)
                    pixel_data = image.pixel_data
                    image_set.providers.append(provider)
                    m.add_image_measurement(
                            "_".join((C_MD5_DIGEST, image_name)),
                            provider.get_md5_hash(m))
                    m.add_image_measurement("_".join((C_SCALING, image_name)),
                                            image.scale)
                    m.add_image_measurement("_".join((C_HEIGHT, image_name)),
                                            int(pixel_data.shape[0]))
                    m.add_image_measurement("_".join((C_WIDTH, image_name)),
                                            int(pixel_data.shape[1]))
                    if image_size is None:
                        image_size = tuple(pixel_data.shape[:2])
                        first_image_filename = filename
                    elif image_size != tuple(pixel_data.shape[:2]):
                        warning = bad_sizes_warning(image_size, first_image_filename,
                                                    pixel_data.shape[:2], filename)
                        if cellprofiler.preferences.get_headless():
                            print(warning)
                        elif self.show_window:
                            workspace.display_data.warning = warning
                else:
                    ################
                    #
                    #  Interpret file as objects
                    #
                    # Get the frame count from the OME-XML metadata
                    #
                    from bioformats.formatreader import get_omexml_metadata
                    import bioformats.omexml

                    md = get_omexml_metadata(provider.get_full_name())
                    md = bioformats.omexml.OMEXML(md)
                    mdpixels = md.image().Pixels
                    if (mdpixels.channel_count == 1 and mdpixels.Channel().SamplesPerPixel == 3):
                        #
                        # Single interleaved color image
                        #
                        n_frames = 1
                    else:
                        n_frames = mdpixels.SizeZ * mdpixels.SizeC * mdpixels.SizeT
                    series = provider.series
                    ijv = numpy.zeros((0, 3), int)
                    offset = 0
                    for index in range(n_frames):
                        provider.index = index
                        provider.rescale = False
                        labels = provider.provide_image(None).pixel_data
                        shape = labels.shape[:2]
                        labels = convert_image_to_objects(labels)
                        i, j = numpy.mgrid[0:labels.shape[0], 0:labels.shape[1]]
                        ijv = numpy.vstack((
                            ijv, numpy.column_stack((i[labels != 0],
                                                     j[labels != 0],
                                                     labels[labels != 0] + offset))))
                        if ijv.shape[0] > 0:
                            offset = numpy.max(ijv[:, 2])
                    o = cellprofiler.object.Objects()
                    o.set_ijv(ijv, shape)
                    object_set = workspace.object_set
                    assert isinstance(object_set, cellprofiler.object.ObjectSet)
                    object_name = channel.object_name.value
                    object_set.add_objects(o, object_name)
                    provider.release_memory()
                    row[0] = object_name
                    identify.add_object_count_measurements(m, object_name, o.count)
                    identify.add_object_location_measurements_ijv(m, object_name, ijv)
                    if channel.wants_outlines:
                        outlines = numpy.zeros(shape, bool)
                        for l, c in o.get_labels():
                            outlines |= centrosome.outline.outline(l).astype(
                                    outlines.dtype)
                        outline_image = cellprofiler.image.Image(outlines,
                                                                 path_name=path,
                                                                 file_name=filename)
                        workspace.image_set.add(channel.outlines_name.value, outline_image)

                for tag in tags:
                    if tag in metadata:
                        row.append(metadata[tag])
                    elif tag in image_set_metadata:
                        row.append(image_set_metadata[tag])
                    else:
                        row.append("")
                statistics.append(row)
        workspace.display_data.statistics = statistics
        workspace.display_data.col_labels = header

    def display(self, workspace, figure):
        if self.show_window:
            # if hasattr(workspace.display_data, "warning"):
            #     show_warning("Images have different sizes",
            #                  workspace.display_data.warning,
            #                  get_show_report_bad_sizes_dlg,
            #                  set_show_report_bad_sizes_dlg)

            figure.set_subplots((1, 1))
            figure.subplot_table(0, 0,
                                 workspace.display_data.statistics,
                                 col_labels=workspace.display_data.col_labels)

    def get_filename_metadata(self, fd, filename, path):
        """Get the filename and path metadata for a given image

        fd - file/image dictionary
        filename - filename to be parsed
        path - path to be parsed
        """
        metadata = {}
        if self.has_file_metadata(fd):
            metadata.update(cellprofiler.measurement.extract_metadata(fd.file_metadata.value,
                                                                      filename))
        if self.has_path_metadata(fd):
            path = os.path.abspath(os.path.join(self.image_directory(), path))
            metadata.update(cellprofiler.measurement.extract_metadata(fd.path_metadata.value,
                                                                      path))
        if needs_well_metadata(metadata.keys()):
            well_row_token, well_column_token = well_metadata_tokens(metadata.keys())
            metadata[cellprofiler.measurement.FTR_WELL] = (metadata[well_row_token] +
                                                           metadata[well_column_token])
        return metadata

    def write_measurements(self, measurements, image_number,
                           image_settings, full_path,
                           series=None, frame=None, channel_name=None):
        '''Write the image filename, path, url and metadata meas. for a file

        measurements - write measurements into this measurements structure.
                       if None, return the measurements as a dictionary

        image_number - image number for this image

        image_settings - the image settings group for the channel

        full_path - the full file path to the image

        series - For formats like TIFF which can have multiple image stacks,
                 this is the index to the image stack.

        frame - this is the frame in a movie or the index within an image stack.

        channel - write measurements for all channels if None, else write only
                  for this channel.
        '''
        if measurements is not None:
            assert isinstance(measurements, cellprofiler.measurement.Measurements)

            def add_fn(feature, value):
                measurements.add_measurement(
                        cellprofiler.measurement.IMAGE, feature, value,
                        image_set_number=image_number)
        else:
            d = {}

            def add_fn(feature, value):
                d[feature] = value
        path, filename = os.path.split(full_path)
        url = pathname2url(full_path)
        metadata = self.get_filename_metadata(image_settings, filename, path)
        for channel in image_settings.channels:
            if (channel_name is not None and channel_name != channel.get_image_name()):
                continue
            if self.channel_wants_images(channel):
                path_name_category = cellprofiler.measurement.C_PATH_NAME
                file_name_category = cellprofiler.measurement.C_FILE_NAME
                url_category = cellprofiler.measurement.C_URL
            else:
                path_name_category = cellprofiler.measurement.C_OBJECTS_PATH_NAME
                file_name_category = cellprofiler.measurement.C_OBJECTS_FILE_NAME
                url_category = cellprofiler.measurement.C_OBJECTS_URL
            image_data = [(path_name_category, path),
                          (file_name_category, filename),
                          (url_category, url)]
            if series is not None:
                image_data.append((C_SERIES, series))
            if frame is not None:
                image_data.append((C_FRAME, frame))
            for category, value in image_data:
                add_fn("_".join((category, channel.get_image_name())), value)
        for key in metadata.keys():
            add_fn("_".join((cellprofiler.measurement.C_METADATA, key)), metadata[key])
        if measurements is None:
            return d

    def get_frame_count(self, pathname):
        """Return the # of frames in a movie"""
        import bioformats.omexml
        from bioformats.formatreader import get_omexml_metadata
        if self.file_types in (FF_AVI_MOVIES, FF_OTHER_MOVIES, FF_STK_MOVIES):
            url = pathname2url(pathname)
            xml = get_omexml_metadata(pathname)
            omexml = bioformats.omexml.OMEXML(xml)
            frame_count = omexml.image(0).Pixels.SizeT
            if frame_count == 1:
                frame_count = omexml.image(0).Pixels.SizeZ
            return frame_count

        raise NotImplementedError("get_frame_count not implemented for %s" % self.file_types)

    @staticmethod
    def has_file_metadata(fd):
        '''True if the metadata choice is either M_FILE_NAME or M_BOTH

        fd - one of the image file descriptors from self.images
        '''
        return fd.metadata_choice in (M_FILE_NAME, M_BOTH)

    @staticmethod
    def has_path_metadata(fd):
        '''True if the metadata choice is either M_PATH or M_BOTH

        fd - one of the image file descriptors from self.images
        '''
        return fd.metadata_choice in (M_PATH, M_BOTH)

    def get_metadata_tags(self, fd=None):
        """Find the metadata tags for the indexed image

        fd - an image file directory from self.images
        """
        if not fd:
            s = set()
            for fd in self.images:
                s.update(self.get_metadata_tags(fd))
            tags = list(s)
            tags.sort()
            return tags

        tags = []
        if self.has_file_metadata(fd):
            tags += cellprofiler.measurement.find_metadata_tokens(fd.file_metadata.value)
        if self.has_path_metadata(fd):
            tags += cellprofiler.measurement.find_metadata_tokens(fd.path_metadata.value)
        if self.file_types == FF_OTHER_MOVIES:
            tags += [M_Z, M_T, C_SERIES]
        elif self.file_types in (FF_AVI_MOVIES, FF_STK_MOVIES):
            tags += [M_T]
        if needs_well_metadata(tags):
            tags += [cellprofiler.measurement.FTR_WELL]
        return tags

    def get_groupings(self, workspace):
        '''Return the groupings as indicated by the metadata_fields setting

        returns a tuple of key_names and group_list:
        key_names - the names of the keys that identify the groupings
        group_list - a sequence composed of two-tuples.
                     the first element of the tuple is a dictionary giving
                     the metadata values for the metadata keys
                     the second element of the tuple is a sequence of
                     image numbers comprising the image sets of the group
        For instance, an experiment might have key_names of 'Metadata_Row'
        and 'Metadata_Column' and a group_list of:
        [ ({'Metadata_Row':'A','Metadata_Column':'01'}, [1,97,193]),
          ({'Metadata_Row':'A','Metadata_Column':'02'), [2,98,194]),... ]

        Returns None to indicate that the module does not contribute any
        groupings.
        '''
        image_name = self.images[0].channels[0].get_image_name()
        url_feature = '_'.join((cellprofiler.measurement.C_URL, image_name))
        if self.do_group_by_metadata:
            keys = ['_'.join((cellprofiler.measurement.C_METADATA, s))
                    for s in self.metadata_fields.selections]
            mapping = dict([(key, s) for key, s in zip(
                    keys, self.metadata_fields.selections)])
            if len(keys) == 0:
                return None
        elif self.load_movies() and self.file_types == FF_OTHER_MOVIES:
            #
            # Pick the first channel for grouping movie frames
            #
            series_feature = '_'.join((C_SERIES, image_name))
            # Default for Flex is to group by file name and series
            keys = (url_feature, series_feature)
            mapping = dict(((url_feature, cellprofiler.measurement.C_URL), (series_feature, C_SERIES)))

        elif self.load_movies():
            keys = (url_feature,)
            mapping = dict(((url_feature, cellprofiler.measurement.C_URL),))
        else:
            return None
        groupings = workspace.measurements.get_groupings(keys)
        #
        # Rewrite the grouping dictionaries using the preferred key names
        #
        groupings = [(dict([(mapping[k], g[k]) for k in g]), image_numbers)
                     for g, image_numbers in groupings]
        return tuple([mapping[k] for k in keys]), groupings

    def load_images(self):
        """Return true if we're loading images
        """
        return self.file_types == FF_INDIVIDUAL_IMAGES

    def load_movies(self):
        """Return true if we're loading movies
        """
        return self.file_types != FF_INDIVIDUAL_IMAGES

    def load_choice(self):
        """Return the way to match against files: MS_EXACT_MATCH, MS_REGULAR_EXPRESSIONS or MS_ORDER
        """
        return self.match_method.value

    def analyze_sub_dirs(self):
        """Return True if we should analyze subdirectories in addition to the root image directory
        """
        return self.descend_subdirectories != SUB_NONE

    def collect_files(self, workspace):
        """Collect the files that match the filter criteria

        Collect the files that match the filter criteria, starting at the image directory
        and descending downward if AnalyzeSubDirs allows it.

        Returns a list of two-tuples where the first element of the tuple is the path
        from the root directory, including the file name, the second element is the
        index within the image settings (e.g., ImageNameVars).
        """
        global cached_file_lists
        can_cache = workspace.pipeline.test_mode
        frame = workspace.frame
        root = self.image_directory()
        use_cached = False
        if can_cache and frame is not None and root in cached_file_lists:
            how_long, files = cached_file_lists[root]
            if how_long > 3:
                import wx
                if wx.MessageBox(
                        ("The last time you started test mode it took %f seconds\n" 
                         "to find all of the image sets. Do you want to find the\n" 
                         'files again? Choose "No" if you are in a hurry.') % how_long,
                        "Do you want to wait %f seconds again?" % how_long,
                        wx.YES_NO | wx.NO_DEFAULT | wx.ICON_QUESTION, frame) == wx.NO:
                    use_cached = True
        if not use_cached:
            import time
            start_time = time.clock()

            def listdir(path):
                return [x for x in os.listdir(path) if os.path.isfile(os.path.join(path, x))]
            my_relpath = os.path.relpath
            realpath = os.path.realpath
            join = os.path.join

            if root.lower().startswith("http:") or root.lower().startswith("https:"):
                w = None
            elif sys.version_info[0] == 2 and sys.version_info[1] < 6:
                w = os.walk(root, topdown=True)
            else:
                w = os.walk(root, topdown=True, followlinks=True)

            if self.analyze_sub_dirs():
                if self.descend_subdirectories == SUB_SOME:
                    prohibited = self.subdirectory_filter.get_selections()
                else:
                    prohibited = []
                files = []
                seen_dirs = set()
                for dirpath, dirnames, filenames in w:
                    path = my_relpath(dirpath, root)
                    if (path in prohibited) or (os.path.realpath(dirpath) in seen_dirs):
                        # Don't descend into prohibited directories, and avoid infinite loops from links
                        del dirnames[:]
                        continue
                    # update list of visited paths
                    seen_dirs.add(realpath(dirpath))
                    dirnames.sort()  # try to ensure consistent behavior.  Is this needed?
                    if path == os.path.curdir:
                        files += [(file_name, file_name)
                                  for file_name in filenames]
                    else:
                        files += [(join(path, file_name), file_name)
                                  for file_name in filenames]
            else:
                files = [(file_name, file_name)
                         for file_name in sorted(listdir(root))]
            how_long = time.clock() - start_time
            cached_file_lists[self.image_directory()] = (how_long, files)

        if self.load_choice() == MS_EXACT_MATCH:
            files = [(path, self.assign_filename_by_exact_match(file_name))
                     for path, file_name in files]
        elif self.load_choice() == MS_REGEXP:
            files = [(path, self.assign_filename_by_regexp(file_name))
                     for path, file_name in files]
        else:
            # Load by order.
            files = [path for path, file_name in files
                     if self.filter_filename(file_name)]
            files = [(path, self.assign_filename_by_order(idx))
                     for idx, path in enumerate(files)]

        files = [(path, idx) for path, idx in files
                 if idx is not None]
        files.sort()
        if len(files) == 0:
            message = (
                "CellProfiler did not find any image files that "
                'matched your matching pattern: "%s"' %
                self.images[0].common_text.value)
            workspace.pipeline.report_prepare_run_error(self, message)
        return files

    def image_directory(self):
        """Return the image directory
        """
        return self.location.get_absolute_path()

    def image_name_vars(self):
        """Return the list of values in the image name field (the name that later modules see)
        """
        result = []
        for image in self.images:
            if (self.is_multichannel or
                    (self.load_movies() and image.wants_movie_frame_grouping)):
                result += [channel.get_image_name() for channel in image.channels]
            else:
                result += [image.channels[0].get_image_name()]
        return result

    def text_to_find_vars(self):
        """Return the list of values in the image name field (the name that later modules see)
        """
        return [fd.common_text for fd in self.images]

    def text_to_exclude(self):
        """Return the text to match against the file name to exclude it from the set
        """
        return self.match_exclude.value

    def filter_filename(self, filename):
        """Returns True if the file extension is correct

        Returns true if in image mode and an image extension
        or if in movie mode and extension is a movie extension.
        """
        if self.file_types in (FF_AVI_MOVIES, FF_STK_MOVIES, FF_OTHER_MOVIES):
            if not is_movie(filename):
                return False
        elif not is_image(filename):
            return False
        if ((self.text_to_exclude() != cellprofiler.setting.DO_NOT_USE) and
                self.exclude and (filename.find(self.text_to_exclude()) >= 0)):
            return False
        return True

    def assign_filename_by_exact_match(self, filename):
        '''Assign the file name to an image by matching a portion exactly

        filename - filename in question

        Returns either the index of the image or None if no match
        '''
        if not self.filter_filename(filename):
            return None
        ttfs = self.text_to_find_vars()
        for i, ttf in enumerate(ttfs):
            if filename.find(ttf.value) >= 0:
                return i
        return None

    def assign_filename_by_regexp(self, filename):
        '''Assign the file name to an image by regular expression matching

        filename - filename in question

        Returns either the index of the image or None if no match
        '''
        ttfs = self.text_to_find_vars()
        for i, ttf in enumerate(ttfs):
            if re.search(ttf.value, filename):
                return i
        return None

    def assign_filename_by_order(self, index):
        '''Assign the file name to an image by alphabetical order

        index - the order in which it appears in the list

        Returns either the image index or None if the index, modulo the
        # of files in a group is greater than the number of images.
        '''
        result = (index % self.order_group_size.value) + 1
        for i, fd in enumerate(self.images):
            if result == fd.order_position:
                return i
        return None

    def get_categories(self, pipeline, object_name):
        '''Return the categories of measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        '''
        res = []
        object_names = sum(
                [[channel.object_name.value for channel in image.channels
                  if channel.image_object_choice == IO_OBJECTS]
                 for image in self.images], [])
        has_image_name = any([any(
                [True for channel in image.channels
                 if channel.image_object_choice == IO_IMAGES])
                              for image in self.images])

        if object_name == cellprofiler.measurement.IMAGE:
            if has_image_name:
                res += [
                    cellprofiler.measurement.C_FILE_NAME,
                    cellprofiler.measurement.C_PATH_NAME,
                    cellprofiler.measurement.C_URL,
                    C_MD5_DIGEST,
                    C_SCALING,
                    C_HEIGHT,
                    C_WIDTH
                ]
            has_metadata = (self.file_types in
                            (FF_AVI_MOVIES, FF_STK_MOVIES, FF_OTHER_MOVIES))
            if self.file_types == FF_OTHER_MOVIES:
                res += [C_SERIES, C_FRAME]
            elif self.load_movies():
                res += [C_FRAME]
            for fd in self.images:
                if fd.metadata_choice != M_NONE:
                    has_metadata = True
            if has_metadata:
                res += [cellprofiler.measurement.C_METADATA]
            if len(object_names) > 0:
                res += [cellprofiler.measurement.C_OBJECTS_FILE_NAME, cellprofiler.measurement.C_OBJECTS_PATH_NAME,
                        cellprofiler.measurement.C_OBJECTS_URL, cellprofiler.measurement.C_COUNT]
        elif object_name in object_names:
            res += [cellprofiler.measurement.C_LOCATION, cellprofiler.measurement.C_NUMBER]
        return res

    def get_measurements(self, pipeline, object_name, category):
        '''Return the measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        '''
        result = []
        object_names = sum(
                [[channel.object_name.value for channel in image.channels
                  if channel.image_object_choice == IO_OBJECTS]
                 for image in self.images], [])
        if object_name == cellprofiler.measurement.IMAGE:
            if category == cellprofiler.measurement.C_COUNT:
                result += object_names
            else:
                result += [c[1].split('_', 1)[1]
                           for c in self.get_measurement_columns(pipeline)
                           if c[1].split('_')[0] == category]
        elif object_name in object_names:
            if category == cellprofiler.measurement.C_NUMBER:
                result += [cellprofiler.measurement.FTR_OBJECT_NUMBER]
            elif category == cellprofiler.measurement.C_LOCATION:
                result += [cellprofiler.measurement.FTR_CENTER_X, cellprofiler.measurement.FTR_CENTER_Y]
        return result

    def get_measurement_columns(self, pipeline):
        '''Return a sequence describing the measurement columns needed by this module
        '''
        cols = []
        all_tokens = []
        for fd in self.images:
            for channel in fd.channels:
                if not self.channel_wants_images(channel):
                    name = channel.object_name.value
                    cols += identify.get_object_measurement_columns(name)
                    path_name_category = cellprofiler.measurement.C_OBJECTS_PATH_NAME
                    file_name_category = cellprofiler.measurement.C_OBJECTS_FILE_NAME
                    url_category = cellprofiler.measurement.C_URL
                else:
                    name = channel.get_image_name()
                    path_name_category = cellprofiler.measurement.C_PATH_NAME
                    file_name_category = cellprofiler.measurement.C_FILE_NAME
                    url_category = cellprofiler.measurement.C_URL
                    cols += [(cellprofiler.measurement.IMAGE, "_".join((C_MD5_DIGEST, name)),
                              cellprofiler.measurement.COLTYPE_VARCHAR_FORMAT % 32)]
                    cols += [(cellprofiler.measurement.IMAGE, "_".join((C_SCALING, name)),
                              cellprofiler.measurement.COLTYPE_FLOAT)]
                    cols += [(cellprofiler.measurement.IMAGE, "_".join((feature, name)),
                              cellprofiler.measurement.COLTYPE_INTEGER)
                             for feature in (C_HEIGHT, C_WIDTH)]

                cols += [(cellprofiler.measurement.IMAGE, "_".join((file_name_category, name)),
                          cellprofiler.measurement.COLTYPE_VARCHAR_FILE_NAME)]
                cols += [(cellprofiler.measurement.IMAGE, "_".join((path_name_category, name)),
                          cellprofiler.measurement.COLTYPE_VARCHAR_PATH_NAME)]
                cols += [(cellprofiler.measurement.IMAGE, "_".join((url_category, name)),
                          cellprofiler.measurement.COLTYPE_VARCHAR_PATH_NAME)]
                if self.file_types == FF_OTHER_MOVIES:
                    cols += [(cellprofiler.measurement.IMAGE, "_".join((C_SERIES, name)),
                              cellprofiler.measurement.COLTYPE_INTEGER),
                             (cellprofiler.measurement.IMAGE, "_".join((C_FRAME, name)),
                              cellprofiler.measurement.COLTYPE_INTEGER)]
                elif self.load_movies():
                    cols += [(cellprofiler.measurement.IMAGE, "_".join((C_FRAME, name)),
                              cellprofiler.measurement.COLTYPE_INTEGER)]

            if self.has_file_metadata(fd):
                tokens = cellprofiler.measurement.find_metadata_tokens(fd.file_metadata.value)
                cols += [(cellprofiler.measurement.IMAGE, '_'.join((cellprofiler.measurement.C_METADATA, token)),
                          cellprofiler.measurement.COLTYPE_VARCHAR_FILE_NAME)
                         for token in tokens
                         if token not in all_tokens]
                all_tokens += tokens

            if self.has_path_metadata(fd):
                tokens = cellprofiler.measurement.find_metadata_tokens(fd.path_metadata.value)
                cols += [(cellprofiler.measurement.IMAGE, '_'.join((cellprofiler.measurement.C_METADATA, token)),
                          cellprofiler.measurement.COLTYPE_VARCHAR_PATH_NAME)
                         for token in tokens
                         if token not in all_tokens]
                all_tokens += tokens
        #
        # Add a well feature if we have well row and well column
        #
        if needs_well_metadata(all_tokens):
            cols += [
                (cellprofiler.measurement.IMAGE,
                 '_'.join((cellprofiler.measurement.C_METADATA, cellprofiler.measurement.FTR_WELL)),
                 cellprofiler.measurement.COLTYPE_VARCHAR_FILE_NAME)
            ]
        if self.file_types in (FF_AVI_MOVIES, FF_STK_MOVIES):
            cols += [(cellprofiler.measurement.IMAGE, "_".join((cellprofiler.measurement.C_METADATA, M_T)),
                      cellprofiler.measurement.COLTYPE_INTEGER)]
        elif self.file_types == FF_OTHER_MOVIES:
            cols += [(cellprofiler.measurement.IMAGE, "_".join((cellprofiler.measurement.C_METADATA, feature)),
                      cellprofiler.measurement.COLTYPE_INTEGER)
                     for feature in (M_Z, M_T)]

        return cols

    def change_causes_prepare_run(self, setting):
        '''Check to see if changing the given setting means you have to restart

        Some settings, esp in modules like LoadImages, affect more than
        the current image set when changed. For instance, if you change
        the name specification for files, you have to reload your image_set_list.
        Override this and return True if changing the given setting means
        that you'll have to do "prepare_run".
        '''
        #
        # It's safest to say that any change in loadimages requires a restart
        #
        return True

    def needs_conversion(self):
        if self.match_method not in (MS_EXACT_MATCH, MS_REGEXP):
            raise ValueError(
                    "Can't convert a LoadImages module that matches images by %s" %
                    self.match_method.value)
        return True

    def convert(self, pipeline, metadata, namesandtypes, groups):
        '''Convert to the modern format using the input modules

        This method converts any LoadImages modules in the pipeline to
        Images / Metadata / NamesAndTypes and Groups

        metadata - the metadata module
        namesandtypes - the namesandtypes module
        groups - the groups module
        first - True if no images have been added to namesandtypes yet.
        '''
        import cellprofiler.modules.metadata as cpmetadata
        import cellprofiler.modules.namesandtypes as cpnamesandtypes
        import cellprofiler.modules.groups as cpgroups
        assert isinstance(metadata, cpmetadata.Metadata)
        assert isinstance(namesandtypes, cpnamesandtypes.NamesAndTypes)
        assert isinstance(groups, cpgroups.Groups)

        if self.match_method not in (MS_EXACT_MATCH, MS_REGEXP):
            raise ValueError(
                    "Can't convert a LoadImages module that matches images by %s" %
                    self.match_method.value)
        if self.group_by_metadata:
            groups.notes.append(
                    "WARNING: original pipeline used metadata grouping"
                    " which was not converted during processing"
                    " of current pipeline.")
        warn_metadata_match = False
        warn_gray_color = False
        edited_modules = set()
        for group in self.images:
            for channel in group.channels:
                if namesandtypes.assignment_method == cpnamesandtypes.ASSIGN_ALL:
                    namesandtypes.assignment_method.value = \
                        cpnamesandtypes.ASSIGN_RULES
                else:
                    namesandtypes.add_assignment()
                edited_modules.add(namesandtypes)
                assignment = namesandtypes.assignments[-1]
                if channel.image_object_choice == IO_IMAGES:
                    name = assignment.image_name.value = \
                        channel.image_name.value
                    assignment.load_as_choice.value = \
                        cpnamesandtypes.LOAD_AS_GRAYSCALE_IMAGE
                    warn_gray_color = True
                else:
                    name = assignment.object_name.value = \
                        channel.object_name.value
                    assignment.load_as_choice.value = \
                        cpnamesandtypes.LOAD_AS_OBJECTS
                rfilter = assignment.rule_filter
                assert isinstance(rfilter, cellprofiler.setting.Filter)
                structure = [cellprofiler.setting.Filter.AND_PREDICATE]
                fp = images.FilePredicate()
                fp_does, fp_does_not = [
                    [d for d in fp.subpredicates if isinstance(d, c)][0]
                    for c in (cellprofiler.setting.Filter.DoesPredicate, cellprofiler.setting.Filter.DoesNotPredicate)]
                if self.exclude:
                    # Exclude with a predicate of
                    # File does not contain <match_exclude>
                    #
                    structure.append([
                        fp, fp_does_not, cellprofiler.setting.Filter.CONTAINS_PREDICATE,
                        self.match_exclude.value])
                if self.match_method == MS_EXACT_MATCH:
                    structure.append([
                        fp, fp_does, cellprofiler.setting.Filter.CONTAINS_PREDICATE,
                        group.common_text.value])
                else:
                    structure.append([
                        fp, fp_does, cellprofiler.setting.Filter.CONTAINS_REGEXP_PREDICATE,
                        group.common_text.value])
                assignment.rule_filter.build(structure)
                my_tags = set()
                for metadata_choice, source, value in (
                        (M_FILE_NAME, cpmetadata.XM_FILE_NAME, group.file_metadata.value),
                        (M_PATH, cpmetadata.XM_FOLDER_NAME, group.path_metadata.value)):
                    if group.metadata_choice in (metadata_choice, M_BOTH):
                        if not metadata.wants_metadata:
                            metadata.wants_metadata.value = True
                        else:
                            metadata.add_extraction_method()
                        edited_modules.add(metadata)
                        mgroup = metadata.extraction_methods[-1]
                        mgroup.source.value = source
                        if metadata_choice == M_FILE_NAME:
                            mgroup.file_regexp.value = value
                        else:
                            mgroup.folder_regexp.value = value
                        mgroup.filter_choice.value = cpmetadata.F_FILTERED_IMAGES
                        mgroup.extraction_method.value = cpmetadata.X_MANUAL_EXTRACTION
                        mgroup.filter.build(structure)
                        my_tags.update(cellprofiler.measurement.find_metadata_tokens(value))
                if namesandtypes.matching_choice == cpnamesandtypes.MATCH_BY_METADATA:
                    # Add our metadata tags to the joiner
                    current = namesandtypes.join.parse()
                    for d in current:
                        for v in d.values():
                            if v in my_tags:
                                d[name] = v
                                my_tags.remove(v)
                                break
                        else:
                            warn_metadata_match = True
                            d[name] = None
                    if len(my_tags) > 0:
                        warn_metadata_match = True
                        if len(current) == 0:
                            for tag in my_tags:
                                current.append({name: tag})
                        else:
                            for tag in my_tags:
                                entry = dict([(k, None) for k in current[0]])
                                entry[name] = tag
                                current.append(entry)
                    namesandtypes.join.build(current)
                    edited_modules.add(namesandtypes)
                elif len(my_tags) > 0:
                    namesandtypes.matching_choice.value = \
                        cpnamesandtypes.MATCH_BY_METADATA
                    namesandtypes.join.build([{name: tag} for tag in my_tags])
                    edited_modules.add(namesandtypes)
        if self.group_by_metadata:
            groups.wants_groups.value = True
            edited_modules.add(groups)
            metadata_fields = self.metadata_fields.get_selections()
            if len(groups.grouping_metadata) > len(metadata_fields):
                del groups.grouping_metadata[len(metadata_fields):]
            else:
                while len(groups.grouping_metadata) < len(metadata_fields):
                    groups.add_grouping_metadata()

            for i, field in enumerate(metadata_fields):
                md_group = groups.grouping_metadata[i]
                md_group.metadata_choice.value = \
                    field

        if warn_metadata_match:
            namesandtypes.notes.append(
                    "WARNING: the metadata matching for this pipeline may not"
                    " have been converted correctly.")
        if warn_gray_color:
            namesandtypes.notes.append(
                    'Please change any color images from "Load as Grayscale image"'
                    ' to "Load as Color image"')
        for module in edited_modules:
            pipeline.edit_module(module.module_num, True)


def well_metadata_tokens(tokens):
    '''Return the well row and well column tokens out of a set of metadata tokens'''

    well_row_token = None
    well_column_token = None
    for token in tokens:
        if cellprofiler.measurement.is_well_row_token(token):
            well_row_token = token
        if cellprofiler.measurement.is_well_column_token(token):
            well_column_token = token
    return well_row_token, well_column_token


def needs_well_metadata(tokens):
    '''Return true if, based on a set of metadata tokens, we need a well token

    Check for a row and column token and the absence of the well token.
    '''
    if cellprofiler.measurement.FTR_WELL.lower() in [x.lower() for x in tokens]:
        return False
    well_row_token, well_column_token = well_metadata_tokens(tokens)
    return (well_row_token is not None) and (well_column_token is not None)


def is_image(filename):
    '''Determine if a filename is a potential image file based on extension'''
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def is_movie(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in SUPPORTED_MOVIE_EXTENSIONS


def is_numpy_file(filename):
    return os.path.splitext(filename)[-1].lower() == ".npy"


def is_matlab_file(filename):
    return os.path.splitext(filename)[-1].lower() == ".mat"


def loadmat(path):
    imgdata = scipy.io.matlab.mio.loadmat(path, struct_as_record=True)
    img = imgdata["Image"]

    return img


def load_data_file(pathname_or_url, load_fn):
    ext = os.path.splitext(pathname_or_url)[-1].lower()

    if any([pathname_or_url.startswith(scheme) for scheme in PASSTHROUGH_SCHEMES]):
        url = cellprofiler.misc.generate_presigned_url(pathname_or_url)

        try:
            src = urllib.urlopen(url)
            fd, path = tempfile.mkstemp(suffix=ext)
            with os.fdopen(fd, mode="wb") as dest:
                shutil.copyfileobj(src, dest)
            img = load_fn(path)
        finally:
            try:
                src.close()
                os.remove(path)
            except NameError:
                pass

        return img

    return load_fn(pathname_or_url)


class LoadImagesImageProvider(cellprofiler.image.AbstractImageProvider):
    """Base for image providers: handle pathname and filename & URLs"""

    def __init__(self, name, pathname, filename, rescale=True, series=None, index=None,
                 channel=None, volume=False, spacing=None):
        """
        :param name: Name of image to be provided
        :type name:
        :param pathname: Path to file or base of URL
        :type pathname:
        :param filename: Filename of file or last chunk of URL
        :type filename:
        :param rescale:
        :type rescale:
        :param series:
        :type series:
        :param index:
        :type index:
        :param channel:
        :type channel:
        :param volume:
        :type volume:
        :param spacing:
        :type spacing:
        """
        if pathname.startswith(FILE_SCHEME):
            pathname = url2pathname(pathname)
        self.__name = name
        self.__pathname = pathname
        self.__filename = filename
        self.__cached_file = None
        self.__is_cached = False
        self.__cacheing_tried = False
        self.__image = None

        if pathname is None:
            self.__url = filename
        elif any([pathname.startswith(s + ":") for s in PASSTHROUGH_SCHEMES]):
            if filename is not None:
                self.__url = pathname + "/" + filename
            else:
                self.__url = pathname
        elif filename is None:
            self.__url = pathname2url(pathname)
        else:
            self.__url = pathname2url(os.path.join(pathname, filename))

        self.rescale = rescale
        self.__series = series
        self.__channel = channel
        self.__index = index
        self.__volume = volume
        self.__spacing = spacing
        self.scale = None

    @property
    def series(self):
        return self.__series

    @series.setter
    def series(self, series):
        # Invalidate the cached image
        self.__image = None
        self.__series = series

    @property
    def channel(self):
        return self.__channel

    @channel.setter
    def channel(self, index):
        # Invalidate the cached image
        self.__image = None
        self.__channel = index

    @property
    def index(self):
        return self.__index

    @index.setter
    def index(self, index):
        # Invalidate the cached image
        self.__image = None
        self.__index = index

    def get_name(self):
        return self.__name

    def get_pathname(self):
        return self.__pathname

    def get_filename(self):
        return self.__filename

    def cache_file(self):
        '''Cache a file that needs to be HTTP downloaded

        Return True if the file has been cached
        '''
        if self.__cacheing_tried:
            return self.__is_cached
        self.__cacheing_tried = True
        #
        # Check to see if the pathname can be accessed as a directory
        # If so, handle normally
        #
        path = self.get_pathname()
        if len(path) == 0:
            filename = self.get_filename()
            if os.path.exists(filename):
                return False
            parsed_path = urlparse.urlparse(filename)
            url = filename
            if len(parsed_path.scheme) < 2:
                raise IOError("Test for access to file failed. File: %s" % filename)
        elif os.path.exists(path):
            return False
        else:
            parsed_path = urlparse.urlparse(path)
            url = '/'.join((path, self.get_filename()))
            #
            # Scheme length == 0 means no scheme
            # Scheme length == 1 - probably DOS drive letter
            #
            if len(parsed_path.scheme) < 2:
                raise IOError("Test for access to directory failed. Directory: %s" % path)
        if parsed_path.scheme == 'file':
            self.__cached_file = url2pathname(path)
        elif is_numpy_file(self.__filename):
            #
            # urlretrieve uses the suffix of the path component of the URL
            # to name the temporary file, so we replicate that behavior
            #
            temp_dir = cellprofiler.preferences.get_temporary_directory()
            tempfd, temppath = tempfile.mkstemp(suffix=".npy", dir=temp_dir)
            self.__cached_file = temppath
            try:
                url = cellprofiler.misc.generate_presigned_url(url)
                self.__cached_file, headers = urllib.urlretrieve(url, filename=temppath)
            finally:
                os.close(tempfd)
        else:
            from bioformats.formatreader import get_image_reader
            rdr = get_image_reader(id(self), url=url)
            self.__cached_file = rdr.path
        self.__is_cached = True
        return True

    def get_full_name(self):
        self.cache_file()
        if self.__is_cached:
            return self.__cached_file
        return os.path.join(self.get_pathname(), self.get_filename())

    def get_url(self):
        '''Get the URL representation of the file location'''
        return self.__url

    def get_md5_hash(self, measurements):
        '''Compute the MD5 hash of the underlying file or use cached value

        measurements - backup for case where MD5 is calculated on image data
                       directly retrieved from URL
        '''
        #
        # Cache the MD5 hash on the image reader
        #
        if is_matlab_file(self.__filename) or is_numpy_file(self.__filename):
            rdr = None
        else:
            from bioformats.formatreader import get_image_reader
            rdr = get_image_reader(None, url=self.get_url())
        if rdr is None or not hasattr(rdr, "md5_hash"):
            hasher = hashlib.md5()
            path = self.get_full_name()
            if not os.path.isfile(path):
                # No file here - hash the image
                image = self.provide_image(measurements)
                hasher.update(image.pixel_data.tostring())
            else:
                with open(self.get_full_name(), "rb") as fd:
                    while True:
                        buf = fd.read(65536)
                        if len(buf) == 0:
                            break
                        hasher.update(buf)
            if rdr is None:
                return hasher.hexdigest()
            rdr.md5_hash = hasher.hexdigest()
        return rdr.md5_hash

    def release_memory(self):
        '''Release any image memory

        Possibly delete the temporary file'''
        if self.__is_cached:
            if is_matlab_file(self.__filename) or is_numpy_file(self.__filename):
                try:
                    os.remove(self.__cached_file)
                except:
                    logger.warning("Could not delete file %s", self.__cached_file,
                                   exc_info=True)
            else:
                from bioformats.formatreader import release_image_reader
                release_image_reader(id(self))
            self.__is_cached = False
            self.__cacheing_tried = False
            self.__cached_file = None
        self.__image = None

    def __del__(self):
        # using __del__ is all kinds of bad, but we need to remove the
        # files to keep the system from filling up.
        self.release_memory()

    def __set_image(self):
        if self.__volume:
            self.__set_image_volume()
            return

        from bioformats.formatreader import get_image_reader
        self.cache_file()
        channel_names = []
        if is_matlab_file(self.__filename):
            img = load_data_file(self.get_full_name(), loadmat)
            self.scale = 1.0
        elif is_numpy_file(self.__filename):
            img = load_data_file(self.get_full_name(), numpy.load)
            self.scale = 1.0
        else:
            url = self.get_url()
            if url.lower().startswith("omero:"):
                rdr = get_image_reader(self.get_name(), url=url)
            else:
                rdr = get_image_reader(
                    self.get_name(), url=self.get_url())
            if numpy.isscalar(self.index) or self.index is None:
                img, self.scale = rdr.read(
                    c=self.channel,
                    series=self.series,
                    index=self.index,
                    rescale=self.rescale,
                    wants_max_intensity=True,
                    channel_names=channel_names)
            else:
                # It's a stack
                stack = []
                if numpy.isscalar(self.series):
                    series_list = [self.series] * len(self.index)
                else:
                    series_list = self.series
                if not numpy.isscalar(self.channel):
                    channel_list = [self.channel] * len(self.index)
                else:
                    channel_list = self.channel
                for series, index, channel in zip(
                        series_list, self.index, channel_list):
                    img, self.scale = rdr.read(
                        c=channel,
                        series=series,
                        index=index,
                        rescale=self.rescale,
                        wants_max_intensity=True,
                        channel_names=channel_names)
                    stack.append(img)
                img = numpy.dstack(stack)
        if isinstance(self.rescale, float):
            # Apply a manual rescale
            img = img.astype(numpy.float32) / self.rescale
        self.__image = cellprofiler.image.Image(img,
                                                path_name=self.get_pathname(),
                                                file_name=self.get_filename(),
                                                scale=self.scale)
        if img.ndim == 3 and len(channel_names) == img.shape[2]:
            self.__image.channel_names = list(channel_names)

    def provide_image(self, image_set):
        """Load an image from a pathname
        """
        if self.__image is None or image_set is not None:
            self.__set_image()
        return self.__image

    def __set_image_volume(self):
        pathname = url2pathname(self.get_url())

        # Volume loading is currently limited to tiffs/numpy files only
        if is_numpy_file(self.__filename):
            data = numpy.load(pathname)
        else:
            data = skimage.external.tifffile.imread(pathname)

        # https://github.com/CellProfiler/python-bioformats/blob/855f2fb7807f00ef41e6d169178b7f3d22530b79/bioformats/formatreader.py#L768-L791
        if data.dtype in [numpy.int8, numpy.uint8]:
            self.scale = 255
        elif data.dtype in [numpy.int16, numpy.uint16]:
            self.scale = 65535
        elif data.dtype == numpy.int32:
            self.scale = 2**32-1
        elif data.dtype == numpy.uint32:
            self.scale = 2**32
        else:
            self.scale = 1

        self.__image = cellprofiler.image.Image(
            image=data,
            path_name=self.get_pathname(),
            file_name=self.get_filename(),
            dimensions=3,
            scale=self.scale,
            spacing=self.__spacing
        )


class LoadImagesImageProviderURL(LoadImagesImageProvider):
    '''Reference an image via a URL'''

    def __init__(self, name, url, rescale=True,
                 series=None, index=None, channel=None, volume=False, spacing=None):
        if url.lower().startswith("file:"):
            path = url2pathname(url)
            pathname, filename = os.path.split(path)
        else:
            pathname = ""
            filename = url
        super(LoadImagesImageProviderURL, self).__init__(
                name, pathname, filename, rescale, series, index, channel, volume, spacing)
        self.url = url

    def get_url(self):
        if self.cache_file():
            return super(LoadImagesImageProviderURL, self).get_url()
        return self.url


class LoadImagesMovieFrameProvider(LoadImagesImageProvider):
    """Provide an image by filename:frame, loading the file as it is requested
    """

    def __init__(self, name, pathname, filename, frame, rescale):
        super(LoadImagesMovieFrameProvider, self).__init__(
                name, pathname, filename, rescale, index=frame)


class LoadImagesFlexFrameProvider(LoadImagesImageProvider):
    """Provide an image by filename:frame, loading the file as it is requested
    """

    def __init__(self, name, pathname, filename, series, index, rescale):
        super(LoadImagesFlexFrameProvider, self).__init__(
                name, pathname, filename,
                rescale=rescale,
                series=series,
                index=index)


class LoadImagesSTKFrameProvider(LoadImagesImageProvider):
    """Provide an image by filename:frame from an STK file"""

    def __init__(self, name, pathname, filename, frame, rescale):
        '''Initialize the provider

        name - name of the provider for access from image set
        pathname - path to the file
        filename - name of the file
        frame - # of the frame to provide
        '''
        super(LoadImagesSTKFrameProvider, self).__init__(
                name, pathname, filename, rescale=rescale, index=frame)


def convert_image_to_objects(image):
    '''Interpret an image as object indices

    image - a grayscale or color image, assumes zero == background

    returns - a similarly shaped integer array with zero representing background
              and other values representing the indices of the associated object.
    '''
    assert isinstance(image, numpy.ndarray)
    if image.ndim == 2:
        unique_indices = numpy.unique(image.ravel())
        if (len(unique_indices) * 2 > max(numpy.max(unique_indices), 254) and
                numpy.all(numpy.abs(numpy.round(unique_indices, 1) - unique_indices) <=
                          numpy.finfo(float).eps)):
            # Heuristic: reinterpret only if sparse and roughly integer
            return numpy.round(image).astype(int)

        def sorting(x):
            return [x]

        def comparison(i0, i1):
            return image.ravel()[i0] != image.ravel()[i1]
    else:
        i, j = numpy.mgrid[0:image.shape[0], 0:image.shape[1]]

        def sorting(x):
            return [x[:, :, 2], x[:, :, 1], x[:, :, 0]]

        def comparison(i0, i1):
            return numpy.any(image[i.ravel()[i0], j.ravel()[i0], :] != image[i.ravel()[i1], j.ravel()[i1], :], 1)

    order = numpy.lexsort([x.ravel() for x in sorting(image)])
    different = numpy.hstack([[False], comparison(order[:-1], order[1:])])
    index = numpy.cumsum(different)
    image = numpy.zeros(image.shape[:2], index.dtype)
    image.ravel()[order] = index
    return image


def bad_sizes_warning(first_size, first_filename,
                      second_size, second_filename):
    '''Return a warning message about sizes being wrong

    first_size: tuple of height / width of first image
    first_filename: file name of first image
    second_size: tuple of height / width of second image
    second_filename: file name of second image
    '''
    warning = ("Warning: loading image files of different dimensions.\n\n"
               "%s: width = %d, height = %d\n"
               "%s: width = %d, height = %d") % (
                  first_filename, first_size[1], first_size[0],
                  second_filename, second_size[1], second_size[0])
    return warning


FILE_SCHEME = "file:"
PASSTHROUGH_SCHEMES = ("http", "https", "ftp", "omero", "s3")


def pathname2url(path):
    '''Convert the unicode path to a file: url'''
    utf8_path = path.encode('utf-8')
    if any([utf8_path.lower().startswith(x) for x in PASSTHROUGH_SCHEMES]):
        return utf8_path
    return FILE_SCHEME + urllib.pathname2url(utf8_path)


def is_file_url(url):
    return url.lower().startswith(FILE_SCHEME)


def url2pathname(url):
    if isinstance(url, six.text_type):
        url = url.encode("utf-8")
    if any([url.lower().startswith(x) for x in PASSTHROUGH_SCHEMES]):
        return url
    assert is_file_url(url)
    utf8_url = urllib.url2pathname(url[len(FILE_SCHEME):])
    return utf8_url.encode('utf-8')


def urlfilename(url):
    '''Return just the file part of a URL

    For instance http://cellprofiler.org/linked_files/file%20has%20spaces.txt
    has a file part of "file has spaces.txt"
    '''
    if is_file_url(url):
        return os.path.split(url2pathname(url))[1]
    path = urlparse.urlparse(url)[2]
    if "/" in path:
        return urllib.unquote(path.rsplit("/", 1)[1])
    else:
        return urllib.unquote(path)


def urlpathname(url):
    '''Return the path part of a URL

    For instance, http://cellprofiler.org/Comma%2Cseparated/foo.txt
    has a path of http://cellprofiler.org/Comma,separated

    A file url has the normal sort of path that you'd expect.
    '''
    if is_file_url(url):
        return os.path.split(url2pathname(url))[0]
    scheme, netloc, path = urlparse.urlparse(url)[:3]
    path = urlparse.urlunparse([scheme, netloc, path, "", "", ""])
    if "/" in path:
        return urllib.unquote(path.rsplit("/", 1)[0])
    else:
        return urllib.unquote(path)
