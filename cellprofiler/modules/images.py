# coding=utf-8

import os
import urllib

import six

import _help
import cellprofiler.gui.help.content
import cellprofiler.icons
import cellprofiler.module
import cellprofiler.pipeline
import cellprofiler.setting
import cellprofiler.utilities.hdf5_dict
import javabridge
import loadimages

__doc__ = """\
Images
======

The **Images** module allows you to specify the location of files to be
analyzed by the pipeline; setting this module correctly is the first
step in creating a new project in CellProfiler. These files can be
located on your hard drive, on a networked computer elsewhere, or
accessible with a URL. You can also provide rules to specify only those
files that you want analyzed out of a larger collection (for example,
from a folder containing both images for analysis and non-image files
that should be disregarded).

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============

What is a “digital image”?
^^^^^^^^^^^^^^^^^^^^^^^^^^

A *digital image* is a set of numbers arranged into a two-dimensional
format of rows and columns; a pixel refers to the row/column location of
a particular point in the image. Pixels in grayscale or monochrome
(black/white) images contain a single intensity value, whereas in color
images, each pixel contains a red, green, and blue (RGB) triplet of
intensity values. Additionally, the term image can be used as short-hand
for an image sequence, that is, an image collection such as a time-lapse
series (2-D + *t*), confocal Z-stacks (3-D), etc.

CellProfiler can read a wide variety of image formats by using a library
called Bio-Formats; see `here`_ for the formats available. Some image
formats are better than others for use in image analysis. Some are
`“lossy”`_ (information is lost in the conversion to the format) like
most JPG/JPEG files; others are `“lossless”`_ (no image information is
lost). For image analysis purposes, a lossless format like TIF or PNG is
recommended.

What do I need as input?
^^^^^^^^^^^^^^^^^^^^^^^^

The most straightforward way to provide image files to the **Images**
module is to simply drag-and-drop them on the file list panel (the blank
space indicated by the text “Drop files and folders here”).

.. image:: {IMG_PANEL_BLANK}
   :width: 100%

Using the file explorer tool of your choice (e.g., Explorer in Windows,
Finder in Mac), you can drag-and-drop individual files and/or entire
folders into this panel. You can also right-click in the File list panel
to bring up a file selection window to browse for individual files; on
the Mac, folders can be drag-and-dropped from this window and you can
select multiple files using Ctrl-A (Windows) or Cmd-A (Mac).

.. image:: {IMG_PANEL_FILLED}
   :width: 100%

Right-clicking on the file list panel will provide a context menu with
options to modify the file list:

-  *Show Selected Image:* Selecting this option (or double-clicking on
   the file) will open the image in a new window.
-  *Remove From List:* Removes the selected file or folder from the
   list. Note that this does not remove the file/folder from the hard
   drive.
-  *Remove Unavailable Files:* Refresh the list by checking for
   existence of file. Note that this does not remove the files from the
   hard drive.
-  *Browse For Images:* Use a dialog box to select an image file (though
   drag-and-drop is recommended).
-  *Refresh:* Shown only if folder is selected. Refresh the list of
   files from the folder. Files that were manually removed from the list
   for that folder are restored.
-  *Expand All Folders:* Expand all trees shown in the file list panel.
-  *Collapse All Folders:* Collapse all folder trees shown in the file
   list panel.
-  *Clear File List:* Remove all files/folders in the file list panel.
   You will be prompted for confirmation beforehand. Note that this does
   not remove the files from the hard drive.

How do I configure the module?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have a subset of files that you want to analyze from the full
listing shown in the panel, you can filter the files according to a set
of rules. This is useful in cases such as:

-  You have dragged a folder of images onto the file list panel, but the
   folder contains images you want to analyze along with non-image files
   that you want to disregard.
-  You have dragged a folder of images onto the file list panel, but the
   folder contains the images from one experiment that you want to
   process along with images from another experiment that you want to
   ignore for now.

You may specify as many rules as necessary to define the desired list of
images.

After you have filtered the file list, press the “Apply” button to
update the view of the file list. You can also toggle the “Show file
excluded by filters” box to modify the display of the files:

-  Checking this box will show all the files in the list, with the files
   that have been filtered out shown as grayed-out entries.
-  Not checking this box will only show the files in the list that pass
   the filter(s).

What do I get as output?
^^^^^^^^^^^^^^^^^^^^^^^^

The final product of the **Images** module is a file list in which any
files that are not intended for further processing have been removed,
whether manually or using filtering. This list will be used when
collecting metadata (if desired) and when assembling the image sets in
NamesAndTypes. The list can be filtered further in NamesAndTypes to
specify, for example, that a subset of these images represents a
particular wavelength.

.. _here: http://www.openmicroscopy.org/site/support/bio-formats5/supported-formats.html
.. _“lossy”: http://www.techterms.com/definition/lossy
.. _“lossless”: http://www.techterms.com/definition/lossless
""".format(**{
    "IMG_PANEL_BLANK": cellprofiler.gui.help.content.image_resource('Images_FilelistPanel_Blank.png'),
    "IMG_PANEL_FILLED": cellprofiler.gui.help.content.image_resource('Images_FilelistPanel_Filled.png')
})

FILTER_CHOICE_NONE = "No filtering"
FILTER_CHOICE_IMAGES = "Images only"
FILTER_CHOICE_CUSTOM = "Custom"
FILTER_CHOICE_ALL = [FILTER_CHOICE_NONE, FILTER_CHOICE_IMAGES,
                     FILTER_CHOICE_CUSTOM]

FILTER_DEFAULT = 'and (extension does isimage) (directory doesnot containregexp "[\\\\\\\\/]\\\\.")'


class Images(cellprofiler.module.Module):
    variable_revision_number = 2
    module_name = "Images"
    category = "File Processing"

    MI_SHOW_IMAGE = "Show image"
    MI_REMOVE = cellprofiler.setting.FileCollectionDisplay.DeleteMenuItem("Remove from list")
    MI_REFRESH = "Refresh"

    def create_settings(self):
        self.workspace = None
        module_explanation = [
            "To begin creating your project, use the %s module to compile" % self.module_name,
            "a list of files and/or folders that you want to analyze. You can also specify a set of rules",
            "to include only the desired files in your selected folders."]
        self.set_notes([" ".join(module_explanation)])

        self.path_list_display = cellprofiler.setting.PathListDisplay()
        predicates = [FilePredicate(),
                      DirectoryPredicate(),
                      ExtensionPredicate()]

        self.filter_choice = cellprofiler.setting.Choice(
            "Filter images?",
            FILTER_CHOICE_ALL,
            value=FILTER_CHOICE_IMAGES,
            doc="""\
The **Images** module will pass all the files specified in the file list
panel downstream to have a meaningful name assigned to it (so other
modules can access it) or optionally, to define the relationships
between images and associated metadata. Enabling file filtering will
allow you to specify a subset of the files from the file list panel by
defining rules to filter the files. This approach is useful if, for
example, you drag-and-dropped a folder onto the file list panel which
contains a mixture of images that you want to analyze and other files
that you want to ignore.

Several options are available for this setting:

-  *{FILTER_CHOICE_NONE}:* Do not enable filtering; all files in the
   File list panel will be passed to downstream modules for processing.
   This option can be selected if you are sure that only images are
   specified in the list.
-  *{FILTER_CHOICE_IMAGES}:* Only image files will be passed to
   downstream modules. The permissible image formats are provided by a
   library called Bio-Formats; see `here`_ for the formats available.
-  *{FILTER_CHOICE_CUSTOM}:* Specify custom rules for selecting a
   subset of the files from the File list panel. This approach is useful
   if, for example, you drag-and-dropped a folder onto the File list
   panel which contains a mixture of images that you want to analyze and
   other files that you want to ignore.

.. _here: http://www.openmicroscopy.org/site/support/bio-formats5/supported-formats.html
""".format(**{
                "FILTER_CHOICE_CUSTOM": FILTER_CHOICE_CUSTOM,
                "FILTER_CHOICE_IMAGES": FILTER_CHOICE_IMAGES,
                "FILTER_CHOICE_NONE": FILTER_CHOICE_NONE
            })
        )

        self.filter = cellprofiler.setting.Filter(
            "Select the rule criteria",
            predicates,
            FILTER_DEFAULT,
            doc="""\
Specify a set of rules to narrow down the files to be analyzed.

{FILTER_RULES_BUTTONS_HELP}
""".format(**{
                "FILTER_RULES_BUTTONS_HELP": _help.FILTER_RULES_BUTTONS_HELP
            })
        )

        self.update_button = cellprofiler.setting.PathListRefreshButton(
            "Apply filters to the file list",
            "Apply filters to the file list",
            doc="""\
*(Only displayed if filtering based on rules)*

Re-display the file list, removing or graying out the files that do not
pass the current filter.
""")

    def help_settings(self):
        return [
            self.filter,
            self.update_button
        ]

    @staticmethod
    def modpath_to_url(modpath):
        if modpath[0] in ("http", "https", "ftp", "s3"):
            if len(modpath) == 1:
                return modpath[0] + ":"
            elif len(modpath) == 2:
                return modpath[0] + ":" + modpath[1]
            else:
                return modpath[0] + ":" + modpath[1] + "/" + "/".join(
                    [urllib.quote(part) for part in modpath[2:]])
        path = os.path.join(*modpath)
        return loadimages.pathname2url(path)

    @staticmethod
    def url_to_modpath(url):
        if not url.lower().startswith("file:"):
            schema, rest = cellprofiler.utilities.hdf5_dict.HDF5FileList.split_url(url)
            return [schema] + rest[0:1] + [urllib.unquote(part) for part in rest[1:]]
        path = urllib.url2pathname(url[5:])
        parts = []
        while True:
            new_path, part = os.path.split(path)
            if len(new_path) == 0 or len(part) == 0:
                parts.insert(0, path)
                break
            parts.insert(0, part)
            path = new_path
        return parts

    @classmethod
    def make_modpath_from_path(cls, path):
        result = []
        while True:
            new_path, part = os.path.split(path)
            if len(new_path) == 0 or len(part) == 0:
                return [path] + result
            result.insert(0, part)
            path = new_path

    def settings(self):
        return [self.path_list_display, self.filter_choice, self.filter]

    def visible_settings(self):
        result = [self.path_list_display, self.filter_choice]
        if self.filter_choice == FILTER_CHOICE_CUSTOM:
            result += [self.filter, self.update_button]
            self.path_list_display.using_filter = True
        elif self.filter_choice == FILTER_CHOICE_IMAGES:
            result += [self.update_button]
            self.path_list_display.using_filter = True
        else:
            self.path_list_display.using_filter = False
        return result

    def change_causes_prepare_run(self, setting):
        """Return True if a change to the settings requires a call to prepare_run

        Images should return True if any setting changes because that
        will affect the image plane descriptors passed onto later modules
        which will change the image set produced by the pipeline.
        """
        return setting in self.settings()

    @classmethod
    def is_input_module(self):
        return True

    def prepare_run(self, workspace):
        """Create an IPD for every url that passes the filter"""
        if workspace.pipeline.in_batch_mode():
            return True
        file_list = workspace.pipeline.file_list
        if self.filter_choice != FILTER_CHOICE_NONE:
            if self.filter_choice == FILTER_CHOICE_IMAGES:
                expression = FILTER_DEFAULT
            else:
                expression = self.filter.value_text
            env = javabridge.get_env()
            ifcls = javabridge.class_for_name("org.cellprofiler.imageset.ImageFile")
            scls = env.find_class("java/lang/String")
            iffilter = javabridge.make_instance(
                "org/cellprofiler/imageset/filter/Filter",
                "(Ljava/lang/String;Ljava/lang/Class;)V",
                expression, ifcls)
            file_array = env.make_object_array(len(file_list), scls)
            for i, url in enumerate(file_list):
                if url.startswith("s3:"):
                    url = url.replace(" ", "+")

                if isinstance(url, six.text_type):
                    ourl = env.new_string(url)
                else:
                    ourl = env.new_string_utf(url)
                env.set_object_array_element(file_array, i, ourl)
            passes_filter = javabridge.call(
                iffilter, "filterURLs",
                "([Ljava/lang/String;)[Z", file_array)
            if isinstance(passes_filter, javabridge.JB_Object):
                passes_filter = javabridge.get_env().get_boolean_array_elements(
                    passes_filter)
            file_list = [f for f, passes in zip(file_list, passes_filter)
                         if passes]
        workspace.pipeline.set_filtered_file_list(file_list, self)
        return True

    def run(self, workspace):
        pass

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        """Upgrade pipeline settings from a previous revision

        setting_values - the text values of the module's settings

        variable_revision_number - revision # of module version that saved them

        module_name / from_matlab - ignore please

        Returns upgraded setting values, revision number and matlab flag
        """
        if variable_revision_number == 1:
            # Changed from yes/no for filter to a choice
            filter_choice = \
                FILTER_CHOICE_CUSTOM if setting_values[1] == cellprofiler.setting.YES else FILTER_CHOICE_NONE
            setting_values = \
                setting_values[:1] + [filter_choice] + setting_values[2:]
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab

    def volumetric(self):
        return True


class DirectoryPredicate(cellprofiler.setting.Filter.FilterPredicate):
    """A predicate that only filters directories"""

    def __init__(self):
        subpredicates = (
            cellprofiler.setting.Filter.CONTAINS_PREDICATE,
            cellprofiler.setting.Filter.CONTAINS_REGEXP_PREDICATE,
            cellprofiler.setting.Filter.STARTS_WITH_PREDICATE,
            cellprofiler.setting.Filter.ENDSWITH_PREDICATE,
            cellprofiler.setting.Filter.EQ_PREDICATE)
        predicates = [cellprofiler.setting.Filter.DoesPredicate(subpredicates),
                      cellprofiler.setting.Filter.DoesNotPredicate(subpredicates)]
        cellprofiler.setting.Filter.FilterPredicate.__init__(self,
                                                             'directory', "Directory", self.fn_filter,
                                                             predicates, doc="Apply the rule to directories")

    def fn_filter(self, node_type__modpath__module, *args):
        """The DirectoryPredicate filter function

        The arg slot expects a tuple of node_type and modpath.
        The predicate returns None (= agnostic about filtering) if
        the node is not a directory, otherwise it composites the
        modpath into a file path and applies it to the rest of
        the args.
        """
        (node_type, modpath, module) = node_type__modpath__module
        if isinstance(modpath[-1], tuple) and len(modpath[-1]) == 3:
            path = os.path.join(*modpath[:-2])
        else:
            path = os.path.join(*modpath[:-1])
        return args[0](path, *args[1:])

    def test_valid(self, pipeline, *args):
        self((cellprofiler.setting.FileCollectionDisplay.NODE_FILE,
              ["/imaging", "image.tif"], None), *args)


class FilePredicate(cellprofiler.setting.Filter.FilterPredicate):
    """A predicate that only filters files"""

    def __init__(self):
        subpredicates = (
            cellprofiler.setting.Filter.CONTAINS_PREDICATE,
            cellprofiler.setting.Filter.CONTAINS_REGEXP_PREDICATE,
            cellprofiler.setting.Filter.STARTS_WITH_PREDICATE,
            cellprofiler.setting.Filter.ENDSWITH_PREDICATE,
            cellprofiler.setting.Filter.EQ_PREDICATE)
        predicates = [cellprofiler.setting.Filter.DoesPredicate(subpredicates),
                      cellprofiler.setting.Filter.DoesNotPredicate(subpredicates)]
        cellprofiler.setting.Filter.FilterPredicate.__init__(self,
                                                             'file', "File", self.fn_filter, predicates,
                                                             doc="Apply the rule to files")

    def fn_filter(self, node_type__modpath__module, *args):
        """The FilePredicate filter function

        The arg slot expects a tuple of node_type and modpath.
        The predicate returns None (= agnostic about filtering) if
        the node is not a directory, otherwise it composites the
        modpath into a file path and applies it to the rest of
        the args
        """
        (node_type, modpath, module) = node_type__modpath__module
        if node_type == cellprofiler.setting.FileCollectionDisplay.NODE_DIRECTORY:
            return None
        elif isinstance(modpath[-1], tuple) and len(modpath[-1]) == 3:
            filename = modpath[-2]
        else:
            filename = modpath[-1]
        return args[0](filename, *args[1:])

    def test_valid(self, pipeline, *args):
        self((cellprofiler.setting.FileCollectionDisplay.NODE_FILE,
              ["/imaging", "test.tif"], None), *args)


def is_image_extension(suffix):
    """Return True if the extension is one of those recongized by bioformats"""
    extensions = javabridge.get_collection_wrapper(
        javabridge.static_call("org/cellprofiler/imageset/filter/IsImagePredicate",
                               "getImageSuffixes", "()Ljava/util/Set;"))
    return extensions.contains(suffix.lower())


class ExtensionPredicate(cellprofiler.setting.Filter.FilterPredicate):
    """A predicate that operates on file extensions"""
    IS_TIF_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
        "istif", '"tif", "tiff", "ome.tif" or "ome.tiff"',
        lambda x: x.lower() in ("tif", "tiff", "ome.tif", "ome.tiff"), [],
        doc="The extension is associated with TIFF image files")
    IS_JPEG_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
        "isjpeg", '"jpg" or "jpeg"',
        lambda x: x.lower() in ("jpg", "jpeg"), [],
        doc="The extension is associated with JPEG image files")
    IS_PNG_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
        "ispng", '"png"',
        lambda x: x.lower() == "png", [],
        doc="The extension is associated with PNG image files")
    IS_IMAGE_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
        'isimage', 'the extension of an image file',
        is_image_extension, [],
        'Is an extension commonly associated with image files')
    IS_FLEX_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
        'isflex', '"flex"',
        lambda x: x.lower() == "flex", [],
        doc="The extension is associated with .flex files")
    IS_MOVIE_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
        "ismovie", '"mov" or "avi"',
        lambda x: x.lower() in ("mov", "avi"), [],
        doc="The extension is associated with movie files")

    def __init__(self):
        subpredicates = (
            self.IS_TIF_PREDICATE,
            self.IS_JPEG_PREDICATE,
            self.IS_PNG_PREDICATE,
            self.IS_IMAGE_PREDICATE,
            self.IS_FLEX_PREDICATE,
            self.IS_MOVIE_PREDICATE)
        predicates = [cellprofiler.setting.Filter.DoesPredicate(subpredicates, "Is"),
                      cellprofiler.setting.Filter.DoesNotPredicate(subpredicates, "Is not")]
        cellprofiler.setting.Filter.FilterPredicate.__init__(self,
                                                             'extension', "Extension", self.fn_filter, predicates,
                                                             doc="The rule applies to the file extension")

    def fn_filter(self, node_type__modpath__module, *args):
        """The ExtensionPredicate filter function

        If the element is a file, try the different predicates on
        all possible extension parsings.
        """
        (node_type, modpath, module) = node_type__modpath__module
        if node_type == cellprofiler.setting.FileCollectionDisplay.NODE_DIRECTORY:
            return None
        elif isinstance(modpath[-1], tuple) and len(modpath[-1]) == 3:
            filename = modpath[-2]
        else:
            filename = modpath[-1]
        exts = []
        while True:
            filename, ext = os.path.splitext(filename)
            if len(filename) == 0 or len(ext) == 0:
                return False
            exts.insert(0, ext[1:])
            ext = '.'.join(exts)
            if args[0](ext, *args[1:]):
                return True

    def test_valid(self, pipeline, *args):
        self((cellprofiler.setting.FileCollectionDisplay.NODE_FILE,
              ["/imaging", "test.tif"], None), *args)


class ImagePredicate(cellprofiler.setting.Filter.FilterPredicate):
    """A predicate that applies subpredicates to image plane details"""
    IS_COLOR_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
        "iscolor", "Color",
        lambda x: (
                cellprofiler.pipeline.ImagePlaneDetails.MD_COLOR_FORMAT in x.metadata and
                x.metadata[cellprofiler.pipeline.ImagePlaneDetails.MD_COLOR_FORMAT] ==
                cellprofiler.pipeline.ImagePlaneDetails.MD_RGB), [],
        doc="The image is an interleaved color image (for example, a PNG image)")

    IS_MONOCHROME_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
        "ismonochrome", "Monochrome",
        lambda x: (
                cellprofiler.pipeline.ImagePlaneDetails.MD_COLOR_FORMAT in x.metadata and
                x.metadata[cellprofiler.pipeline.ImagePlaneDetails.MD_COLOR_FORMAT] ==
                cellprofiler.pipeline.ImagePlaneDetails.MD_MONOCHROME), [],
        doc="The image is monochrome")

    @staticmethod
    def is_stack(x):
        if (cellprofiler.pipeline.ImagePlaneDetails.MD_SIZE_T in x.metadata and
                x.metadata[cellprofiler.pipeline.ImagePlaneDetails.MD_SIZE_T] > 1):
            return True
        if (cellprofiler.pipeline.ImagePlaneDetails.MD_SIZE_Z in x.metadata and
                x.metadata[cellprofiler.pipeline.ImagePlaneDetails.MD_SIZE_Z] > 1):
            return True
        return False

    IS_STACK_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
        "isstack", "Stack", lambda x: ImagePredicate.is_stack(x), [],
        doc="The image is a Z-stack or movie")

    IS_STACK_FRAME_PREDICATE = cellprofiler.setting.Filter.FilterPredicate(
        "isstackframe", "Stack frame", lambda x: x.index is not None, [],
        doc="The image is a frame of a movie or a plane of a Z-stack")

    def __init__(self):
        subpredicates = (self.IS_COLOR_PREDICATE,
                         self.IS_MONOCHROME_PREDICATE,
                         self.IS_STACK_PREDICATE,
                         self.IS_STACK_FRAME_PREDICATE)
        predicates = [pred_class(subpredicates, text)
                      for pred_class, text in (
                          (cellprofiler.setting.Filter.DoesPredicate, "Is"),
                          (cellprofiler.setting.Filter.DoesNotPredicate, "Is not"))]
        cellprofiler.setting.Filter.FilterPredicate.__init__(
            self,
            'image',
            "Image",
            self.fn_filter,
            predicates,
            doc="Filter based on image characteristics"
        )

    def fn_filter(self, node_type__modpath__module, *args):
        (node_type, modpath, module) = node_type__modpath__module
        if node_type == cellprofiler.setting.FileCollectionDisplay.NODE_DIRECTORY:
            return None
        ipd = module.get_image_plane_details(modpath)
        if ipd is None:
            return None
        return args[0](ipd, *args[1:])

    class FakeModule(cellprofiler.module.Module):
        """A fake module for setting validation"""

        def get_image_plane_details(self, modpath):
            url = Images.modpath_to_url(modpath)
            return cellprofiler.pipeline.ImagePlaneDetails(url)

    def test_valid(self, pipeline, *args):
        self((cellprofiler.setting.FileCollectionDisplay.NODE_FILE,
              ["/imaging", "test.tif"], self.FakeModule()), *args)
