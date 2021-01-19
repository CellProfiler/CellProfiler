# coding=utf-8

import os
import urllib.parse
import urllib.request

import javabridge

from ..constants.module import FILTER_RULES_BUTTONS_HELP
from ..constants.modules.images import FILTER_CHOICE_ALL
from ..constants.modules.images import FILTER_CHOICE_CUSTOM
from ..constants.modules.images import FILTER_CHOICE_IMAGES
from ..constants.modules.images import FILTER_CHOICE_NONE
from ..constants.modules.images import FILTER_DEFAULT
from ..module import Module
from ..setting import FileCollectionDisplay
from ..setting import PathListDisplay
from ..setting.choice import Choice
from ..setting.do_something import PathListRefreshButton
from ..setting.filter import DirectoryPredicate
from ..setting.filter import ExtensionPredicate
from ..setting.filter import FilePredicate
from ..setting.filter import Filter
from ..utilities.hdf5_dict import HDF5FileList
from ..utilities.image import image_resource
from ..utilities.pathname import pathname2url

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
""".format(
    **{
        "IMG_PANEL_BLANK": image_resource("Images_FilelistPanel_Blank.png"),
        "IMG_PANEL_FILLED": image_resource("Images_FilelistPanel_Filled.png"),
    }
)


class Images(Module):
    variable_revision_number = 2
    module_name = "Images"
    category = "File Processing"

    MI_SHOW_IMAGE = "Show image"
    MI_REMOVE = FileCollectionDisplay.DeleteMenuItem("Remove from list")
    MI_REFRESH = "Refresh"


    def create_settings(self):
        self.workspace = None
        module_explanation = [
            "To begin creating your project, use the %s module to compile"
            % self.module_name,
            "a list of files and/or folders that you want to analyze. You can also specify a set of rules",
            "to include only the desired files in your selected folders.",
        ]
        self.set_notes([" ".join(module_explanation)])

        self.path_list_display = PathListDisplay()
        predicates = [FilePredicate(), DirectoryPredicate(), ExtensionPredicate()]

        self.filter_choice = Choice(
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
""".format(
                **{
                    "FILTER_CHOICE_CUSTOM": FILTER_CHOICE_CUSTOM,
                    "FILTER_CHOICE_IMAGES": FILTER_CHOICE_IMAGES,
                    "FILTER_CHOICE_NONE": FILTER_CHOICE_NONE,
                }
            ),
        )

        self.filter = Filter(
            "Select the rule criteria",
            predicates,
            FILTER_DEFAULT,
            doc="""\
Specify a set of rules to narrow down the files to be analyzed.

{FILTER_RULES_BUTTONS_HELP}
""".format(
                **{"FILTER_RULES_BUTTONS_HELP": FILTER_RULES_BUTTONS_HELP}
            ),
        )

        self.update_button = PathListRefreshButton(
            "Apply filters to the file list",
            "Apply filters to the file list",
            doc="""\
*(Only displayed if filtering based on rules)*

Re-display the file list, removing or graying out the files that do not
pass the current filter.
""",
        )

    def help_settings(self):
        return [self.filter_choice, self.filter, self.update_button]

    @staticmethod
    def modpath_to_url(modpath):
        if modpath[0] in ("http", "https", "ftp", "s3"):
            if len(modpath) == 1:
                return modpath[0] + ":"
            elif len(modpath) == 2:
                return modpath[0] + ":" + modpath[1]
            else:
                return (
                    modpath[0]
                    + ":"
                    + modpath[1]
                    + "/"
                    + "/".join([urllib.parse.quote(part) for part in modpath[2:]])
                )
        path = os.path.join(*modpath)
        return pathname2url(path)

    @staticmethod
    def url_to_modpath(url):
        if not url.lower().startswith("file:"):
            schema, rest = HDF5FileList.split_url(url)
            return (
                [schema] + rest[0:1] + [urllib.parse.unquote(part) for part in rest[1:]]
            )
        path = urllib.request.url2pathname(url[5:])
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
                expression,
                ifcls,
            )
            file_array = env.make_object_array(len(file_list), scls)
            for i, url in enumerate(file_list):
                if url.startswith("s3:"):
                    url = url.replace(" ", "+")

                if isinstance(url, str):
                    ourl = env.new_string(url)
                else:
                    ourl = env.new_string_utf(url)
                env.set_object_array_element(file_array, i, ourl)
            passes_filter = javabridge.call(
                iffilter, "filterURLs", "([Ljava/lang/String;)[Z", file_array
            )
            if isinstance(passes_filter, javabridge.JB_Object):
                passes_filter = javabridge.get_env().get_boolean_array_elements(
                    passes_filter
                )
            file_list = [f for f, passes in zip(file_list, passes_filter) if passes]
        workspace.pipeline.set_filtered_file_list(file_list, self)
        return True

    def run(self, workspace):
        pass

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        """Upgrade pipeline settings from a previous revision

        setting_values - the text values of the module's settings

        variable_revision_number - revision # of module version that saved them

        Returns upgraded setting values, revision number and matlab flag
        """
        if variable_revision_number == 1:
            # Changed from yes/no for filter to a choice
            filter_choice = (
                FILTER_CHOICE_CUSTOM
                if setting_values[1] == "Yes"
                else FILTER_CHOICE_NONE
            )
            setting_values = setting_values[:1] + [filter_choice] + setting_values[2:]
            variable_revision_number = 2
        return setting_values, variable_revision_number

    def volumetric(self):
        return True
