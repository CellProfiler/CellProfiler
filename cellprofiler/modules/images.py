import cellprofiler.gui.help
import cellprofiler.icons
import cellprofiler.module
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.setting
import cellprofiler.utilities.hdf5_dict
import cellprofiler.utilities.url
import cellprofiler.workspace
import javabridge
import cellprofiler.utilities.predicate

__doc__ = """
The <b>Images</b> module specifies the location of image files to be analyzed by your pipeline.
<hr>
The <b>Images</b> module allows you to specify the location of files to be analyzed by the pipeline;
setting this module correctly is the first step in creating a new project in CellProfiler.
These files can be located on your hard drive, on a networked computer elsewhere,
or accessible with a URL. You can also provide rules to specify only those files that you want
analyzed out of a larger collection (for example, from a folder containing both images for
analysis and non-image files that should be disregarded).

<h4>What is a "digital image"?</h4>
A <i>digital image</i> is a set of numbers arranged into a two-dimensional format of rows and columns;
a pixel refers to the row/column location of a particular point in the image. Pixels in grayscale or monochrome
(black/white) images contain a single intensity value, whereas in color images, each pixel contains a red,
green, and blue (RGB) triplet of intensity values. Additionally, the term image can be used as short-hand
for an image sequence, that is, an image collection such as a time-lapse series (2-D + <i>t</i>), confocal Z-stacks
(3-D), etc.

<p>CellProfiler can read a wide variety of image formats by using a library called Bio-Formats;
see <a href="http://www.openmicroscopy.org/site/support/bio-formats5/supported-formats.html">here</a>
for the formats available. Some image formats are better than others for use in image analysis. Some are
<a href="http://www.techterms.com/definition/lossy">"lossy"</a> (information is lost in the conversion
to the format) like most JPG/JPEG files; others are
<a href="http://www.techterms.com/definition/lossless">"lossless"</a> (no image information is lost).
For image analysis purposes, a lossless format like TIF or PNG is recommended.</p>

<h4>What do I need as input?</h4>
The most straightforward way to provide image files to the <b>Images</b> module is to simply drag-and-drop
them on the file list panel (the blank space indicated by the text "Drop files and folders here").
<table cellpadding="0" width="100%%">
<tr align="center"><td><img src="memory:{images_filelist_blank}"></td></tr>
</table>

<p>Using the file explorer tool of your choice (e.g., Explorer in Windows, Finder in Mac), you can drag-and-drop
individual files and/or entire folders into this panel. You can also right-click in the File list panel to
bring up a file selection window to browse for individual files; on the Mac, folders can be drag-and-dropped
from this window and you can select multiple files using Ctrl-A (Windows) or Cmd-A (Mac).
<table cellpadding="0" width="100%%">
<tr align="center"><td><img src="memory:{images_filelist_filled}"></td></tr>
</table>
Right-clicking on the file list panel will provide a context menu with options to modify the file list:
<ul>
<li><i>Show Selected Image:</i> Selecting this option (or double-clicking on the file) will open the image
in a new window.</li>
<li><i>Remove From List:</i> Removes the selected file or folder from the list. Note that this does not remove
the file/folder from the hard drive.</li>
<li><i>Remove Unavailable Files:</i> Refresh the list by checking for existence of file. Note that this does not remove
the file from the hard drive.</li>
<li><i>Browse For Images:</i> Use a dialog box to select an image file (though drag-and-drop is recommended).</li>
<li><i>Refresh:</i> Shown only if folder is selected. Refresh the list of files from the folder. Files that were
manually removed from the list for that folder are restored.</li>
<li><i>Expand All Folders:</i> Expand all trees shown in the file list panel.</li>
<li><i>Collapse All Folders:</i> Collapse all folder trees shown in the file list panel.</li>
<li><i>Clear File List:</i> Remove all files/folders in the file list panel. You will be prompted for
confirmation beforehand.</li>
</ul></p>

<h4>What do the settings mean?</h4>
If you have a subset of files that you want to analyze from the full listing shown in the
panel, you can filter the files according to a set of rules. This is useful in cases such as:
<ul>
<li>You have dragged a folder of images onto the file list panel, but the folder contains images
you want to analyze along with non-image files that you want to disregard.</li>
<li>You have dragged a folder of images onto the file list panel, but the folder contains the images
from one experiment that you want to process along with images from another experiment that you
want to ignore for now. </li>
</ul>
You may specify as many rules as necessary to define the desired list of images.

<p>After you have filtered the file list, press the "Apply" button to update the view of the
file list. You can also toggle the "Show file excluded by filters" box to modify the display of the files:
<ul>
<li>Checking this box will show all the files in the list, with the files that have been filtered out
shown as grayed-out entries.</li>
<li>Not checking this box will only show the files in the list that pass the filter(s).</li>
</ul></p>

<h4>What do I get as output?</h4>
The final product of the <b>Images</b> module is a file list in which any files that are not intended for
further processing have been removed, whether manually or using filtering. This list will be used when
collecting metadata (if desired) and when assembling the image sets in NamesAndTypes. The list can be
filtered further in NamesAndTypes to specify, for example, that a subset of these images represents a
particular wavelength.
""".format(**{
    'images_filelist_blank': cellprofiler.gui.help.IMAGES_FILELIST_BLANK,
    'images_filelist_filled': cellprofiler.gui.help.IMAGES_FILELIST_FILLED
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
        predicates = [cellprofiler.utilities.predicate.FilePredicate(),
                      cellprofiler.utilities.predicate.DirectoryPredicate(),
                      cellprofiler.utilities.predicate.ExtensionPredicate()]

        self.filter_choice = cellprofiler.setting.Choice(
                "Filter images?", FILTER_CHOICE_ALL, value=FILTER_CHOICE_IMAGES,
                doc="""
            The <b>Images</b> module will pass all the files specified in the file list
            panel downstream to have a meaningful name assigned to it (so other modules can
            access it) or optionally, to define the relationships between images and associated
            metadata. Enabling file filtering will allow you to specify a subset of the files
            from the file list panel by defining rules to filter the files. This approach is
            useful if, for example, you drag-and-dropped a folder onto the file list panel
            which contains a mixture of images that you want to analyze and other files that
            you want to ignore.
            <p>Several options are available for this setting:
            <ul>
            <li><i>{filter_choice_none}:</i> Do not enable filtering; all files in the File list
            panel will be passed to downstream modules for processing. This option can be
            selected if you are sure that only images are specified in the list.</li>
            <li><i>{filter_choice_images}:</i> Only image files will be passed to downstream
            modules. The permissible image formats are provided by a library called Bio-Formats; see
            <a href="http://www.openmicroscopy.org/site/support/bio-formats5/supported-formats.html">here</a> for the formats available.</li>
            <li><i>{filter_choice_custom}:</i> Specify custom rules for selecting a subset of
            the files from the File list panel. This approach is useful if, for example, you
            drag-and-dropped a folder onto the File list panel which contains a mixture of images
            that you want to analyze and other files that you want to ignore.</li>
            </ul></p>""".format(**{
                    'filter_choice_none': FILTER_CHOICE_NONE,
                    'filter_choice_images': FILTER_CHOICE_IMAGES,
                    'filter_choice_custom': FILTER_CHOICE_CUSTOM
                }))

        self.filter = cellprofiler.setting.Filter("Select the rule criteria", predicates,
                                                  FILTER_DEFAULT, doc="""
            Specify a set of rules to narrow down the files to be analyzed.
            <p>{filter_rules_buttons_help}</p>""".format(**{
                'filter_rules_buttons_help': cellprofiler.gui.help.FILTER_RULES_BUTTONS_HELP
            }))

        self.update_button = cellprofiler.setting.PathListRefreshButton(
                "", "Apply filters to the file list", doc="""
            <i>(Only displayed if filtering based on rules)</i><br>
            Re-display the file list, removing or graying out the files
            that do not pass the current filter.
            """)

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
                if isinstance(url, unicode):
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
                FILTER_CHOICE_CUSTOM if setting_values[1] == cellprofiler.setting.YES else \
                    FILTER_CHOICE_NONE
            setting_values = \
                setting_values[:1] + [filter_choice] + setting_values[2:]
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab
