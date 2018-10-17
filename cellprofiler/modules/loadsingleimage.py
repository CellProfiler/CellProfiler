# coding=utf-8

"""
LoadSingleImage
===============

**LoadSingleImage** loads a single image for use in all image cycles.

This module tells CellProfiler where to retrieve a single image and
gives the image a meaningful name by which the other modules can access
it. The module executes only the first time through the pipeline;
thereafter the image is accessible to all subsequent processing cycles.
This is particularly useful for loading an image like an illumination
correction image for use by the **CorrectIlluminationApply** module,
when that single image will be used to correct all images in the
analysis run.

*Disclaimer:* Please note that the Input modules (i.e., **Images**,
**Metadata**, **NamesAndTypes** and **Groups**) largely supersedes this
module. However, old pipelines loaded into CellProfiler that contain
this module will provide the option of preserving them; these pipelines
will operate exactly as before.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

See also
^^^^^^^^

See also the **Input** modules (**Images**, **NamesAndTypes**,
**MetaData**, **Groups**), **LoadImages**, and **LoadData**.

Measurements made by this module
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

-  *Pathname, Filename:* The full path and the filename of each image.
-  *Metadata:* The metadata information extracted from the path and/or
   filename, if requested.
-  *Scaling:* The maximum possible intensity value for the image format.
-  *Height, Width:* The height and width of images loaded by this module.

Technical notes
^^^^^^^^^^^^^^^

For most purposes, you will probably want to use the **LoadImages**
module, not **LoadSingleImage**. The reason is that **LoadSingleImage**
does not actually create image sets (or even a single image set).
Instead, it adds the single image to every image cycle for an *already
existing* image set. Hence **LoadSingleImage** should never be used as
the only image-loading module in a pipeline; attempting to do so will
display a warning message in the module settings.

If you have a single file to load in the pipeline (and only that file),
you will want to use **LoadImages** or **LoadData** with a single,
hardcoded file name.

"""

import hashlib
import os
import re

import centrosome.outline
import numpy as np

import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.object as cpo
import cellprofiler.object as cpo
import cellprofiler.preferences as cpprefs
import cellprofiler.setting as cps
from cellprofiler.modules._help import USING_METADATA_HELP_REF, USING_METADATA_TAGS_REF, IO_FOLDER_CHOICE_HELP_TEXT, \
    IO_WITH_METADATA_HELP_TEXT
from cellprofiler.preferences import standardize_default_folder_names, \
    DEFAULT_INPUT_FOLDER_NAME, DEFAULT_OUTPUT_FOLDER_NAME
from cellprofiler.setting import YES, NO
from cellprofiler.measurement import C_LOCATION, C_NUMBER, C_COUNT, FTR_CENTER_X, FTR_CENTER_Y, FTR_OBJECT_NUMBER
from cellprofiler.modules.identify import add_object_count_measurements, add_object_location_measurements
from cellprofiler.modules.identify import get_object_measurement_columns
from cellprofiler.modules.loadimages import C_HEIGHT, C_WIDTH, C_MD5_DIGEST, IO_IMAGES, IO_OBJECTS, IO_ALL
from cellprofiler.measurement import C_OBJECTS_FILE_NAME, C_OBJECTS_URL, C_PATH_NAME, C_URL,\
    C_OBJECTS_PATH_NAME, C_FILE_NAME
from cellprofiler.modules.loadimages import IMAGE_FOR_OBJECTS_F
from cellprofiler.modules.loadimages import IO_IMAGES, IO_OBJECTS, IO_ALL
from cellprofiler.modules.loadimages import LoadImagesImageProvider, C_SCALING
from cellprofiler.modules.loadimages import convert_image_to_objects, pathname2url
from cellprofiler.modules import images

DIR_CUSTOM_FOLDER = "Custom folder"
DIR_CUSTOM_WITH_METADATA = "Custom with metadata"

FILE_TEXT = "Filename of the image to load (Include the extension, e.g., .tif)"
URL_TEXT = "URL of the image to load (Include the extension, e.g., .tif)"

S_FIXED_SETTINGS_COUNT_V5 = 1
S_FIXED_SETTINGS_COUNT = 1
S_FILE_SETTINGS_COUNT_V4 = 3
S_FILE_SETTINGS_COUNT_V5 = 7
S_FILE_SETTINGS_COUNT = 7
S_FILE_NAME_OFFSET_V4 = 0
S_IMAGE_NAME_OFFSET_V4 = 1
S_RESCALE_OFFSET_V4 = 2


class LoadSingleImage(cpm.Module):
    module_name = "LoadSingleImage"
    category = "File Processing"
    variable_revision_number = 5

    def create_settings(self):
        """Create the settings during initialization

        """
        self.directory = cps.DirectoryPath(
            "Input image file location",
            support_urls=True,
            doc="""\
Choose the folder containing the image(s) to be loaded. Generally, it is
best to store the image you want to load in either the Default Input or
Output Folder, so that the correct image is loaded into the pipeline and
typos are avoided.

{IO_FOLDER_CHOICE_HELP_TEXT}

{IO_WITH_METADATA_HELP_TEXT}
""".format(**{
                "IO_FOLDER_CHOICE_HELP_TEXT": IO_FOLDER_CHOICE_HELP_TEXT,
                "IO_WITH_METADATA_HELP_TEXT": IO_WITH_METADATA_HELP_TEXT
            })
        )

        self.file_settings = []
        self.add_file(can_remove=False)
        self.add_button = cps.DoSomething("", "Add another image", self.add_file)

    def add_file(self, can_remove=True):
        """Add settings for another file to the list"""
        group = cps.SettingsGroup()
        if can_remove:
            group.append("divider", cps.Divider(line=False))

        def get_directory_fn():
            return self.directory.get_absolute_path()

        group.append(
            "file_name",
            cps.FilenameText(
                FILE_TEXT,
                cps.NONE,
                metadata=True,
                get_directory_fn=get_directory_fn,
                exts=[("TIF - Tagged Image File format (*.tif,*.tiff)", "*.tif;*.tiff"),
                      ("PNG - Portable Network Graphics (*.png)", "*.png"),
                      ("JPG/JPEG file (*.jpg,*.jpeg)", "*.jpg,*.jpeg"),
                      ("BMP - Windows Bitmap (*.bmp)", "*.bmp"),
                      ("Compuserve GIF file (*.gif)", "*.gif"),
                      ("MATLAB image (*.mat)", "*.mat"),
                      ("NumPy array (*.npy)", "*.npy"),
                      ("All files (*.*)", "*.*")],
                doc="""\
The filename can be constructed in one of two ways:

-  As a fixed filename (e.g., *Exp1\_D03f00d0.tif*).
-  Using the metadata associated with an image set in **LoadImages** or
   **LoadData**. This is especially useful if you want your output given
   a unique label according to the metadata corresponding to an image
   group. The name of the metadata to substitute is included in a
   special tag format embedded in your file specification.

{USING_METADATA_TAGS_REF}

{USING_METADATA_HELP_REF}

Keep in mind that in either case, the image file extension, if any, must
be included.
""".format(**{
                    "USING_METADATA_TAGS_REF": USING_METADATA_TAGS_REF,
                    "USING_METADATA_HELP_REF": USING_METADATA_HELP_REF
                })
            )
        )

        group.append(
            "image_objects_choice",
            cps.Choice(
                'Load as images or objects?',
                IO_ALL,
                doc="""\
This setting determines whether you load an image as image data or as
segmentation results (i.e., objects):

-  *{IO_IMAGES}:* The input image will be given the name you specify, by
   which it will be referred downstream. This is the most common usage
   for this module.
-  *{IO_OBJECTS}:* Use this option if the input image is a label matrix
   and you want to obtain the objects that it defines. A *label matrix*
   is a grayscale or color image in which the connected regions share
   the same label, and defines how objects are represented in
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
            cps.FileImageNameProvider(
                "Name the image that will be loaded",
                "OrigBlue",
                doc="""\
*(Used only if an image is output)*

Enter the name of the image that will be loaded. You can use this name
to select the image in downstream modules.
"""
            )
        )

        group.append(
            "rescale", cps.Binary(
                "Rescale intensities?",
                True,
                doc="""\
*(Used only if an image is output)*

This option determines whether image metadata should be used to
rescale the image’s intensities. Some image formats save the maximum
possible intensity value along with the pixel data. For instance, a
microscope might acquire images using a 12-bit A/D converter which
outputs intensity values between zero and 4095, but stores the values
in a field that can take values up to 65535.

Select *{YES}* to rescale the image intensity so that saturated values
are rescaled to 1.0 by dividing all pixels in the image by the maximum
possible intensity value.

Select *{NO}* to ignore the image metadata and rescale the image to 0 –
1.0 by dividing by 255 or 65535, depending on the number of bits used to
store the image.
""".format(**{
                    "NO": NO,
                    "YES": YES
                })
            )
        )

        group.append("objects_name", cps.ObjectNameProvider(
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
            cps.Binary(
                "Retain outlines of loaded objects?",
                False,
                doc="""\
*(Used only if objects are output)*

Select *{YES}* if you want to save an image of the outlines of the
loaded objects.
""".format(**{
                    "YES": YES
                })
            )
        )

        group.append(
            "outlines_name",
            cps.OutlineNameProvider(
                'Name the outlines',
                'NucleiOutlines',
                doc="""\
*(Used only if objects are output)*

Enter a name that will allow the outlines to be selected later in the
pipeline.
"""
            )
        )

        if can_remove:
            group.append("remove", cps.RemoveSettingButton("", "Remove this image", self.file_settings, group))
        self.file_settings.append(group)

    def settings(self):
        """Return the settings in the order in which they appear in a pipeline file"""
        result = [self.directory]
        for file_setting in self.file_settings:
            result += [file_setting.file_name, file_setting.image_objects_choice,
                       file_setting.image_name, file_setting.objects_name,
                       file_setting.wants_outlines, file_setting.outlines_name,
                       file_setting.rescale]
        return result

    def help_settings(self):
        result = [self.directory]
        image_group = self.file_settings[0]
        result += [image_group.file_name,
                   image_group.image_objects_choice,
                   image_group.image_name,
                   image_group.rescale,
                   image_group.objects_name,
                   image_group.wants_outlines,
                   image_group.outlines_name]
        return result

    def visible_settings(self):
        result = [self.directory]
        for file_setting in self.file_settings:
            url_based = (self.directory.dir_choice == cps.URL_FOLDER_NAME)
            file_setting.file_name.set_browsable(not url_based)
            file_setting.file_name.text = URL_TEXT if url_based else FILE_TEXT
            result += [
                file_setting.file_name, file_setting.image_objects_choice]
            if file_setting.image_objects_choice == IO_IMAGES:
                result += [file_setting.image_name, file_setting.rescale]
            else:
                result += [file_setting.objects_name, file_setting.wants_outlines]
                if file_setting.wants_outlines:
                    result += [file_setting.outlines_name]
            if hasattr(file_setting, "remove"):
                result += [file_setting.remove]
        result.append(self.add_button)
        return result

    def prepare_settings(self, setting_values):
        """Adjust the file_settings depending on how many files there are"""
        count = ((len(setting_values) - S_FIXED_SETTINGS_COUNT) /
                 S_FILE_SETTINGS_COUNT)
        del self.file_settings[count:]
        while len(self.file_settings) < count:
            self.add_file()

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        '''Prepare to create a batch file

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
        '''
        self.directory.alter_for_create_batch_files(fn_alter_path)
        return True

    def get_base_directory(self, workspace):
        return self.directory.get_absolute_path(workspace.measurements)

    def get_file_names(self, workspace, image_set_number=None):
        """Get the files for the current image set

        workspace - workspace for current image set

        returns a dictionary of image_name keys and file path values
        """
        result = {}
        for file_setting in self.file_settings:
            file_pattern = file_setting.file_name.value
            file_name = workspace.measurements.apply_metadata(file_pattern,
                                                              image_set_number)
            if file_setting.image_objects_choice == IO_IMAGES:
                image_name = file_setting.image_name.value
            else:
                image_name = file_setting.objects_name.value
            result[image_name] = file_name

        return result

    def get_file_settings(self, image_name):
        '''Get the file settings associated with a given image name'''
        for file_setting in self.file_settings:
            if (file_setting.image_objects_choice == IO_IMAGES and
                        file_setting.image_name == image_name):
                return file_setting
            if (file_setting.image_objects_choice == IO_OBJECTS and
                        file_setting.objects_name == image_name):
                return file_setting
        return None

    def file_wants_images(self, file_setting):
        '''True if the file_setting produces images, false if it produces objects'''
        return file_setting.image_objects_choice == IO_IMAGES

    def is_load_module(self):
        return True

    def prepare_run(self, workspace):
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        root = self.get_base_directory(workspace)

        if m.image_set_count == 0:
            # Oh bad - using LoadSingleImage to load one image and process it
            image_numbers = [1]
        else:
            image_numbers = m.get_image_numbers()

        for image_number in image_numbers:
            dict = self.get_file_names(workspace, image_set_number=image_number)
            for image_name in dict.keys():
                file_settings = self.get_file_settings(image_name)
                if file_settings.image_objects_choice == IO_IMAGES:
                    #
                    # Add measurements
                    #
                    path_name_category = C_PATH_NAME
                    file_name_category = C_FILE_NAME
                    url_category = C_URL
                else:
                    #
                    # Add measurements
                    #
                    path_name_category = C_OBJECTS_PATH_NAME
                    file_name_category = C_OBJECTS_FILE_NAME
                    url_category = C_OBJECTS_URL

                url = pathname2url(os.path.join(root, dict[image_name]))
                for category, value in (
                        (path_name_category, root),
                        (file_name_category, dict[image_name]),
                        (url_category, url)):
                    measurement_name = "_".join((category, image_name))
                    m.add_measurement(cpmeas.IMAGE, measurement_name, value,
                                      image_set_number=image_number)
        return True

    def run(self, workspace):
        statistics = []
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        #
        # Hack: if LoadSingleImage is first, no paths are populated
        #
        if self.file_wants_images(self.file_settings[0]):
            m_path = "_".join((C_PATH_NAME,
                               self.file_settings[0].image_name.value))
        else:
            m_path = "_".join((C_OBJECTS_PATH_NAME,
                               self.file_settings[0].objects_name.value))
        if m.get_current_image_measurement(m_path) is None:
            self.prepare_run(workspace)

        image_set = workspace.image_set
        for file_setting in self.file_settings:
            wants_images = self.file_wants_images(file_setting)
            image_name = file_setting.image_name.value if wants_images else \
                file_setting.objects_name.value
            m_path, m_file, m_md5_digest, m_scaling, m_height, m_width = [
                "_".join((c, image_name)) for c in (
                    C_PATH_NAME if wants_images else C_OBJECTS_PATH_NAME,
                    C_FILE_NAME if wants_images else C_OBJECTS_FILE_NAME,
                    C_MD5_DIGEST, C_SCALING, C_HEIGHT, C_WIDTH)]
            pathname = m.get_current_image_measurement(m_path)
            filename = m.get_current_image_measurement(m_file)
            rescale = (wants_images and file_setting.rescale.value)

            provider = LoadImagesImageProvider(
                    image_name, pathname, filename, rescale)
            image = provider.provide_image(image_set)
            pixel_data = image.pixel_data
            if wants_images:
                md5 = provider.get_md5_hash(m)
                m.add_image_measurement("_".join((C_MD5_DIGEST, image_name)),
                                        md5)
                m.add_image_measurement("_".join((C_SCALING, image_name)),
                                        image.scale)
                m.add_image_measurement("_".join((C_HEIGHT, image_name)),
                                        int(pixel_data.shape[0]))
                m.add_image_measurement("_".join((C_WIDTH, image_name)),
                                        int(pixel_data.shape[1]))
                image_set.providers.append(provider)
            else:
                #
                # Turn image into objects
                #
                labels = convert_image_to_objects(pixel_data)
                objects = cpo.Objects()
                objects.segmented = labels
                object_set = workspace.object_set
                assert isinstance(object_set, cpo.ObjectSet)
                object_set.add_objects(objects, image_name)
                add_object_count_measurements(m, image_name, objects.count)
                add_object_location_measurements(m, image_name, labels)
                #
                # Add outlines if appropriate
                #
                if file_setting.wants_outlines:
                    outlines = centrosome.outline.outline(labels)
                    outline_image = cpi.Image(outlines.astype(bool))
                    workspace.image_set.add(file_setting.outlines_name.value,
                                            outline_image)
            statistics += [(image_name, filename)]
        workspace.display_data.col_labels = ("Image name", "File")
        workspace.display_data.statistics = statistics

    def is_interactive(self):
        return False

    def display(self, workspace, figure):
        statistics = workspace.display_data.statistics
        col_labels = workspace.display_data.col_labels
        title = "Load single image: image cycle # %d" % (
            workspace.measurements.image_set_number + 1)
        figure.set_subplots((1, 1))
        figure.subplot_table(0, 0, statistics, col_labels=col_labels)

    def get_measurement_columns(self, pipeline):
        columns = []
        for file_setting in self.file_settings:
            if file_setting.image_objects_choice == IO_IMAGES:
                image_name = file_setting.image_name.value
                path_name_category = C_PATH_NAME
                file_name_category = C_FILE_NAME
                columns += [
                    (cpmeas.IMAGE, "_".join((C_MD5_DIGEST, image_name)), cpmeas.COLTYPE_VARCHAR_FORMAT % 32),
                    (cpmeas.IMAGE, "_".join((C_SCALING, image_name)), cpmeas.COLTYPE_FLOAT),
                    (cpmeas.IMAGE, "_".join((C_HEIGHT, image_name)), cpmeas.COLTYPE_INTEGER),
                    (cpmeas.IMAGE, "_".join((C_WIDTH, image_name)), cpmeas.COLTYPE_INTEGER)]
            else:
                image_name = file_setting.objects_name.value
                path_name_category = C_OBJECTS_PATH_NAME
                file_name_category = C_OBJECTS_FILE_NAME
                columns += get_object_measurement_columns(image_name)

            columns += [(cpmeas.IMAGE, '_'.join((feature, image_name)), coltype)
                        for feature, coltype in (
                            (file_name_category, cpmeas.COLTYPE_VARCHAR_FILE_NAME),
                            (path_name_category, cpmeas.COLTYPE_VARCHAR_PATH_NAME),
                        )]
        return columns

    @property
    def wants_images(self):
        '''True if any file setting loads images'''
        return any([True for file_setting in self.file_settings
                    if file_setting.image_objects_choice == IO_IMAGES])

    @property
    def wants_objects(self):
        '''True if any file setting loads objects'''
        return any([True for file_setting in self.file_settings
                    if file_setting.image_objects_choice == IO_OBJECTS])

    def get_categories(self, pipeline, object_name):
        result = []
        if object_name == cpmeas.IMAGE:
            if self.wants_images:
                result += [C_FILE_NAME, C_MD5_DIGEST, C_PATH_NAME, C_SCALING, C_HEIGHT, C_WIDTH]
            if self.wants_objects:
                result += [C_COUNT, C_OBJECTS_FILE_NAME, C_OBJECTS_PATH_NAME]
        if any([True for file_setting in self.file_settings
                if file_setting.image_objects_choice == IO_OBJECTS and
                                object_name == file_setting.objects_name]):
            result += [C_LOCATION, C_NUMBER]
        return result

    def get_measurements(self, pipeline, object_name, category):
        '''Return the measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        '''
        result = []
        if object_name == cpmeas.IMAGE:
            if category in (C_FILE_NAME, C_MD5_DIGEST, C_PATH_NAME, C_SCALING, C_HEIGHT, C_WIDTH):
                result += [file_setting.image_name.value
                           for file_setting in self.file_settings
                           if file_setting.image_objects_choice == IO_IMAGES]
            if category in (C_OBJECTS_FILE_NAME, C_OBJECTS_PATH_NAME, C_COUNT):
                result += [file_setting.objects_name.value
                           for file_setting in self.file_settings
                           if file_setting.image_objects_choice == IO_OBJECTS]
        elif any([file_setting.image_objects_choice == IO_OBJECTS and
                                  file_setting.objects_name == object_name
                  for file_setting in self.file_settings]):
            if category == C_NUMBER:
                result += [FTR_OBJECT_NUMBER]
            elif category == C_LOCATION:
                result += [FTR_CENTER_X, FTR_CENTER_Y]
        return result

    def validate_module(self, pipeline):
        # Keep users from using LoadSingleImage to define image sets
        if not any([x.is_load_module() for x in pipeline.modules()]):
            raise cps.ValidationError(
                    "LoadSingleImage cannot be used to run a pipeline on one "
                    "image file. Please use LoadImages or LoadData instead.",
                    self.directory)

        # Make sure LoadSingleImage appears after all other load modules
        after = False
        for module in pipeline.modules():
            if module is self:
                after = True
            elif after and module.is_load_module():
                raise cps.ValidationError(
                        "LoadSingleImage must appear after all other Load modules in your pipeline\n"
                        "Please move %s before LoadSingleImage" % module.module_name,
                        self.directory)

        # Make sure metadata tags exist
        for group in self.file_settings:
            text_str = group.file_name.value
            undefined_tags = pipeline.get_undefined_metadata_tags(text_str)
            if len(undefined_tags) > 0:
                raise cps.ValidationError(
                        "%s is not a defined metadata tag. Check the metadata specifications in your load modules" %
                        undefined_tags[0],
                        group.file_name)

    def validate_module_warnings(self, pipeline):
        '''Check for potentially dangerous settings'''
        # Check that user-specified names don't have bad characters
        invalid_chars_pattern = "^[A-Za-z][A-Za-z0-9_]+$"
        warning_text = "The image name has questionable characters. The pipeline can use this name " \
                       "and produce results, but downstream programs that use this data (e.g, MATLAB, MySQL) may error."
        for file_setting in self.file_settings:
            if file_setting.image_objects_choice == IO_IMAGES:
                if not re.match(invalid_chars_pattern, file_setting.image_name.value):
                    raise cps.ValidationError(warning_text, file_setting.image_name)

    def needs_conversion(self):
        return True

    def convert(self, pipeline, metadata, namesandtypes, groups):
        '''Convert from legacy to modern'''
        import cellprofiler.modules.metadata as cpmetadata
        import cellprofiler.modules.namesandtypes as cpnamesandtypes
        import cellprofiler.modules.groups as cpgroups
        assert isinstance(metadata, cpmetadata.Metadata)
        assert isinstance(namesandtypes, cpnamesandtypes.NamesAndTypes)
        assert isinstance(groups, cpgroups.Groups)

        edited_modules = set()
        for group in self.file_settings:
            tags = []
            file_name = group.file_name.value
            if group.image_objects_choice == IO_IMAGES:
                name = group.image_name.value
            else:
                name = group.objects_name.value
            loc = 0
            regexp = "^"
            while True:
                m = re.search('\\\\g[<](.+?)[>]', file_name[loc:])
                if m is None:
                    break
                tag = m.groups()[0]
                tags.append(tag)
                start = loc + m.start()
                end = loc + m.end()
                regexp += re.escape(file_name[loc:start])
                regexp += "(?P<%s>.+?)" % tag
                loc = end
                if loc == len(file_name):
                    break
            regexp += re.escape(file_name[loc:]) + "$"
            if namesandtypes.assignment_method != cpnamesandtypes.ASSIGN_RULES:
                namesandtypes.assignment_method.value = cpnamesandtypes.ASSIGN_RULES
            else:
                namesandtypes.add_assignment()
            edited_modules.add(namesandtypes)
            assignment = namesandtypes.assignments[-1]
            structure = [cps.Filter.AND_PREDICATE]
            fp = images.FilePredicate()
            fp_does, fp_does_not = [
                [d for d in fp.subpredicates if isinstance(d, c)][0]
                for c in (cps.Filter.DoesPredicate, cps.Filter.DoesNotPredicate)]
            if len(tags) == 0:
                structure.append([fp, fp_does, cps.Filter.EQ_PREDICATE,
                                  file_name])
            else:
                #
                # Unfortunately, we can't replace metadata in the file name.
                # We have to extract metadata from files that match the
                # parts of the file name outside of the metadata region. This
                # isn't what the user intended, but we do the best we can.
                #
                # Examples: file_A01.TIF, file_thumbnail.TIF will both match
                # file_(?P<well>.+?)\.TIF even though the user doesn't want
                # file_thumbnail.TIF.
                #
                metadata.notes.append(
                        "WARNING: LoadSingleImage used metadata matching. The conversion "
                        "might match files that were not matched by the legacy "
                        "module.")
                namesandtypes.notes.append(
                        ("WARNING: LoadSingleImage used metadata matching for the %s "
                         "image. The conversion might match files that were not "
                         "matched by the legacy module.") % name)
                structure.append([
                    fp, fp_does, cps.Filter.CONTAINS_REGEXP_PREDICATE, regexp])
                if not metadata.wants_metadata:
                    metadata.wants_metadata.value = True
                else:
                    metadata.add_extraction_method()
                edited_modules.add(metadata)
                em = metadata.extraction_methods[-1]
                em.extraction_method.value = cpmetadata.X_MANUAL_EXTRACTION
                em.source.value = cpmetadata.XM_FILE_NAME
                em.file_regexp.value = regexp
                em.filter_choice.value = cpmetadata.F_FILTERED_IMAGES
                em.filter.build(structure)
            #
            # If there was metadata to match, namesandtypes should
            # have a metadata joiner.
            #
            if namesandtypes.matching_choice == cpnamesandtypes.MATCH_BY_METADATA:
                joins = namesandtypes.join.parse()
                for d in joins:
                    for v in d.values():
                        if v in tags:
                            d[name] = v
                            tags.remove(v)
                            break
                    else:
                        d[name] = None
                namesandtypes.join.build(joins)

            assignment.rule_filter.build(structure)
            if group.image_objects_choice == IO_IMAGES:
                assignment.image_name.value = name
                assignment.load_as_choice.value = cpnamesandtypes.LOAD_AS_GRAYSCALE_IMAGE
            else:
                assignment.object_name.value = name
                assignment.load_as_choice.value = cpnamesandtypes.LOAD_AS_OBJECTS
        for module in edited_modules:
            pipeline.edit_module(module.module_num, True)

    def upgrade_settings(self, setting_values, variable_revision_number, module_name, from_matlab):
        if from_matlab and variable_revision_number == 4:
            new_setting_values = list(setting_values)
            # The first setting was blank in Matlab. Now it contains
            # the directory choice
            if setting_values[1] == '.':
                new_setting_values[0] = cps.DEFAULT_INPUT_FOLDER_NAME
            elif setting_values[1] == '&':
                new_setting_values[0] = cps.DEFAULT_OUTPUT_FOLDER_NAME
            else:
                new_setting_values[0] = DIR_CUSTOM_FOLDER
            #
            # Remove "Do not use" images
            #
            for i in [8, 6, 4]:
                if new_setting_values[i + 1] == cps.DO_NOT_USE:
                    del new_setting_values[i:i + 2]
            setting_values = new_setting_values
            from_matlab = False
            variable_revision_number = 1
        #
        # Minor revision: default image folder -> default input folder
        #
        if variable_revision_number == 1 and not from_matlab:
            if setting_values[0].startswith("Default image"):
                dir_choice = cps.DEFAULT_INPUT_FOLDER_NAME
                custom_directory = setting_values[1]
            elif setting_values[0] in (DIR_CUSTOM_FOLDER, DIR_CUSTOM_WITH_METADATA):
                custom_directory = setting_values[1]
                if custom_directory[0] == ".":
                    dir_choice = cps.DEFAULT_INPUT_SUBFOLDER_NAME
                elif custom_directory[0] == "&":
                    dir_choice = cps.DEFAULT_OUTPUT_SUBFOLDER_NAME
                    custom_directory = "." + custom_directory[1:]
                else:
                    dir_choice = cps.ABSOLUTE_FOLDER_NAME
            else:
                dir_choice = setting_values[0]
                custom_directory = setting_values[1]
            directory = cps.DirectoryPath.static_join_string(
                    dir_choice, custom_directory)
            setting_values = [directory] + setting_values[2:]
            variable_revision_number = 2

        # Standardize input/output directory name references
        SLOT_DIR = 0
        setting_values[SLOT_DIR] = cps.DirectoryPath.upgrade_setting(
                setting_values[SLOT_DIR])

        if variable_revision_number == 2 and (not from_matlab):
            # changes to DirectoryPath and URL handling
            dir = setting_values[0]
            dir_choice, custom_dir = cps.DirectoryPath.split_string(dir)
            if dir_choice == cps.URL_FOLDER_NAME:
                dir = cps.DirectoryPath.static_join_string(dir_choice, '')

                filenames = setting_values[1::2]
                imagenames = setting_values[2::2]
                setting_values = [dir] + sum(
                        [[custom_dir + '/' + filename, image_name]
                         for filename, image_name in zip(filenames, imagenames)], [])
            variable_revision_number = 3

        if variable_revision_number == 3 and (not from_matlab):
            # Added rescale option
            new_setting_values = setting_values[:1]
            for i in range(1, len(setting_values), 2):
                new_setting_values += setting_values[i:(i + 2)] + [cps.YES]
            setting_values = new_setting_values
            variable_revision_number = 4

        if variable_revision_number == 4 and (not from_matlab):
            # Added load objects
            new_setting_values = setting_values[:1]
            for i in range(1, len(setting_values), S_FILE_SETTINGS_COUNT_V4):
                new_setting_values += [
                    setting_values[i + S_FILE_NAME_OFFSET_V4],
                    IO_IMAGES,
                    setting_values[i + S_IMAGE_NAME_OFFSET_V4],
                    "Nuclei",
                    cps.NO,
                    "NucleiOutlines",
                    setting_values[i + S_RESCALE_OFFSET_V4]]
            setting_values = new_setting_values
            variable_revision_number = 5

        return setting_values, variable_revision_number, from_matlab
