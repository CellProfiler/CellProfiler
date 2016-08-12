"""<b>Load Single Image</b> loads a single image for use in all image cycles.
<hr>
<p>This module tells CellProfiler where to retrieve a single image and gives the image a
meaningful name by which the other modules can access it. The module
executes only the first time through the pipeline; thereafter the image
is accessible to all subsequent processing cycles. This is
particularly useful for loading an image like an illumination correction
image for use by the <b>CorrectIlluminationApply</b> module, when that single
image will be used to correct all images in the analysis run.</p>

<p><i>Disclaimer:</i> Please note that the Input modules (i.e., <b>Images</b>, <b>Metadata</b>, <b>NamesAndTypes</b>
and <b>Groups</b>) largely supercedes this module. However, old pipelines loaded into
CellProfiler that contain this module will provide the option of preserving them;
these pipelines will operate exactly as before.</p>

<h4>Available measurements</h4>
<ul>
<li><i>Pathname, Filename:</i> The full path and the filename of each image.</li>
<li><i>Metadata:</i> The metadata information extracted from the path and/or
filename, if requested.</li>
<li><i>Scaling:</i> The maximum possible intensity value for the image format.</li>
<li><i>Height, Width:</i> The height and width of the current image.</li>
</ul>

<h4>Technical notes</h4>

<p>For most purposes, you will probably want to use the <b>LoadImages</b> module, not
<b>LoadSingleImage</b>. The reason is that <b>LoadSingleImage</b> does not actually
create image sets (or even a single image set). Instead, it adds the single image
to every image cycle for an <i>already existing</i> image set. Hence
<b>LoadSingleImage</b> should never be used as the only image-loading module in a
pipeline; attempting to do so will display a warning message in the module settings.
<p>If you have a single file to load in the pipeline (and only that file), you
will want to use <b>LoadImages</b> or <b>LoadData</b> with a single, hardcoded file name. </p>

See also the <b>Input</b> modules, <b>LoadImages</b>,<b>LoadData</b>.

"""

import os
import re

import cellprofiler.gui.help
import cellprofiler.identify
import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.metadata
import cellprofiler.module
import cellprofiler.modules
import cellprofiler.modules.namesandtypes
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.setting
import cellprofiler.utilities.predicate
import cellprofiler.utilities.url
import centrosome.outline

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


class LoadSingleImage(cellprofiler.module.Module):
    module_name = "LoadSingleImage"
    category = "File Processing"
    variable_revision_number = 5

    def create_settings(self):
        """Create the settings during initialization

        """
        self.directory = cellprofiler.setting.DirectoryPath(
            "Input image file location",
            support_urls=True,
            doc='''
            Select the folder containing the image(s) to be loaded. Generally,
            it is best to store the image you want to load in either the Default Input or
            Output Folder, so that the correct image is loaded into the pipeline
            and typos are avoided. {io_folder_choice_help_text}

            <p>{io_with_metadata_help_text} {using_metadata_tags} For instance,
            if you have a "Plate" metadata tag, and your single files are
            organized in subfolders named with the "Plate" tag, you can select one of the
            subfolder options and then specify a subfolder name of "\g&lt;Plate&gt;"
            to get the files from the subfolder associated with that image's plate. The module will
            substitute the metadata values for the current image set for any metadata tags in the
            folder name. {using_metadata_help}.</p>'''.format(**{
                'io_folder_choice_help_text': cellprofiler.preferences.IO_FOLDER_CHOICE_HELP_TEXT,
                'io_with_metadata_help_text': cellprofiler.preferences.IO_WITH_METADATA_HELP_TEXT,
                'using_metadata_tags': cellprofiler.gui.help.USING_METADATA_TAGS_REF,
                'using_metadata_help': cellprofiler.gui.help.USING_METADATA_HELP_REF
            })
        )

        self.file_settings = []
        self.add_file(can_remove=False)
        self.add_button = cellprofiler.setting.DoSomething("", "Add another image", self.add_file)

    def add_file(self, can_remove=True):
        """Add settings for another file to the list"""
        group = cellprofiler.setting.SettingsGroup()
        if can_remove:
            group.append("divider", cellprofiler.setting.Divider(line=False))

        def get_directory_fn():
            return self.directory.get_absolute_path()

        group.append(
            "file_name",
            cellprofiler.setting.FilenameText(
                FILE_TEXT,
                cellprofiler.setting.NONE,
                metadata=True,
                get_directory_fn=get_directory_fn,
                exts=[("TIF - Tagged Image File format (*.tif,*.tiff)", "*.tif;*.tiff"),
                      ("PNG - Portable Network Graphics (*.png)", "*.png"),
                      ("JPG/JPEG file (*.jpg,*.jpeg)", "*.jpg,*.jpeg"),
                      ("BMP - Windows Bitmap (*.bmp)", "*.bmp"),
                      ("Compuserve GIF file (*.gif)", "*.gif"),
                      ("MATLAB image (*.mat)", "*.mat"),
                      ("All files (*.*)", "*.*")],
                doc="""
                The filename can be constructed in one of two ways:
                <ul>
                <li>As a fixed filename (e.g., <i>Exp1_D03f00d0.tif</i>). </li>
                <li>Using the metadata associated with an image set in
                <b>LoadImages</b> or <b>LoadData</b>. This is especially useful
                if you want your output given a unique label according to the
                metadata corresponding to an image group. The name of the metadata
                to substitute is included in a special tag format embedded
                in your file specification. {using_metadata_tags}{using_metadata_help}.</li>
                </ul>
                <p>Keep in mind that in either case, the image file extension, if any, must be included.""".format(**{
                    'using_metadata_tags': cellprofiler.gui.help.USING_METADATA_TAGS_REF,
                    'using_metadata_help': cellprofiler.gui.help.USING_METADATA_HELP_REF
                })
            )
        )

        group.append("image_objects_choice", cellprofiler.setting.Choice(
                'Load as images or objects?', cellprofiler.image.IO_ALL, doc="""
                    This setting determines whether you load an image as image data
                    or as segmentation results (i.e., objects):
                    <ul>
                    <li><i>%(IO_IMAGES)s:</i> The input image will be given a user-specified name by
                    which it will be refered downstream. This is the most common usage for this
                    module.</li>
                    <li><i>%(IO_OBJECTS)s:</i> Use this option if the input image is a label matrix
                    and you want to obtain the objects that it defines. A <i>label matrix</i>
                    is a grayscale or color image in which the connected regions share the
                    same label, and defines how objects are represented in CellProfiler.
                    The labels are integer values greater than or equal to 0.
                    The elements equal to 0 are the background, whereas the elements equal to 1
                    make up one object, the elements equal to 2 make up a second object, and so on.
                    This option allows you to use the objects without needing to insert an
                    <b>Identify</b> module to extract them first. See <b>IdentifyPrimaryObjects</b>
                    for more details.</li>
                    </ul>""" % globals()))

        group.append("image_name", cellprofiler.setting.FileImageNameProvider("Name the image that will be loaded",
                                                             "OrigBlue", doc='''
                    <i>(Used only if an image is output)</i><br>
                    Enter the name of the image that will be loaded.
                    You can use this name to select the image in downstream modules.'''))

        group.append(
            "rescale",
            cellprofiler.setting.Binary(
                "Rescale intensities?",
                True,
                doc="""
                <i>(Used only if an image is output)</i><br>
                This option determines whether image metadata should be
                used to rescale the image's intensities. Some image formats
                save the maximum possible intensity value along with the pixel data.
                For instance, a microscope might acquire images using a 12-bit
                A/D converter which outputs intensity values between zero and 4095,
                but stores the values in a field that can take values up to 65535.
                <p>Select <i>{yes}</i> to rescale the image intensity so that
                saturated values are rescaled to 1.0 by dividing all pixels
                in the image by the maximum possible intensity value. </p>
                <p>Select <i>{no}</i> to ignore the image metadata and rescale the image
                to 0 &ndash; 1.0 by dividing by 255 or 65535, depending on the number
                of bits used to store the image.</p>""".format(**{
                    'yes': cellprofiler.setting.YES,
                    'no': cellprofiler.setting.NO
                })
            )
        )

        group.append("objects_name", cellprofiler.setting.ObjectNameProvider(
                'Name this loaded object',
                "Nuclei",
                doc="""<i>(Used only if objects are output)</i><br>
                    This is the name for the objects loaded from your image"""))

        group.append(
            "wants_outlines",
            cellprofiler.setting.Binary(
                "Retain outlines of loaded objects?",
                False,
                doc="""
                <i>(Used only if objects are output)</i><br>
                Select <i>{}</i> if you want to save an image of the outlines
                of the loaded objects.""".format(cellprofiler.setting.YES)
            )
        )

        group.append("outlines_name", cellprofiler.setting.OutlineNameProvider(
                'Name the outlines',
                'NucleiOutlines', doc="""
                    <i>(Used only if objects are output)</i><br>
                    Enter a name that will allow the outlines to be selected later in the pipeline."""))

        if can_remove:
            group.append("remove", cellprofiler.setting.RemoveSettingButton("", "Remove this image", self.file_settings, group))
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
            url_based = (self.directory.dir_choice == cellprofiler.preferences.URL_FOLDER_NAME)
            file_setting.file_name.set_browsable(not url_based)
            file_setting.file_name.text = URL_TEXT if url_based else FILE_TEXT
            result += [
                file_setting.file_name, file_setting.image_objects_choice]
            if file_setting.image_objects_choice == cellprofiler.image.IO_IMAGES:
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
            if file_setting.image_objects_choice == cellprofiler.image.IO_IMAGES:
                image_name = file_setting.image_name.value
            else:
                image_name = file_setting.objects_name.value
            result[image_name] = file_name

        return result

    def get_file_settings(self, image_name):
        """Get the file settings associated with a given image name"""
        for file_setting in self.file_settings:
            if (file_setting.image_objects_choice == cellprofiler.image.IO_IMAGES and
                        file_setting.image_name == image_name):
                return file_setting
            if (file_setting.image_objects_choice == cellprofiler.image.IO_OBJECTS and
                        file_setting.objects_name == image_name):
                return file_setting
        return None

    def file_wants_images(self, file_setting):
        """True if the file_setting produces images, false if it produces objects"""
        return file_setting.image_objects_choice == cellprofiler.image.IO_IMAGES

    def is_load_module(self):
        return True

    def prepare_run(self, workspace):
        m = workspace.measurements
        assert isinstance(m, cellprofiler.measurement.Measurements)
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
                if file_settings.image_objects_choice == cellprofiler.image.IO_IMAGES:
                    #
                    # Add measurements
                    #
                    path_name_category = cellprofiler.measurement.C_PATH_NAME
                    file_name_category = cellprofiler.measurement.C_FILE_NAME
                    url_category = cellprofiler.measurement.C_URL
                else:
                    #
                    # Add measurements
                    #
                    path_name_category = cellprofiler.measurement.C_OBJECTS_PATH_NAME
                    file_name_category = cellprofiler.measurement.C_OBJECTS_FILE_NAME
                    url_category = cellprofiler.measurement.C_OBJECTS_URL

                url = cellprofiler.utilities.url.pathname2url(os.path.join(root, dict[image_name]))
                for category, value in (
                        (path_name_category, root),
                        (file_name_category, dict[image_name]),
                        (url_category, url)):
                    measurement_name = "_".join((category, image_name))
                    m.add_measurement(cellprofiler.measurement.IMAGE, measurement_name, value,
                                      image_set_number=image_number)
        return True

    def run(self, workspace):
        statistics = []
        m = workspace.measurements
        assert isinstance(m, cellprofiler.measurement.Measurements)
        #
        # Hack: if LoadSingleImage is first, no paths are populated
        #
        if self.file_wants_images(self.file_settings[0]):
            m_path = "_".join((cellprofiler.measurement.C_PATH_NAME,
                               self.file_settings[0].image_name.value))
        else:
            m_path = "_".join((cellprofiler.measurement.C_OBJECTS_PATH_NAME,
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
                    cellprofiler.measurement.C_PATH_NAME if wants_images else cellprofiler.measurement.C_OBJECTS_PATH_NAME,
                    cellprofiler.measurement.C_FILE_NAME if wants_images else cellprofiler.measurement.C_OBJECTS_FILE_NAME,
                    cellprofiler.measurement.C_MD5_DIGEST, cellprofiler.measurement.C_SCALING,
                    cellprofiler.measurement.C_HEIGHT, cellprofiler.measurement.C_WIDTH)]
            pathname = m.get_current_image_measurement(m_path)
            filename = m.get_current_image_measurement(m_file)
            rescale = (wants_images and file_setting.rescale.value)

            provider = cellprofiler.image.LoadImagesImageProvider(
                    image_name, pathname, filename, rescale)
            image = provider.provide_image(image_set)
            pixel_data = image.pixel_data
            if wants_images:
                md5 = provider.get_md5_hash(m)
                m.add_image_measurement("_".join((cellprofiler.measurement.C_MD5_DIGEST, image_name)),
                                        md5)
                m.add_image_measurement("_".join((cellprofiler.measurement.C_SCALING, image_name)),
                                        image.scale)
                m.add_image_measurement("_".join((cellprofiler.measurement.C_HEIGHT, image_name)),
                                        int(pixel_data.shape[0]))
                m.add_image_measurement("_".join((cellprofiler.measurement.C_WIDTH, image_name)),
                                        int(pixel_data.shape[1]))
                image_set.providers.append(provider)
            else:
                #
                # Turn image into objects
                #
                labels = cellprofiler.image.convert_image_to_objects(pixel_data)
                objects = cellprofiler.region.Region()
                objects.segmented = labels
                object_set = workspace.object_set
                assert isinstance(object_set, cellprofiler.region.Set)
                object_set.add_objects(objects, image_name)
                cellprofiler.identify.add_object_count_measurements(m, image_name, objects.count)
                cellprofiler.identify.add_object_location_measurements(m, image_name, labels)
                #
                # Add outlines if appropriate
                #
                if file_setting.wants_outlines:
                    outlines = centrosome.outline.outline(labels)
                    outline_image = cellprofiler.image.Image(outlines.astype(bool))
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
            if file_setting.image_objects_choice == cellprofiler.image.IO_IMAGES:
                image_name = file_setting.image_name.value
                path_name_category = cellprofiler.measurement.C_PATH_NAME
                file_name_category = cellprofiler.measurement.C_FILE_NAME
                columns += [
                    (cellprofiler.measurement.IMAGE, "_".join((cellprofiler.measurement.C_MD5_DIGEST, image_name)), cellprofiler.measurement.COLTYPE_VARCHAR_FORMAT % 32),
                    (cellprofiler.measurement.IMAGE, "_".join((cellprofiler.measurement.C_SCALING, image_name)), cellprofiler.measurement.COLTYPE_FLOAT),
                    (cellprofiler.measurement.IMAGE, "_".join((cellprofiler.measurement.C_HEIGHT, image_name)), cellprofiler.measurement.COLTYPE_INTEGER),
                    (cellprofiler.measurement.IMAGE, "_".join((cellprofiler.measurement.C_WIDTH, image_name)), cellprofiler.measurement.COLTYPE_INTEGER)]
            else:
                image_name = file_setting.objects_name.value
                path_name_category = cellprofiler.measurement.C_OBJECTS_PATH_NAME
                file_name_category = cellprofiler.measurement.C_OBJECTS_FILE_NAME
                columns += cellprofiler.identify.get_object_measurement_columns(image_name)

            columns += [(cellprofiler.measurement.IMAGE, '_'.join((feature, image_name)), coltype)
                        for feature, coltype in (
                            (file_name_category, cellprofiler.measurement.COLTYPE_VARCHAR_FILE_NAME),
                            (path_name_category, cellprofiler.measurement.COLTYPE_VARCHAR_PATH_NAME),
                        )]
        return columns

    @property
    def wants_images(self):
        """True if any file setting loads images"""
        return any([True for file_setting in self.file_settings
                    if file_setting.image_objects_choice == cellprofiler.image.IO_IMAGES])

    @property
    def wants_objects(self):
        """True if any file setting loads objects"""
        return any([True for file_setting in self.file_settings
                    if file_setting.image_objects_choice == cellprofiler.image.IO_OBJECTS])

    def get_categories(self, pipeline, object_name):
        result = []
        if object_name == cellprofiler.measurement.IMAGE:
            if self.wants_images:
                result += [cellprofiler.measurement.C_FILE_NAME, cellprofiler.measurement.C_MD5_DIGEST, cellprofiler.measurement.C_PATH_NAME,
                           cellprofiler.measurement.C_SCALING, cellprofiler.measurement.C_HEIGHT,
                           cellprofiler.measurement.C_WIDTH]
            if self.wants_objects:
                result += [cellprofiler.identify.C_COUNT, cellprofiler.measurement.C_OBJECTS_FILE_NAME, cellprofiler.measurement.C_OBJECTS_PATH_NAME]
        if any([True for file_setting in self.file_settings
                if file_setting.image_objects_choice == cellprofiler.image.IO_OBJECTS and
                                object_name == file_setting.objects_name]):
            result += [cellprofiler.identify.C_LOCATION, cellprofiler.identify.C_NUMBER]
        return result

    def get_measurements(self, pipeline, object_name, category):
        """Return the measurements that this module produces

        object_name - return measurements made on this object (or 'Image' for image measurements)
        category - return measurements made in this category
        """
        result = []
        if object_name == cellprofiler.measurement.IMAGE:
            if category in (cellprofiler.measurement.C_FILE_NAME, cellprofiler.measurement.C_MD5_DIGEST, cellprofiler.measurement.C_PATH_NAME,
                            cellprofiler.measurement.C_SCALING, cellprofiler.measurement.C_HEIGHT,
                            cellprofiler.measurement.C_WIDTH):
                result += [file_setting.image_name.value
                           for file_setting in self.file_settings
                           if file_setting.image_objects_choice == cellprofiler.image.IO_IMAGES]
            if category in (cellprofiler.measurement.C_OBJECTS_FILE_NAME, cellprofiler.measurement.C_OBJECTS_PATH_NAME, cellprofiler.identify.C_COUNT):
                result += [file_setting.objects_name.value
                           for file_setting in self.file_settings
                           if file_setting.image_objects_choice == cellprofiler.image.IO_OBJECTS]
        elif any([file_setting.image_objects_choice == cellprofiler.image.IO_OBJECTS and
                                  file_setting.objects_name == object_name
                  for file_setting in self.file_settings]):
            if category == cellprofiler.identify.C_NUMBER:
                result += [cellprofiler.identify.FTR_OBJECT_NUMBER]
            elif category == cellprofiler.identify.C_LOCATION:
                result += [cellprofiler.identify.FTR_CENTER_X, cellprofiler.identify.FTR_CENTER_Y]
        return result

    def validate_module(self, pipeline):
        # Keep users from using LoadSingleImage to define image sets
        if not any([x.is_load_module() for x in pipeline.modules()]):
            raise cellprofiler.setting.ValidationError(
                    "LoadSingleImage cannot be used to run a pipeline on one "
                    "image file. Please use LoadImages or LoadData instead.",
                    self.directory)

        # Make sure LoadSingleImage appears after all other load modules
        after = False
        for module in pipeline.modules():
            if module is self:
                after = True
            elif after and module.is_load_module():
                raise cellprofiler.setting.ValidationError(
                        "LoadSingleImage must appear after all other Load modules in your pipeline\n"
                        "Please move %s before LoadSingleImage" % module.module_name,
                        self.directory)

        # Make sure metadata tags exist
        for group in self.file_settings:
            text_str = group.file_name.value
            undefined_tags = pipeline.get_undefined_metadata_tags(text_str)
            if len(undefined_tags) > 0:
                raise cellprofiler.setting.ValidationError(
                        "%s is not a defined metadata tag. Check the metadata specifications in your load modules" %
                        undefined_tags[0],
                        group.file_name)

    def validate_module_warnings(self, pipeline):
        """Check for potentially dangerous settings"""
        # Check that user-specified names don't have bad characters
        invalid_chars_pattern = "^[A-Za-z][A-Za-z0-9_]+$"
        warning_text = "The image name has questionable characters. The pipeline can use this name " \
                       "and produce results, but downstream programs that use this data (e.g, MATLAB, MySQL) may error."
        for file_setting in self.file_settings:
            if file_setting.image_objects_choice == cellprofiler.image.IO_IMAGES:
                if not re.match(invalid_chars_pattern, file_setting.image_name.value):
                    raise cellprofiler.setting.ValidationError(warning_text, file_setting.image_name)

    def needs_conversion(self):
        return True

    def convert(self, pipeline, metadata, namesandtypes, groups):
        """Convert from legacy to modern"""
        edited_modules = set()

        for group in self.file_settings:
            tags = []
            file_name = group.file_name.value
            if group.image_objects_choice == cellprofiler.image.IO_IMAGES:
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
            if namesandtypes.assignment_method != cellprofiler.modules.namesandtypes.ASSIGN_RULES:
                namesandtypes.assignment_method.value = cellprofiler.modules.namesandtypes.ASSIGN_RULES
            else:
                namesandtypes.add_assignment()
            edited_modules.add(namesandtypes)
            assignment = namesandtypes.assignments[-1]
            structure = [cellprofiler.setting.Filter.AND_PREDICATE]
            fp = cellprofiler.utilities.predicate.FilePredicate()
            fp_does, fp_does_not = [
                [d for d in fp.subpredicates if isinstance(d, c)][0]
                for c in (cellprofiler.setting.Filter.DoesPredicate, cellprofiler.setting.Filter.DoesNotPredicate)]
            if len(tags) == 0:
                structure.append([fp, fp_does, cellprofiler.setting.Filter.EQ_PREDICATE,
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
                    fp, fp_does, cellprofiler.setting.Filter.CONTAINS_REGEXP_PREDICATE, regexp])
                if not metadata.wants_metadata:
                    metadata.wants_metadata.value = True
                else:
                    metadata.add_extraction_method()
                edited_modules.add(metadata)
                em = metadata.extraction_methods[-1]
                em.extraction_method.value = cellprofiler.metadata.X_MANUAL_EXTRACTION
                em.source.value = cellprofiler.metadata.XM_FILE_NAME
                em.file_regexp.value = regexp
                em.filter_choice.value = cellprofiler.metadata.F_FILTERED_IMAGES
                em.filter.build(structure)
            #
            # If there was metadata to match, namesandtypes should
            # have a metadata joiner.
            #
            if namesandtypes.matching_choice == cellprofiler.modules.namesandtypes.MATCH_BY_METADATA:
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
            if group.image_objects_choice == cellprofiler.image.IO_IMAGES:
                assignment.image_name.value = name
                assignment.load_as_choice.value = cellprofiler.modules.namesandtypes.LOAD_AS_GRAYSCALE_IMAGE
            else:
                assignment.object_name.value = name
                assignment.load_as_choice.value = cellprofiler.modules.namesandtypes.LOAD_AS_OBJECTS
        for module in edited_modules:
            pipeline.edit_module(module.module_num, True)

    def upgrade_settings(self, setting_values, variable_revision_number, module_name, from_matlab):
        if from_matlab and variable_revision_number == 4:
            new_setting_values = list(setting_values)
            # The first setting was blank in Matlab. Now it contains
            # the directory choice
            if setting_values[1] == '.':
                new_setting_values[0] = cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME
            elif setting_values[1] == '&':
                new_setting_values[0] = cellprofiler.preferences.DEFAULT_OUTPUT_FOLDER_NAME
            else:
                new_setting_values[0] = DIR_CUSTOM_FOLDER
            #
            # Remove "Do not use" images
            #
            for i in [8, 6, 4]:
                if new_setting_values[i + 1] == cellprofiler.setting.DO_NOT_USE:
                    del new_setting_values[i:i + 2]
            setting_values = new_setting_values
            from_matlab = False
            variable_revision_number = 1
        #
        # Minor revision: default image folder -> default input folder
        #
        if variable_revision_number == 1 and not from_matlab:
            if setting_values[0].startswith("Default image"):
                dir_choice = cellprofiler.preferences.DEFAULT_INPUT_FOLDER_NAME
                custom_directory = setting_values[1]
            elif setting_values[0] in (DIR_CUSTOM_FOLDER, DIR_CUSTOM_WITH_METADATA):
                custom_directory = setting_values[1]
                if custom_directory[0] == ".":
                    dir_choice = cellprofiler.preferences.DEFAULT_INPUT_SUBFOLDER_NAME
                elif custom_directory[0] == "&":
                    dir_choice = cellprofiler.preferences.DEFAULT_OUTPUT_SUBFOLDER_NAME
                    custom_directory = "." + custom_directory[1:]
                else:
                    dir_choice = cellprofiler.preferences.ABSOLUTE_FOLDER_NAME
            else:
                dir_choice = setting_values[0]
                custom_directory = setting_values[1]
            directory = cellprofiler.setting.DirectoryPath.static_join_string(
                    dir_choice, custom_directory)
            setting_values = [directory] + setting_values[2:]
            variable_revision_number = 2

        # Standardize input/output directory name references
        SLOT_DIR = 0
        setting_values[SLOT_DIR] = cellprofiler.setting.DirectoryPath.upgrade_setting(
                setting_values[SLOT_DIR])

        if variable_revision_number == 2 and (not from_matlab):
            # changes to DirectoryPath and URL handling
            dir = setting_values[0]
            dir_choice, custom_dir = cellprofiler.setting.DirectoryPath.split_string(dir)
            if dir_choice == cellprofiler.preferences.URL_FOLDER_NAME:
                dir = cellprofiler.setting.DirectoryPath.static_join_string(dir_choice, '')

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
                new_setting_values += setting_values[i:(i + 2)] + [cellprofiler.setting.YES]
            setting_values = new_setting_values
            variable_revision_number = 4

        if variable_revision_number == 4 and (not from_matlab):
            # Added load objects
            new_setting_values = setting_values[:1]
            for i in range(1, len(setting_values), S_FILE_SETTINGS_COUNT_V4):
                new_setting_values += [
                    setting_values[i + S_FILE_NAME_OFFSET_V4],
                    cellprofiler.image.IO_IMAGES,
                    setting_values[i + S_IMAGE_NAME_OFFSET_V4],
                    "Nuclei",
                    cellprofiler.setting.NO,
                    "NucleiOutlines",
                    setting_values[i + S_RESCALE_OFFSET_V4]]
            setting_values = new_setting_values
            variable_revision_number = 5

        return setting_values, variable_revision_number, from_matlab
