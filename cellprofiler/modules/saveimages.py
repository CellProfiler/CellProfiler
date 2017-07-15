# coding=utf-8

"""
**Save Images** saves image or movie files.

Because CellProfiler usually performs many image analysis steps on many
groups of images, it does *not* save any of the resulting images to the
hard drive unless you specifically choose to do so with the
**SaveImages** module. You can save any of the processed images created
by CellProfiler during the analysis using this module.

You can choose from many different image formats for saving your files.
This allows you to use the module as a file format converter, by loading
files in their original format and then saving them in an alternate
format.

Note that saving images in 12-bit format is not supported, and 16-bit
format is supported for TIFF only.

See also **NamesAndTypes**.
"""

import os
import os.path
import sys

import bioformats.formatwriter
import bioformats.omexml
import numpy
import skimage.io
import skimage.util

import cellprofiler.gui.help
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.loadimages
import cellprofiler.preferences
import cellprofiler.setting


IF_IMAGE = "Image"
IF_MASK = "Mask"
IF_CROPPING = "Cropping"
IF_MOVIE = "Movie"
IF_ALL = [IF_IMAGE, IF_MASK, IF_CROPPING, IF_MOVIE]

BIT_DEPTH_8 = "8-bit integer"
BIT_DEPTH_16 = "16-bit integer"
BIT_DEPTH_FLOAT = "64-bit floating point"

FN_FROM_IMAGE = "From image filename"
FN_SEQUENTIAL = "Sequential numbers"
FN_SINGLE_NAME = "Single name"

SINGLE_NAME_TEXT = "Enter single file name"
SEQUENTIAL_NUMBER_TEXT = "Enter file prefix"

FF_JPEG = "jpeg"
FF_PNG = "png"
FF_TIFF = "tiff"

PC_WITH_IMAGE = "Same folder as image"

WS_EVERY_CYCLE = "Every cycle"
WS_FIRST_CYCLE = "First cycle"
WS_LAST_CYCLE = "Last cycle"


class SaveImages(cellprofiler.module.Module):
    module_name = "SaveImages"

    variable_revision_number = 12

    category = "File Processing"

    def create_settings(self):
        self.save_image_or_figure = cellprofiler.setting.Choice(
            "Select the type of image to save",
            IF_ALL,
            IF_IMAGE,
            doc="""
            The following types of images can be saved as a file on the hard drive:
            <ul>
                <li><i>{IF_IMAGE}:</i> Any of the images produced upstream of <b>SaveImages</b> can be selected
                for saving. Outlines created by <b>Identify</b> modules can also be saved with this option, but
                you must select "Retain outlines..." of identified objects within the <b>Identify</b> module.
                You might also want to use the <b>OverlayOutlines</b> module prior to saving images.</li>
                <li><i>{IF_MASK}:</i> Relevant only if the <b>Crop</b> module is used. The <b>Crop</b> module
                creates a mask of the pixels of interest in the image. Saving the mask will produce a binary
                image in which the pixels of interest are set to 1; all other pixels are set to 0.</li>
                <li><i>{IF_CROPPING}:</i> Relevant only if the <b>Crop</b> module is used. The <b>Crop</b>
                module also creates a cropping image which is typically the same size as the original image.
                However, since the <b>Crop</b> permits removal of the rows and columns that are left blank, the
                cropping can be of a different size than the mask.</li>
                <li><i>{IF_MOVIE}:</i> A sequence of images can be saved as a TIFF stack.</li>
            </ul>
            """.format(**{
                "IF_CROPPING": IF_CROPPING,
                "IF_IMAGE": IF_IMAGE,
                "IF_MASK": IF_MASK,
                "IF_MOVIE": IF_MOVIE
            })
        )

        self.image_name = cellprofiler.setting.ImageNameSubscriber(
            "Select the image to save",
            doc="Select the image you want to save."
        )

        self.file_name_method = cellprofiler.setting.Choice(
            "Select method for constructing file names",
            [
                FN_FROM_IMAGE,
                FN_SEQUENTIAL,
                FN_SINGLE_NAME
            ],
            FN_FROM_IMAGE,
            doc="""
            <i>(Used only if saving non-movie files)</i><br>
            Several choices are available for constructing the image file name:
            <ul>
                <li>
                    <i>{FN_FROM_IMAGE}:</i> The filename will be constructed based on the original filename of
                    an input image specified in <b>NamesAndTypes</b>. You will have the opportunity to prefix
                    or append additional text.
                    <p>If you have metadata associated with your images, you can append an text to the image
                    filename using a metadata tag. This is especially useful if you want your output given a
                    unique label according to the metadata corresponding to an image group. The name of the
                    metadata to substitute can be provided for each image for each cycle using the
                    <b>Metadata</b> module. {USING_METADATA_TAGS_REF}{USING_METADATA_HELP_REF}.</p>
                </li>
                <li><i>{FN_SEQUENTIAL}:</i> Same as above, but in addition, each filename will have a number
                appended to the end that corresponds to the image cycle number (starting at 1).</li>
                <li><i>{FN_SINGLE_NAME}:</i> A single name will be given to the file. Since the filename is
                fixed, this file will be overwritten with each cycle. In this case, you would probably want to
                save the image on the last cycle (see the <i>Select how often to save</i> setting). The
                exception to this is to use a metadata tag to provide a unique label, as mentioned in the
                <i>{FN_FROM_IMAGE}</i> option.</li>
            </ul>
            """.format(**{
                "FN_FROM_IMAGE": FN_FROM_IMAGE,
                "FN_SEQUENTIAL": FN_SEQUENTIAL,
                "FN_SINGLE_NAME": FN_SINGLE_NAME,
                "USING_METADATA_HELP_REF": cellprofiler.gui.help.USING_METADATA_HELP_REF,
                "USING_METADATA_TAGS_REF": cellprofiler.gui.help.USING_METADATA_TAGS_REF
            })
        )

        self.file_image_name = cellprofiler.setting.FileImageNameSubscriber(
            "Select image name for file prefix",
            cellprofiler.setting.NONE,
            doc="""
            <i>(Used only when "{FN_FROM_IMAGE}" is selected for contructing the filename)</i><br>
            Select an image loaded using <b>NamesAndTypes</b>. The original filename will be
            used as the prefix for the output filename.
            """.format(**{
                "FN_FROM_IMAGE": FN_FROM_IMAGE
            })
        )

        self.single_file_name = cellprofiler.setting.Text(
            SINGLE_NAME_TEXT,
            "OrigBlue",
            metadata=True,
            doc="""
            <i>(Used only when "{FN_SEQUENTIAL}" or "{FN_SINGLE_NAME}" are selected for contructing the
            filename)</i><br>
            Specify the filename text here. If you have metadata associated with your images, enter the
            filename text with the metadata tags. {USING_METADATA_TAGS_REF}<br>
            Do not enter the file extension in this setting; it will be appended automatically.
            """.format(**{
                "FN_SEQUENTIAL": FN_SEQUENTIAL,
                "FN_SINGLE_NAME": FN_SINGLE_NAME,
                "USING_METADATA_TAGS_REF": cellprofiler.gui.help.USING_METADATA_TAGS_REF
            })
        )

        self.number_of_digits = cellprofiler.setting.Integer(
            "Number of digits",
            4,
            doc="""
            <i>(Used only when "{FN_SEQUENTIAL}" is selected for contructing the filename)</i><br>
            Specify the number of digits to be used for the sequential numbering. Zeros will be
            used to left-pad the digits. If the number specified here is less than that needed to
            contain the number of image sets, the latter will override the value entered.
            """.format(**{
                "FN_SEQUENTIAL": FN_SEQUENTIAL
            })
        )

        self.wants_file_name_suffix = cellprofiler.setting.Binary(
            "Append a suffix to the image file name?",
            False,
            doc="""
            Select <i>{YES}</i> to add a suffix to the image's file name.
            Select <i>{NO}</i> to use the image name as-is.
            """.format(**{
                "NO": cellprofiler.setting.NO,
                "YES": cellprofiler.setting.YES
            })
        )

        self.file_name_suffix = cellprofiler.setting.Text(
            "Text to append to the image name",
            "",
            metadata=True,
            doc="""
            <i>(Used only when constructing the filename from the image filename)</i><br>
            Enter the text that should be appended to the filename specified above.
            """
        )

        self.file_format = cellprofiler.setting.Choice(
            "Saved file format",
            [
                FF_JPEG,
                FF_PNG,
                FF_TIFF
            ],
            value=FF_TIFF,
            doc="""
            <i>(Used only when saving non-movie files)</i><br>
            Select the image or movie format to save the image(s). Most common image formats are available.
            """
        )

        self.pathname = SaveImagesDirectoryPath(
            "Output file location",
            self.file_image_name,
            doc="""
            This setting lets you choose the folder for the output files. {IO_FOLDER_CHOICE_HELP_TEXT}
            <p>An additional option is the following:</p>
            <ul>
                <li><i>Same folder as image</i>: Place the output file in the same folder that the source image
                is located.</li>
            </ul>
            <p></p>
            <p>{IO_WITH_METADATA_HELP_TEXT} {USING_METADATA_TAGS_REF}. For instance, if you have a metadata tag
            named "Plate", you can create a per-plate folder by selecting one the subfolder options and then
            specifying the subfolder name as "\g&lt;Plate&gt;". The module will substitute the metadata values
            for the current image set for any metadata tags in the folder name.{USING_METADATA_HELP_REF}.</p>
            <p>If the subfolder does not exist when the pipeline is run, CellProfiler will create it.</p>
            <p>If you are creating nested subfolders using the sub-folder options, you can specify the
            additional folders separated with slashes. For example, "Outlines/Plate1" will create a "Plate1"
            folder in the "Outlines" folder, which in turn is under the Default Input/Output Folder. The use of
            a forward slash ("/") as a folder separator will avoid ambiguity between the various operating
            systems.</p>
            """.format(**{
                "IO_FOLDER_CHOICE_HELP_TEXT": cellprofiler.preferences.IO_FOLDER_CHOICE_HELP_TEXT,
                "IO_WITH_METADATA_HELP_TEXT": cellprofiler.preferences.IO_WITH_METADATA_HELP_TEXT,
                "USING_METADATA_HELP_REF": cellprofiler.gui.help.USING_METADATA_HELP_REF,
                "USING_METADATA_TAGS_REF": cellprofiler.gui.help.USING_METADATA_TAGS_REF
            })
        )

        self.bit_depth = cellprofiler.setting.Choice(
            "Image bit depth",
            [
                BIT_DEPTH_8,
                BIT_DEPTH_16,
                BIT_DEPTH_FLOAT
            ],
            doc="""
            Select the bit-depth at which you want to save the images.
            <i>{BIT_DEPTH_FLOAT}</i> saves the image as floating-point decimals
            with 64-bit precision in its raw form, typically scaled between
            0 and 1.
            <b>{BIT_DEPTH_16} and {BIT_DEPTH_FLOAT} images are supported only
            for TIFF formats. Currently, saving images in 12-bit is not supported.</b>
            """.format(**{
                "BIT_DEPTH_FLOAT": BIT_DEPTH_FLOAT,
                "BIT_DEPTH_16": BIT_DEPTH_16
            })
        )

        self.overwrite = cellprofiler.setting.Binary(
            "Overwrite existing files without warning?",
            False,
            doc="""
            Select <i>{YES}</i> to automatically overwrite a file if it already exists.
            Select <i>{NO}</i> to be prompted for confirmation first.
            <p>If you are running the pipeline on a computing cluster,
            select <i>{YES}</i> since you will not be able to intervene and answer the confirmation prompt.</p>
            """.format(**{
                "NO": cellprofiler.setting.NO,
                "YES": cellprofiler.setting.YES
            })
        )

        self.when_to_save = cellprofiler.setting.Choice(
            "When to save",
            [
                WS_EVERY_CYCLE,
                WS_FIRST_CYCLE,
                WS_LAST_CYCLE
            ],
            WS_EVERY_CYCLE,
            doc="""
            <a id="when_to_save" name='when_to_save'><i>(Used only when saving non-movie files)</i><br>
            Specify at what point during pipeline execution to save file(s).</a>
            <ul>
                <li><i>{WS_EVERY_CYCLE}:</i> Useful for when the image of interest is created every cycle and
                is not dependent on results from a prior cycle.</li>
                <li><i>{WS_FIRST_CYCLE}:</i> Useful for when you are saving an aggregate image created on the
                first cycle, e.g., <b>CorrectIlluminationCalculate</b> with the <i>All</i> setting used on
                images obtained directly from <b>NamesAndTypes</b>.</li>
                <li><i>{WS_LAST_CYCLE}</i> Useful for when you are saving an aggregate image completed on the
                last cycle, e.g., <b>CorrectIlluminationCalculate</b> with the <i>All</i> setting used on
                intermediate images generated during each cycle.</li>
            </ul>
            """.format(**{
                "WS_EVERY_CYCLE": WS_EVERY_CYCLE,
                "WS_FIRST_CYCLE": WS_FIRST_CYCLE,
                "WS_LAST_CYCLE": WS_LAST_CYCLE
            })
        )

        self.update_file_names = cellprofiler.setting.Binary(
            "Record the file and path information to the saved image?",
            False,
            doc="""
            Select <i>{YES}</i> to store filename and pathname data for each of the new files created via this
            module as a per-image measurement.
            <p>Instances in which this information may be useful include:</p>
            <ul>
                <li>Exporting measurements to a database, allowing access to the saved image. If you are using
                the machine-learning tools or image viewer in CellProfiler Analyst, for example, you will want
                to enable this setting if you want the saved images to be displayed along with the original
                images.</li>
            </ul>
            """.format(**{
                "YES": cellprofiler.setting.YES
            })
        )

        self.create_subdirectories = cellprofiler.setting.Binary(
            "Create subfolders in the output folder?",
            False,
            doc="""
            Select <i>{YES}</i> to create subfolders to match the input image folder structure.
            """.format(**{
                "YES": cellprofiler.setting.YES
            })
        )

        self.root_dir = cellprofiler.setting.DirectoryPath(
            "Base image folder",
            doc="""
            <i>Used only if creating subfolders in the output folder</i> In subfolder mode, <b>SaveImages</b>
            determines the folder for an image file by examining the path of the matching input file. The path
            that SaveImages uses is relative to the image folder chosen using this setting. As an example,
            input images might be stored in a folder structure of "images{sep}<i>experiment-name</i>{sep}
            <i>date</i>{sep}<i>plate-name</i>". If the image folder is "images", <b>SaveImages</b> will store
            images in the subfolder, "<i>experiment-name</i>{sep}<i>date</i>{sep}<i>plate-name</i>". If the
            image folder is "images{sep}<i>experiment-name</i>", <b>SaveImages</b> will store images in the
            subfolder, <i>date</i>{sep}<i>plate-name</i>".
            """.format(sep=os.path.sep)
        )

    def settings(self):
        """Return the settings in the order to use when saving"""
        return [self.save_image_or_figure, self.image_name,
                self.file_name_method, self.file_image_name,
                self.single_file_name, self.number_of_digits,
                self.wants_file_name_suffix,
                self.file_name_suffix, self.file_format,
                self.pathname, self.bit_depth,
                self.overwrite, self.when_to_save,
                self.update_file_names, self.create_subdirectories,
                self.root_dir]

    def visible_settings(self):
        """Return only the settings that should be shown"""
        result = [
            self.save_image_or_figure,
            self.image_name,
            self.file_name_method
        ]

        if self.file_name_method == FN_FROM_IMAGE:
            result += [self.file_image_name, self.wants_file_name_suffix]
            if self.wants_file_name_suffix:
                result.append(self.file_name_suffix)
        elif self.file_name_method == FN_SEQUENTIAL:
            self.single_file_name.text = SEQUENTIAL_NUMBER_TEXT
            # XXX - Change doc, as well!
            result.append(self.single_file_name)
            result.append(self.number_of_digits)
        elif self.file_name_method == FN_SINGLE_NAME:
            self.single_file_name.text = SINGLE_NAME_TEXT
            result.append(self.single_file_name)
        else:
            raise NotImplementedError("Unhandled file name method: %s" % self.file_name_method)
        result.append(self.file_format)
        supports_16_bit = (self.file_format == FF_TIFF and self.save_image_or_figure == IF_IMAGE)
        if supports_16_bit:
            # TIFF supports 8 & 16-bit, all others are written 8-bit
            result.append(self.bit_depth)
        result.append(self.pathname)
        result.append(self.overwrite)
        if self.save_image_or_figure != IF_MOVIE:
            result.append(self.when_to_save)
        result.append(self.update_file_names)
        if self.file_name_method == FN_FROM_IMAGE:
            result.append(self.create_subdirectories)
            if self.create_subdirectories:
                result.append(self.root_dir)
        return result

    @property
    def module_key(self):
        return "%s_%d" % (self.module_name, self.module_num)

    def prepare_group(self, workspace, grouping, image_numbers):
        d = self.get_dictionary(workspace.image_set_list)
        if self.save_image_or_figure == IF_MOVIE:
            d['N_FRAMES'] = len(image_numbers)
            d['CURRENT_FRAME'] = 0
        return True

    def prepare_to_create_batch(self, workspace, fn_alter_path):
        self.pathname.alter_for_create_batch_files(fn_alter_path)
        if self.create_subdirectories:
            self.root_dir.alter_for_create_batch_files(fn_alter_path)

    def run(self, workspace):
        """Run the module

        pipeline     - instance of CellProfiler.Pipeline for this run
        workspace    - the workspace contains:
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - display within this frame (or None to not display)
        """
        if self.save_image_or_figure.value in (IF_IMAGE, IF_MASK, IF_CROPPING):
            should_save = self.run_image(workspace)
        elif self.save_image_or_figure == IF_MOVIE:
            should_save = self.run_movie(workspace)
        else:
            raise NotImplementedError(("Saving a %s is not yet supported" %
                                       self.save_image_or_figure))
        workspace.display_data.filename = self.get_filename(
                workspace, make_dirs=False, check_overwrite=False)

    def is_aggregation_module(self):
        '''SaveImages is an aggregation module when it writes movies'''
        return self.save_image_or_figure == IF_MOVIE or \
               self.when_to_save == WS_LAST_CYCLE

    def display(self, workspace, figure):
        if self.show_window:
            if self.save_image_or_figure == IF_MOVIE:
                return
            figure.set_subplots((1, 1))
            outcome = ("Wrote %s" if workspace.display_data.wrote_image
                       else "Did not write %s")
            figure.subplot_table(0, 0, [[outcome %
                                         workspace.display_data.filename]])

    def run_image(self, workspace):
        """Handle saving an image"""
        #
        # First, check to see if we should save this image
        #
        if self.when_to_save == WS_FIRST_CYCLE:
            d = self.get_dictionary(workspace.image_set_list)
            if workspace.measurements[cellprofiler.measurement.IMAGE, cellprofiler.measurement.GROUP_INDEX] > 1:
                workspace.display_data.wrote_image = False
                self.save_filename_measurements(workspace)
                return
            d["FIRST_IMAGE"] = False

        elif self.when_to_save == WS_LAST_CYCLE:
            workspace.display_data.wrote_image = False
            self.save_filename_measurements(workspace)
            return
        self.save_image(workspace)
        return True

    def run_movie(self, workspace):
        out_file = self.get_filename(workspace, check_overwrite=False)
        # overwrite checks are made only for first frame.
        d = self.get_dictionary(workspace.image_set_list)
        if d["CURRENT_FRAME"] == 0 and os.path.exists(out_file):
            if not self.check_overwrite(out_file, workspace):
                d["CURRENT_FRAME"] = "Ignore"
                return
            else:
                # Have to delete the old movie before making the new one
                os.remove(out_file)
        elif d["CURRENT_FRAME"] == "Ignore":
            return

        image = workspace.image_set.get_image(self.image_name.value)
        pixels = image.pixel_data
        pixels = pixels * 255
        frames = d['N_FRAMES']
        current_frame = d["CURRENT_FRAME"]
        d["CURRENT_FRAME"] += 1
        self.do_save_image(workspace, out_file, pixels, bioformats.omexml.PT_UINT8,
                           t=current_frame, size_t=frames)

    def post_group(self, workspace, *args):
        if (self.when_to_save == WS_LAST_CYCLE and self.save_image_or_figure != IF_MOVIE):
            self.save_image(workspace)

    def do_save_image(self, workspace, filename, pixels, pixel_type,
                      c=0, z=0, t=0,
                      size_c=1, size_z=1, size_t=1,
                      channel_names=None):
        '''Save image using bioformats

        workspace - the current workspace

        filename - save to this filename

        pixels - the image to save

        pixel_type - save using this pixel type

        c - the image's channel index

        z - the image's z index

        t - the image's t index

        sizeC - # of channels in the stack

        sizeZ - # of z stacks

        sizeT - # of timepoints in the stack

        channel_names - names of the channels (make up names if not present
        '''
        bioformats.formatwriter.write_image(filename, pixels, pixel_type,
                                            c=c, z=z, t=t,
                                            size_c=size_c, size_z=size_z, size_t=size_t,
                                            channel_names=channel_names)

    def save_image(self, workspace):
        if self.show_window:
            workspace.display_data.wrote_image = False

        image = workspace.image_set.get_image(self.image_name.value)

        if image.volumetric and self.file_format.value != FF_TIFF:
            raise RuntimeError(
                "Unsupported file format {} for 3D pipeline. Use {} format when processing images as 3D.".format(
                    self.file_format.value,
                    FF_TIFF
                )
            )

        if self.save_image_or_figure.value == IF_IMAGE:
            pixels = image.pixel_data
        elif self.save_image_or_figure.value == IF_MASK:
            pixels = image.mask
        elif self.save_image_or_figure.value == IF_CROPPING:
            pixels = image.crop_mask

        if self.get_bit_depth() == BIT_DEPTH_8:
            pixels = skimage.util.img_as_ubyte(pixels)
        elif self.get_bit_depth() == BIT_DEPTH_16:
            pixels = skimage.util.img_as_uint(pixels)
        elif self.get_bit_depth() == BIT_DEPTH_FLOAT:
            pixels = skimage.util.img_as_float(pixels)

        filename = self.get_filename(workspace)

        if filename is None:  # failed overwrite check
            return

        skimage.io.imsave(filename, pixels)

        if self.show_window:
            workspace.display_data.wrote_image = True

        if self.when_to_save != WS_LAST_CYCLE:
            self.save_filename_measurements(workspace)

    def check_overwrite(self, filename, workspace):
        '''Check to see if it's legal to overwrite a file

        Throws an exception if can't overwrite and no interaction available.
        Returns False if can't overwrite, otherwise True.
        '''
        if not self.overwrite.value and os.path.isfile(filename):
            try:
                return workspace.interaction_request(self, workspace.measurements.image_set_number, filename) == "Yes"
            except workspace.NoInteractionException:
                raise ValueError(
                        'SaveImages: trying to overwrite %s in headless mode, but Overwrite files is set to "No"' % (
                            filename))
        return True

    def handle_interaction(self, image_set_number, filename):
        '''handle an interaction request from check_overwrite()'''
        import wx
        dlg = wx.MessageDialog(wx.GetApp().TopWindow,
                               "%s #%d, set #%d - Do you want to overwrite %s?" % \
                               (self.module_name, self.module_num, image_set_number, filename),
                               "Warning: overwriting file", wx.YES_NO | wx.ICON_QUESTION)
        result = dlg.ShowModal() == wx.ID_YES
        return "Yes" if result else "No"

    def save_filename_measurements(self, workspace):
        if self.update_file_names.value:
            filename = self.get_filename(workspace, make_dirs=False,
                                         check_overwrite=False)
            pn, fn = os.path.split(filename)
            url = cellprofiler.modules.loadimages.pathname2url(filename)
            workspace.measurements.add_measurement(cellprofiler.measurement.IMAGE,
                                                   self.file_name_feature,
                                                   fn,
                                                   can_overwrite=True)
            workspace.measurements.add_measurement(cellprofiler.measurement.IMAGE,
                                                   self.path_name_feature,
                                                   pn,
                                                   can_overwrite=True)
            workspace.measurements.add_measurement(cellprofiler.measurement.IMAGE,
                                                   self.url_feature,
                                                   url,
                                                   can_overwrite=True)

    @property
    def file_name_feature(self):
        return '_'.join((cellprofiler.modules.loadimages.C_FILE_NAME, self.image_name.value))

    @property
    def path_name_feature(self):
        return '_'.join((cellprofiler.modules.loadimages.C_PATH_NAME, self.image_name.value))

    @property
    def url_feature(self):
        return '_'.join((cellprofiler.modules.loadimages.C_URL, self.image_name.value))

    @property
    def source_file_name_feature(self):
        '''The file name measurement for the exemplar disk image'''
        return '_'.join((cellprofiler.modules.loadimages.C_FILE_NAME, self.file_image_name.value))

    def source_path(self, workspace):
        '''The path for the image data, or its first parent with a path'''
        if self.file_name_method.value == FN_FROM_IMAGE:
            path_feature = '%s_%s' % (cellprofiler.modules.loadimages.C_PATH_NAME, self.file_image_name.value)
            assert workspace.measurements.has_feature(cellprofiler.measurement.IMAGE, path_feature), \
                "Image %s does not have a path!" % self.file_image_name.value
            return workspace.measurements.get_current_image_measurement(path_feature)

        # ... otherwise, chase the cpimage hierarchy looking for an image with a path
        cur_image = workspace.image_set.get_image(self.image_name.value)
        while cur_image.path_name is None:
            cur_image = cur_image.parent_image
            assert cur_image is not None, "Could not determine source path for image %s' % (self.image_name.value)"
        return cur_image.path_name

    def get_measurement_columns(self, pipeline):
        if self.update_file_names.value:
            return [(cellprofiler.measurement.IMAGE,
                     self.file_name_feature,
                     cellprofiler.measurement.COLTYPE_VARCHAR_FILE_NAME),
                    (cellprofiler.measurement.IMAGE,
                     self.path_name_feature,
                     cellprofiler.measurement.COLTYPE_VARCHAR_PATH_NAME)]
        else:
            return []

    def get_filename(self, workspace, make_dirs=True, check_overwrite=True):
        "Concoct a filename for the current image based on the user settings"

        measurements = workspace.measurements
        if self.file_name_method == FN_SINGLE_NAME:
            filename = self.single_file_name.value
            filename = workspace.measurements.apply_metadata(filename)
        elif self.file_name_method == FN_SEQUENTIAL:
            filename = self.single_file_name.value
            filename = workspace.measurements.apply_metadata(filename)
            n_image_sets = workspace.measurements.image_set_count
            ndigits = int(numpy.ceil(numpy.log10(n_image_sets + 1)))
            ndigits = max((ndigits, self.number_of_digits.value))
            padded_num_string = str(measurements.image_set_number).zfill(ndigits)
            filename = '%s%s' % (filename, padded_num_string)
        else:
            file_name_feature = self.source_file_name_feature
            filename = measurements.get_current_measurement('Image',
                                                            file_name_feature)
            filename = os.path.splitext(filename)[0]
            if self.wants_file_name_suffix:
                suffix = self.file_name_suffix.value
                suffix = workspace.measurements.apply_metadata(suffix)
                filename += suffix

        filename = "%s.%s" % (filename, self.get_file_format())
        pathname = self.pathname.get_absolute_path(measurements)
        if self.create_subdirectories:
            image_path = self.source_path(workspace)
            subdir = os.path.relpath(image_path, self.root_dir.get_absolute_path())
            pathname = os.path.join(pathname, subdir)
        if len(pathname) and not os.path.isdir(pathname) and make_dirs:
            try:
                os.makedirs(pathname)
            except:
                #
                # On cluster, this can fail if the path was created by
                # another process after this process found it did not exist.
                #
                if not os.path.isdir(pathname):
                    raise
        result = os.path.join(pathname, filename)
        if check_overwrite and not self.check_overwrite(result, workspace):
            return

        if check_overwrite and os.path.isfile(result):
            try:
                os.remove(result)
            except:
                import bioformats
                bioformats.clear_image_reader_cache()
                os.remove(result)
        return result

    def get_file_format(self):
        """Return the file format associated with the extension in self.file_format
        """
        if self.save_image_or_figure == IF_MOVIE:
            return FF_TIFF

        return self.file_format.value

    def get_bit_depth(self):
        if (self.save_image_or_figure == IF_IMAGE and self.get_file_format() == FF_TIFF):
            return self.bit_depth.value
        else:
            return BIT_DEPTH_8

    def upgrade_settings(self, setting_values, variable_revision_number, module_name, from_matlab):
        if variable_revision_number == 11:
            if setting_values[0] == "Objects":
                raise NotImplementedError(
                    "Unsupported image type: Objects. Use <i>ConvertObjectsToImage</i> to create an image."
                )

            if setting_values[10] in ("bmp", "mat"):
                raise NotImplementedError("Unsupported file format: {}".format(setting_values[10]))
            elif setting_values[10] == "tif":
                setting_values[10] = FF_TIFF
            elif setting_values[10] == "jpg":
                setting_values[10] = FF_JPEG

            new_setting_values = setting_values[:2]
            new_setting_values += setting_values[4:15]
            new_setting_values += setting_values[18:-1]

            setting_values = new_setting_values

            variable_revision_number = 12

        return setting_values, variable_revision_number, False

    def validate_module(self, pipeline):
        if (self.save_image_or_figure in (IF_IMAGE, IF_MASK, IF_CROPPING) and
                    self.when_to_save in (WS_FIRST_CYCLE, WS_EVERY_CYCLE)):
            #
            # Make sure that the image name is available on every cycle
            #
            for setting in cellprofiler.setting.get_name_providers(pipeline,
                                                                   self.image_name):
                if setting.provided_attributes.get(cellprofiler.setting.AVAILABLE_ON_LAST_ATTRIBUTE):
                    #
                    # If we fell through, then you can only save on the last cycle
                    #
                    raise cellprofiler.setting.ValidationError("%s is only available after processing all images in an image group" %
                                                               self.image_name.value,
                                                               self.when_to_save)

        # XXX - should check that if file_name_method is
        # FN_FROM_IMAGE, that the named image actually has the
        # required path measurement

        # Make sure metadata tags exist
        if self.file_name_method == FN_SINGLE_NAME or \
                (self.file_name_method == FN_FROM_IMAGE and self.wants_file_name_suffix.value):
            text_str = self.single_file_name.value if self.file_name_method == FN_SINGLE_NAME else self.file_name_suffix.value
            undefined_tags = pipeline.get_undefined_metadata_tags(text_str)
            if len(undefined_tags) > 0:
                raise cellprofiler.setting.ValidationError(
                        "%s is not a defined metadata tag. Check the metadata specifications in your load modules" %
                        undefined_tags[0],
                        self.single_file_name if self.file_name_method == FN_SINGLE_NAME else self.file_name_suffix)

    def volumetric(self):
        return True


class SaveImagesDirectoryPath(cellprofiler.setting.DirectoryPath):
    '''A specialized version of DirectoryPath to handle saving in the image dir'''

    def __init__(self, text, file_image_name, doc):
        '''Constructor
        text - explanatory text to display
        file_image_name - the file_image_name setting so we can save in same dir
        doc - documentation for user
        '''
        super(SaveImagesDirectoryPath, self).__init__(
                text, dir_choices=[
                    cellprofiler.setting.DEFAULT_OUTPUT_FOLDER_NAME, cellprofiler.setting.DEFAULT_INPUT_FOLDER_NAME,
                    PC_WITH_IMAGE, cellprofiler.setting.ABSOLUTE_FOLDER_NAME,
                    cellprofiler.setting.DEFAULT_OUTPUT_SUBFOLDER_NAME,
                    cellprofiler.setting.DEFAULT_INPUT_SUBFOLDER_NAME], doc=doc)
        self.file_image_name = file_image_name

    def get_absolute_path(self, measurements=None, image_set_index=None):
        if self.dir_choice == PC_WITH_IMAGE:
            path_name_feature = "PathName_%s" % self.file_image_name.value
            return measurements.get_current_image_measurement(path_name_feature)
        return super(SaveImagesDirectoryPath, self).get_absolute_path(
                measurements, image_set_index)

    def test_valid(self, pipeline):
        if self.dir_choice not in self.dir_choices:
            raise cellprofiler.setting.ValidationError("%s is not a valid directory option" %
                                                       self.dir_choice, self)
