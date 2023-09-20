"""
RunImageJMacro
==============

**RunImageJMacro** exports image(s), executes an ImageJ macro on them and
then loads resulting image(s) back into CellProfiler.

To operate, this module requires that the user has installed ImageJ (or FIJI)
elsewhere on their system. It can be downloaded `here`_.

You should point the module to the ImageJ executable in it's installation folder.

The ImageJ macro itself should specify which input images and variables are needed.

On running, CellProfiler saves required images into a temporary folder, executes the
macro and then attempts to load images which the macro should save into that same
temporary folder.

See `this guide`_ for a full tutorial.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

.. _here: https://imagej.nih.gov/ij/download.html
.. _this guide: https://github.com/CellProfiler/CellProfiler/wiki/RunImageJMacro

"""
import logging

import itertools
import os
import subprocess

from cellprofiler_core.image import Image
from cellprofiler.modules import _help
from cellprofiler_core.module import Module
from cellprofiler_core.setting.text import Filename, ImageName, Text, Directory
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting._settings_group import SettingsGroup
from cellprofiler_core.setting import Divider, HiddenCount, Binary
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.preferences import get_default_output_directory, get_headless

import random
import skimage.io


LOGGER = logging.getLogger(__name__)

class RunImageJMacro(Module):
    module_name = "RunImageJMacro"
    variable_revision_number = 1
    category = "Advanced"
    doi = {"Please cite the following when using RunImageJMacro:": 'https://doi.org/10.1038/nmeth.2089'}

    def create_settings(self):

        self.executable_directory = Directory(
            "Executable directory", allow_metadata=False, doc="""\
Select the folder containing the executable. MacOS users should select the directory where Fiji.app lives. Windows users 
should select the directory containing ImageJ-win64.exe (usually corresponding to the Fiji.app folder).

{IO_FOLDER_CHOICE_HELP_TEXT}
""".format(**{
                "IO_FOLDER_CHOICE_HELP_TEXT": _help.IO_FOLDER_CHOICE_HELP_TEXT
            }))

        def set_directory_fn_executable(path):
            dir_choice, custom_path = self.executable_directory.get_parts_from_path(path)
            self.executable_directory.join_parts(dir_choice, custom_path)

        self.executable_file = Filename(
            "Executable", "ImageJ.exe", doc="Select your executable. MacOS users should select the Fiji.app "
                                            "application. Windows user should select the ImageJ-win64.exe executable",
            get_directory_fn=self.executable_directory.get_absolute_path,
            set_directory_fn=set_directory_fn_executable,
            browse_msg="Choose executable file"
        )

        self.macro_directory = Directory(
            "Macro directory", allow_metadata=False, doc=f"""Select the folder containing the macro.
{_help.IO_FOLDER_CHOICE_HELP_TEXT}""")

        def set_directory_fn_macro(path):
            dir_choice, custom_path = self.macro_directory.get_parts_from_path(path)
            self.macro_directory.join_parts(dir_choice, custom_path)

        self.macro_file = Filename(
            "Macro", "macro.py", doc="Select your macro file.",
            get_directory_fn=self.macro_directory.get_absolute_path,
            set_directory_fn=set_directory_fn_macro,
            browse_msg="Choose macro file"
        )

        self.debug_mode = Binary(
            "Debug mode: Prevent deletion of temporary files",
            False,
            doc="This setting only applies when running in Test Mode."
                "If enabled, temporary folders used to communicate with ImageJ will not be cleared automatically."
                "You'll need to remove them manually. This can be helpful when trying to debug a macro."
                "Temporary folder location will be printed to the console."
        )

        self.add_directory = Text(
            "What variable in your macro defines the folder ImageJ should use?",
            "Directory",
            doc="""Because CellProfiler will save the output images in a temporary directory, this directory should be 
specified as a variable in the macro script. It is assumed that the macro will use this directory variable 
to obtain the full path to the inputted image. Enter the variable name here. CellProfiler will create a 
temporary directory and assign its path as a value to this variable."""
        )

        self.image_groups_in = []
        self.image_groups_out = []

        self.macro_variables_list = []

        self.image_groups_in_count = HiddenCount(self.image_groups_in)
        self.image_groups_out_count = HiddenCount(self.image_groups_out)
        self.macro_variable_count = HiddenCount(self.macro_variables_list)

        self.add_image_in(can_delete=False)
        self.add_image_button_in = DoSomething("", 'Add another input image', self.add_image_in)

        self.add_image_out(can_delete=False)
        self.add_image_button_out = DoSomething("", 'Add another output image', self.add_image_out)

        self.add_variable_button_out = DoSomething("Does your macro expect variables?", "Add another variable", self.add_macro_variables)

    def add_macro_variables(self, can_delete=True):
        group = SettingsGroup()
        if can_delete:
            group.append("divider", Divider(line=False))
        group.append(
            "variable_name",
            Text(
                'What variable name is your macro expecting?',
                "None",
                doc='Enter the variable name that your macro is expecting. '
            )
        )
        group.append(
            "variable_value",
            Text(
                "What value should this variable have?",
                "None",
                doc="Enter the desire value for this variable."),
        )
        if len(self.macro_variables_list) == 0:  # Insert space between 1st two images for aesthetics
            group.append("extra_divider", Divider(line=False))

        if can_delete:
            group.append("remover", RemoveSettingButton("", "Remove this variable", self.macro_variables_list, group))

        self.macro_variables_list.append(group)

    def add_image_in(self, can_delete=True):
        """Add an image to the image_groups collection
        can_delete - set this to False to keep from showing the "remove"
                     button for images that must be present.
        """
        group = SettingsGroup()
        if can_delete:
            group.append("divider", Divider(line=False))
        group.append(
            "image_name",
            ImageSubscriber(
                'Select an image to send to your macro',
                "None",
                doc="Select an image to send to your macro. "
            )
        )
        group.append(
            "output_filename",
            Text(
                "What should this image temporarily saved as?",
                "None.tiff",
                doc='Enter the filename of the image to be used by the macro. This should be set to the name expected '
                    'by the macro file.'),
        )
        if len(self.image_groups_in) == 0:  # Insert space between 1st two images for aesthetics
            group.append("extra_divider", Divider(line=False))

        if can_delete:
            group.append("remover", RemoveSettingButton("", "Remove this image",  self.image_groups_in, group))

        self.image_groups_in.append(group)

    def add_image_out(self, can_delete=True):
        """Add an image to the image_groups collection
        can_delete - set this to False to keep from showing the "remove"
                     button for images that must be present.
        """
        group = SettingsGroup()
        if can_delete:
            group.append("divider", Divider(line=False))
        group.append(
            "input_filename",
            Text(
                "What is the image filename CellProfiler should load?",
                "None.tiff",
                doc="Enter the image filename CellProfiler should load. This should be set to the output filename "
                    "written in the macro file. The image written by the macro will be saved in a temporary directory "
                    "and read by CellProfiler."),
        )

        group.append(
            "image_name",
            ImageName(
                r'What should CellProfiler call the loaded image?',
                "None",
                doc='Enter a name to assign to the new image loaded by CellProfiler. This image will be added to your '
                    'workspace. '
            )
        )

        if len(self.image_groups_out) == 0:  # Insert space between 1st two images for aesthetics
            group.append("extra_divider", Divider(line=False))

        if can_delete:
            group.append("remover", RemoveSettingButton("", "Remove this image",  self.image_groups_out, group))

        self.image_groups_out.append(group)

    def settings(self):
        result = [self.image_groups_in_count, self.image_groups_out_count, self.macro_variable_count]
        result += [self.executable_directory, self.executable_file, self.macro_directory, self.macro_file, self.add_directory]
        for image_group_in in self.image_groups_in:
            result += [image_group_in.image_name, image_group_in.output_filename]
        for image_group_out in self.image_groups_out:
            result += [image_group_out.input_filename, image_group_out.image_name]
        for macro_variable in self.macro_variables_list:
            result +=[macro_variable.variable_name, macro_variable.variable_value]
        return result

    def visible_settings(self):
        visible_settings = [self.executable_directory, self.executable_file, self.macro_directory, self.macro_file,
                            self.debug_mode, self.add_directory]
        for image_group_in in self.image_groups_in:
            visible_settings += image_group_in.visible_settings()
        visible_settings += [self.add_image_button_in]
        for image_group_out in self.image_groups_out:
            visible_settings += image_group_out.visible_settings()
        visible_settings += [self.add_image_button_out]
        for macro_variable in self.macro_variables_list:
            visible_settings += macro_variable.visible_settings()
        visible_settings += [self.add_variable_button_out]
        return visible_settings

    def prepare_settings(self, setting_values):
        image_groups_in_count = int(setting_values[0])
        image_groups_out_count = int(setting_values[1])
        macro_variable_count = int(setting_values[2])

        del self.image_groups_in[image_groups_in_count:]
        del self.image_groups_out[image_groups_out_count:]
        del self.macro_variables_list[macro_variable_count:]

        while len(self.image_groups_in) < image_groups_in_count:
            self.add_image_in()
        while len(self.image_groups_out) < image_groups_out_count:
            self.add_image_out()
        while len(self.macro_variables_list) < macro_variable_count:
            self.add_macro_variables()


    def stringify_metadata(self, dir):
        met_string = ""
        met_string += self.add_directory.value + "='" + dir + "', "
        for var in self.macro_variables_list:
            met_string += var.variable_name.value + "='" + var.variable_value.value + "', "
        return met_string[:-2]

    def run(self, workspace):
        default_output_directory = get_default_output_directory()
        tag = "runimagejmacro_" + str(random.randint(100000, 999999))
        tempdir = os.path.join(default_output_directory, tag)
        os.makedirs(tempdir, exist_ok=True)
        try:
            for image_group in self.image_groups_in:
                image = workspace.image_set.get_image(image_group.image_name.value)
                image_pixels = image.pixel_data
                skimage.io.imsave(os.path.join(tempdir, image_group.output_filename.value), image_pixels)

            if self.executable_file.value[-4:] == ".app":
                executable = os.path.join(default_output_directory, self.executable_directory.value.split("|")[1], self.executable_file.value, "Contents/MacOS/ImageJ-macosx")
            else:
                executable = os.path.join(default_output_directory, self.executable_directory.value.split("|")[1], self.executable_file.value)
            cmd = [executable, "--headless", "console", "--run", os.path.join(default_output_directory, self.macro_directory.value.split("|")[1], self.macro_file.value)]

            cmd += [self.stringify_metadata(tempdir)]

            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for image_group in self.image_groups_out:
                if not os.path.exists(os.path.join(tempdir, image_group.input_filename.value)):
                    # Cleanup the error logs for display, we want to remove less-useful lines to keep it succinct.
                    reject = ('console:', 'Java Hot', 'at org', 'at java', '[WARNING]', '\t')
                    # ImageJ tends to report the same few lines over and over, so we'll use a dict as an ordered set.
                    err = {}
                    for line in result.stdout.splitlines():
                        if len(line.strip()) > 0 and not line.startswith(reject):
                            err[line] = None
                    if len(err) > 1:
                        # Error appears when file loading fails, but can also show up if the macro failed to generate
                        # an output image. We remove this if it wasn't the only error, as it can be confusing.
                        err.pop('Unsupported format or not found', None)
                    err = "\n".join(err.keys())
                    msg = f"CellProfiler couldn't find the output expected from the ImageJ Macro," \
                          f"\n File {image_group.input_filename.value} was missing."
                    if err:
                        msg += f"\n\nImageJ logs contained the following: \n{err}"
                    raise FileNotFoundError("Missing file", msg)
                image_pixels = skimage.io.imread(os.path.join(tempdir, image_group.input_filename.value))
                workspace.image_set.add(image_group.image_name.value, Image(image_pixels, convert=False))
        finally:
            want_delete = True
            # Optionally clean up temp directory regardless of macro success
            if workspace.pipeline.test_mode and self.debug_mode:
                want_delete = False
                if not get_headless():
                    import wx
                    message = f"Debugging was enabled.\nTemporary folder was not deleted automatically" \
                              f"\n\nTemporary subfolder is {os.path.split(tempdir)[-1]} in your Default Output Folder\n\nDo you want to delete it now?"
                    with wx.Dialog(None, title="RunImageJMacro Debug Mode") as dlg:
                        text_sizer = dlg.CreateTextSizer(message)
                        sizer = wx.BoxSizer(wx.VERTICAL)
                        dlg.SetSizer(sizer)
                        button_sizer = dlg.CreateStdDialogButtonSizer(flags=wx.YES | wx.NO)
                        open_temp_folder_button = wx.Button(
                            dlg, -1, "Open temporary folder"
                        )
                        button_sizer.Insert(0, open_temp_folder_button)

                        def on_open_temp_folder(event):
                            import sys
                            if sys.platform == "win32":
                                os.startfile(tempdir)
                            else:
                                import subprocess
                                subprocess.call(["open", tempdir, ])

                        open_temp_folder_button.Bind(wx.EVT_BUTTON, on_open_temp_folder)
                        sizer.Add(text_sizer, 0, wx.EXPAND | wx.ALL, 10)
                        sizer.Add(button_sizer, 0, wx.EXPAND | wx.ALL, 10)
                        dlg.SetEscapeId(wx.ID_NO)
                        dlg.SetAffirmativeId(wx.ID_YES)
                        dlg.Fit()
                        dlg.CenterOnParent()
                        if dlg.ShowModal() == wx.ID_YES:
                            want_delete = True
            if want_delete:
                try:
                    for subdir, dirs, files in os.walk(tempdir):
                        for file in files:
                            os.remove(os.path.join(tempdir, file))
                    os.removedirs(tempdir)
                except:
                    LOGGER.error("Unable to delete temporary directory, files may be in use by another program.")
                    LOGGER.error("Temp folder is subfolder {tempdir} in your Default Output Folder.\nYou may need to remove it manually.")
            else:
                LOGGER.error(f"Debugging was enabled.\nDid not remove temporary folder at {tempdir}")

        pixel_data = []
        image_names = []

        if self.show_window:
            for x in itertools.chain(self.image_groups_in, self.image_groups_out):
                pixel_data.append(workspace.image_set.get_image(x.image_name.value).pixel_data)
                image_names.append(x.image_name.value)

        workspace.display_data.pixel_data = pixel_data
        workspace.display_data.display_names = image_names
        workspace.display_data.dimensions = workspace.image_set.get_image(
            self.image_groups_out[0].image_name.value).dimensions

    def display(self, workspace, figure):
        import matplotlib.cm

        pixel_data = workspace.display_data.pixel_data
        display_names = workspace.display_data.display_names

        columns = (len(pixel_data) + 1) // 2

        figure.set_subplots((columns, 2), dimensions=workspace.display_data.dimensions)

        for i in range(len(pixel_data)):
            if pixel_data[i].shape[-1] in (3, 4):
                cmap = None
            elif pixel_data[i].dtype.kind == "b":
                cmap = matplotlib.cm.binary_r
            else:
                cmap = matplotlib.cm.Greys_r

            figure.subplot_imshow(
                i % columns,
                int(i / columns),
                pixel_data[i],
                title=display_names[i],
                sharexy=figure.subplot(0, 0),
                colormap=cmap,
            )




