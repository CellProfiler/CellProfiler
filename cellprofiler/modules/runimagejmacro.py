import itertools
import os
import subprocess

from cellprofiler_core.image import Image

from cellprofiler.modules import _help
from cellprofiler_core.module import Module
from cellprofiler_core.setting.text import Pathname, Filename, ImageName, Text, Directory
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting._settings_group import SettingsGroup
from cellprofiler_core.setting import Divider, HiddenCount
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.preferences import DEFAULT_OUTPUT_FOLDER_NAME, get_default_output_directory

import random
import skimage.io


class RunImageJMacro(Module):
    module_name = "RunImageJMacro"
    variable_revision_number = 1
    category = "Advanced"

    def create_settings(self):

        self.executable_directory = Directory(
            "Executable directory", allow_metadata=False, doc="""\
Select the folder containing the executable. {IO_FOLDER_CHOICE_HELP_TEXT}
""".format(**{
                "IO_FOLDER_CHOICE_HELP_TEXT": _help.IO_FOLDER_CHOICE_HELP_TEXT
            }))

        def set_directory_fn_executable(path):
            dir_choice, custom_path = self.executable_directory.get_parts_from_path(path)
            self.executable_directory.join_parts(dir_choice, custom_path)

        self.executable_file = Filename(
            "Executable", "ImageJ.exe", doc="TODO",
            get_directory_fn=self.executable_directory.get_absolute_path,
            set_directory_fn=set_directory_fn_executable,
            browse_msg="Choose executable file"
        )

        self.macro_directory = Directory(
            "Macro directory", allow_metadata=False, doc="""\
        Select the folder containing the macro. {IO_FOLDER_CHOICE_HELP_TEXT}
        """.format(**{
                "IO_FOLDER_CHOICE_HELP_TEXT": _help.IO_FOLDER_CHOICE_HELP_TEXT
            }))


        def set_directory_fn_macro(path):
            dir_choice, custom_path = self.macro_directory.get_parts_from_path(path)
            self.macro_directory.join_parts(dir_choice, custom_path)

        self.macro_file = Filename(
            "Macro", "macro.py", doc="TODO",
            get_directory_fn=self.macro_directory.get_absolute_path,
            set_directory_fn=set_directory_fn_macro,
            browse_msg="Choose macro file"
        )

        self.image_groups_in = []
        self.image_groups_out = []

        self.macro_variables_list = []

        self.image_groups_in_count = HiddenCount(self.image_groups_in)
        self.image_groups_out_count = HiddenCount(self.image_groups_out)
        self.macro_variable_count = HiddenCount(self.macro_variables_list)

        self.add_image_in(can_delete=False)
        self.add_image_button_in = DoSomething("", 'Add another image', self.add_image_in)

        self.add_image_out(can_delete=False)
        self.add_image_button_out = DoSomething("", 'Add another image', self.add_image_out)


        self.add_directory = Text("What variable in your macro defines the folder ImageJ should use?",
                                  "Directory",
                                  doc="What variable in your macro defines the folder ImageJ should use?")
        self.add_variable_button_out = DoSomething("", "Add another variable", self.add_macro_variables)


    def add_macro_variables(self, can_delete=True):
        group = SettingsGroup()
        if can_delete:
            group.append("divider", Divider(line=False))
        group.append(
            "variable_name",
            Text(
                'What variable name is your macro expecting?',
                "None",
                doc='What variable name is your macro expecting?'
            )
        )
        group.append(
            "variable_value",
            Text(
                "What value should this variable have?",
                "None",
                doc="What value should this variable have?"),
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
                doc='Select an image to send to your macro'
            )
        )
        group.append(
            "output_filename",
            Text(
                "What should this image temporarily saved as?",
                "None.tiff",
                doc="What should this image temporarily saved as?"),
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
                "What is the image name CellProfiler should load?",
                "None.tiff",
                doc="What is the image name CellProfiler should load?"),
        )

        group.append(
            "image_name",
            ImageName(
                r'What should CellProfiler call the loaded image?',
                "None",
                doc='What should CellProfiler call the loaded image?'
            )
        )

        if len(self.image_groups_out) == 0:  # Insert space between 1st two images for aesthetics
            group.append("extra_divider", Divider(line=False))

        if can_delete:
            group.append("remover", RemoveSettingButton("", "Remove this image",  self.image_groups_out, group))

        self.image_groups_out.append(group)

    def settings(self):
        result = [self.image_groups_in_count, self.image_groups_out_count, self.macro_variable_count]
        result += [self.executable_directory, self.executable_file, self.macro_directory, self.macro_file]
        for image_group_in in self.image_groups_in:
            result += [image_group_in.image_name, image_group_in.output_filename]
        for image_group_out in self.image_groups_out:
            result += [image_group_out.input_filename, image_group_out.image_name]
        result += [self.add_directory]
        for macro_variable in self.macro_variables_list:
            result +=[macro_variable.variable_name, macro_variable.variable_value]
        return result

    def visible_settings(self):
        visible_settings = [self.executable_directory, self.executable_file, self.macro_directory, self.macro_file]
        for image_group_in in self.image_groups_in:
            visible_settings += image_group_in.visible_settings()
        visible_settings += [self.add_image_button_in]
        for image_group_out in self.image_groups_out:
            visible_settings += image_group_out.visible_settings()
        visible_settings += [self.add_image_button_out]
        visible_settings += [self.add_directory]
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
        met_string += self.add_directory.value + "=\"" + dir +  "\", "
        for var in self.macro_variables_list:
            met_string += var.variable_name.value + "='" + var.variable_value.value + "', "
        return met_string[:-2]

    def run(self, workspace):

        default_output_directory = get_default_output_directory()
        #Making a temp directory
        tag = str(random.randint(100000, 999999))
        tempdir = os.path.join(default_output_directory, tag)
        os.makedirs(tempdir, exist_ok=True)

        # Save image to the temp directory
        for image_group in self.image_groups_in:
            image = workspace.image_set.get_image(image_group.image_name.value)
            image_pixels = image.pixel_data
            skimage.io.imsave(os.path.join(tempdir, image_group.output_filename.value), image_pixels)

        # Execute the macro
        if self.executable_file.value[-4:] == ".app":
            executable = os.path.join(self.executable_directory.value.split("|")[1], self.executable_file.value, "Contents/MacOS/ImageJ-macosx")
        else:
            executable = os.path.join(self.executable_directory.value.split("|")[1], self.executable_file.value)
        cmd = [executable, "--headless", "console", "--run", os.path.join(default_output_directory, self.macro_directory.value.split("|")[1], self.macro_file.value)]

        cmd += [self.stringify_metadata(tempdir)]

        subp = subprocess.call(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        # Load images from the temp directory
        for image_group in self.image_groups_out:
            image_pixels = skimage.io.imread(os.path.join(tempdir,image_group.input_filename.value))
            workspace.image_set.add(image_group.image_name.value, Image(image_pixels, convert=False))

        #remove temp content and directory
        for subdir, dirs, files in os.walk(tempdir):
            for file in files:
                os.remove(os.path.join(tempdir, file))
        os.removedirs(tempdir)

        # Display results
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




