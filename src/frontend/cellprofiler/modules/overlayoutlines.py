"""
OverlayOutlines
===============

**OverlayOutlines** places outlines of objects over a desired image.

This module places outlines of objects on any desired image (grayscale, color, or blank).
The resulting image can be saved using the **SaveImages** module.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============
"""

from cellprofiler_core.image import Image
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Binary, Divider, SettingsGroup, Color
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting.subscriber import ImageSubscriber, LabelSubscriber
from cellprofiler_core.setting.text import ImageName

from cellprofiler_library.modules._overlayoutlines import overlay_outlines
from cellprofiler_library.opts.overlayoutlines import BrightnessMode, OutlineMode, COLOR_ORDER

FROM_IMAGES = "Image"
FROM_OBJECTS = "Objects"

NUM_FIXED_SETTINGS_V1 = 5
NUM_FIXED_SETTINGS_V2 = 6
NUM_FIXED_SETTINGS_V3 = 6
NUM_FIXED_SETTINGS_V4 = 6
NUM_FIXED_SETTINGS = 6

NUM_OUTLINE_SETTINGS_V2 = 2
NUM_OUTLINE_SETTINGS_V3 = 4
NUM_OUTLINE_SETTINGS_V4 = 2
NUM_OUTLINE_SETTINGS = 2


class OverlayOutlines(Module):
    module_name = "OverlayOutlines"
    variable_revision_number = 4
    category = "Image Processing"

    def create_settings(self):
        self.blank_image = Binary(
            "Display outlines on a blank image?",
            False,
            doc="""\
Select "*{YES}*" to produce an image of the outlines on a black background.

Select "*{NO}*" to overlay the outlines on an image you choose.
""".format(
                **{"YES": "Yes", "NO": "No"}
            ),
        )

        self.image_name = ImageSubscriber(
            "Select image on which to display outlines",
            "None",
            doc="""\
*(Used only when a blank image has not been selected)*

Choose the image to serve as the background for the outlines. You can
choose from images that were loaded or created by modules previous to
this one.
""",
        )

        self.line_mode = Choice(
            "How to outline",
            ["Inner", "Outer", "Thick"],
            value="Inner",
            doc="""\
Specify how to mark the boundaries around an object:

-  *Inner:* outline the pixels just inside of objects, leaving
   background pixels untouched.
-  *Outer:* outline pixels in the background around object boundaries.
   When two objects touch, their boundary is also marked.
-  *Thick:* any pixel not completely surrounded by pixels of the same
   label is marked as a boundary. This results in boundaries that are 2
   pixels thick.
""",
        )

        self.output_image_name = ImageName(
            "Name the output image",
            "OrigOverlay",
            doc="""\
Enter the name of the output image with the outlines overlaid. This
image can be selected in later modules (for instance, **SaveImages**).
""",
        )

        self.wants_color = Choice(
            "Outline display mode",
            [OutlineMode.COLOR.value, OutlineMode.GRAYSCALE.value],
            doc="""\
Specify how to display the outline contours around your objects. Color
outlines produce a clearer display for images where the cell borders
have a high intensity, but take up more space in memory. Grayscale
outlines are displayed with either the highest possible intensity or the
same intensity as the brightest pixel in the image.
""",
        )

        self.spacer = Divider(line=False)

        self.max_type = Choice(
            "Select method to determine brightness of outlines",
            [BrightnessMode.MAX_IMAGE.value, BrightnessMode.MAX_POSSIBLE.value],
            doc="""\
*(Used only when outline display mode is grayscale)*

The following options are possible for setting the intensity
(brightness) of the outlines:

-  *{MAX_IMAGE}:* Set the brightness to the the same as the brightest
   point in the image.
-  *{MAX_POSSIBLE}:* Set to the maximum possible value for this image
   format.

If your image is quite dim, then putting bright white lines onto it may
not be useful. It may be preferable to make the outlines equal to the
maximal brightness already occurring in the image.
""".format(
                **{"MAX_IMAGE": BrightnessMode.MAX_IMAGE.value, "MAX_POSSIBLE": BrightnessMode.MAX_POSSIBLE.value}
            ),
        )

        self.outlines = []

        self.add_outline(can_remove=False)

        self.add_outline_button = DoSomething(
            "", "Add another outline", self.add_outline
        )

    def add_outline(self, can_remove=True):
        group = SettingsGroup()
        if can_remove:
            group.append("divider", Divider(line=False))

        group.append(
            "objects_name",
            LabelSubscriber(
                "Select objects to display",
                "None",
                doc="Choose the objects whose outlines you would like to display.",
            ),
        )

        default_color = (
            COLOR_ORDER[len(self.outlines)]
            if len(self.outlines) < len(COLOR_ORDER)
            else COLOR_ORDER[0]
        )

        group.append(
            "color",
            Color(
                "Select outline color",
                default_color,
                doc="Objects will be outlined in this color.",
            ),
        )

        if can_remove:
            group.append(
                "remover",
                RemoveSettingButton("", "Remove this outline", self.outlines, group),
            )

        self.outlines.append(group)

    def prepare_settings(self, setting_values):
        num_settings = (
            len(setting_values) - NUM_FIXED_SETTINGS
        ) // NUM_OUTLINE_SETTINGS
        if len(self.outlines) == 0:
            self.add_outline(False)
        elif len(self.outlines) > num_settings:
            del self.outlines[num_settings:]
        else:
            for i in range(len(self.outlines), num_settings):
                self.add_outline()

    def settings(self):
        result = [
            self.blank_image,
            self.image_name,
            self.output_image_name,
            self.wants_color,
            self.max_type,
            self.line_mode,
        ]
        for outline in self.outlines:
            result += [outline.color, outline.objects_name]
        return result

    def visible_settings(self):
        result = [self.blank_image]
        if not self.blank_image.value:
            result += [self.image_name]
        result += [
            self.output_image_name,
            self.wants_color,
            self.line_mode,
            self.spacer,
        ]
        if self.wants_color.value == OutlineMode.GRAYSCALE and not self.blank_image.value:
            result += [self.max_type]
        for outline in self.outlines:
            result += [outline.objects_name]
            if self.wants_color.value == OutlineMode.COLOR:
                result += [outline.color]
            if hasattr(outline, "remover"):
                result += [outline.remover]
        result += [self.add_outline_button]
        return result

    def run(self, workspace):
        # Parameter extraction for dispatcher call
        
        # Extract base image parameters
        if self.blank_image.value:
            outline = self.outlines[0]
            objects = workspace.object_set.get_objects(outline.objects_name.value)

            obj_shape = objects.shape
            obj_dimensions = objects.dimensions
            im_pixel_data = None
            im_multichannel = False
            im_dimensions = None
            
        else:
            image = workspace.image_set.get_image(self.image_name.value)

            obj_shape = None
            obj_dimensions = None
            im_pixel_data = image.pixel_data
            im_multichannel = image.multichannel
            im_dimensions = image.dimensions
        
        # Extract outline data for both color and BW modes
        object_labels_list = []
        colors_list = []
        is_volumetric = (im_dimensions or obj_dimensions) > 2
        
        for outline in self.outlines:
            objects = workspace.object_set.get_objects(outline.objects_name.value)
            obj_labels_list = objects.get_labels()
            
            # For BW mode
            object_labels_list.append(obj_labels_list)
            
            # For color mode
            color_rgb = outline.color.to_rgb()
            colors_list.append(color_rgb)

        if self.wants_color.value == OutlineMode.GRAYSCALE:
            colors_list = None
        
        # Call dispatcher with all extracted parameters
        pixel_data, base_image, dimensions = overlay_outlines(
            BrightnessMode.MAX_POSSIBLE if self.blank_image.value else self.max_type.value,
            self.line_mode.value,
            obj_shape,
            obj_dimensions,
            im_pixel_data,
            im_multichannel,
            im_dimensions,
            object_labels_list,
            colors_list,
            is_volumetric
        )
        
        # UI/Framework-specific logic - create output image and update workspace
        output_image = Image(pixel_data, dimensions=dimensions)
        workspace.image_set.add(self.output_image_name.value, output_image)
        
        if not self.blank_image.value:
            image = workspace.image_set.get_image(self.image_name.value)
            output_image.parent_image = image
        
        if self.show_window:
            workspace.display_data.pixel_data = pixel_data
            workspace.display_data.image_pixel_data = base_image
            workspace.display_data.dimensions = dimensions

    def display(self, workspace, figure):
        dimensions = workspace.display_data.dimensions

        if self.blank_image.value:
            figure.set_subplots((1, 1), dimensions=dimensions)

            if self.wants_color.value == OutlineMode.COLOR:
                figure.subplot_imshow(
                    0,
                    0,
                    workspace.display_data.pixel_data,
                    self.output_image_name.value,
                )
            else:
                figure.subplot_imshow_bw(
                    0,
                    0,
                    workspace.display_data.pixel_data,
                    self.output_image_name.value,
                )
        else:
            figure.set_subplots((2, 1), dimensions=dimensions)

            figure.subplot_imshow_bw(
                0, 0, workspace.display_data.image_pixel_data, self.image_name.value
            )

            if self.wants_color.value == OutlineMode.COLOR:
                figure.subplot_imshow(
                    1,
                    0,
                    workspace.display_data.pixel_data,
                    self.output_image_name.value,
                    sharexy=figure.subplot(0, 0),
                )
            else:
                figure.subplot_imshow_bw(
                    1,
                    0,
                    workspace.display_data.pixel_data,
                    self.output_image_name.value,
                    sharexy=figure.subplot(0, 0),
                )


    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            #
            # Added line width
            #
            setting_values = (
                setting_values[:NUM_FIXED_SETTINGS_V1]
                + ["1"]
                + setting_values[NUM_FIXED_SETTINGS_V1:]
            )
            variable_revision_number = 2

        if variable_revision_number == 2:
            #
            # Added overlay image / objects choice
            #
            new_setting_values = setting_values[:NUM_FIXED_SETTINGS_V2]
            for i in range(
                NUM_FIXED_SETTINGS_V2, len(setting_values), NUM_OUTLINE_SETTINGS_V2
            ):
                new_setting_values += setting_values[i : (i + NUM_OUTLINE_SETTINGS_V2)]
                new_setting_values += [FROM_IMAGES, "None"]
            setting_values = new_setting_values
            variable_revision_number = 3

        if variable_revision_number == 3:
            new_setting_values = setting_values[: NUM_FIXED_SETTINGS_V3 - 1]

            new_setting_values += ["Inner"]

            colors = setting_values[
                NUM_FIXED_SETTINGS_V3 + 1 :: NUM_OUTLINE_SETTINGS_V3
            ]

            names = setting_values[NUM_FIXED_SETTINGS_V3 + 3 :: NUM_OUTLINE_SETTINGS_V3]

            for color, name in zip(colors, names):
                new_setting_values += [color, name]

            setting_values = new_setting_values

            variable_revision_number = 4

        return setting_values, variable_revision_number


    def volumetric(self):
        return True
