"""
<b>Overlay Outlines</b> places outlines produced by an <b>Identify</b> module over a desired image.
<hr>
This module places outlines (in a special format produced by an <b>Identify</b> module) on any
desired image (grayscale, color, or blank). The resulting image can be saved using the
<b>SaveImages</b> module. See also <b>IdentifyPrimaryObjects, IdentifySecondaryObjects,
IdentifyTertiaryObjects</b>.
"""

import numpy
import scipy.ndimage
import skimage.color
import skimage.segmentation

import cellprofiler.image
import cellprofiler.module
import cellprofiler.setting

WANTS_COLOR = "Color"
WANTS_GRAYSCALE = "Grayscale"

MAX_IMAGE = "Max of image"
MAX_POSSIBLE = "Max possible"

COLORS = {"White": (1, 1, 1),
          "Black": (0, 0, 0),
          "Red": (1, 0, 0),
          "Green": (0, 1, 0),
          "Blue": (0, 0, 1),
          "Yellow": (1, 1, 0)}

COLOR_ORDER = ["Red", "Green", "Blue", "Yellow", "White", "Black"]

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


class OverlayOutlines(cellprofiler.module.Module):
    module_name = 'OverlayOutlines'
    variable_revision_number = 4
    category = "Image Processing"

    def create_settings(self):
        self.blank_image = cellprofiler.setting.Binary(
            "Display outlines on a blank image?",
            False,
            doc="""
            Select <i>{YES}</i> to produce an image of the outlines on a black background.
            <p>Select <i>{NO}</i>, the module will overlay the outlines on an image of your choosing.</p>
            """.format(**{
                "YES": cellprofiler.setting.YES,
                "NO": cellprofiler.setting.NO
            })
        )

        self.image_name = cellprofiler.setting.ImageNameSubscriber(
            "Select image on which to display outlines",
            cellprofiler.setting.NONE,
            doc="""
            <i>(Used only when a blank image has not been selected)</i><br>
            Choose the image to serve as the background for the outlines. You can choose from images that were
            loaded or created by modules previous to this one.
            """
        )

        self.line_width = cellprofiler.setting.Float(
            "Width of outlines",
            "1",
            doc="""
            Enter the width, in pixels, of the outlines to be displayed on the image.
            """
        )

        self.output_image_name = cellprofiler.setting.ImageNameProvider(
            "Name the output image",
            "OrigOverlay",
            doc="""
            Enter the name of the output image with the outlines overlaid. This image can be selected in later
            modules (for instance, <b>SaveImages</b>).
            """
        )

        self.wants_color = cellprofiler.setting.Choice(
            "Outline display mode",
            [WANTS_COLOR, WANTS_GRAYSCALE],
            doc="""
            Specify how to display the outline contours around your objects. Color outlines produce a clearer
            display for images where the cell borders have a high intensity, but take up more space in memory.
            Grayscale outlines are displayed with either the highest possible intensity or the same intensity
            as the brightest pixel in the image.
            """
        )

        self.spacer = cellprofiler.setting.Divider(line=False)

        self.max_type = cellprofiler.setting.Choice(
            "Select method to determine brightness of outlines",
            [MAX_IMAGE, MAX_POSSIBLE],
            doc="""
            <i>(Used only when outline display mode is grayscale)</i><br>
            The following options are possible for setting the intensity (brightness) of the outlines:
            <ul>
                <li><i>{MAX_IMAGE}:</i> Set the brighness to the the same as the brightest point in the
                image.</li>
                <li><i>{MAX_POSSIBLE}:</i> Set to the maximum possible value for this image format.</li>
            </ul>If your image is quite dim, then putting bright white lines onto it may not be useful. It may
            be preferable to make the outlines equal to the maximal brightness already occurring in the image.
            """.format(**{
                "MAX_IMAGE": MAX_IMAGE,
                "MAX_POSSIBLE": MAX_POSSIBLE
            })
        )

        self.outlines = []

        self.add_outline(can_remove=False)

        self.add_outline_button = cellprofiler.setting.DoSomething("", "Add another outline", self.add_outline)

    def add_outline(self, can_remove=True):
        group = cellprofiler.setting.SettingsGroup()
        if can_remove:
            group.append("divider", cellprofiler.setting.Divider(line=False))

        group.append(
            "objects_name",
            cellprofiler.setting.ObjectNameSubscriber(
                "Select objects to display",
                cellprofiler.setting.NONE,
                doc="Choose the objects whose outlines you would like to display."
            )
        )

        default_color = (COLOR_ORDER[len(self.outlines)] if len(self.outlines) < len(COLOR_ORDER) else COLOR_ORDER[0])

        group.append("color", cellprofiler.setting.Color("Select outline color", default_color))

        if can_remove:
            group.append(
                "remover",
                cellprofiler.setting.RemoveSettingButton("", "Remove this outline", self.outlines, group)
            )

        self.outlines.append(group)

    def prepare_settings(self, setting_values):
        num_settings = (len(setting_values) - NUM_FIXED_SETTINGS) / NUM_OUTLINE_SETTINGS
        if len(self.outlines) == 0:
            self.add_outline(False)
        elif len(self.outlines) > num_settings:
            del self.outlines[num_settings:]
        else:
            for i in range(len(self.outlines), num_settings):
                self.add_outline()

    def settings(self):
        result = [self.blank_image, self.image_name, self.output_image_name,
                  self.wants_color, self.max_type, self.line_width]
        for outline in self.outlines:
            result += [outline.color, outline.objects_name]
        return result

    def visible_settings(self):
        result = [self.blank_image]
        if not self.blank_image.value:
            result += [self.image_name]
        result += [self.output_image_name, self.wants_color, self.line_width, self.spacer]
        if (self.wants_color.value == WANTS_GRAYSCALE and not self.blank_image.value):
            result += [self.max_type]
        for outline in self.outlines:
            result += [outline.objects_name]
            if self.wants_color.value == WANTS_COLOR:
                result += [outline.color]
            if hasattr(outline, "remover"):
                result += [outline.remover]
        result += [self.add_outline_button]
        return result

    def run(self, workspace):
        if self.wants_color.value == WANTS_COLOR:
            pixel_data = self.run_color(workspace)
        else:
            pixel_data = self.run_bw(workspace)
        if self.blank_image.value:
            output_image = cellprofiler.image.Image(pixel_data)
            workspace.image_set.add(self.output_image_name.value, output_image)
        else:
            image = workspace.image_set.get_image(self.image_name.value)
            output_image = cellprofiler.image.Image(pixel_data, parent_image=image)
            workspace.image_set.add(self.output_image_name.value, output_image)
            workspace.display_data.image_pixel_data = image.pixel_data
        if self.show_window:
            workspace.display_data.labels = {}
            for outline in self.outlines:
                name = outline.objects_name.value
                objects = workspace.object_set.get_objects(name)
                workspace.display_data.labels[name] = \
                    [labels for labels, indexes in objects.get_labels()]

        workspace.display_data.pixel_data = pixel_data

    def display(self, workspace, figure):
        from cellprofiler.gui.figure import CPLD_LABELS, CPLD_NAME, \
            CPLD_OUTLINE_COLOR, CPLD_MODE, CPLDM_OUTLINES, CPLD_LINE_WIDTH

        figure.set_subplots((1, 1))

        if self.blank_image:
            pixel_data = numpy.zeros(workspace.display_data.pixel_data.shape)
        else:
            pixel_data = workspace.display_data.image_pixel_data
        cplabels = []
        ldict = workspace.display_data.labels
        for outline in self.outlines:
            name = outline.objects_name.value
            if self.wants_color.value == WANTS_COLOR:
                color = numpy.array(outline.color.to_rgb(), float)
            else:
                color = numpy.ones(3) * 255.0
            d = {CPLD_NAME: name,
                 CPLD_LABELS: ldict[name],
                 CPLD_OUTLINE_COLOR: color,
                 CPLD_MODE: CPLDM_OUTLINES,
                 CPLD_LINE_WIDTH: self.line_width.value}
            cplabels.append(d)
        if self.blank_image.value:
            if self.wants_color.value == WANTS_COLOR:
                figure.subplot_imshow(0, 0, pixel_data,
                                      self.output_image_name.value,
                                      cplabels=cplabels)
            else:
                figure.subplot_imshow_bw(0, 0, pixel_data,
                                         self.output_image_name.value,
                                         cplabels=cplabels)
        else:
            figure.set_subplots((2, 1))

            image_pixel_data = workspace.display_data.image_pixel_data
            if image_pixel_data.ndim == 2:
                figure.subplot_imshow_bw(0, 0, image_pixel_data,
                                         "Original: %s" %
                                         self.image_name.value)
            else:
                figure.subplot_imshow_color(0, 0, image_pixel_data,
                                            "Original: %s" %
                                            self.image_name.value)
            if self.wants_color.value == WANTS_COLOR:
                if cplabels is not None and pixel_data.ndim == 2:
                    fn = figure.subplot_imshow_grayscale
                else:
                    fn = figure.subplot_imshow
                fn(1, 0, pixel_data,
                   self.output_image_name.value,
                   sharexy=figure.subplot(0, 0),
                   cplabels=cplabels)
            else:
                figure.subplot_imshow_bw(1, 0, pixel_data,
                                         self.output_image_name.value,
                                         sharexy=figure.subplot(0, 0),
                                         cplabels=cplabels)

    def run_bw(self, workspace):
        image_set = workspace.image_set
        if self.blank_image.value:
            shape = self.get_outline(workspace, self.outlines[0]).shape[:2]
            pixel_data = numpy.zeros(shape)
            maximum = 1
        else:
            image = image_set.get_image(self.image_name.value,
                                        must_be_grayscale=True)
            pixel_data = image.pixel_data
            maximum = 1 if self.max_type == MAX_POSSIBLE else numpy.max(pixel_data)
            pixel_data = pixel_data.copy()
        for outline in self.outlines:
            mask = self.get_outline(workspace, outline)
            i_max = min(mask.shape[0], pixel_data.shape[0])
            j_max = min(mask.shape[1], pixel_data.shape[1])
            mask = mask[:i_max, :j_max]
            pixel_data[:i_max, :j_max][mask] = maximum
        return pixel_data

    def run_color(self, workspace):
        outline = self.outlines[0]

        objects = workspace.object_set.get_objects(outline.objects_name.value)

        if self.blank_image.value:
            pixel_data = numpy.zeros(objects.shape + (3,))
        else:
            image = workspace.image_set.get_image(self.image_name.value)

            if image.multichannel:
                pixel_data = image.pixel_data
            else:
                pixel_data = skimage.color.gray2rgb(image.pixel_data)

        color = tuple(c / 255.0 for c in outline.color.to_rgb())

        for labels, _ in objects.get_labels():
            if objects.volumetric:
                for index, plane in enumerate(labels):
                    pixel_data[index] = skimage.segmentation.mark_boundaries(
                        pixel_data[index],
                        plane,
                        color=color,
                        mode="inner"
                    )
            else:
                pixel_data = skimage.segmentation.mark_boundaries(
                    pixel_data,
                    labels,
                    color=color,
                    mode="inner"
                )

        return pixel_data

    def get_outline(self, workspace, outline):
        '''Get outline, with aliasing and taking widths into account'''
        name = outline.objects_name.value
        objects = workspace.object_set.get_objects(name)
        pixel_data = numpy.zeros(objects.shape, bool)
        for labels, _ in objects.get_labels():
            pixel_data = pixel_data | skimage.segmentation.find_boundaries(labels, mode="inner")
        if self.wants_color == WANTS_GRAYSCALE:
            return pixel_data.astype(bool)
        color = numpy.array(outline.color.to_rgb(), float) / 255.0
        if pixel_data.ndim == 2:
            if len(color) == 3:
                color = numpy.hstack((color, [1]))
            pixel_data = pixel_data > 0
            output_image = color[numpy.newaxis, numpy.newaxis, :] * pixel_data[:, :, numpy.newaxis]
        else:
            output_image = numpy.dstack([pixel_data[:, :, i] for i in range(3)] + [numpy.sum(pixel_data, 2) > 0])
        # float16s are slower, but since we're potentially allocating an image
        # 4 times larger than our input, the tradeoff is worth it.
        if hasattr(numpy, 'float16'):
            output_image = output_image.astype(numpy.float16)
        if self.line_width.value > 1:
            half_line_width = float(self.line_width.value) / 2
            d, (i, j) = scipy.ndimage.distance_transform_edt(output_image[:, :, 3] == 0, return_indices=True)
            mask = (d > 0) & (d <= half_line_width - .5)
            output_image[mask, :] = output_image[i[mask], j[mask], :]
            #
            # Do a little aliasing here using an alpha channel
            #
            mask = ((d > max(0, half_line_width - .5)) & (d < half_line_width + .5))
            d = half_line_width + .5 - d
            output_image[mask, :3] = output_image[i[mask], j[mask], :3]
            output_image[mask, 3] = d[mask]

        return output_image

    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if from_matlab and variable_revision_number == 2:
            # Order is
            # image_name
            # outline name
            # max intensity
            # output_image_name
            # color
            setting_values = [cellprofiler.setting.YES if setting_values[0] == "Blank" else cellprofiler.setting.NO,
                              setting_values[0],
                              setting_values[3],
                              WANTS_COLOR,
                              setting_values[2],
                              setting_values[1],
                              setting_values[4]]
            from_matlab = False
            variable_revision_number = 1
        if (not from_matlab) and variable_revision_number == 1:
            #
            # Added line width
            #
            setting_values = setting_values[:NUM_FIXED_SETTINGS_V1] + \
                             ["1"] + setting_values[NUM_FIXED_SETTINGS_V1:]
            variable_revision_number = 2

        if (not from_matlab) and variable_revision_number == 2:
            #
            # Added overlay image / objects choice
            #
            new_setting_values = setting_values[:NUM_FIXED_SETTINGS_V2]
            for i in range(NUM_FIXED_SETTINGS_V2, len(setting_values),
                           NUM_OUTLINE_SETTINGS_V2):
                new_setting_values += \
                    setting_values[i:(i + NUM_OUTLINE_SETTINGS_V2)]
                new_setting_values += [FROM_IMAGES, cellprofiler.setting.NONE]
            setting_values = new_setting_values
            variable_revision_number = 3

        if (not from_matlab) and variable_revision_number == 3:
            new_setting_values = setting_values[:NUM_FIXED_SETTINGS_V3]

            colors = setting_values[NUM_FIXED_SETTINGS_V3 + 1::NUM_OUTLINE_SETTINGS_V3]

            names = setting_values[NUM_FIXED_SETTINGS_V3 + 3::NUM_OUTLINE_SETTINGS_V3]

            for color, name in zip(colors, names):
                new_setting_values += [color, name]

            setting_values = new_setting_values

            variable_revision_number = 4

        return setting_values, variable_revision_number, from_matlab
