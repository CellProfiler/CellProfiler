"""
Tile
====

**Tile** tiles images together to form large montage images.

This module allows more than one image to be placed next to each other
in a grid layout you specify. It might be helpful, for example, to place
images adjacent to each other when multiple fields of view have been
imaged for the same sample. Images can be tiled either across cycles
(multiple fields of view, for example) or within a cycle (multiple
channels of the same field of view, for example).

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

Tiling images to create a montage with this module generates an image
that is roughly the size of all the images’ sizes added together. For
large numbers of images, this may cause memory errors, which might be
avoided by the following suggestions:

-  Resize the images to a fraction of their original size, using the
   **Resize** module prior to this module in the pipeline.
-  Rescale the images to 8-bit using the **RescaleIntensity** module,
   which diminishes image quality by decreasing the number of graylevels
   in the image (that is, bit depth) but also decreases the size of the
   image.

Please also note that this module does not perform *image stitching*
(i.e., intelligent adjustment of the alignment between adjacent images).
For image stitching, you may find the following list of software
packages useful:

-  `Photomerge Feature in Photoshop`_
-  `PTGui`_
-  `Autostitch`_
-  `ImageJ with the MosaicJ plugin`_

Other packages are referenced `here`_.

.. _Photomerge Feature in Photoshop: https://helpx.adobe.com/photoshop/using/create-panoramic-images-photomerge.html
.. _PTGui: http://www.ptgui.com/
.. _Autostitch: http://matthewalunbrown.com/autostitch/autostitch.html
.. _ImageJ with the MosaicJ plugin: http://bigwww.epfl.ch/thevenaz/mosaicj/
.. _here: http://graphicssoft.about.com/od/panorama/Panorama_Creation_and_Stitching_Tools.htm

|

============ ============
Supports 2D? Supports 3D?
============ ============
YES          NO
============ ============

"""

import numpy
from cellprofiler_core.image import Image
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting import Divider
from cellprofiler_core.setting import SettingsGroup
from cellprofiler_core.setting import ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.do_something import DoSomething, RemoveSettingButton
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import ImageName, Integer

T_WITHIN_CYCLES = "Within cycles"
T_ACROSS_CYCLES = "Across cycles"
T_ALL = (T_WITHIN_CYCLES, T_ACROSS_CYCLES)

P_TOP_LEFT = "top left"
P_BOTTOM_LEFT = "bottom left"
P_TOP_RIGHT = "top right"
P_BOTTOM_RIGHT = "bottom right"
P_ALL = (P_TOP_LEFT, P_BOTTOM_LEFT, P_TOP_RIGHT, P_BOTTOM_RIGHT)

S_ROW = "row"
S_COL = "column"
S_ALL = (S_ROW, S_COL)

"""Module dictionary keyword for storing the # of images in the group when tiling"""
IMAGE_COUNT = "ImageCount"
"""Dictionary keyword for storing the current image number in the group"""
IMAGE_NUMBER = "ImageNumber"
"""Module dictionary keyword for the image being tiled"""
TILED_IMAGE = "TiledImage"
TILE_WIDTH = "TileWidth"
TILE_HEIGHT = "TileHeight"

FIXED_SETTING_COUNT = 10


class Tile(Module):
    module_name = "Tile"
    category = "Image Processing"
    variable_revision_number = 1

    def create_settings(self):
        self.input_image = ImageSubscriber(
            "Select an input image",
            "None",
            doc="""Select the image to be tiled. Additional images within the cycle can be
added later by choosing the "*%(T_ACROSS_CYCLES)s*" option below.
"""
            % globals(),
        )

        self.output_image = ImageName(
            "Name the output image",
            "TiledImage",
            doc="""Enter a name for the final tiled image.""",
        )

        self.additional_images = []

        self.add_button = DoSomething(
            "",
            "Add another image",
            self.add_image,
            doc="""Add images from other channels to perform similar tiling""",
        )

        self.tile_method = Choice(
            "Tile assembly method",
            T_ALL,
            doc="""\
This setting controls the method by which the final tiled image is
assembled:

-  *%(T_WITHIN_CYCLES)s:* If you have loaded more than one image for
   each cycle using modules upstream in the pipeline, the images can be
   tiled. For example, you may tile three different channels (OrigRed,
   OrigBlue, and OrigGreen), and a new tiled image will be created for
   every image cycle.
-  *%(T_ACROSS_CYCLES)s:* If you want to tile images from multiple
   cycles together, select this option. For example, you may tile all
   the images of the same type (e.g., OrigBlue) across all fields of
   view in your experiment, which will result in one final tiled image
   when processing is complete.
"""
            % globals(),
        )

        self.rows = Integer(
            "Final number of rows",
            8,
            doc="""\
Specify the number of rows would you like to have in the tiled image.
For example, if you want to show your images in a 96-well format, enter
8.

*Special cases:* Let *M* be the total number of slots for images (i.e,
number of rows x number of columns) and *N* be the number of actual
images.

-  If *M* > *N*, blanks will be used for the empty slots.
-  If the *M* < *N*, an error will occur since there are not enough
   image slots. Check “Automatically calculate number of rows?” to avoid
   this error.
""",
        )

        self.columns = Integer(
            "Final number of columns",
            12,
            doc="""\
Specify the number of columns you like to have in the tiled image. For
example, if you want to show your images in a 96-well format, enter 12.

*Special cases:* Let *M* be the total number of slots for images (i.e,
number of rows x number of columns) and *N* be the number of actual
images.

-  If *M* > *N*, blanks will be used for the empty slots.
-  If the *M* < *N*, an error will occur since there are not enough
   image slots. Check “Automatically calculate number of columns?” to
   avoid this error.
""",
        )

        self.place_first = Choice(
            "Image corner to begin tiling",
            P_ALL,
            doc="""Where do you want the first image to be placed? Begin in the upper
left-hand corner for a typical multi-well plate format where the first image is A01.
""",
        )

        self.tile_style = Choice(
            "Direction to begin tiling",
            S_ALL,
            doc="""This setting specifies the order that the images are to be arranged. For example, if
your images are named A01, A02, etc, enter "*%(S_ROW)s*".
"""
            % globals(),
        )

        self.meander = Binary(
            "Use meander mode?",
            False,
            doc="""\
Select "*Yes*" to tile adjacent images in one direction, then the next
row/column is tiled in the opposite direction. Some microscopes capture
images in this fashion. The default mode is “comb”, or “typewriter”
mode; in this mode, when one row is completely tiled in one direction,
the next row starts near where the first row started and tiles again in
the same direction.
"""
            % globals(),
        )

        self.wants_automatic_rows = Binary(
            "Automatically calculate number of rows?",
            False,
            doc="""\
**Tile** can automatically calculate the number of rows in the grid
based on the number of image cycles that will be processed. Select
"*Yes*" to create a grid that has the number of columns that you
entered and enough rows to display all of your images. Select "*No*"
to specify the number of rows.

If you check both automatic rows and automatic columns, **Tile** will
create a grid that has roughly the same number of rows and columns.
"""
            % globals(),
        )

        self.wants_automatic_columns = Binary(
            "Automatically calculate number of columns?",
            False,
            doc="""\
**Tile** can automatically calculate the number of columns in the grid
from the number of image cycles that will be processed. Select "*Yes*"
to create a grid that has the number of rows that you entered and enough
columns to display all of your images. Select "*No*" to specify the
number of rows.

If you check both automatic rows and automatic columns, **Tile** will
create a grid that has roughly the same number of rows and columns.
"""
            % globals(),
        )

    def add_image(self, can_remove=True):
        """Add an image + associated questions and buttons"""
        group = SettingsGroup()
        if can_remove:
            group.append("divider", Divider(line=True))

        group.append(
            "input_image_name",
            ImageSubscriber(
                "Select an additional image to tile",
                "None",
                doc="""Select an additional image to tile?""",
            ),
        )
        if can_remove:
            group.append(
                "remover",
                RemoveSettingButton(
                    "", "Remove above image", self.additional_images, group
                ),
            )
        self.additional_images.append(group)

    def settings(self):
        result = [
            self.input_image,
            self.output_image,
            self.tile_method,
            self.rows,
            self.columns,
            self.place_first,
            self.tile_style,
            self.meander,
            self.wants_automatic_rows,
            self.wants_automatic_columns,
        ]

        for additional in self.additional_images:
            result += [additional.input_image_name]
        return result

    def prepare_settings(self, setting_values):
        assert (len(setting_values) - FIXED_SETTING_COUNT) % 1 == 0
        n_additional = (len(setting_values) - FIXED_SETTING_COUNT) / 1
        del self.additional_images[:]
        while len(self.additional_images) < n_additional:
            self.add_image()

    def visible_settings(self):
        result = [
            self.input_image,
            self.output_image,
            self.tile_method,
            self.wants_automatic_rows,
        ]
        if not self.wants_automatic_rows:
            result += [self.rows]
        result += [self.wants_automatic_columns]
        if not self.wants_automatic_columns:
            result += [self.columns]

        result += [self.place_first, self.tile_style, self.meander]

        if self.tile_method == T_WITHIN_CYCLES:
            for additional in self.additional_images:
                result += additional.visible_settings()
            result += [self.add_button]
        return result

    def help_settings(self):
        result = [
            self.input_image,
            self.output_image,
            self.tile_method,
            self.wants_automatic_rows,
            self.rows,
            self.wants_automatic_columns,
            self.columns,
            self.place_first,
            self.tile_style,
            self.meander,
        ]

        return result

    def is_aggregation_module(self):
        return self.tile_method == T_ACROSS_CYCLES

    def prepare_group(self, workspace, grouping, image_numbers):
        """Prepare to handle a group of images when tiling"""
        d = self.get_dictionary(workspace.image_set_list)
        d[IMAGE_COUNT] = len(image_numbers)
        d[IMAGE_NUMBER] = 0
        d[TILED_IMAGE] = None

    def run(self, workspace):
        """do the image analysis"""
        if self.tile_method == T_WITHIN_CYCLES:
            output_pixels = self.place_adjacent(workspace)
        else:
            output_pixels = self.tile(workspace)
        output_image = Image(output_pixels)
        workspace.image_set.add(self.output_image.value, output_image)
        if self.show_window:
            workspace.display_data.image = output_pixels

    def post_group(self, workspace, grouping):
        if self.tile_method == T_ACROSS_CYCLES:
            image_set = workspace.image_set
            if self.output_image.value not in image_set.names:
                d = self.get_dictionary(workspace.image_set_list)
                image_set.add(self.output_image.value, Image(d[TILED_IMAGE]))

    def is_aggregation_module(self):
        """Need to run all cycles in same worker if across cycles"""
        return self.tile_method == T_ACROSS_CYCLES

    def display(self, workspace, figure):
        """Display
        """
        figure.set_subplots((1, 1))
        pixels = workspace.display_data.image
        name = self.output_image.value
        if pixels.ndim == 3:
            figure.subplot_imshow(0, 0, pixels, title=name)
        else:
            figure.subplot_imshow_grayscale(0, 0, pixels, title=name)

    def tile(self, workspace):
        """Tile images across image cycles
        """
        d = self.get_dictionary(workspace.image_set_list)
        rows, columns = self.get_grid_dimensions(d[IMAGE_COUNT])
        image_set = workspace.image_set
        image = image_set.get_image(self.input_image.value)
        pixels = image.pixel_data
        if d[TILED_IMAGE] is None:
            tile_width = pixels.shape[1]
            tile_height = pixels.shape[0]
            height = tile_height * rows
            width = tile_width * columns
            if pixels.ndim == 3:
                shape = (height, width, pixels.shape[2])
            else:
                shape = (height, width)
            output_pixels = numpy.zeros(shape)
            d[TILED_IMAGE] = output_pixels
            d[TILE_WIDTH] = tile_width
            d[TILE_HEIGHT] = tile_height
        else:
            output_pixels = d[TILED_IMAGE]
            tile_width = d[TILE_WIDTH]
            tile_height = d[TILE_HEIGHT]

        image_index = d[IMAGE_NUMBER]
        d[IMAGE_NUMBER] = image_index + 1
        self.put_tile(pixels, output_pixels, image_index, rows, columns)
        return output_pixels

    def put_tile(self, pixels, output_pixels, image_index, rows, columns):
        tile_height = int(output_pixels.shape[0] / rows)
        tile_width = int(output_pixels.shape[1] / columns)
        tile_i, tile_j = self.get_tile_ij(image_index, rows, columns)
        tile_i *= tile_height
        tile_j *= tile_width
        img_height = min(tile_height, pixels.shape[0])
        img_width = min(tile_width, pixels.shape[1])
        if output_pixels.ndim == 2:
            output_pixels[
                tile_i : (tile_i + img_height), tile_j : (tile_j + img_width)
            ] = pixels[:img_height, :img_width]
        elif pixels.ndim == 3:
            output_pixels[
                tile_i : (tile_i + img_height), tile_j : (tile_j + img_width), :
            ] = pixels[:img_height, :img_width, :]
        else:
            for k in range(output_pixels.shape[2]):
                output_pixels[
                    tile_i : (tile_i + img_height), tile_j : (tile_j + img_width), k
                ] = pixels[:img_height, :img_width]
        return output_pixels

    def place_adjacent(self, workspace):
        """Place images from the same image set adjacent to each other"""
        rows, columns = self.get_grid_dimensions()
        image_names = [self.input_image.value] + [
            g.input_image_name.value for g in self.additional_images
        ]
        pixel_data = [
            workspace.image_set.get_image(name).pixel_data for name in image_names
        ]
        tile_width = 0
        tile_height = 0
        colors = 0
        for p in pixel_data:
            tile_width = max(tile_width, p.shape[1])
            tile_height = max(tile_height, p.shape[0])
            if p.ndim > 2:
                colors = 3
        height = tile_height * rows
        width = tile_width * columns
        if colors > 0:
            output_pixels = numpy.zeros((height, width, colors))
        else:
            output_pixels = numpy.zeros((height, width))
        for i, p in enumerate(pixel_data):
            self.put_tile(p, output_pixels, i, rows, columns)
        return output_pixels

    def get_tile_ij(self, image_index, rows, columns):
        """Get the I/J coordinates for an image

        returns i,j where 0 < i < self.rows and 0 < j < self.columns
        """
        if self.tile_style == S_ROW:
            tile_i = int(image_index / columns)
            tile_j = image_index % columns
            if self.meander and tile_i % 2 == 1:
                # Reverse the direction if in meander mode
                tile_j = columns - tile_j - 1
        else:
            tile_i = image_index % rows
            tile_j = int(image_index / rows)
            if self.meander and tile_j % 2 == 1:
                # Reverse the direction if in meander mode
                tile_i = rows - tile_i - 1
        if self.place_first in (P_BOTTOM_LEFT, P_BOTTOM_RIGHT):
            tile_i = rows - tile_i - 1
        if self.place_first in (P_TOP_RIGHT, P_BOTTOM_RIGHT):
            tile_j = columns - tile_j - 1
        if tile_i < 0 or tile_i >= rows or tile_j < 0 or tile_j >= columns:
            raise ValueError(
                (
                    "The current image falls outside of the grid boundaries. \n"
                    "Grid dimensions: %d, %d\n"
                    "Tile location: %d, %d\n"
                )
                % (columns, rows, tile_j, tile_i)
            )
        return tile_i, tile_j

    def get_grid_dimensions(self, image_count=None):
        """Get the dimensions of the grid in i,j format

        image_count - # of images in the grid. If None, use info from settings.
        """
        assert (image_count is not None) or self.tile_method == T_WITHIN_CYCLES, (
            "Must specify image count for %s method" % self.tile_method.value
        )
        if image_count is None:
            image_count = len(self.additional_images) + 1
        if self.wants_automatic_rows:
            if self.wants_automatic_columns:
                #
                # Take the square root of the # of images & assign as rows.
                # Maybe add 1 to get # of columns.
                #
                i = int(numpy.sqrt(image_count))
                j = int((image_count + i - 1) / i)
                return i, j
            else:
                j = self.columns.value
                i = int((image_count + j - 1) / j)
                return i, j
        elif self.wants_automatic_columns:
            i = self.rows.value
            j = int((image_count + i - 1) / i)
            return i, j
        else:
            return self.rows.value, self.columns.value

    def get_measurement_columns(self, pipeline):
        """return the measurements"""
        columns = []
        return columns

    def validate_module(self, pipeline):
        """Make sure the settings are consistent

        Check to make sure that we have enough rows and columns if
        we are in PlaceAdjacent mode.
        """
        if (
            self.tile_method == T_WITHIN_CYCLES
            and (not self.wants_automatic_rows)
            and (not self.wants_automatic_columns)
            and self.rows.value * self.columns.value < len(self.additional_images) + 1
        ):
            raise ValidationError(
                "There are too many images (%d) for a %d by %d grid"
                % (
                    len(self.additional_images) + 1,
                    self.columns.value,
                    self.rows.value,
                ),
                self.rows,
            )
