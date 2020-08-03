"""
InvertForPrinting
=================

**InvertForPrinting** inverts fluorescent images into
brightfield-looking images for printing.

This module turns a single or multi-channel immunofluorescent-stained
image into an image that resembles a brightfield image stained with
similarly colored stains, which generally prints better. You can operate
on up to three grayscale images (representing the red, green, and blue
channels of a color image) or on an image that is already a color image.
The module can produce either three grayscale images or one color image
as output. If you want to invert the grayscale intensities of an image,
use **ImageMath**.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           NO
============ ============ ===============

"""

import numpy
from cellprofiler_core.image import Image
from cellprofiler_core.module import Module
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting import ValidationError
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.text import ImageName

CC_GRAYSCALE = "Grayscale"
CC_COLOR = "Color"
CC_ALL = [CC_COLOR, CC_GRAYSCALE]


class InvertForPrinting(Module):
    module_name = "InvertForPrinting"
    category = "Image Processing"
    variable_revision_number = 1

    def create_settings(self):
        # Input settings
        self.input_color_choice = Choice(
            "Input image type",
            CC_ALL,
            doc="Specify whether you are combining several grayscale images or loading a single color image.",
        )

        self.wants_red_input = Binary(
            "Use a red image?",
            True,
            doc="""\
*(Used only if input image type is "{CC_GRAYSCALE}")*

Select "*Yes*" to specify an image to use for the red channel.
""".format(
                **{"CC_GRAYSCALE": CC_GRAYSCALE}
            ),
        )

        self.red_input_image = ImageSubscriber(
            "Select the red image",
            "None",
            doc="""\
*(Used only if input image type is "{CC_GRAYSCALE}" and a red image is used)*

Provide an image for the red channel.
""".format(
                **{"CC_GRAYSCALE": CC_GRAYSCALE}
            ),
        )

        self.wants_green_input = Binary(
            "Use a green image?",
            True,
            doc="""\
*(Used only if input image type is "{CC_GRAYSCALE}")*

Select "*Yes*" to specify an image to use for the green channel.
""".format(
                **{"CC_GRAYSCALE": CC_GRAYSCALE}
            ),
        )

        self.green_input_image = ImageSubscriber(
            "Select the green image",
            "None",
            doc="""\
*(Used only if input image type is "{CC_GRAYSCALE}" and a green image is used)*

Provide an image for the green channel.
""".format(
                **{"CC_GRAYSCALE": CC_GRAYSCALE}
            ),
        )

        self.wants_blue_input = Binary(
            "Use a blue image?",
            True,
            doc="""\
*(Used only if input image type is "{CC_GRAYSCALE}")*

Select "*Yes*" to specify an image to use for the blue channel.
""".format(
                **{"CC_GRAYSCALE": CC_GRAYSCALE}
            ),
        )

        self.blue_input_image = ImageSubscriber(
            "Select the blue image",
            "None",
            doc="""\
*(Used only if input image type is "{CC_GRAYSCALE}" and a blue image is used)*

Provide an image for the blue channel.
""".format(
                **{"CC_GRAYSCALE": CC_GRAYSCALE}
            ),
        )

        self.color_input_image = ImageSubscriber(
            "Select the color image",
            "None",
            doc="""
*(Used only if input image type is "{CC_COLOR}")*

Select the color image to use.
""".format(
                **{"CC_COLOR": CC_COLOR}
            ),
        )

        # Output settings
        self.output_color_choice = Choice(
            "Output image type",
            CC_ALL,
            doc="Specify whether you want to produce several grayscale images or one color image.",
        )

        self.wants_red_output = Binary(
            'Select "*Yes*" to produce a red image.',
            True,
            doc="""\
*(Used only if output image type is "{CC_GRAYSCALE}")*

Select "*Yes*" to produce a grayscale image corresponding to the inverted red channel.
""".format(
                **{"CC_GRAYSCALE": CC_GRAYSCALE}
            ),
        )

        self.red_output_image = ImageName(
            "Name the red image",
            "InvertedRed",
            doc="""\
*(Used only if output image type is "{CC_GRAYSCALE}" and a red image is output)*

Provide a name for the inverted red channel image.
""".format(
                **{"CC_GRAYSCALE": CC_GRAYSCALE}
            ),
        )

        self.wants_green_output = Binary(
            'Select "*Yes*" to produce a green image.',
            True,
            doc="""\
*(Used only if output image type is "{CC_GRAYSCALE}")*

Select "*Yes*" to produce a grayscale image corresponding to the inverted green channel.
""".format(
                **{"CC_GRAYSCALE": CC_GRAYSCALE}
            ),
        )

        self.green_output_image = ImageName(
            "Name the green image",
            "InvertedGreen",
            doc="""\
*(Used only if output image type is "{CC_GRAYSCALE}" and a green image is output)*

Provide a name for the inverted green channel image.
""".format(
                **{"CC_GRAYSCALE": CC_GRAYSCALE}
            ),
        )

        self.wants_blue_output = Binary(
            'Select "*Yes*" to produce a blue image.',
            True,
            doc="""\
*(Used only if output image type is "{CC_GRAYSCALE}")*

Select "*Yes*" to produce a grayscale image corresponding to the inverted blue channel.
""".format(
                **{"CC_GRAYSCALE": CC_GRAYSCALE}
            ),
        )

        self.blue_output_image = ImageName(
            "Name the blue image",
            "InvertedBlue",
            doc="""\
*(Used only if output image type is "{CC_GRAYSCALE}" and a blue image is output)*

Provide a name for the inverted blue channel image.
""".format(
                **{"CC_GRAYSCALE": CC_GRAYSCALE}
            ),
        )

        self.color_output_image = ImageName(
            "Name the inverted color image",
            "InvertedColor",
            doc="""\
*(Used only when producing a color output image)*

Enter a name for the inverted color image.
""",
        )

    def settings(self):
        """Return the settings as saved in the pipeline"""
        return [
            self.input_color_choice,
            self.wants_red_input,
            self.red_input_image,
            self.wants_green_input,
            self.green_input_image,
            self.wants_blue_input,
            self.blue_input_image,
            self.color_input_image,
            self.output_color_choice,
            self.wants_red_output,
            self.red_output_image,
            self.wants_green_output,
            self.green_output_image,
            self.wants_blue_output,
            self.blue_output_image,
            self.color_output_image,
        ]

    def help_settings(self):
        return [
            self.input_color_choice,
            self.wants_red_input,
            self.red_input_image,
            self.wants_green_input,
            self.green_input_image,
            self.wants_blue_input,
            self.blue_input_image,
            self.color_input_image,
            self.output_color_choice,
            self.color_output_image,
            self.wants_red_output,
            self.red_output_image,
            self.wants_green_output,
            self.green_output_image,
            self.wants_blue_output,
            self.blue_output_image,
        ]

    def visible_settings(self):
        """Return the settings as displayed in the UI"""
        result = [self.input_color_choice]
        if self.input_color_choice == CC_GRAYSCALE:
            for wants_input, input_image in (
                (self.wants_red_input, self.red_input_image),
                (self.wants_green_input, self.green_input_image),
                (self.wants_blue_input, self.blue_input_image),
            ):
                result += [wants_input]
                if wants_input.value:
                    result += [input_image]
        else:
            result += [self.color_input_image]
        result += [self.output_color_choice]
        if self.output_color_choice == CC_GRAYSCALE:
            for wants_output, output_image in (
                (self.wants_red_output, self.red_output_image),
                (self.wants_green_output, self.green_output_image),
                (self.wants_blue_output, self.blue_output_image),
            ):
                result += [wants_output]
                if wants_output.value:
                    result += [output_image]
        else:
            result += [self.color_output_image]
        return result

    def validate_module(self, pipeline):
        """Make sure the user has at least one of the grayscale boxes checked"""
        if (
            self.input_color_choice == CC_GRAYSCALE
            and (not self.wants_red_input.value)
            and (not self.wants_green_input.value)
            and (not self.wants_blue_input.value)
        ):
            raise ValidationError(
                "You must supply at least one grayscale input", self.wants_red_input
            )

    def run(self, workspace):
        image_set = workspace.image_set
        shape = None
        if self.input_color_choice == CC_GRAYSCALE:
            if self.wants_red_input.value:
                red_image = image_set.get_image(
                    self.red_input_image.value, must_be_grayscale=True
                ).pixel_data
                shape = red_image.shape
            else:
                red_image = 0
            if self.wants_green_input.value:
                green_image = image_set.get_image(
                    self.green_input_image.value, must_be_grayscale=True
                ).pixel_data
                shape = green_image.shape
            else:
                green_image = 0
            if self.wants_blue_input.value:
                blue_image = image_set.get_image(
                    self.blue_input_image.value, must_be_grayscale=True
                ).pixel_data
                shape = blue_image.shape
            else:
                blue_image = 0
            color_image = numpy.zeros((shape[0], shape[1], 3))
            color_image[:, :, 0] = red_image
            color_image[:, :, 1] = green_image
            color_image[:, :, 2] = blue_image
            red_image = color_image[:, :, 0]
            green_image = color_image[:, :, 1]
            blue_image = color_image[:, :, 2]
        elif self.input_color_choice == CC_COLOR:
            color_image = image_set.get_image(
                self.color_input_image.value, must_be_color=True
            ).pixel_data
            red_image = color_image[:, :, 0]
            green_image = color_image[:, :, 1]
            blue_image = color_image[:, :, 2]
        else:
            raise ValueError(
                "Unimplemented color choice: %s" % self.input_color_choice.value
            )
        inverted_red = (1 - green_image) * (1 - blue_image)
        inverted_green = (1 - red_image) * (1 - blue_image)
        inverted_blue = (1 - red_image) * (1 - green_image)
        inverted_color = numpy.dstack((inverted_red, inverted_green, inverted_blue))
        if self.output_color_choice == CC_GRAYSCALE:
            for wants_output, output_image_name, output_image in (
                (self.wants_red_output, self.red_output_image, inverted_red),
                (self.wants_green_output, self.green_output_image, inverted_green),
                (self.wants_blue_output, self.blue_output_image, inverted_blue),
            ):
                if wants_output.value:
                    image = Image(output_image)
                    image_set.add(output_image_name.value, image)
        elif self.output_color_choice == CC_COLOR:
            image = Image(inverted_color)
            image_set.add(self.color_output_image.value, image)
        else:
            raise ValueError(
                "Unimplemented color choice: %s" % self.output_color_choice.value
            )

        if self.show_window:
            workspace.display_data.color_image = color_image
            workspace.display_data.inverted_color = inverted_color

    def display(self, workspace, figure):
        figure.set_subplots((2, 1))
        color_image = workspace.display_data.color_image
        inverted_color = workspace.display_data.inverted_color
        figure.subplot_imshow(0, 0, color_image, "Original image")
        figure.subplot_imshow(
            1, 0, inverted_color, "Color-inverted image", sharexy=figure.subplot(0, 0)
        )
