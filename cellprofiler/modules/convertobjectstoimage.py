"""<b>Convert Objects To Image </b> converts objects you have identified into an image.
<hr>

This module allows you to take previously identified objects and convert
them into an image according to a colormap you select, which can then be saved
with the <b>SaveImages</b> modules.

<p>If you would like to save your objects but do not need a colormap,
you can by bypass this module and use the <b>SaveImages</b> module directly
by specifying "Objects" as the type of image to save.
"""

import centrosome.cpmorphology
import matplotlib.cm
import numpy

import cellprofiler.image
import cellprofiler.module
import cellprofiler.preferences
import cellprofiler.setting

DEFAULT_COLORMAP = "Default"


class ConvertObjectsToImage(cellprofiler.module.Module):
    module_name = "ConvertObjectsToImage"

    category = "Object Processing"

    variable_revision_number = 1

    def create_settings(self):
        self.object_name = cellprofiler.setting.ObjectNameSubscriber(
            "Select the input objects",
            cellprofiler.setting.NONE,
            doc="""Choose the name of the objects you want to convert to an image."""
        )

        self.image_name = cellprofiler.setting.ImageNameProvider(
            "Name the output image",
            "CellImage",
            doc="""Enter the name of the resulting image."""
        )

        self.image_mode = cellprofiler.setting.Choice(
            "Select the color format",
            [
                "Color",
                "Binary (black & white)",
                "Grayscale",
                "uint16"
            ],
            doc="""Select which colors the resulting image should use. You have the following options:
            <ul>
                <li>
                    <i>Color:</i> Allows you to choose a colormap that will produce jumbled colors for your objects.
                </li>
                <li>
                    <i>Binary (black & white):</i> All object pixels will be assigned 1 and all background pixels will
                    be assigned 0, creating a binary image.
                </li>
                <li>
                    <i>Grayscale:</i> Gives each object a graylevel pixel intensity value corresponding to its
                    number (also called label), so it usually results in objects on the left side of the image being
                    very dark, progressing toward white on the right side of the image.
                </li>
                <li>
                    <i>uint16:</i> Assigns each object a different number, from 1 to 65535 (the numbers that
                    you can put in a 16-bit integer) and numbers all pixels in each object with the object's number.
                    This format can be written out as a .mat or .tiff file if you want to process the label matrix
                    image using another program.
                </li>
            </ul>
            You can choose <i>Color</i> with a <i>Gray</i> colormap to produce jumbled gray objects."""
        )

        self.colormap = cellprofiler.setting.Colormap(
            "Select the colormap",
            doc="""<i>(Used only if "<i>Color</i>" output image selected)</i>
            <br>
            Choose the colormap to be used, which affects how the objects are colored. You can look up your default
            colormap under <i>File > Preferences</i>."""
        )

    def settings(self):
        return [
            self.object_name,
            self.image_name,
            self.image_mode,
            self.colormap
        ]

    def visible_settings(self):
        settings = [
            self.object_name,
            self.image_name,
            self.image_mode
        ]

        if self.image_mode == "Color":
            settings = settings + [
                self.colormap
            ]

        return settings

    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.object_name.value)

        alpha = numpy.zeros(objects.shape)

        convert = True

        if self.image_mode == "Binary (black & white)":
            pixel_data = numpy.zeros(objects.shape, bool)
        elif self.image_mode == "Grayscale":
            pixel_data = numpy.zeros(objects.shape)
        elif self.image_mode == "uint16":
            pixel_data = numpy.zeros(objects.shape, numpy.int32)
        else:
            pixel_data = numpy.zeros(objects.shape + (3,))

        for labels, _ in objects.get_labels():
            mask = labels != 0

            if numpy.all(~ mask):
                continue

            if self.image_mode == "Binary (black & white)":
                pixel_data[mask] = True

                alpha[mask] = 1
            elif self.image_mode == "Grayscale":
                pixel_data[mask] = labels[mask].astype(float) / numpy.max(labels)

                alpha[mask] = 1
            elif self.image_mode == "Color":
                if self.colormap.value == DEFAULT_COLORMAP:
                    cm_name = cellprofiler.preferences.get_default_colormap()
                elif self.colormap.value == "colorcube":
                    # Colorcube missing from matplotlib
                    cm_name = "gist_rainbow"
                elif self.colormap.value == "lines":
                    # Lines missing from matplotlib and not much like it,
                    # Pretty boring palette anyway, hence
                    cm_name = "Pastel1"
                elif self.colormap.value == "white":
                    # White missing from matplotlib, it's just a colormap
                    # of all completely white... not even different kinds of
                    # white. And, isn't white just a uniform sampling of
                    # frequencies from the spectrum?
                    cm_name = "Spectral"
                else:
                    cm_name = self.colormap.value

                cm = matplotlib.cm.get_cmap(cm_name)

                mapper = matplotlib.cm.ScalarMappable(cmap=cm)

                if labels.ndim == 3:
                    for index, plane in enumerate(mask):
                        pixel_data[index, plane, :] = mapper.to_rgba(
                            centrosome.cpmorphology.distance_color_labels(labels[index])
                        )[plane, :3]
                else:
                    pixel_data[mask, :] += mapper.to_rgba(
                        centrosome.cpmorphology.distance_color_labels(labels)
                    )[mask, :3]

                alpha[mask] += 1
            elif self.image_mode == "uint16":
                pixel_data[mask] = labels[mask]

                alpha[mask] = 1

                convert = False

        mask = alpha > 0

        if self.image_mode == "Color":
            pixel_data[mask, :] = pixel_data[mask, :] / alpha[mask][:, numpy.newaxis]
        elif self.image_mode != "Binary (black & white)":
            pixel_data[mask] = pixel_data[mask] / alpha[mask]

        image = cellprofiler.image.Image(
            pixel_data,
            parent_image=objects.parent_image,
            convert=convert,
            dimensions=len(objects.shape)
        )

        workspace.image_set.add(self.image_name.value, image)

        if self.show_window:
            if image.dimensions == 2:
                workspace.display_data.ijv = objects.ijv
            else:
                workspace.display_data.segmented = objects.segmented

            workspace.display_data.pixel_data = pixel_data

            workspace.display_data.dimensions = image.dimensions

    def display(self, workspace, figure):
        pixel_data = workspace.display_data.pixel_data

        dimensions = workspace.display_data.dimensions

        cmap = None if self.image_mode == "Color" else "gray"

        figure.set_subplots((2, 1), dimensions=dimensions)

        # TODO: volumetric IJV
        if dimensions == 2:
            figure.subplot_imshow_ijv(
                0,
                0,
                workspace.display_data.ijv,
                shape=workspace.display_data.pixel_data.shape[:2],
                title="Original: %s" % self.object_name.value
            )
        else:
            figure.subplot_imshow(
                0,
                0,
                workspace.display_data.segmented,
                title="Original: %s" % self.object_name.value,
                dimensions=dimensions
            )

        figure.subplot_imshow(
            1,
            0,
            pixel_data,
            self.image_name.value,
            colormap=cmap,
            dimensions=dimensions
        )

    def upgrade_settings(self, setting_values, variable_revision_number, module_name, from_matlab):
        if variable_revision_number == 1 and from_matlab:
            from_matlab = False

        return setting_values, variable_revision_number, from_matlab

    def volumetric(self):
        return True

#
# Backwards compatability
#
ConvertToImage = ConvertObjectsToImage
