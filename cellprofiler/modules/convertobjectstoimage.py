'''<b>Convert Objects To Image </b> converts objects you have identified into an image.
<hr>

This module allows you to take previously identified objects and convert
them into an image according to a colormap you select, which can then be saved
with the <b>SaveImages</b> modules.

<p>If you would like to save your objects but do not need a colormap,
you can by bypass this module and use the <b>SaveImages</b> module directly
by specifying "Objects" as the type of image to save.
'''

import numpy as np

import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.preferences as cpprefs
import cellprofiler.settings as cps

DEFAULT_COLORMAP = "Default"
COLORCUBE = "colorcube"
LINES = "lines"
WHITE = "white"
COLORMAPS = ["Default", "autumn", "bone", COLORCUBE, "cool", "copper",
             "flag", "gray", "hot", "hsv", "jet", LINES,"pink", "prism",
             "spring", "summer", WHITE, "winter" ]

IM_COLOR = "Color"
IM_BINARY = "Binary (black & white)"
IM_GRAYSCALE = "Grayscale"
IM_UINT16 = "uint16"
IM_ALL = [IM_COLOR, IM_BINARY, IM_GRAYSCALE, IM_UINT16]

class ConvertObjectsToImage(cpm.CPModule):

    module_name = "ConvertObjectsToImage"
    category = "Object Processing"
    variable_revision_number = 1

    def create_settings(self):
        self.object_name = cps.ObjectNameSubscriber(
            "Select the input objects",cps.NONE,doc="""
            Choose the name of the objects you want to convert to an image.""")

        self.image_name = cps.ImageNameProvider(
            "Name the output image", "CellImage",doc="""
            Enter the name of the resulting image.""")

        self.image_mode = cps.Choice(
            "Select the color format",
            IM_ALL,doc="""
            Select which colors the resulting image should use. You have the following
            options:
            <ul>
            <li><i>%(IM_COLOR)s:</i> Allows you to choose a colormap that will
            produce jumbled colors for your objects. </li>
            <li><i>%(IM_BINARY)s:</i> All object pixels will be assigned 1 and all
            background pixels will be assigned 0, creating a binary image.</li>
            <li><i>%(IM_GRAYSCALE)s:</i> Gives each object
            a graylevel pixel intensity value corresponding to its number (also
            called label), so it usually results in objects on the left side of the
            image being very dark, progressing toward white on the right side of
            the image. </li>
            <li><i>%(IM_UINT16)s:</i> Assigns each object a different number,
            from 1 to 65535 (the numbers that you can put in
            a 16-bit integer) and numbers all pixels in each
            object with the object's number. This format can
            be written out as a .mat or .tiff file if you
            want to process the label matrix image using
            another program.</li>
            </ul>
            You can choose <i>%(IM_COLOR)s</i> with a <i>Gray</i> colormap to produce
            jumbled gray objects."""%globals())

        self.colormap = cps.Colormap(
            "Select the colormap",doc="""
            <i>(Used only if "<i>%(IM_COLOR)s</i>" output image selected)</i><br>
            Choose the colormap to be used, which affects how the objects are colored.
            You can look up your default colormap under <i>File > Preferences</i>.
            """%globals())

    def settings(self):
        return [self.object_name, self.image_name, self.image_mode,
                self.colormap]

    def visible_settings(self):
        if self.image_mode == IM_COLOR:
            return [self.object_name, self.image_name, self.image_mode,
                    self.colormap]
        else:
            return [self.object_name, self.image_name, self.image_mode]

    def run(self, workspace):
        objects = workspace.object_set.get_objects(self.object_name.value)
        alpha = np.zeros(objects.shape)
        if self.image_mode == IM_BINARY:
            pixel_data = np.zeros(objects.shape, bool)
        elif self.image_mode == IM_GRAYSCALE:
            pixel_data = np.zeros(objects.shape)
        elif self.image_mode == IM_UINT16:
            pixel_data = np.zeros(objects.shape, np.int32)
        else:
            pixel_data = np.zeros((objects.shape[0], objects.shape[1], 3))
        convert = True
        for labels, indices in objects.get_labels():
            mask = labels != 0
            if np.all(~ mask):
                continue
            if self.image_mode == IM_BINARY:
                pixel_data[mask] = True
                alpha[mask] = 1
            elif self.image_mode == IM_GRAYSCALE:
                pixel_data[mask] = labels[mask].astype(float) / np.max(labels)
                alpha[mask] = 1
            elif self.image_mode == IM_COLOR:
                import matplotlib.cm
                from cellprofiler.gui.cpfigure_tools import renumber_labels_for_display
                if self.colormap.value == DEFAULT_COLORMAP:
                    cm_name = cpprefs.get_default_colormap()
                elif self.colormap.value == COLORCUBE:
                    # Colorcube missing from matplotlib
                    cm_name = "gist_rainbow"
                elif self.colormap.value == LINES:
                    # Lines missing from matplotlib and not much like it,
                    # Pretty boring palette anyway, hence
                    cm_name = "Pastel1"
                elif self.colormap.value == WHITE:
                    # White missing from matplotlib, it's just a colormap
                    # of all completely white... not even different kinds of
                    # white. And, isn't white just a uniform sampling of
                    # frequencies from the spectrum?
                    cm_name = "Spectral"
                else:
                    cm_name = self.colormap.value
                cm = matplotlib.cm.get_cmap(cm_name)
                mapper = matplotlib.cm.ScalarMappable(cmap=cm)
                pixel_data[mask, :] += \
                    mapper.to_rgba(renumber_labels_for_display(labels))[mask, :3]
                alpha[mask] += 1
            elif self.image_mode == IM_UINT16:
                pixel_data[mask] = labels[mask]
                alpha[mask] = 1
                convert = False
        mask = alpha > 0
        if self.image_mode == IM_BINARY:
            pass
        elif self.image_mode == IM_COLOR:
            pixel_data[mask, :] = pixel_data[mask, :] / alpha[mask][:, np.newaxis]
        else:
            pixel_data[mask] = pixel_data[mask] / alpha[mask]
        image = cpi.Image(pixel_data, parent_image = objects.parent_image,
                          convert = convert)
        workspace.image_set.add(self.image_name.value, image)
        if self.show_window:
            workspace.display_data.ijv = objects.ijv
            workspace.display_data.pixel_data = pixel_data

    def display(self, workspace, figure):
        pixel_data = workspace.display_data.pixel_data
        figure.set_subplots((2, 1))
        figure.subplot_imshow_ijv(
            0, 0, workspace.display_data.ijv,
            shape = workspace.display_data.pixel_data.shape[:2],
                title = "Original: %s"%self.object_name.value)
        if self.image_mode == IM_BINARY:
            figure.subplot_imshow_bw(1, 0, pixel_data,
                                            self.image_name.value,
                                            sharexy = figure.subplot(0, 0))
        elif pixel_data.shape[1] == 2:
            figure.subplot_imshow_grayscale(1, 0, pixel_data,
                                            self.image_name.value,
                                            sharexy = figure.subplot(0, 0))
        else:
            figure.subplot_imshow_grayscale(1, 0, pixel_data,
                                            self.image_name.value,
                                            sharexy = figure.subplot(0, 0))


    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if variable_revision_number == 1 and from_matlab:
            from_matlab = False
        return setting_values, variable_revision_number, from_matlab

#
# Backwards compatability
#
ConvertToImage = ConvertObjectsToImage
