"""
FindMaxima
==========

**FindMaxima** isolates local peaks of high intensity from an image.

The returned image will feature single pixels at each position where
a peak of intensity was found in the input image.

This can be useful for finding particular points of interest,
identifying very small objects or generating markers for segmentation
with the Watershed module.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          YES          NO
============ ============ ===============
"""

import numpy
from skimage.feature import peak_local_max
from skimage.morphology import disk, ball, dilation

from cellprofiler_core.image import Image
from cellprofiler_core.module import ImageProcessing
from cellprofiler_core.setting import Color, Divider
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.subscriber import ImageSubscriber, LabelSubscriber
from cellprofiler_core.setting.text import Integer, Float
from cellprofiler_core.utilities.core.object import overlay_labels

MODE_THRESHOLD = "Threshold"
MODE_MASK = "Mask"
MODE_OBJECTS = "Within Objects"


class FindMaxima(ImageProcessing):
    category = "Advanced"

    module_name = "FindMaxima"

    variable_revision_number = 1

    def create_settings(self):
        super(FindMaxima, self).create_settings()

        self.min_distance = Integer(
            text="Minimum distance between maxima",
            value=5,
            minval=0,
            doc="""Choose the minimum distance between accepted local maxima"""
        )

        self.exclude_mode = Choice(
            "Method for excluding background",
            [MODE_THRESHOLD, MODE_MASK, MODE_OBJECTS],
            value="Threshold",
            doc=f"""\
By default, local maxima will be searched for across the whole image. This means that maxima will be found in 
areas that consist entirely of background. To resolve this we have several methods to exclude background.

**{MODE_THRESHOLD}** allows you to specify a minimum pixel intensity to be considered as a peak. Setting this to 0
effectively uses no threshold.

**{MODE_MASK}** will restrict peaks to areas which are within a provided mask image. This mask will typically come from 
the threshold module or another means of finding background.

**{MODE_OBJECTS}** will restrict peaks to areas within an existing set of objects.
"""

        )

        self.min_intensity = Float(
            "Specify the minimum intensity of a peak",
            0,
            minval=0,
            maxval=99,
            doc="""\
Intensity peaks below this threshold value will be excluded. Use this to ensure that your local 
maxima are within objects of interest."""

        )

        self.mask_image = ImageSubscriber(
            "Select the image to use as a mask", doc="Select the image you want to use. This should be a binary image."
        )

        self.mask_objects = LabelSubscriber(
            "Select the objects to search within", doc="Select the objects within which to search for peaks."
        )

        self.maxima_color = Color(
            "Select maxima preview color",
            "Red",
            doc="Maxima will be displayed in this color.",
        )

        self.maxima_size = Integer(
            "Select maxima preview size",
            value=1,
            minval=1,
            doc="Size of the markers for each maxima in the preview. Positive pixels will be"
                "expanded by this radius."
                "You may want to increase this when working with large images.",
        )

        self.spacer = Divider(line=True)

    def settings(self):
        __settings__ = super(FindMaxima, self).settings()

        return __settings__ + [self.min_distance, self.exclude_mode,
                               self.min_intensity, self.mask_image, self.mask_objects]

    def visible_settings(self):
        __settings__ = super(FindMaxima, self).visible_settings()

        result = __settings__ + [self.min_distance, self.exclude_mode]

        if self.exclude_mode == MODE_THRESHOLD:
            result.append(self.min_intensity)
        elif self.exclude_mode == MODE_MASK:
            result.append(self.mask_image)
        elif self.exclude_mode == MODE_OBJECTS:
            result.append(self.mask_objects)

        result += [self.spacer, self.maxima_color, self.maxima_size]

        return result

    def run(self, workspace):

        x_name = self.x_name.value

        y_name = self.y_name.value

        images = workspace.image_set

        x = images.get_image(x_name)

        dimensions = x.dimensions

        x_data_orig = x.pixel_data

        x_data = x_data_orig.copy()

        th_abs = None

        if self.exclude_mode.value == MODE_THRESHOLD:
            th_abs = self.min_intensity.value
        elif self.exclude_mode.value == MODE_MASK:
            mask = images.get_image(self.mask_image.value).pixel_data.astype(bool)
            x_data[~mask] = 0
        elif self.exclude_mode.value == MODE_OBJECTS:
            mask_objects = workspace.object_set.get_objects(self.mask_objects.value)
            mask = mask_objects.segmented.astype(bool)
            x_data[~mask] = 0
        else:
            raise NotImplementedError("Invalid background method choice")

        y_data = peak_local_max(
            x_data,
            min_distance=self.min_distance.value,
            threshold_abs=th_abs,
            indices=False
        )

        y = Image(dimensions=dimensions, image=y_data, parent_image=x, convert=False)

        images.add(y_name, y)

        if self.show_window:
            workspace.display_data.x_data = x_data_orig

            workspace.display_data.y_data = y_data

            workspace.display_data.dimensions = dimensions

            workspace.display_data.overlay_base = x_data

    def display(self, workspace, figure, cmap=None):
        """Display the image and labeling"""
        layout = (2, 2)
        dimensions = workspace.display_data.dimensions

        figure.set_subplots(
            dimensions=dimensions, subplots=layout
        )

        title = "Input image, cycle #%d" % (workspace.measurements.image_number,)
        image = workspace.display_data.x_data
        maxima_image = workspace.display_data.y_data.astype(int)
        overlay_base = workspace.display_data.overlay_base

        ax = figure.subplot_imshow_grayscale(0, 0, image, title)
        figure.subplot_imshow_grayscale(1, 0, maxima_image, self.y_name.value, sharexy=ax)

        # Generate static colormap for alpha overlay
        from matplotlib.colors import ListedColormap

        cmap = ListedColormap(self.maxima_color.value)
        if self.maxima_size.value > 1:
            if dimensions == 2:
                strel = disk(self.maxima_size.value - 1)
            else:
                strel = ball(self.maxima_size.value - 1)
            labels = dilation(maxima_image, footprint=strel)
        else:
            labels = maxima_image

        overlay_display = overlay_labels(overlay_base, labels, opacity=numpy.max(overlay_base))
        
        title = "Overlay"
        figure.subplot_imshow_grayscale(
            0, 1, overlay_display, title, colormap=None, sharexy=ax
        )

    def update_settings(self, settings):
        return settings

    def volumetric(self):
        return True
