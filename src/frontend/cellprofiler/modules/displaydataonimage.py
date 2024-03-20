"""
DisplayDataOnImage
==================

**DisplayDataOnImage** produces an image with measured data on top of
identified objects.

This module displays either a single image measurement on an image of
your choosing, or one object measurement per object on top of every
object in an image. The display itself is an image which you can save to
a file using **SaveImages**.

|

============ ============ ===============
Supports 2D? Supports 3D? Respects masks?
============ ============ ===============
YES          NO           YES
============ ============ ===============

"""

import matplotlib.axes
import matplotlib.cm
import matplotlib.figure
import matplotlib.text
import numpy
from cellprofiler_core.constants.measurement import C_FILE_NAME
from cellprofiler_core.constants.measurement import C_PATH_NAME
from cellprofiler_core.constants.measurement import M_LOCATION_CENTER_X
from cellprofiler_core.constants.measurement import M_LOCATION_CENTER_Y
from cellprofiler_core.image import FileImage
from cellprofiler_core.image import Image
from cellprofiler_core.module import Module
from cellprofiler_core.preferences import get_default_colormap
from cellprofiler_core.setting import Binary
from cellprofiler_core.setting import Color
from cellprofiler_core.setting import Measurement
from cellprofiler_core.setting.choice import Choice
from cellprofiler_core.setting.choice import Colormap
from cellprofiler_core.setting.range import FloatRange
from cellprofiler_core.setting.subscriber import ImageSubscriber
from cellprofiler_core.setting.subscriber import LabelSubscriber
from cellprofiler_core.setting.text import ImageName
from cellprofiler_core.setting.text import Integer

OI_OBJECTS = "Object"
OI_IMAGE = "Image"

E_FIGURE = "Figure"
E_AXES = "Axes"
E_IMAGE = "Image"

CT_COLOR = "Color"
CT_TEXT = "Text"

F_WEIGHT_NORMAL = "normal"
F_WEIGHT_BOLD = "bold"

CMS_USE_MEASUREMENT_RANGE = "Use this image's measurement range"
CMS_MANUAL = "Manual"

# Load fonts available to matplotlob in alphabetical order
font_list = sorted(set([font.name for font in matplotlib.font_manager.fontManager.ttflist]))
class DisplayDataOnImage(Module):
    module_name = "DisplayDataOnImage"
    category = "Data Tools"
    variable_revision_number = 6

    def create_settings(self):
        """Create your settings by subclassing this function

        create_settings is called at the end of initialization.

        You should create the setting variables for your module here:
            # Ask the user for the input image
            self.image_name = .ImageSubscriber(...)
            # Ask the user for the name of the output image
            self.output_image = .ImageName(...)
            # Ask the user for a parameter
            self.smoothing_size = .Float(...)
        """
        self.objects_or_image = Choice(
            "Display object or image measurements?",
            [OI_OBJECTS, OI_IMAGE],
            doc="""\
-  *%(OI_OBJECTS)s* displays measurements made on objects.
-  *%(OI_IMAGE)s* displays a single measurement made on an image.
"""
            % globals(),
        )

        self.objects_name = LabelSubscriber(
            "Select the input objects",
            "None",
            doc="""\
*(Used only when displaying object measurements)*

Choose the name of objects identified by some previous module (such as
**IdentifyPrimaryObjects** or **IdentifySecondaryObjects**).
""",
        )

        def object_fn():
            if self.objects_or_image == OI_OBJECTS:
                return self.objects_name.value
            else:
                return "Image"

        self.measurement = Measurement(
            "Measurement to display",
            object_fn,
            doc="""\
Choose the measurement to display. This will be a measurement made by
some previous module on either the whole image (if displaying a single
image measurement) or on the objects you selected.
""",
        )

        self.wants_image = Binary(
            "Display background image?",
            True,
            doc="""\
Choose whether or not to display the measurements on
a background image. Usually, you will want to see the image
context for the measurements, but it may be useful to save
just the overlay of the text measurements and composite the
overlay image and the original image later. Choose "Yes" to
display the measurements on top of a background image or "No"
to display the measurements on a black background.""",
        )

        self.image_name = ImageSubscriber(
            "Select the image on which to display the measurements",
            "None",
            doc="""\
Choose the image to be displayed behind the measurements.
This can be any image created or loaded by a previous module.
If you have chosen not to display the background image, the image
will only be used to determine the dimensions of the displayed image.""",
        )

        self.color_or_text = Choice(
            "Display mode",
            [CT_TEXT, CT_COLOR],
            doc="""\
*(Used only when displaying object measurements)*

Choose how to display the measurement information. If you choose
%(CT_TEXT)s, **DisplayDataOnImage** will display the numeric value on
top of each object. If you choose %(CT_COLOR)s, **DisplayDataOnImage**
will convert the image to grayscale, if necessary, and display the
portion of the image within each object using a hue that indicates the
measurement value relative to the other objects in the set using the
default color map.
"""
            % globals(),
        )

        self.colormap = Colormap(
            "Color map",
            doc="""\
*(Used only when displaying object measurements)*

This is the color map used as the color gradient for coloring the
objects by their measurement values. See `this page`_ for pictures
of the available colormaps.

.. _this page: http://matplotlib.org/users/colormaps.html
            """,
        )
        self.text_color = Color(
            "Text color",
            "red",
            doc="""This is the color that will be used when displaying the text.""",
        )

        self.display_image = ImageName(
            "Name the output image that has the measurements displayed",
            "DisplayImage",
            doc="""\
The name that will be given to the image with the measurements
superimposed. You can use this name to refer to the image in subsequent
modules (such as **SaveImages**).
""",
        )
        self.sci_notation = Binary(
            "Use scientific notation?",
            False,
            doc="""Choose whether to display data in scientific notation.
""",
        )

        self.font_choice = Choice(
            "Font",
            font_list,
            doc="""\
Set the font of the text to be displayed. 

Note: The fonts will be loaded from the system running CellProfiler. 
Not all fonts that are loaded will have the required glyphs, leading to 
blank or incomplete data displays. Moreover, not all fonts will support 
font weight changes. 
""",
        )
        self.font_weight = Choice(
            "Font weight",
            [F_WEIGHT_NORMAL, F_WEIGHT_BOLD],
            value="normal",
            doc="""Set the font weight of the text to be displayed""",
        )

        self.font_size = Integer(
            "Font size (points)",
            10,
            minval=1,
            doc="""Set the font size of the letters to be displayed.""",
        )

        self.decimals = Integer(
            "Number of decimals",
            2,
            minval=0,
            doc="""Set how many decimals to be displayed, for example 2 decimals for 0.01; 3 decimals for 0.001.""",
        )

        self.saved_image_contents = Choice(
            "Image elements to save",
            [E_IMAGE, E_FIGURE, E_AXES],
            doc="""\
This setting controls the level of annotation on the image:

-  *%(E_IMAGE)s:* Saves the image with the overlaid measurement
   annotations.
-  *%(E_AXES)s:* Adds axes with tick marks and image coordinates.
-  *%(E_FIGURE)s:* Adds a title and other decorations.
"""
            % globals(),
        )

        self.offset = Integer(
            "Annotation offset (in pixels)",
            0,
            doc="""\
Add a pixel offset to the measurement. Normally, the text is
placed at the object (or image) center, which can obscure relevant features of
the object. This setting adds a specified offset to the text, in a random
direction.""",
        )

        self.color_map_scale_choice = Choice(
            "Color map scale",
            [CMS_USE_MEASUREMENT_RANGE, CMS_MANUAL],
            doc="""\
*(Used only when displaying object measurements as a colormap)*

**DisplayDataOnImage** assigns a color to each object’s measurement
value from a colormap when in colormap-mode, mapping the value to a
color along the colormap’s continuum. This mapping has implicit upper
and lower bounds to its range which are the extremes of the colormap.
This setting determines whether the extremes are the minimum and
maximum values of the measurement from among the objects in the
current image or manually-entered extremes.

-  *%(CMS_USE_MEASUREMENT_RANGE)s:* Use the full range of colors to
   get the maximum contrast within the image.
-  *%(CMS_MANUAL)s:* Manually set the upper and lower bounds so that
   images with different maxima and minima can be compared by a uniform
   color mapping.
"""
            % globals(),
        )
        self.color_map_scale = FloatRange(
            "Color map range",
            value=(0.0, 1.0),
            doc="""\
*(Used only when setting a manual colormap range)*

This setting determines the lower and upper bounds of the values for the
color map.
""",
        )

    def settings(self):
        """Return the settings to be loaded or saved to/from the pipeline

        These are the settings (from cellprofiler_core.settings) that are
        either read from the strings in the pipeline or written out
        to the pipeline. The settings should appear in a consistent
        order so they can be matched to the strings in the pipeline.
        """
        return [
            self.objects_or_image,
            self.objects_name,
            self.measurement,
            self.image_name,
            self.text_color,
            self.display_image,
            self.font_size,
            self.decimals,
            self.saved_image_contents,
            self.offset,
            self.color_or_text,
            self.colormap,
            self.wants_image,
            self.color_map_scale_choice,
            self.color_map_scale,
            self.font_choice,
            self.sci_notation,
            self.font_weight
        ]

    def visible_settings(self):
        """The settings that are visible in the UI
        """
        result = [self.objects_or_image]
        if self.objects_or_image == OI_OBJECTS:
            result += [self.objects_name]
        result += [self.measurement, self.wants_image, self.image_name]
        if self.objects_or_image == OI_OBJECTS:
            result += [self.color_or_text]
        if self.use_color_map():
            result += [self.colormap, self.color_map_scale_choice]
            if self.color_map_scale_choice == CMS_MANUAL:
                result += [self.color_map_scale]
        else:
            result += [self.font_choice, self.font_weight, self.sci_notation, self.text_color, self.font_size, self.decimals, self.offset]
        result += [self.display_image, self.saved_image_contents]
        return result

    def use_color_map(self):
        """True if the measurement values are rendered using a color map"""
        return self.objects_or_image == OI_OBJECTS and self.color_or_text == CT_COLOR

    def run(self, workspace):
        import matplotlib
        import matplotlib.cm
        import matplotlib.backends.backend_agg
        import matplotlib.transforms
        from cellprofiler.gui.tools import figure_to_image, only_display_image

        #
        # Get the image
        #
        image = workspace.image_set.get_image(self.image_name.value)
        if self.wants_image:
            pixel_data = image.pixel_data
        else:
            pixel_data = numpy.zeros(image.pixel_data.shape[:2])
        object_set = workspace.object_set
        if self.objects_or_image == OI_OBJECTS:
            if self.objects_name.value in object_set.get_object_names():
                objects = object_set.get_objects(self.objects_name.value)
            else:
                objects = None
        workspace.display_data.pixel_data = pixel_data
        if self.use_color_map():
            workspace.display_data.labels = objects.segmented
        #
        # Get the measurements and positions
        #
        measurements = workspace.measurements
        if self.objects_or_image == OI_IMAGE:
            value = measurements.get_current_image_measurement(self.measurement.value)
            values = [value]
            x = [pixel_data.shape[1] / 2]
            x_offset = numpy.random.uniform(high=1.0, low=-1.0)
            x[0] += x_offset
            y = [pixel_data.shape[0] / 2]
            y_offset = numpy.sqrt(1 - x_offset ** 2)
            y[0] += y_offset
        else:
            values = measurements.get_current_measurement(
                self.objects_name.value, self.measurement.value
            )
            if objects is not None:
                if len(values) < objects.count:
                    temp = numpy.zeros(objects.count, values.dtype)
                    temp[: len(values)] = values
                    temp[len(values) :] = numpy.nan
                    values = temp
                elif len(values) > objects.count:
                    # If the values for something (say, object number) are greater
                    # than the actual number of objects we have, some might have been
                    # filtered out/removed. We'll need to diff the arrays to figure out
                    # what objects to remove
                    indices = objects.indices
                    diff = numpy.setdiff1d(indices, numpy.unique(objects.segmented))
                    values = numpy.delete(values, diff)
            x = measurements.get_current_measurement(
                self.objects_name.value, M_LOCATION_CENTER_X
            )
            x_offset = numpy.random.uniform(high=1.0, low=-1.0, size=x.shape)
            y_offset = numpy.sqrt(1 - x_offset ** 2)
            x += self.offset.value * x_offset
            y = measurements.get_current_measurement(
                self.objects_name.value, M_LOCATION_CENTER_Y
            )
            y += self.offset.value * y_offset
            if numpy.issubdtype(values.dtype, str):
                if self.use_color_map():
                    raise NotImplementedError("Cannot interpret a text measurement for display with a color scale")
                mask = ~(numpy.isnan(x) | numpy.isnan(y))
            else:
                mask = ~(numpy.isnan(values) | numpy.isnan(x) | numpy.isnan(y))
            values = values[mask]
            x = x[mask]
            y = y[mask]
            workspace.display_data.mask = mask
        workspace.display_data.values = values
        workspace.display_data.x = x
        workspace.display_data.y = y
        fig = matplotlib.figure.Figure()
        axes = fig.add_subplot(1, 1, 1)

        def imshow_fn(pixel_data):
            # Note: requires typecast to avoid failure during
            #       figure_to_image (IMG-764)
            img = pixel_data * 255
            img[img < 0] = 0
            img[img > 255] = 255
            img = img.astype(numpy.uint8)
            axes.imshow(img, cmap=matplotlib.cm.get_cmap("Greys"))

        self.display_on_figure(workspace, axes, imshow_fn)

        canvas = matplotlib.backends.backend_agg.FigureCanvasAgg(fig)
        if self.saved_image_contents == E_AXES:
            fig.set_frameon(False)
            if not self.use_color_map():
                fig.subplots_adjust(0.1, 0.1, 0.9, 0.9, 0, 0)
            shape = pixel_data.shape
            width = float(shape[1]) / fig.dpi
            height = float(shape[0]) / fig.dpi
            fig.set_figheight(height)
            fig.set_figwidth(width)
        elif self.saved_image_contents == E_IMAGE:
            if self.use_color_map():
                fig.axes[1].set_visible(False)
            only_display_image(fig, pixel_data.shape)
        else:
            if not self.use_color_map():
                fig.subplots_adjust(0.1, 0.1, 0.9, 0.9, 0, 0)

        pixel_data = figure_to_image(fig, dpi=fig.dpi)
        image = Image(pixel_data)
        workspace.image_set.add(self.display_image.value, image)

    def run_as_data_tool(self, workspace):
        # Note: workspace.measurements.image_set_number contains the image
        #    number that should be displayed.
        import wx
        import os.path

        im_id = self.image_name.value

        m = workspace.measurements
        image_name = self.image_name.value
        pathname_feature = "_".join((C_PATH_NAME, image_name))
        filename_feature = "_".join((C_FILE_NAME, image_name))
        if not all(
            [m.has_feature("Image", f) for f in (pathname_feature, filename_feature)]
        ):
            with wx.FileDialog(
                None,
                message="Image file for display",
                wildcard="Image files (*.tif, *.png, *.jpg)|*.tif;*.png;*.jpg|"
                "All files (*.*)|*.*",
            ) as dlg:
                if dlg.ShowModal() != wx.ID_OK:
                    return
            pathname, filename = os.path.split(dlg.Path)
        else:
            pathname = m.get_current_image_measurement(pathname_feature)
            filename = m.get_current_image_measurement(filename_feature)

        # Add the image to the workspace ImageSetList
        image_set_list = workspace.image_set_list
        image_set = image_set_list.get_image_set(0)
        ip = FileImage(im_id, pathname, filename)
        image_set.add_provider(ip)

        self.run(workspace)

    def display(self, workspace, figure):
        figure.set_subplots((1, 1))
        ax = figure.subplot(0, 0)
        title = "%s_%s" % (
            self.objects_name.value if self.objects_or_image == OI_OBJECTS else "Image",
            self.measurement.value,
        )

        def imshow_fn(pixel_data):
            if pixel_data.ndim == 3:
                figure.subplot_imshow_color(0, 0, pixel_data, title=title)
            else:
                figure.subplot_imshow_grayscale(0, 0, pixel_data, title=title)

        self.display_on_figure(workspace, ax, imshow_fn)

    def display_on_figure(self, workspace, axes, imshow_fn):
        if self.use_color_map():
            labels = workspace.display_data.labels
            if self.wants_image:
                pixel_data = workspace.display_data.pixel_data
            else:
                pixel_data = (labels != 0).astype(numpy.float32)
            if pixel_data.ndim == 3:
                pixel_data = numpy.sum(pixel_data, 2) / pixel_data.shape[2]
            colormap_name = self.colormap.value
            if colormap_name == "Default":
                colormap_name = get_default_colormap()
            colormap = matplotlib.cm.get_cmap(colormap_name)
            values = workspace.display_data.values
            vmask = workspace.display_data.mask
            colors = numpy.ones((len(vmask) + 1, 4))
            colors[1:][~vmask, :3] = 1
            sm = matplotlib.cm.ScalarMappable(cmap=colormap)
            if self.color_map_scale_choice == CMS_MANUAL:
                sm.set_clim(self.color_map_scale.min, self.color_map_scale.max)
            sm.set_array(values)
            colors[1:][vmask, :] = sm.to_rgba(values)
            img = colors[labels, :3] * pixel_data[:, :, numpy.newaxis]
            imshow_fn(img)
            assert isinstance(axes, matplotlib.axes.Axes)
            figure = axes.get_figure()
            assert isinstance(figure, matplotlib.figure.Figure)
            figure.colorbar(sm, ax=axes)
        else:
            imshow_fn(workspace.display_data.pixel_data)
            for x, y, value in zip(
                workspace.display_data.x,
                workspace.display_data.y,
                workspace.display_data.values,
            ):
                if self.sci_notation:
                    svalue = f"{value:.{self.decimals.value}e}"
                else:
                    try:
                        svalue = "%.*f" % (self.decimals.value, value)
                    except:
                        svalue = str(value)
                text = matplotlib.text.Text(
                    x=x,
                    y=y,
                    text=svalue,
                    size=self.font_size.value,
                    color=self.text_color.value,
                    verticalalignment="center",
                    horizontalalignment="center",
                    fontname=self.font_choice.value,
                    weight=self.font_weight.value,
                )
                axes.add_artist(text)

    def upgrade_settings(self, setting_values, variable_revision_number, module_name):
        if variable_revision_number == 1:
            (
                objects_or_image,
                objects_name,
                measurement,
                image_name,
                text_color,
                display_image,
                dpi,
                saved_image_contents,
            ) = setting_values
            setting_values = [
                objects_or_image,
                objects_name,
                measurement,
                image_name,
                text_color,
                display_image,
                10,
                2,
                saved_image_contents,
            ]
            variable_revision_number = 2

        if variable_revision_number == 2:
            """Added annotation offset"""
            setting_values = setting_values + ["0"]
            variable_revision_number = 3

        if variable_revision_number == 3:
            # Added color map mode
            setting_values = setting_values + [
                CT_TEXT,
                get_default_colormap(),
            ]
            variable_revision_number = 4

        if variable_revision_number == 4:
            # added wants_image
            setting_values = setting_values + ["Yes"]
            variable_revision_number = 5
        if variable_revision_number == 5:
            # added color_map_scale_choice and color_map_scale
            setting_values = setting_values + [CMS_USE_MEASUREMENT_RANGE, "0.0,1.0"]
            variable_revision_number = 6
        return setting_values, variable_revision_number
