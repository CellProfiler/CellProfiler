'''<b>Display DataOn Image</b> 
produces an image with measured data on top of identified objects
<hr>
This module displays either a single image measurement on an image of
your choosing, or one object measurement per object on top
of every object in an image. The display itself is an image which you
can save to a file using <b>SaveImages</b>.
'''

# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2010
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org

__version__="$Revision: 1 $"

import numpy as np

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.preferences as cpprefs
from cellprofiler.modules.identify import M_LOCATION_CENTER_X, M_LOCATION_CENTER_Y

OI_OBJECTS = "Object"
OI_IMAGE = "Image"

E_FIGURE = "Figure"
E_AXES = "Axes"
E_IMAGE = "Image"

class DisplayDataOnImage(cpm.CPModule):
    module_name = 'DisplayDataOnImage'
    category = 'Other'
    variable_revision_number = 1
    
    def create_settings(self):
        """Create your settings by subclassing this function
        
        create_settings is called at the end of initialization.
        
        You should create the setting variables for your module here:
            # Ask the user for the input image
            self.image_name = cellprofiler.settings.ImageNameSubscriber(...)
            # Ask the user for the name of the output image
            self.output_image = cellprofiler.settings.ImageNameProvider(...)
            # Ask the user for a parameter
            self.smoothing_size = cellprofiler.settings.Float(...)
        """
        self.objects_or_image = cps.Choice(
            "Do you want to display object or image measurements?",
            [OI_OBJECTS, OI_IMAGE],
            doc = """Choose <i>Image</i> to display a single measurement made
            on an image. Choose <i>Object</i> to display measurements made on
            objects.""")
        self.objects_name = cps.ObjectNameSubscriber(
            "Which objects' measurements do you want to display?", "None",
            doc = """Choose the name of objects identified by some previous
            module (such as <b>IdentifyPrimAutomatic</b> or
            <b>IdentifySecondary</b>).""")
        def object_fn():
            if self.objects_or_image == OI_OBJECTS:
                return self.objects_name.value
            else:
                return cpmeas.IMAGE
        self.measurement = cps.Measurement(
            "Measurement to display", object_fn,
            doc="""Choose the measurement to display. This will be a measurement
            made by some previous module on either the whole image (if
            displaying a single image measurement) or on the objects you
            selected.""")
        self.image_name = cps.ImageNameSubscriber(
            "Select the image on which to display the measurements", "None",
            doc="""Choose the image to be displayed behind the measurements.
            This can be any image created or loaded by a previous module.""")
        self.text_color = cps.Text(
            "Text color","red",
            doc="""This is the color that will be used when displaying the text.
            There are several different ways to specify the color:<br>
            <ul><li>Single letter: b=blue, g=green, r=red, c=cyan, m=magenta,
            y=yellow, k=black, w=white</li>
            <li>By name: You can use any name supported by HTML. The following
            link has a list of colors:
            http://www.w3schools.com/html/html_colornames.asp
            </li>
            <li>By RGB code: You can specify the color as a combination of
            the red, green and blue intensities, for instance, "#FFFF00"
            for yellow (yellow = (red:FF, green:FF, blue:00) where FF is
            hexadecimal for 255, the highest intensity). See
            http://www.w3schools.com/html/html_colors.asp for a more detailed
            explanation</li></ul>""")
        self.display_image = cps.ImageNameProvider(
            "Name the output image, which has the measurements displayed","DisplayImage",
            doc="""This is the name that will be given to the image with
            the measurements superimposed. You can use this name to refer to the image in
            subsequent modules (such as <b>SaveImages</b>).""")
        self.dpi = cps.Float(
            "Resolution (pixels per inch)",96.0,minval=1.0,
            doc="""This is the resolution to be used when displaying the image
            (in pixels per inch).""")
        self.saved_image_contents = cps.Choice(
            "What elements do you want to save?",
            [E_IMAGE, E_FIGURE, E_AXES],
            doc="""This setting controls the level of annotation on the image:
            <ul><li>
            <i>Image</i>: The module will save the image with
            the overlaid measurement annotations.</li>
            <li><i>Axes</i>:
            The module adds axes with tick marks and image coordinates.</li>
            <li><i>Figure</i>: The module adds a title and other
            decorations.</li></ul>""")
        
    def settings(self):
        """Return the settings to be loaded or saved to/from the pipeline
        
        These are the settings (from cellprofiler.settings) that are
        either read from the strings in the pipeline or written out
        to the pipeline. The settings should appear in a consistent
        order so they can be matched to the strings in the pipeline.
        """
        return [self.objects_or_image, self.objects_name, self.measurement,
                self.image_name, self.text_color, self.display_image,
                self.dpi, self.saved_image_contents]
    
    def visible_settings(self):
        """The settings that are visible in the UI
        """
        result = [self.objects_or_image]
        if self.objects_or_image == OI_OBJECTS:
            result += [self.objects_name]
        result += [self.measurement, self.image_name, self.text_color,
                   self.display_image, self.dpi, self.saved_image_contents]
        return result
        
    def is_interactive(self):
        return False
    
    def run(self, workspace):
        import matplotlib
        import matplotlib.backends.backend_wxagg
        from cellprofiler.gui.cpfigure import figure_to_image
        #
        # Get the image
        #
        image = workspace.image_set.get_image(self.image_name.value)
        workspace.display_data.pixel_data = image.pixel_data
        #
        # Get the measurements and positions
        #
        measurements = workspace.measurements
        if self.objects_or_image == OI_IMAGE:
            value = measurements.get_current_image_measurement(
                self.measurement.value)
            values = [value]
            x = [image.pixel_data.shape[1] / 2]
            y = [image.pixel_data.shape[0] / 2]
        else:
            values = measurements.get_current_measurement(
                self.objects_name.value,
                self.measurement.value)
            x = measurements.get_current_measurement(
                self.objects_name.value, M_LOCATION_CENTER_X)
            y = measurements.get_current_measurement(
                self.objects_name.value, M_LOCATION_CENTER_Y)
            mask = ~(np.isnan(values) | np.isnan(x) | np.isnan(y))
            values = values[mask]
            x = x[mask]
            y = y[mask]
        workspace.display_data.values = values
        workspace.display_data.x = x
        workspace.display_data.y = y
        if self.saved_image_contents != E_FIGURE:
            # Set the aspect ratio 
            my_dpi = self.dpi.value * 2
            height = float(image.pixel_data.shape[0]) * 4.0 / my_dpi
            width = float(image.pixel_data.shape[1]) *4.0 / my_dpi
            figure = matplotlib.figure.Figure(figsize=(width, height))
        else:
            # Set the aspect ratio 
            my_dpi = self.dpi.value * 2
            height = float(image.pixel_data.shape[0]) * 4.0 / my_dpi / .8
            width = float(image.pixel_data.shape[1]) *4.0 / my_dpi / .8
            figure = matplotlib.figure.Figure(figsize=(width, height))
        figure.set_dpi(my_dpi)
        self.display_on_figure(workspace, figure)
        if self.saved_image_contents == E_AXES:
            figure.subplots_adjust(0,0,1,1,0,0)
        elif self.saved_image_contents == E_IMAGE:
            figure.subplots_adjust(0,0,1,1,0,0)
            axes = figure.axes[0]
            assert isinstance(axes,matplotlib.axes.Axes)
            axes.set_axis_off()
        else:
            figure.subplots_adjust(.1,.1,.9,.9,0,0)
            
        canvas = matplotlib.backends.backend_wxagg.FigureCanvasAgg(figure)
        pixel_data = figure_to_image(figure)
        image = cpi.Image(pixel_data)
        workspace.image_set.add(self.display_image.value, image)
        
    def display(self, workspace):
        figure_frame = workspace.create_or_find_figure()
        figure_frame.clf()
        figure_frame.figure.set_dpi(self.dpi.value)
        self.display_on_figure(workspace, figure_frame.figure)
        
    def display_on_figure(self, workspace, figure):
        import matplotlib
        axes = figure.add_subplot(1,1,1)
        assert isinstance(axes, matplotlib.axes.Axes)
        axes.set_title("%s\n%s" % 
                       (self.objects_name.value, self.measurement.value),
                       fontname = cpprefs.get_title_font_name(),
                       fontsize = cpprefs.get_title_font_size())
        axes_image = axes.imshow(workspace.display_data.pixel_data,
                                 cmap = matplotlib.cm.Greys)
        for x, y, value in zip(workspace.display_data.x,
                               workspace.display_data.y,
                               workspace.display_data.values):
            try:
                fvalue = float(value)
                if round(fvalue) == value:
                    svalue = str(value)
                else:
                    svalue = round(fvalue,3)
            except:
                svalue = str(value)
            
            text = matplotlib.text.Text(x=x, y=y, text=svalue,
                                        color=self.text_color.value,
                                        verticalalignment='center',
                                        horizontalalignment='center')
            axes.add_artist(text)
        
                
            
    def upgrade_settings(self, setting_values, variable_revision_number, 
                         module_name, from_matlab):
        if from_matlab and (variable_revision_number == 2):
            object_name, category, feature_nbr, image_name, size_scale, \
                display_image, data_image, dpi_to_save, \
                saved_image_contents = setting_values
            objects_or_image = (OI_IMAGE if object_name == cpmeas.IMAGE
                                else OI_OBJECTS)
            measurement = '_'.join((category, feature_nbr, image_name, size_scale))
            setting_values = [
                objects_or_image, object_name, measurement, display_image,
                "red", data_image, dpi_to_save, saved_image_contents]
            from_matlab = False
            variable_revision_number = 1
        
        return setting_values, variable_revision_number, from_matlab
        
