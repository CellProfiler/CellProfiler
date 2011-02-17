'''<b>Display Data On Image</b> 
produces an image with measured data on top of identified objects
<hr>

This module displays either a single image measurement on an image of
your choosing, or one object measurement per object on top
of every object in an image. The display itself is an image which you
can save to a file using <b>SaveImages</b>.

<i>Note:</i> The module ShowDataOnImage and data tool DisplayDataOnImage from 
CellProfiler 1.0 were merged into DisplayDataOnImage in the CellProfiler 2.0.
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

__version__="$Revision$"

import numpy as np

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.cpimage as cpi
import cellprofiler.objects as cpo
import cellprofiler.workspace as cpw
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
    category = 'Data Tools'
    variable_revision_number = 2
    
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
            "Display object or image measurements?",
            [OI_OBJECTS, OI_IMAGE],
            doc = """<ul><li> <i>Image</i> displays a single measurement made
            on an image.</li> <li><i>Object</i> displays measurements made on
            objects.</li></ul>""")
        
        self.objects_name = cps.ObjectNameSubscriber(
            "Select the input objects", "None",
            doc = """<i>(Used only when displaying object measurements)</i><br>Choose the name of objects identified by some previous
            module (such as <b>IdentifyPrimaryObjects</b> or
            <b>IdentifySecondaryObjects</b>).""")
        
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
            There are several different ways by which you can specify the color:<br>
            <ul><li><i>Single letter.</i> "b"=blue, "g"=green, "r"=red, "c"=cyan, "m"=magenta,
            "y"=yellow, "k"=black, "w"=white</li>
            <li><i>Name.</i> You can use any name supported by HTML; a list of colors is shown on this:
            <a href="http://www.w3schools.com/html/html_colors.asp">page</a>.
            </li>
            <li><i>RGB code.</i> You can specify the color as a combination of
            the red, green, and blue intensities, for instance, "#FFFF00"
            for yellow; yellow = red("FF") + green("FF") + blue("00"), where <i>FF</i> is
            hexadecimal for 255, the highest intensity. See 
            <a href="http://www.w3schools.com/html/html_colors.asp">here</a> for a more detailed
            explanation</li></ul>""")
        
        self.display_image = cps.ImageNameProvider(
            "Name the output image that has the measurements displayed","DisplayImage",
            doc="""The name that will be given to the image with
            the measurements superimposed. You can use this name to refer to the image in
            subsequent modules (such as <b>SaveImages</b>).""")
        
        self.font_size = cps.Integer(
            "Font size (points)", 10, minval=1)
        
        self.decimals = cps.Integer(
            "Number of decimals", 2, minval=0)
        
        self.saved_image_contents = cps.Choice(
            "Image elements to save",
            [E_IMAGE, E_FIGURE, E_AXES],
            doc="""This setting controls the level of annotation on the image:
            <ul><li>
            <i>Image</i>: Saves the image with
            the overlaid measurement annotations.</li>
            <li><i>Axes</i>:
            Adds axes with tick marks and image coordinates.</li>
            <li><i>Figure</i>: Adds a title and other
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
                self.font_size, self.decimals, self.saved_image_contents]
    
    def visible_settings(self):
        """The settings that are visible in the UI
        """
        result = [self.objects_or_image]
        if self.objects_or_image == OI_OBJECTS:
            result += [self.objects_name]
        result += [self.measurement, self.image_name, self.text_color,
                   self.display_image, self.font_size, self.decimals,
                   self.saved_image_contents]
        return result
        
    def is_interactive(self):
        return False
    
    def run(self, workspace):
        import matplotlib
        import matplotlib.cm
        import matplotlib.backends.backend_wxagg
        import matplotlib.transforms
        from cellprofiler.gui.cpfigure import figure_to_image, only_display_image
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
        fig = matplotlib.figure.Figure()
        axes = fig.add_subplot(1,1,1)
        def imshow_fn(pixel_data):
            # Note: requires typecast to avoid failure during
            #       figure_to_image (IMG-764)
            img = pixel_data * 255
            img[img < 0] = 0
            img[img > 255] = 255
            img = img.astype(np.uint8)
            axes.imshow(img, cmap = matplotlib.cm.Greys_r)
        self.display_on_figure(workspace, axes, imshow_fn)

        canvas = matplotlib.backends.backend_wxagg.FigureCanvasAgg(fig)
        if self.saved_image_contents == E_AXES:
            fig.set_frameon(False)
            fig.subplots_adjust(0.1,.1,.9,.9,0,0)
            shape = workspace.display_data.pixel_data.shape
            width = float(shape[1]) / fig.dpi
            height = float(shape[0]) / fig.dpi
            fig.set_figheight(height)
            fig.set_figwidth(width)
        elif self.saved_image_contents == E_IMAGE:
            only_display_image(fig, workspace.display_data.pixel_data.shape)
        else:
            fig.subplots_adjust(.1,.1,.9,.9,0,0)
            
        pixel_data = figure_to_image(fig, dpi=fig.dpi)
        image = cpi.Image(pixel_data)
        workspace.image_set.add(self.display_image.value, image)
        
    def run_as_data_tool(self, workspace):
        # Note: workspace.measurements.image_set_number contains the image
        #    number that should be displayed.
        import loadimages as LI
        import os.path
        im_id = self.image_name.value
        image_features = workspace.measurements.get_feature_names(cpmeas.IMAGE)
        
        try:
            filecol = [x for x in image_features if x.startswith(LI.C_FILE_NAME) 
                        and x.endswith(im_id)][0]
            pathcol = [x for x in image_features if x.startswith(LI.C_PATH_NAME) 
                        and x.endswith(im_id)][0]
        except:
            raise Exception('DisplayDataOnImage failed to find your image path and filename features in the supplied measurements.')
        
        index = workspace.measurements.image_set_index
        filename = workspace.measurements.get_measurement(cpmeas.IMAGE, filecol, index)
        pathname = workspace.measurements.get_measurement(cpmeas.IMAGE, pathcol, index)
        
        # Add the image to the workspace ImageSetList
        image_set_list = workspace.image_set_list
        image_set = image_set_list.get_image_set(0)
        ip = LI.LoadImagesImageProvider(im_id, pathname, filename)
        image_set.providers.append(ip)
        
        self.run(workspace)
        self.display(workspace)
        
    def display(self, workspace):
        fig = workspace.create_or_find_figure(title="DisplayDataOnImage, image cycle #%d"%(
                workspace.measurements.image_set_number),
                                              subplots=(1,1))
        fig.clf()
        title = "%s_%s" % (self.objects_name.value, self.measurement.value)
        def imshow_fn(pixel_data):
            if pixel_data.ndim == 3:
                fig.subplot_imshow_color(0, 0, pixel_data, title=title,
                                         use_imshow = True)
            else:
                fig.subplot_imshow_grayscale(0, 0, pixel_data, title=title,
                                             use_imshow = True)

        self.display_on_figure(workspace, fig.subplot(0,0), imshow_fn)
        fig.figure.canvas.draw_idle()
        
    def display_on_figure(self, workspace, axes, imshow_fn):
        import matplotlib
        imshow_fn(workspace.display_data.pixel_data)
        for x, y, value in zip(workspace.display_data.x,
                               workspace.display_data.y,
                               workspace.display_data.values):
            try:
                fvalue = float(value)
                svalue = "%.*f"%(self.decimals.value, value)
            except:
                svalue = str(value)
            
            text = matplotlib.text.Text(x=x, y=y, text=svalue,
                                        size=self.font_size.value,
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
        if variable_revision_number == 1:
            objects_or_image, objects_name, measurement, \
                image_name, text_color, display_image, \
                dpi, saved_image_contents = setting_values
            setting_values = [objects_or_image, objects_name, measurement,
                              image_name, text_color, display_image,
                              10, 2, saved_image_contents]
            variable_revision_number = 2
        return setting_values, variable_revision_number, from_matlab
        

    
if __name__ == "__main__":
    ''' For debugging purposes only...
    '''
    import wx
    from cellprofiler.gui.datatoolframe import DataToolFrame
    app = wx.PySimpleApp()

    tool_name = 'DisplayDataOnImage'
    dlg = wx.FileDialog(None, "Choose data output file for %s data tool" %
                        tool_name, wildcard="*.mat",
                        style=(wx.FD_OPEN | wx.FILE_MUST_EXIST))
    if dlg.ShowModal() == wx.ID_OK:
        data_tool_frame = DataToolFrame(None, module_name=tool_name, measurements_file_name = dlg.Path)
    data_tool_frame.Show()
    
    app.MainLoop()
