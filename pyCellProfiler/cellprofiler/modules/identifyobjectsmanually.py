'''<b>IdentifyObjectsManually</b> identifies objects in an image by selecting
them using the user interface<br>
<hr>
<br>
This module lets you outline the objects in an image using the mouse. The
user interface has several mouse tools:<br>
<ul><li><i>Outline:</i> Lets you draw an outline around an
object. Press the left mouse button at the start of the outline and draw
the outline around your object. The tool will close your outline when
you release the left mouse button.</li>
<li><i>Zoom in:</i> Lets you draw a rectangle and zoom the
display to within that rectangle.</li>
<li><i>Zoom out:</i> Reverses the effect of the last zoom-in.</li>
<li><i>Erase:</i> Erases an object if you click on it.</li></ul>
'''

__version__ = "$Revision$"

import numpy as np

import cellprofiler.cpmodule as cpm
import cellprofiler.objects as cpo
import cellprofiler.preferences as cpprefs
import identify as I
import cellprofiler.settings as cps
from cellprofiler.cpmath.outline import outline
from cellprofiler.cpmath.cpmorphology import draw_line
from cellprofiler.cpmath.cpmorphology import fill_labeled_holes

TOOL_OUTLINE = "Outline"
TOOL_ZOOM_IN = "Zoom in"
TOOL_ERASE = "Erase"

class IdentifyObjectsManually(I.Identify):
    
    category = "Object Processing"
    module_name = "IdentifyObjectsManually"
    variable_revision_number = 1
    
    def create_settings(self):
        self.image_name = cps.ImageNameSubscriber(
            "Select the input image", "None",
            doc = """Choose the name of the image to display in the object
            selection user interface. This can be an image from a file loaded
            by <b>LoadImages</b> or <b>LoadData</b> or a derived image
            from a module previous to this one.""")
        
        self.objects_name = cps.ObjectNameProvider(
            "Select the input objects", "Cells",
            doc = """Enter the name that you want to use for the objects
            that you outline using this module. You can use this name to
            refer to your objects in any subsequent module that uses objects
            as an input.""")
        
        self.wants_outlines = cps.Binary(
            "Save outlines of the identified objects?", False,
            doc = """Check this setting to save the outlines around the objects
            as a binary image.""")
        
        self.outlines_name = cps.ImageNameProvider(
            "Name the outlines", "CellOutlines",
            doc = """This is the name of the outlines image. You can refer to
            this image in any subsequent module that uses images. You can
            save the outlines image using the <b>SaveImages</b> module.""")
        
    def settings(self):
        '''The settings as saved in the pipeline'''
        return [ self.image_name, self.objects_name, self.wants_outlines,
                 self.outlines_name]
    
    def visible_settings(self):
        '''The settings as displayed in the UI'''
        result = [ self.image_name, self.objects_name, self.wants_outlines]
        if self.wants_outlines:
            result += [ self.outlines_name ]
        return result
    
    def prepare_to_create_batch(self, pipeline, image_set_list, fn_alter_path):
        '''This module cannot be used in a batch context'''
        raise ValueError("The IdentifyObjectsManually module cannot be run in batch mode")
    
    def is_interactive(self):
        return True
    
    def run(self, workspace):
        image_name    = self.image_name.value
        objects_name  = self.objects_name.value
        outlines_name = self.outlines_name.value
        image         = workspace.image_set.get_image(image_name)
        pixel_data    = image.pixel_data
        
        labels = np.zeros(pixel_data.shape[:2], int)
        self.do_ui(workspace, pixel_data, labels)
        objects = cpo.Objects()
        objects.segmented = labels
        workspace.object_set.add_objects(objects, objects_name)

        ##################
        #
        # Add measurements
        #
        m = workspace.measurements
        #
        # The object count
        #
        object_count = np.max(labels)
        I.add_object_count_measurements(m, objects_name, object_count)
        #
        # The object locations
        #
        I.add_object_location_measurements(m, objects_name, labels)
        #
        # Outlines if we want them
        #
        if self.wants_outlines:
            outlines_name = self.outlines_name.value
            outlines = outline(labels)
            outlines_image = cpi.Image(outlines.astype(bool))
            workspace.image_set.add(outlines_name, outlines_image)
        #
        # Do the drawing here
        #
        if workspace.frame is not None:
            figure = workspace.create_or_find_figure(subplots=(2,1))
            figure.subplot_imshow_labels(0, 0, labels, objects_name)
            figure.subplot_imshow(1, 0, self.draw_outlines(pixel_data, labels))

    def draw_outlines(self, pixel_data, labels):
        '''Draw a color image that shows the objects
        
        pixel_data - image, either b & w or color
        labels - labels for image
        
        returns - color image of same size as pixel_data
        '''
        from cellprofiler.gui.cpfigure import renumber_labels_for_display
        import matplotlib
        
        labels = renumber_labels_for_display(labels)
        outlines = outline(labels)
        
        if pixel_data.ndim == 3:
            image = pixel_data.copy()
        else:
            image = np.dstack([pixel_data]*3)
        #
        # make labeled pixels a grayscale times the label color
        #
        cm = matplotlib.cm.get_cmap(cpprefs.get_default_colormap())
        sm = matplotlib.cm.ScalarMappable(cmap = cm)
        labels_image = sm.to_rgba(labels)[:,:,:3]
        
        lmask = labels > 0
        gray = (image[lmask,0] + image[lmask,1] + image[lmask,2]) / 3
        
        for i in range(3):
            image[lmask,i] = gray * labels_image[lmask, i]
        #
        # Make the outline pixels a solid color
        #
        outlines_image = sm.to_rgba(outlines)[:,:,:3]
        image[outlines > 0,:] = outlines_image[outlines > 0,:]
        return image
        
    def do_ui(self, workspace, pixel_data, labels):
        '''Display a UI for editing'''
        import matplotlib
        from matplotlib.widgets import Lasso, RectangleSelector
        from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
        import wx
        
        style = wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER
        dialog_box = wx.Dialog(workspace.frame, -1,
                               "Identify objects manually",
                               style = style)
        sizer = wx.BoxSizer(wx.VERTICAL)
        dialog_box.SetSizer(sizer)
        sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
        sizer.Add(sub_sizer, 1, wx.EXPAND)
        figure = matplotlib.figure.Figure()
        axes = figure.add_subplot(1,1,1)
        panel = FigureCanvasWxAgg(dialog_box, -1, figure)
        sub_sizer.Add(panel, 1, wx.EXPAND)
        #
        # The controls are the radio buttons for tool selection and
        # a zoom out button
        #
        controls_sizer = wx.BoxSizer(wx.VERTICAL)
        sub_sizer.Add(controls_sizer, 0, 
                      wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_TOP | 
                      wx.EXPAND|wx.ALL, 10)
        
        tool_choice = wx.RadioBox(dialog_box, -1, "Active tool",
                                  style = wx.RA_VERTICAL,
                                  choices = [TOOL_OUTLINE, TOOL_ZOOM_IN, 
                                             TOOL_ERASE])
        tool_choice.SetSelection(0)
        controls_sizer.Add(tool_choice, 0, wx.ALIGN_LEFT | wx.ALIGN_TOP)
        zoom_out_button = wx.Button(dialog_box, -1, "Zoom out")
        zoom_out_button.Disable()
        controls_sizer.Add(zoom_out_button, 0, wx.ALIGN_CENTER_HORIZONTAL)
        
        zoom_stack = []
        
        ########################
        #
        # The drawing function
        #
        ########################
        def draw():
            '''Draw the current display'''
            assert isinstance(axes, matplotlib.axes.Axes)
            if len(axes.images) > 0:
                del axes.images[0]
            image = self.draw_outlines(pixel_data, labels)
            axes.imshow(image)
            if len(zoom_stack) > 0:
                axes.set_xlim(zoom_stack[-1][0][0], zoom_stack[-1][1][0])
                axes.set_ylim(zoom_stack[-1][0][1], zoom_stack[-1][1][1])
            else:
                axes.set_xlim(0, pixel_data.shape[1])
                axes.set_ylim(0, pixel_data.shape[0])
            figure.canvas.draw()
            panel.Refresh()
        
        ##################################
        #
        # The zoom-out button
        #
        ##################################
        def on_zoom_out(event):
            zoom_stack.pop()
            if len(zoom_stack) == 0:
                zoom_out_button.Disable()
            draw()
            
        dialog_box.Bind(wx.EVT_BUTTON, on_zoom_out, zoom_out_button)

        ##################################
        #
        # Zoom selector callback
        #
        ##################################
        
        def on_zoom_in(event_click, event_release):
            xmin = min(event_click.xdata, event_release.xdata)
            xmax = max(event_click.xdata, event_release.xdata)
            ymin = min(event_click.ydata, event_release.ydata)
            ymax = max(event_click.ydata, event_release.ydata)
            zoom_stack.append(((xmin, ymin), (xmax, ymax)))
            draw()
            zoom_out_button.Enable()
        
        zoom_selector = RectangleSelector(axes, on_zoom_in, drawtype='box',
                                          rectprops = dict(edgecolor='red', 
                                                           fill=False),
                                          useblit=True,
                                          minspanx=2, minspany=2,
                                          spancoords='data')
        zoom_selector.set_active(False)
        
        ##################################
        #
        # Lasso selector callback
        #
        ##################################
        
        current_lasso = []
        def on_lasso(vertices):
            mask = np.zeros(pixel_data.shape, int)
            new_label = np.max(labels) + 1
            for i in range(len(vertices)):
                v0 = (int(vertices[i][1]), int(vertices[i][0]))
                i_next = (i+1) % len(vertices)
                v1 = (int(vertices[i_next][1]), int(vertices[i_next][0]))
                draw_line(mask, v0, v1, new_label)
            mask = fill_labeled_holes(mask)
            labels[mask != 0] = new_label
            draw()
            
        ##################################
        #
        # Left mouse button down
        #
        ##################################
        
        def on_left_mouse_down(event):
            if figure.canvas.widgetlock.locked():
                return
            if event.inaxes != axes:
                return
            
            idx = tool_choice.GetSelection()
            tool = tool_choice.GetItemLabel(idx)
            if tool == TOOL_OUTLINE:
                lasso = Lasso(axes, (event.xdata, event.ydata), on_lasso)
                lasso.line.set_color('red')
                current_lasso.append(lasso)
                figure.canvas.widgetlock(lasso)
            elif tool == TOOL_ERASE:
                erase_label = labels[int(event.ydata), int(event.xdata)]
                if erase_label > 0:
                    labels[labels == erase_label] = 0
                    labels[labels > erase_label] -= 1
                draw()
            
        def on_left_mouse_up(event):
            if len(current_lasso) > 0:
                lasso = current_lasso.pop()
                figure.canvas.widgetlock.release(lasso)
                
        figure.canvas.mpl_connect('button_press_event', on_left_mouse_down)
        figure.canvas.mpl_connect('button_release_event', on_left_mouse_up)
        
        ######################################
        #
        # Radio box change
        #
        ######################################
        def on_radio(event):
            idx = tool_choice.GetSelection()
            tool = tool_choice.GetItemLabel(idx)
            if tool == TOOL_ZOOM_IN:
                zoom_selector.set_active(True)
        
        tool_choice.Bind(wx.EVT_RADIOBOX, on_radio)
        
        button_sizer = wx.StdDialogButtonSizer()
        sizer.Add(button_sizer, 0, wx.ALIGN_RIGHT | wx.EXPAND | wx.ALL, 10)
        button_sizer.AddButton(wx.Button(dialog_box, wx.ID_OK))
        button_sizer.Realize()
        draw()
        dialog_box.Fit()
        dialog_box.ShowModal()
        
    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        if from_matlab and variable_revision_number == 2:
            image_name, object_name, max_resolution, save_outlines = setting_values
            wants_outlines = \
                    (cps.YES if save_outlines.lower() == cps.DO_NOT_USE.lower() 
                     else cps.NO)
            setting_values = [ image_name, object_name, wants_outlines,
                               save_outlines]
            variable_revision_number = 1
            from_matlab = False
        return setting_values, variable_revision_number, from_matlab
            
