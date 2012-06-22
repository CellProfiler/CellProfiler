'''<b>Edit Objects Manually</b> allows you to remove specific objects
from each image by pointing and clicking
<hr>

This module allows you to remove specific objects via a user interface 
where you point and click to select objects for removal. The
module displays three images: the objects as originally identified,
the objects that have not been removed, and the objects that have been
removed.

If you click on an object in the "not removed" image, it moves to the
"removed" image and will be removed. If you click on an object in the
"removed" image, it moves to the "not removed" image and will not be
removed. Clicking on an object in the original image 
toggles its "removed" state.

The pipeline pauses once per processed image when it reaches this module.
You must press the <i>Continue</i> button to accept the selected objects
and continue the pipeline.

<h4>Available measurements</h4>
<i>Image features:</i>
<ul>
<li><i>Count:</i> The number of edited objects in the image.</li>
</ul>
<i>Object features:</i>
<ul>
<li><i>Location_X, Location_Y:</i> The pixel (X,Y) coordinates of the center of mass of the edited objects.</li>
</ul>

See also <b>FilterObjects</b>, <b>MaskObject</b>, <b>OverlayOutlines</b>, <b>ConvertToImage</b>.
'''

__version__="$Revision$"

import numpy as np

import cellprofiler.preferences as cpprefs
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps
import cellprofiler.workspace as cpw
from cellprofiler.cpmath.outline import outline

import identify as I

###########################################
#
# Choices for the "do you want to renumber your objects" setting
#
###########################################
R_RENUMBER = "Renumber"
R_RETAIN = "Retain"

class EditObjectsManually(I.Identify):
    category = "Object Processing"
    variable_revision_number = 2
    module_name = 'EditObjectsManually'
    
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
        self.object_name = cps.ObjectNameSubscriber("Select the objects to be edited", "None",
                                                    doc="""
            Choose a set of previously identified objects
            for editing, such as those produced by one of the
            <b>Identify</b> modules.""")
        
        self.filtered_objects = cps.ObjectNameProvider(
            "Name the edited objects","EditedObjects",
            doc="""What do you want to call the objects that remain
            after editing? These objects will be available for use by
            subsequent modules.""")
        
        self.wants_outlines = cps.Binary(
            "Retain outlines of the edited objects?", False,
            doc="""Check this box if you want to keep images of the outlines
            of the objects that remain after editing. This image
            can be saved by downstream modules or overlayed on other images
            using the <b>OverlayOutlines</b> module.""")
        
        self.outlines_name = cps.OutlineNameProvider(
            "Name the outline image", "EditedObjectOutlines",
            doc="""<i>(Used only if you have selected to retain outlines of edited objects)</i><br>
            What do you want to call the outline image?""")
        
        self.renumber_choice = cps.Choice(
            "Numbering of the edited objects",
            [R_RENUMBER, R_RETAIN],
            doc="""Choose how to number the objects that 
            remain after editing, which controls how edited objects are associated with their predecessors:
            <p>
            If you choose <i>Renumber</i>,
            this module will number the objects that remain 
            using consecutive numbers. This
            is a good choice if you do not plan to use measurements from the
            original objects and you only want to use the edited objects in downstream modules; the
            objects that remain after editing will not have gaps in numbering
            where removed objects are missing.
            <p>
            If you choose <i>Retain</i>,
            this module will retain each object's original number so that the edited object's number matches its original number. This allows any measurements you make from 
            the edited objects to be directly aligned with measurements you might 
            have made of the original, unedited objects (or objects directly 
            associated with them).""")
        
        self.wants_image_display = cps.Binary(
            "Display a guiding image?", True,
            doc = """Check this setting to display an image and outlines
            of the objects. Leave the setting unchecked if you do not
            want a guide image while editing""")
        
        self.image_name = cps.ImageNameSubscriber(
            "Select the guiding image", "None",
            doc = """
            <i>(Used only if a guiding image is desired)</i><br>
            This is the image that will appear when editing objects.
            Choose an image supplied by a previous module.""")
    
    def settings(self):
        """Return the settings to be loaded or saved to/from the pipeline
        
        These are the settings (from cellprofiler.settings) that are
        either read from the strings in the pipeline or written out
        to the pipeline. The settings should appear in a consistent
        order so they can be matched to the strings in the pipeline.
        """
        return [self.object_name, self.filtered_objects, self.wants_outlines,
                self.outlines_name, self.renumber_choice, 
                self.wants_image_display, self.image_name]
    
    def visible_settings(self):
        """The settings that are visible in the UI
        """
        #
        # Only display the outlines_name if wants_outlines is true
        #
        result = [self.object_name, self.filtered_objects, self.wants_outlines]
        if self.wants_outlines:
            result.append(self.outlines_name)
        result += [ self.renumber_choice, self.wants_image_display]
        if self.wants_image_display:
            result += [self.image_name]
        return result
    
    def run(self, workspace):
        """Run the module
        
        workspace    - The workspace contains
            pipeline     - instance of cpp for this run
            image_set    - the images in the image set being processed
            object_set   - the objects (labeled masks) in this image set
            measurements - the measurements for this run
            frame        - the parent frame to whatever frame is created. None means don't draw.
        """
        orig_objects_name = self.object_name.value
        filtered_objects_name = self.filtered_objects.value
        
        orig_objects = workspace.object_set.get_objects(orig_objects_name)
        assert isinstance(orig_objects, cpo.Objects)
        orig_labels = orig_objects.segmented
        mask = orig_labels != 0

        try:
            if self.wants_image_display.value:
                guide_image = workspace.image_set.get_image(self.image_name.value).pixel_data
            else:
                guide_image = None

            filtered_labels = workspace.interaction_request(self, orig_labels, guide_image=guide_image)
        except workspace.NoInteractionException:
            # Accept the labels as-is
            filtered_labels = orig_labels

        #
        # Renumber objects consecutively if asked to do so
        #
        unique_labels = np.unique(filtered_labels)
        unique_labels = unique_labels[unique_labels != 0]
        object_count = len(unique_labels)
        if self.renumber_choice == R_RENUMBER:
            mapping = np.zeros(1 if len(unique_labels) == 0 else np.max(unique_labels)+1, int)
            mapping[unique_labels] = np.arange(1,object_count + 1)
            filtered_labels = mapping[filtered_labels]
        #
        # Make the objects out of the labels
        #
        filtered_objects = cpo.Objects()
        filtered_objects.segmented = filtered_labels
        filtered_objects.unedited_segmented = orig_objects.unedited_segmented
        filtered_objects.parent_image = orig_objects.parent_image
        workspace.object_set.add_objects(filtered_objects, 
                                         filtered_objects_name)
        #
        # Add parent/child & other measurements
        #
        m = workspace.measurements
        child_count, parents = orig_objects.relate_children(filtered_objects)
        m.add_measurement(filtered_objects_name,
                          I.FF_PARENT%(orig_objects_name),
                          parents)
        m.add_measurement(orig_objects_name,
                          I.FF_CHILDREN_COUNT%(filtered_objects_name),
                          child_count)
        #
        # The object count
        #
        I.add_object_count_measurements(m, filtered_objects_name,
                                        object_count)
        #
        # The object locations
        #
        I.add_object_location_measurements(m, filtered_objects_name,
                                           filtered_labels)
        #
        # Outlines if we want them
        #
        if self.wants_outlines:
            outlines_name = self.outlines_name.value
            outlines = outline(filtered_labels)
            outlines_image = cpi.Image(outlines.astype(bool))
            workspace.image_set.add(outlines_name, outlines_image)

        workspace.display_data.orig_labels = orig_labels
        workspace.display_data.filtered_labels = filtered_labels

    def display(self, workspace, figure):
        orig_objects_name = self.object_name.value
        filtered_objects_name = self.filtered_objects.value
        orig_labels = workspace.display_data.orig_labels
        filtered_labels = workspace.display_data.filtered_labels
        figure.set_subplots((2, 1))
        figure.subplot_imshow_labels(0, 0, orig_labels, orig_objects_name)
        figure.subplot_imshow_labels(1, 0, filtered_labels,
                                     filtered_objects_name,
                                     sharex = figure.subplot(0,0),
                                     sharey = figure.subplot(0,0))

    def handle_interaction(self, orig_labels, guide_image):
        import wx
        import matplotlib
        from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
        from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg
        from cellprofiler.gui.cpfigure import renumber_labels_for_display
        
        orig_objects_name = self.object_name.value
        #
        # Get the labels matrix and make a mask of objects to keep from it
        #
        #
        # Display a UI for choosing objects
        #
        style = wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER
        dialog_box = wx.Dialog(wx.GetApp().TopWindow, -1,
                               "Choose objects to keep",
                               style=style)
        sizer = wx.BoxSizer(wx.VERTICAL)
        dialog_box.SetSizer(sizer)
        figure = matplotlib.figure.Figure()
        panel = FigureCanvasWxAgg(dialog_box, -1, figure)
        sizer.Add(panel, 1, wx.EXPAND)
        toolbar = NavigationToolbar2WxAgg(panel)
        sizer.Add(toolbar, 0, wx.EXPAND)
        mask = orig_labels != 0
        #
        # Make 3 axes
        #
        orig_axes = figure.add_subplot(2, 2, 1)
        orig_axes._adjustable = 'box-forced'
        keep_axes = figure.add_subplot(2, 2, 2,
                                       sharex = orig_axes,
                                       sharey = orig_axes)
        remove_axes = figure.add_subplot(2, 2, 4,
                                         sharex = orig_axes,
                                         sharey = orig_axes)
        for axes in (orig_axes, keep_axes, remove_axes):
            axes._adjustable = 'box-forced'
            
        info_axes = figure.add_subplot(2, 2, 3)
        assert isinstance(info_axes, matplotlib.axes.Axes)
        info_axes.set_axis_off()
        #
        # Add an explanation and possibly a checkbox to the info axis
        #
        ui_text = ("Keep or remove objects by clicking\n"
                   "on them with the mouse.\n"
                   'Press the "Done" button when\nediting is complete.')
        info_axes.text(0,0, ui_text, size="small")
        wants_image_display = [self.wants_image_display.value]
        sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
        #
        # Need padding on top because tool bar is wonky about its height
        #
        sizer.Add(sub_sizer, 0, wx.EXPAND | wx.TOP, 10)
        
        resume_id = 100
        cancel_id = 101
        keep_all_id = 102
        remove_all_id = 103
        reverse_select = 104
        #########################################
        #
        # Buttons for keep / remove / toggle
        #
        #########################################
        
        keep_button = wx.Button(dialog_box, keep_all_id, "Keep all")
        sub_sizer.Add(keep_button, 0, wx.ALIGN_CENTER)
        
        remove_button = wx.Button(dialog_box, remove_all_id, "Remove all")
        sub_sizer.Add(remove_button,0, wx.ALIGN_CENTER)
        
        toggle_button = wx.Button(dialog_box, reverse_select, "Reverse selection")
        sub_sizer.Add(toggle_button,0, wx.ALIGN_CENTER)
        
        ######################################
        #
        # Buttons for resume and cancel
        #
        ######################################
        button_sizer = wx.StdDialogButtonSizer()
        resume_button = wx.Button(dialog_box, resume_id, "Done")
        button_sizer.AddButton(resume_button)
        sub_sizer.Add(button_sizer, 0, wx.ALIGN_CENTER)
        def on_resume(event):
            dialog_box.EndModal(wx.OK)
        dialog_box.Bind(wx.EVT_BUTTON, on_resume, resume_button)
        button_sizer.SetAffirmativeButton(resume_button)
        
        cancel_button = wx.Button(dialog_box, cancel_id, "Cancel")
        button_sizer.AddButton(cancel_button)
        def on_cancel(event):
            dialog_box.EndModal(wx.CANCEL)
        dialog_box.Bind(wx.EVT_BUTTON, on_cancel, cancel_button)
        button_sizer.SetNegativeButton(cancel_button)
        button_sizer.Realize()
        
        ################### d i s p l a y #######
        #
        # The following is a function that we can call to refresh
        # the figure's appearance based on the mask and the original labels
        #
        ##########################################
        cm = matplotlib.cm.get_cmap(cpprefs.get_default_colormap())
        cm.set_bad((0,0,0))
        
        def display():
            if len(orig_axes.images) > 0:
                # Save zoom and scale if coming through here a second time
                x0, x1 = orig_axes.get_xlim()
                y0, y1 = orig_axes.get_ylim()
                set_lim = True
            else:
                set_lim = False
            for axes, labels, title in (
                (orig_axes, orig_labels, "Original: %s"%orig_objects_name),
                (keep_axes, orig_labels * mask,"Objects to keep"),
                (remove_axes, orig_labels * (~ mask), "Objects to remove")):
                
                assert isinstance(axes, matplotlib.axes.Axes)
                labels = renumber_labels_for_display(labels)
                axes.clear()
                if np.all(labels == 0):
                    use_cm = matplotlib.cm.gray
                    is_blank = True
                else:
                    use_cm = cm
                    is_blank = False
                if wants_image_display[0]:
                    outlines = outline(labels)
                    image = guide_image.astype(np.float)
                    image, _ = cpo.size_similarly(labels, image)
                    if image.ndim == 2:
                        image = np.dstack((image, image, image))
                    if not is_blank:
                        mappable = matplotlib.cm.ScalarMappable(cmap=use_cm)
                        mappable.set_clim(1,labels.max())
                        limage = mappable.to_rgba(labels)[:,:,:3]
                        image[outlines != 0,:] = limage[outlines != 0, :]
                    axes.imshow(image)
                    
                else:
                    axes.imshow(labels, cmap = use_cm)
                axes.set_title(title,
                               fontname=cpprefs.get_title_font_name(),
                               fontsize=cpprefs.get_title_font_size())
            if set_lim:
                orig_axes.set_xlim((x0, x1))
                orig_axes.set_ylim((y0, y1))
            figure.canvas.draw()
            panel.Refresh()
                
        if self.wants_image_display:
            display_image_checkbox = matplotlib.widgets.CheckButtons(
                info_axes, ["Display image"], [True])
            display_image_checkbox.labels[0].set_size("small")
            r = display_image_checkbox.rectangles[0]
            rwidth = r.get_width()
            rheight = r.get_height()
            rx, ry = r.get_xy()
            new_rwidth = rwidth / 2 
            new_rheight = rheight / 2
            new_rx = rx + rwidth/2
            new_ry = ry + rheight/4
            r.set_width(new_rwidth)
            r.set_height(new_rheight)
            r.set_xy((new_rx, new_ry))
            l1, l2 = display_image_checkbox.lines[0]
            l1.set_data((np.array((new_rx, new_rx+new_rwidth)),
                         np.array((new_ry, new_ry+new_rheight))))
            l2.set_data((np.array((new_rx, new_rx+new_rwidth)),
                         np.array((new_ry + new_rheight, new_ry))))
            
            def on_display_image_clicked(_):
                wants_image_display[0] = not wants_image_display[0]
                display()
            display_image_checkbox.on_clicked(on_display_image_clicked)
            
        def on_click(event):
            if event.inaxes not in (orig_axes, keep_axes, remove_axes):
                return
            x = int(event.xdata)
            y = int(event.ydata)
            if (x < 0 or x >= orig_labels.shape[1] or
                y < 0 or y >= orig_labels.shape[0]):
                return
            lnum = orig_labels[y,x]
            if lnum == 0:
                return
            if event.inaxes == orig_axes:
                mask[orig_labels == lnum] = ~mask[orig_labels == lnum]
            elif event.inaxes == keep_axes:
                mask[orig_labels == lnum] = False
            else:
                mask[orig_labels == lnum] = True
            display()
            
        figure.canvas.mpl_connect('button_press_event', on_click)
        
        ################################
        #
        # Functions for keep / remove/ toggle
        #
        ################################

        def on_keep(event):
            mask[:,:] = (orig_labels != 0)
            display()
        dialog_box.Bind(wx.EVT_BUTTON, on_keep, keep_button)
        
        def on_remove(event):
            mask[:,:] = 0
            display()
        dialog_box.Bind(wx.EVT_BUTTON, on_remove, remove_button)
        
        def on_toggle(event):
            mask[orig_labels != 0] = ~mask[orig_labels != 0]
            display()
        dialog_box.Bind(wx.EVT_BUTTON, on_toggle, toggle_button)
        
        display()
        dialog_box.Fit()
        result = dialog_box.ShowModal()
        dialog_box.Destroy()
        if result != wx.OK:
            raise RuntimeError("User cancelled EditObjectsManually")
        filtered_labels = orig_labels.copy()
        filtered_labels[~mask] = 0
        return filtered_labels
    
    def get_measurement_columns(self, pipeline):
        '''Return information to use when creating database columns'''
        orig_image_name = self.object_name.value
        filtered_image_name = self.filtered_objects.value
        columns = I.get_object_measurement_columns(filtered_image_name)
        columns += [(orig_image_name,
                     I.FF_CHILDREN_COUNT % filtered_image_name,
                     cpmeas.COLTYPE_INTEGER),
                    (filtered_image_name,
                     I.FF_PARENT %  orig_image_name,
                     cpmeas.COLTYPE_INTEGER)]
        return columns
    
    def get_object_dictionary(self):
        '''Return the dictionary that's used by identify.get_object_*'''
        return { self.filtered_objects.value: [ self.object_name.value ] }
    
    def get_categories(self, pipeline, object_name):
        '''Get the measurement categories produced by this module
        
        pipeline - pipeline being run
        object_name - fetch categories for this object
        '''
        categories = self.get_object_categories(pipeline, object_name,
                                                self.get_object_dictionary())
        return categories
    
    def get_measurements(self, pipeline, object_name, category):
        '''Get the measurement features produced by this module
      
        pipeline - pipeline being run
        object_name - fetch features for this object
        category - fetch features for this category
        '''
        measurements = self.get_object_measurements(
            pipeline, object_name, category, self.get_object_dictionary())
        return measurements
    
    def upgrade_settings(self, setting_values, variable_revision_number,
                         module_name, from_matlab):
        '''Upgrade the settings written by a prior version of this module
        
        setting_values - array of string values for the module's settings
        variable_revision_number - revision number of module at time of saving
        module_name - name of module that saved settings
        from_matlab - was a pipeline saved by CP 1.0
        
        returns upgraded settings, new variable revision number and matlab flag
        '''
        if from_matlab and variable_revision_number == 2:
            object_name, filtered_object_name, outlines_name, \
            renumber_or_retain = setting_values
            
            if renumber_or_retain == "Renumber":
                renumber_or_retain = R_RENUMBER
            else:
                renumber_or_retain = R_RETAIN
            
            if outlines_name == cps.DO_NOT_USE:
                wants_outlines = cps.NO
            else:
                wants_outlines = cps.YES
            
            setting_values = [object_name, filtered_object_name,
                              wants_outlines, outlines_name, renumber_or_retain]
            variable_revision_number = 1
            from_matlab = False
            module_name = self.module_name
            
        if (not from_matlab) and variable_revision_number == 1:
            # Added wants image + image
            setting_values = setting_values + [ cps.NO, "None"]
            variable_revision_number = 2
        
        return setting_values, variable_revision_number, from_matlab
