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
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2012 Broad Institute
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org
#
# Some matplotlib interactive editing code is derived from the sample:
#
# http://matplotlib.sourceforge.net/examples/event_handling/poly_editor.html
#
# Copyright 2008, John Hunter, Darren Dale, Michael Droettboom
# 


import logging
logger = logging.getLogger(__name__)

import os
import numpy as np
import sys

import cellprofiler.preferences as cpprefs
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps
import cellprofiler.workspace as cpw
from cellprofiler.cpmath.outline import outline
from cellprofiler.cpmath.cpmorphology import triangle_areas

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
    variable_revision_number = 3
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
        
        self.allow_overlap = cps.Binary(
            "Allow overlapping objects", False,
            doc = """<b>EditObjectsManually</b> can allow you to edit an
            object so that it overlaps another or it can prevent you from
            overlapping one object with another. Objects such as worms or
            the neurites of neurons may cross each other and might need to
            be edited with overlapping allowed, whereas a monolayer of cells
            might be best edited with overlapping off. Check this setting to
            allow overlaps or uncheck it to prevent them.""") 
        
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
                self.wants_image_display, self.image_name, self.allow_overlap]
    
    def visible_settings(self):
        """The settings that are visible in the UI
        """
        #
        # Only display the outlines_name if wants_outlines is true
        #
        result = [self.object_name, self.filtered_objects, 
                  self.allow_overlap, self.wants_outlines]
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
        orig_labels = [l for l, c in orig_objects.get_labels()]

        try:
            if self.wants_image_display:
                guide_image = workspace.image_set.get_image(self.image_name.value)
                guide_image = guide_image.pixel_data
                if np.any(guide_image != np.min(guide_image)):
                    guide_image = (guide_image - np.min(guide_image)) / (np.max(guide_image) - np.min(guide_image))
            else:
                guide_image = None
            filtered_labels = workspace.interaction_request(
                self, orig_labels, guide_image)
        except workspace.NoInteractionException:
            # Accept the labels as-is
            filtered_labels = orig_labels

        #
        # Renumber objects consecutively if asked to do so
        #
        unique_labels = np.unique(np.array(filtered_labels))
        unique_labels = unique_labels[unique_labels != 0]
        object_count = len(unique_labels)
        if self.renumber_choice == R_RENUMBER:
            mapping = np.zeros(1 if len(unique_labels) == 0 else np.max(unique_labels)+1, int)
            mapping[unique_labels] = np.arange(1,object_count + 1)
            filtered_labels = [mapping[l] for l in filtered_labels]
        #
        # Make the objects out of the labels
        #
        filtered_objects = cpo.Objects()
        i, j = np.mgrid[0:filtered_labels[0].shape[0],
                        0:filtered_labels[0].shape[1]]
        ijv = np.zeros((0, 3), filtered_labels[0].dtype)
        for l in filtered_labels:
            ijv = np.vstack((ijv,
                             np.column_stack((i[l != 0],
                                              j[l != 0],
                                              l[l != 0]))))
        filtered_objects.set_ijv(ijv, orig_labels[0].shape)
        if orig_objects.has_unedited_segmented():
            filtered_objects.unedited_segmented = orig_objects.unedited_segmented
        if orig_objects.parent_image is not None:
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
        I.add_object_location_measurements_ijv(m, filtered_objects_name, ijv)
        #
        # Outlines if we want them
        #
        if self.wants_outlines:
            outlines_name = self.outlines_name.value
            outlines = np.zeros((filtered_labels[0].shape[0], 
                                 filtered_labels[0].shape[1]), bool)
            outlines_image = cpi.Image(outlines)
            workspace.image_set.add(outlines_name, outlines_image)

        workspace.display_data.orig_ijv = orig_objects.ijv
        workspace.display_data.filtered_ijv = filtered_objects.ijv
        workspace.display_data.shape = orig_labels[0].shape

    def display(self, workspace, figure):
        orig_ijv = workspace.display_data.orig_ijv
        filtered_ijv = workspace.display_data.filtered_ijv
        shape = workspace.display_data.shape
        figure = workspace.create_or_find_figure(
            title="EditObjectsManually, image cycle #%d"%(
                workspace.measurements.image_set_number),
            subplots=(2,1))
        figure.subplot_imshow_ijv(0, 0, orig_ijv,
                                  shape = shape,
                                  title = self.object_name.value)
        figure.subplot_imshow_ijv(1, 0, filtered_ijv,
                                  shape = shape,
                                  title = self.filtered_objects.value,
                                  sharex = figure.subplot(0,0),
                                  sharey = figure.subplot(0,0))
    
    def run_as_data_tool(self):
        from cellprofiler.gui.editobjectsdlg import EditObjectsDialog
        import wx
        from wx.lib.filebrowsebutton import FileBrowseButton
        from subimager.client import get_image, post_image
        import subimager.omexml as ome
        from cellprofiler.modules.namesandtypes import ObjectsImageProvider
        from cellprofiler.modules.loadimages import pathname2url
        
        with wx.Dialog(None) as dlg:
            dlg.Title = "Choose files for editing"
            dlg.Sizer = wx.BoxSizer(wx.VERTICAL)
            box = wx.StaticBox(dlg, -1, "Choose or create new objects file")
            sub_sizer = wx.StaticBoxSizer(box, wx.HORIZONTAL)
            dlg.Sizer.Add(sub_sizer, 0, wx.EXPAND | wx.ALL, 5)
            new_or_existing_rb = wx.RadioBox(dlg, style=wx.RA_VERTICAL,
                                             choices = ("New", "Existing"))
            sub_sizer.Add(new_or_existing_rb, 0, wx.EXPAND)
            objects_file_fbb = FileBrowseButton(
                dlg, size=(300, -1),
                fileMask="Objects file (*.tif, *.tiff, *.png, *.bmp, *.jpg)|*.tif;*.tiff;*.png;*.bmp;*.jpg",
                dialogTitle="Select objects file",
                labelText="Objects file:")
            objects_file_fbb.Enable(False)
            sub_sizer.AddSpacer(5)
            sub_sizer.Add(objects_file_fbb, 0, wx.ALIGN_TOP | wx.ALIGN_RIGHT)
            def on_radiobox(event):
                objects_file_fbb.Enable(new_or_existing_rb.GetSelection() == 1)
            new_or_existing_rb.Bind(wx.EVT_RADIOBOX, on_radiobox)
            
            image_file_fbb = FileBrowseButton(
                dlg, size=(300, -1),
                fileMask="Objects file (*.tif, *.tiff, *.png, *.bmp, *.jpg)|*.tif;*.tiff;*.png;*.bmp;*.jpg",
                dialogTitle="Select guide image file",
                labelText="Guide image:")
            dlg.Sizer.Add(image_file_fbb, 0, wx.EXPAND | wx.ALL, 5)
            
            allow_overlap_checkbox = wx.CheckBox(dlg, -1, "Allow objects to overlap")
            allow_overlap_checkbox.Value = True
            dlg.Sizer.Add(allow_overlap_checkbox, 0, wx.EXPAND | wx.ALL, 5)
            
            buttons = wx.StdDialogButtonSizer()
            dlg.Sizer.Add(buttons, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT | wx.ALL, 5)
            buttons.Add(wx.Button(dlg, wx.ID_OK))
            buttons.Add(wx.Button(dlg, wx.ID_CANCEL))
            buttons.Realize()
            dlg.Fit()
            result = dlg.ShowModal()
            if result != wx.ID_OK:
                return
            self.allow_overlap.value = allow_overlap_checkbox.Value
            fullname = objects_file_fbb.GetValue()
            guidename = image_file_fbb.GetValue()

        if new_or_existing_rb.GetSelection() == 1:
            provider = ObjectsImageProvider(
                "InputObjects",
                pathname2url(fullname),
                None, None)
            image = provider.provide_image(None)
            pixel_data = image.pixel_data
            shape = pixel_data.shape[:2]
            labels = [pixel_data[:, :, i] for i in range(pixel_data.shape[2])]
        else:
            labels = None
        #
        # Load the guide image
        #
        guide_image = get_image(pathname2url(guidename))
        if np.min(guide_image) != np.max(guide_image):
            guide_image = ((guide_image - np.min(guide_image)) / 
                           (np.max(guide_image)  - np.min(guide_image)))
        if labels is None:
            shape = guide_image.shape[:2]
            labels = [np.zeros(shape, int)]
        with EditObjectsDialog(
            guide_image, labels,
            self.allow_overlap, self.object_name.value) as dialog_box:
            result = dialog_box.ShowModal()
            if result != wx.OK:
                return
            labels = dialog_box.labels
        n_frames = len(labels)
        with wx.FileDialog(None,
                           style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT) as dlg:
        
            dlg.Path = fullname
            dlg.Wildcard = ("Object image file (*.tif,*.tiff)|*.tif;*.tiff|"
                            "Ilastik project file (*.ilp)|*.ilp")
            result = dlg.ShowModal()
            fullname = dlg.Path
            if result == wx.ID_OK:
                if fullname.endswith(".ilp"):
                    self.save_into_ilp(fullname, labels, guidename)
                else:
                    md = ome.OMEXML()
                    mdp = md.image().Pixels
                    mdp.SizeX = shape[1]
                    mdp.SizeY = shape[0]
                    mdp.SizeC = n_frames
                    mdp.SizeT = 1
                    mdp.SizeZ = 1
                    mdp.PixelType = ome.PT_UINT16
                    mdp.channel_count = n_frames
                    if os.path.exists(fullname):
                        os.unlink(fullname)
                    xml = md.to_xml()
                    for i, l in enumerate(labels):
                        post_image(pathname2url(fullname),
                                   l, xml, index = str(i))

    def save_into_ilp(self, project_name, labels, guidename):
        import h5py
        with h5py.File(project_name) as f:
            g = f["DataSets"]
            for k in g:
                data_item = g[k]
                if data_item.attrs.get("fileName") == guidename:
                    break
            else:
                wx.MessageBox("Sorry, could not find the file, %s, in the project, %s" %
                              (guidname, project_name))
            project_labels = data_item["labels"]["data"]
            mask = np.ones(project_labels.shape[2:4], project_labels.dtype)
            for label in labels:
                mask[label != 0] = 2
            #
            # "only" use the first 100,000 points in the image
            #
            subsample = 100000
            npts = np.prod(mask.shape)
            if npts > subsample:
                r = np.random.RandomState()
                r.seed(np.sum(mask))
                i, j = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
                i0 = i[mask==1]
                j0 = j[mask==1]
                i1 = i[mask==2]
                j1 = j[mask==2]
                if len(i1) < subsample / 2:
                    p0 = r.permutation(len(i0))[:(subsample - len(i1))]
                    p1 = np.arange(len(i1))
                elif len(i0) < subsample / 2:
                    p0 = np.arange(len(i0))
                    p1 = r.permutation(len(i1))[:(subsample - len(i0))]
                else:
                    p0 = r.permutation(len(i0))[:(subsample / 2)]
                    p1 = r.permutation(len(i1))[:(subsample / 2)]
                mask_copy = np.zeros(mask.shape, mask.dtype)
                mask_copy[i0[p0], j0[p0]] = 1
                mask_copy[i1[p1], j1[p1]] = 2
                if "prediction" in data_item:
                    prediction = data_item["prediction"]
                    if np.max(prediction[0, 0, :, :, 0]) > .5:
                        # Only do if prediction was done (otherwise all == 0)
                        for n in range(2):
                            p = prediction[0, 0, :, :, n]
                            bad = (p < .5) & (mask == n+1)
                            mask_copy[i[bad], j[bad]] = n+1
                mask = mask_copy
            project_labels[0, 0, :, :, 0] = mask
            
    class InteractionCancelledException(RuntimeError):
        def __init__(self, *args):
            if len(args) == 0:
                args = ["User cancelled EditObjectsManually"]
            super(self.__class__, self).__init__(*args)
            
    def handle_interaction(self, orig_labels, guide_image):
        from cellprofiler.gui.editobjectsdlg import EditObjectsDialog
        from wx import OK
        with EditObjectsDialog(
            guide_image, orig_labels,
            self.allow_overlap, self.object_name.value) as dialog_box:
            result = dialog_box.ShowModal()
            if result != OK:
                raise self.InteractionCancelledException()
            return dialog_box.labels
    
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
            
        if (not from_matlab) and variable_revision_number == 2:
            # Added allow overlap, default = False
            setting_values = setting_values + [ cps.NO ]
            variable_revision_number = 3
        
        return setting_values, variable_revision_number, from_matlab
