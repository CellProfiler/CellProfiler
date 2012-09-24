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

import numpy as np
import os
import sys

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
    
    def is_interactive(self):
        return True
    
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

        if self.wants_image_display:
            guide_image = workspace.image_set.get_image(self.image_name.value)
            guide_image = guide_image.pixel_data
            if np.any(guide_image != np.min(guide_image)):
                guide_image = (guide_image - np.min(guide_image)) / (np.max(guide_image) - np.min(guide_image))
        else:
            guide_image = None
        if workspace.frame is None:
            # Accept the labels as-is
            filtered_labels = orig_labels
        else:
            filtered_labels = self.filter_objects(guide_image, orig_labels)
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
            for l in filtered_labels:
                plane_outlines = outline(l) != 0
                outlines[plane_outlines] = True
            outlines_image = cpi.Image(outlines)
            workspace.image_set.add(outlines_name, outlines_image)
        #
        # Do the drawing here
        #
        if workspace.frame is not None:
            figure = workspace.create_or_find_figure(title="EditObjectsManually, image cycle #%d"%(
                workspace.measurements.image_set_number),subplots=(2,1))
            figure.subplot_imshow_ijv(0, 0, orig_objects.ijv,
                                      shape = orig_labels[0].shape,
                                      title = orig_objects_name)
            figure.subplot_imshow_ijv(1, 0, filtered_objects.ijv,
                                      shape = filtered_labels[0].shape,
                                      title = filtered_objects_name,
                                      sharex = figure.subplot(0,0),
                                      sharey = figure.subplot(0,0))
    
    def run_as_data_tool(self):
        import wx
        from wx.lib.filebrowsebutton import FileBrowseButton
        import cellprofiler.utilities.jutil as jutil
        from cellprofiler.modules.loadimages import load_using_bioformats, convert_image_to_objects
        import bioformats.formatreader as formatreader
        from bioformats.formatwriter import make_ome_tiff_writer_class
        from bioformats.metadatatools import createOMEXMLMetadata, wrap_imetadata_object, make_pixel_type_class
        
        dlg = wx.Dialog(None)
        try:
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
        finally:
            dlg.Destroy()
        if new_or_existing_rb.GetSelection() == 1:
            #
            # Load the objects
            #
            n_frames = 1
            try:
                ImageReader = formatreader.make_image_reader_class()
                rdr = ImageReader()
                rdr.setGroupFiles(False)
                rdr.setId(fullname)
                n_frames = rdr.getImageCount()
            except:
                logger.warn("Failed to get number of frames from %s" %
                            filename)
            ijv = np.zeros((0, 3), int)
            offset = 0
            for index in range(n_frames):
                if n_frames == 1:
                    # Handle special case of interleaved color
                    labels = load_using_bioformats(
                        fullname, rescale = False)
                else:
                    labels = load_using_bioformats(
                        fullname, index = index, rescale = False)
                shape = labels.shape[:2]
                labels = convert_image_to_objects(labels)
                shape = labels.shape[:2]
                i, j = np.mgrid[0:labels.shape[0], 0:labels.shape[1]]
                ijv = np.vstack((
                    ijv, np.column_stack((i[labels!=0],
                                          j[labels!=0],
                                          labels[labels!=0] + offset))))
                if ijv.shape[0] > 0:
                    offset = np.max(ijv[:, 2])
            o = cpo.Objects()
            o.ijv = ijv
            o.shape = shape
            labels = [l for l,c in o.get_labels(shape)]
        else:
            labels = None
        #
        # Load the guide image
        #
        guide_image = load_using_bioformats(
            guidename)
        if np.min(guide_image) != np.max(guide_image):
            guide_image = (guide_image - np.min(guide_image)) / (np.max(guide_image)  - np.min(guide_image))
        if labels is None:
            shape = guide_image.shape[:2]
            labels = [np.zeros(shape, int)]
        labels = self.filter_objects(guide_image, labels)
        n_frames = len(labels)
        dlg = wx.FileDialog(None,
                            style = wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        
        dlg.Path = fullname
        dlg.Wildcard = "Object image file (*.tif,*.tiff)|*.tif;*.tiff"
        result = dlg.ShowModal()
        fullname = dlg.Path
        dlg.Destroy()
        if result == wx.ID_OK:
            md = createOMEXMLMetadata()
            md = wrap_imetadata_object(md)
            md.createRoot()
            md.setPixelsBigEndian(False, 0, 0)
            md.setPixelsDimensionOrder('XYCZT', 0, 0)
            try:
                PixelType = make_pixel_type_class()
                md.setPixelsType(PixelType.UINT16, 0)
            except:
                FormatTools = formatreader.make_format_tools_class()
                md.setPixelsPixelType(FormatTools.UINT16, 0, 0)
            md.setPixelsSizeX(shape[1], 0, 0)
            md.setPixelsSizeY(shape[0], 0, 0)
            md.setPixelsSizeC(n_frames, 0, 0)
            md.setPixelsSizeZ(1, 0, 0)
            md.setPixelsSizeT(1, 0, 0)
            md.setImageID("Image1", 0)
            md.setPixelsID("Pixels1", 0)
            for i in range(n_frames):
                md.setLogicalChannelSamplesPerPixel(1, 0, i)
                md.setChannelID("Channel%d" % (i+1), 0, i)
            ImageWriter = make_ome_tiff_writer_class()
            writer = ImageWriter()
            writer.setMetadataRetrieve(md)
            writer.setInterleaved(False)
            if os.path.exists(fullname):
                os.unlink(fullname)
            writer.setId(fullname)
            for i, l in enumerate(labels):
                s = l.astype("<u2").tostring()
                pixels = np.fromstring(s, dtype = np.uint8)
                ifd = formatreader.jutil.make_instance("loci/formats/tiff/IFD","()V")
                #
                # Need to explicitly set the maximum sample value or images
                # get rescaled inside the TIFF writer.
                #
                min_sample_value = jutil.get_static_field(
                    "loci/formats/tiff/IFD", "MIN_SAMPLE_VALUE", "I")
                jutil.call(ifd, "putIFDValue", "(II)V", min_sample_value, 0)
                max_sample_value = jutil.get_static_field(
                    "loci/formats/tiff/IFD", "MAX_SAMPLE_VALUE","I")
                jutil.call(ifd, "putIFDValue","(II)V",
                           max_sample_value, 65535)
                writer.saveBytesIFD(i, jutil.get_env().make_byte_array(pixels), ifd)
            writer.close()
            
    def filter_objects(self, guide_image, orig_labels):
        import wx
        import wx.html
        import matplotlib
        from matplotlib.lines import Line2D
        from matplotlib.path import Path
        from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
        from matplotlib.backends.backend_wxagg import NavigationToolbar2WxAgg
        import scipy.ndimage
        from cellprofiler.gui.cpfigure import renumber_labels_for_display
        from cellprofiler.icons import get_builtin_image
        from cellprofiler.cpmath.cpmorphology import polygon_lines_to_mask
        from cellprofiler.cpmath.cpmorphology import convex_hull_image
        from cellprofiler.cpmath.cpmorphology import distance2_to_line
        
        class FilterObjectsDialog(wx.Dialog):
            resume_id = wx.NewId()
            cancel_id = wx.NewId()
            keep_all_id = wx.NewId()
            remove_all_id = wx.NewId()
            reverse_select = wx.NewId()
            epsilon = 5 # maximum pixel distance to a vertex for hit test
            FREEHAND_DRAW_MODE = "freehanddrawmode"
            SPLIT_PICK_FIRST_MODE = "split1"
            SPLIT_PICK_SECOND_MODE = "split2"
            NORMAL_MODE = "normal"
            #
            # The object_number for an artist
            #
            K_LABEL = "label"
            #
            # Whether the artist has been edited
            #
            K_EDITED = "edited"
            #
            # Whether the artist is on the outside of the object (True)
            # or is the border of a hole (False)
            #
            K_OUTSIDE = "outside"
            def __init__(self, module, guide_image, orig_labels):
                assert isinstance(module, EditObjectsManually)
                #
                # Get the labels matrix and make a mask of objects to keep from it
                #
                #
                # Display a UI for choosing objects
                #
                frame_size = wx.GetDisplaySize()
                frame_size = [max(frame_size[0], frame_size[1]) / 2] * 2
                style = wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX
                wx.Dialog.__init__(self, None, -1,
                                   "Choose objects to keep",
                                   size=frame_size,
                                   style = style)
                self.module = module
                self.guide_image = guide_image
                self.orig_labels = orig_labels
                self.shape = self.orig_labels[0].shape
                self.background = None # background = None if full repaint needed
                self.reset(display=False)
                self.active_artist = None
                self.active_index = None
                self.mode = self.NORMAL_MODE
                self.split_artist = None
                self.wants_image_display = guide_image != None
                self.pressed_keys = set()
                self.build_ui()
                self.init_labels()
                self.display()
                self.Layout()
                self.Raise()
                self.panel.SetFocus()
                
            def record_undo(self):
                '''Push an undo record onto the undo stack'''
                #
                # The undo record is a diff between the last ijv and
                # the current, plus the current state of the artists.
                #
                ijv = self.calculate_ijv()
                if ijv.shape[0] == 0:
                    ijvx = np.zeros((0, 4), int)
                else:
                    #
                    # Sort the current and last ijv together, adding
                    # an old_new_indicator.
                    #
                    ijvx = np.vstack((
                        np.column_stack(
                            (ijv, np.zeros(ijv.shape[0], ijv.dtype))),
                        np.column_stack(
                            (self.last_ijv,
                             np.ones(self.last_ijv.shape[0], ijv.dtype)))))
                    order = np.lexsort((ijvx[:, 3], 
                                        ijvx[:, 2], 
                                        ijvx[:, 1], 
                                        ijvx[:, 0]))
                    ijvx = ijvx[order, :]
                    #
                    # Then mark all prev and next where i,j,v match (in both sets)
                    #
                    matches = np.hstack(
                        ((np.all(ijvx[:-1, :3] == ijvx[1:, :3], 1) &
                          (ijvx[:-1, 3] == 0) &
                          (ijvx[1:, 3] == 1)), [False]))
                    matches[1:] = matches[1:] | matches[:-1]
                    ijvx = ijvx[~matches, :]
                artist_save = [(a.get_data(), self.artists[a].copy())
                               for a in self.artists]
                self.undo_stack.append((ijvx, self.last_artist_save))
                self.last_artist_save = artist_save
                self.last_ijv = ijv
                self.undo_button.Enable(True)
                
            def undo(self, event=None):
                '''Pop an entry from the undo stack and apply'''
                #
                # Mix what's on the undo ijv with what's in self.last_ijv
                # and remove any 0/1 pairs.
                #
                ijvx, artist_save = self.undo_stack.pop()
                ijvx = np.vstack((
                    ijvx, np.column_stack(
                        (self.last_ijv, np.ones(self.last_ijv.shape[0],
                                                self.last_ijv.dtype)))))
                order = np.lexsort((ijvx[:, 3], ijvx[:, 2], ijvx[:, 1], ijvx[:, 0]))
                ijvx = ijvx[order, :]
                #
                # Then mark all prev and next where i,j,v match (in both sets)
                #
                matches = np.hstack(
                    (np.all(ijvx[:-1, :3] == ijvx[1:, :3], 1), [False]))
                matches[1:] = matches[1:] | matches[:-1]
                ijvx = ijvx[~matches, :]
                self.last_ijv = ijvx[:, :3]
                self.last_artist_save = artist_save
                temp = cpo.Objects()
                temp.ijv = self.last_ijv
                self.labels = [l for l, c in temp.get_labels(self.shape)]
                self.init_labels()
                #
                # replace the artists
                #
                for artist in self.artists:
                    artist.remove()
                self.artists = {}
                for (x, y), d in artist_save:
                    object_number = d[self.K_LABEL]
                    artist = Line2D(x, y,
                                    marker='o', markerfacecolor='r',
                                    markersize=6,
                                    color=self.colormap[object_number, :],
                                    animated = True)
                    self.artists[artist] = d
                    self.orig_axes.add_line(artist)
                self.display()
                if len(self.undo_stack) == 0:
                    self.undo_button.Enable(False)
                
            def calculate_ijv(self):
                '''Return the current IJV representation of the labels'''
                i, j = np.mgrid[0:self.shape[0], 0:self.shape[1]]
                ijv = np.zeros((0, 3), int)
                for l in self.labels:
                    ijv = np.vstack(
                        (ijv,
                         np.column_stack([i[l!=0], j[l!=0], l[l!=0]])))
                return ijv
                
            def build_ui(self):
                sizer = wx.BoxSizer(wx.VERTICAL)
                self.SetSizer(sizer)
                self.figure = matplotlib.figure.Figure()
                self.panel = FigureCanvasWxAgg(self, -1, self.figure)
                sizer.Add(self.panel, 1, wx.EXPAND)
                if True:
                    self.html_frame = wx.MiniFrame(
                        self, style = wx.DEFAULT_MINIFRAME_STYLE | 
                        wx.CLOSE_BOX | wx.SYSTEM_MENU | wx.RESIZE_BORDER)
                    self.html_panel = wx.html.HtmlWindow(self.html_frame)
                    if sys.platform == 'darwin':
                        LEFT_MOUSE = "mouse"
                        LEFT_MOUSE_BUTTON = "mouse button"
                        RIGHT_MOUSE = "[control] + mouse"
                    else:
                        LEFT_MOUSE = "left mouse button"
                        LEFT_MOUSE_BUTTON = LEFT_MOUSE
                        RIGHT_MOUSE = "right mouse button"
                    self.html_panel.SetPage(
                    """<H1>Editing help</H1>
                    The editing user interface lets you create, remove and
                    edit objects. You can remove an object by clicking on it
                    with the %(LEFT_MOUSE)s in the "Objects to keep" window
                    and add it back by clicking on it in the "Objects to
                    remove" window. You can edit objects by selecting them
                    with the %(RIGHT_MOUSE)s. You can move object control points
                    by dragging them while holding the %(LEFT_MOUSE_BUTTON)s
                    down (you cannot move a control point across the boundary
                    of the object you are editing and you cannot move the
                    edges on either side across another control point).
                    When you are finished editing,
                    click on the object again with the %(RIGHT_MOUSE)s to save changes
                    or hit the <i>Esc</i> key to abandon your changes.
                    <br>
                    Press the <i>Done</i> key to save your edits.
                    You can always reset your edits to the original state
                    before editing by pressing the <i>Reset</i> key.
                    <h2>Editing commands</h2>
                    The following keys perform editing commands when pressed:
                    <br><ul>
                    <li><b>1</b>: Toggle between one display (the editing
                    display) and three.</li>
                    <li><b>A</b>: Add a control point to the line nearest the
                    mouse cursor</li>
                    <li><b>C</b>: Join all selected objects into one that forms a
                    convex hull around them all. The convex hull is the smallest
                    shape that has no indentations and encloses all of the
                    objects. You can use this to combine several pieces into
                    one round object.</li>
                    <li><b>D</b>: Delete the control point nearest to the
                    cursor.</li>
                    <li><b>f</b>: Freehand draw. Press down on the %(LEFT_MOUSE)s
                    to draw a new object outline, then release to complete
                    the outline and return to normal editing.</li>
                    <li><b>J</b>: Join all selected objects into one object.</li>
                    <li><b>N</b>: Create a new object under the cursor.</li>
                    <li><b>S</b>: Split an object. Pressing <b>S</b> puts
                    the user interface into <i>Split Mode</i>. The user interface
                    will prompt you to select a first and second point for the
                    split. Two types of splits are allowed: a split between
                    two points on the same contour and a split between the
                    inside and the outside of an object that has a hole in it.
                    The former split creates two separate objects. The latter
                    creates a channel from the hole to the outside of the object.
                    </li>
                    </ul>
                    <br><i>Note: editing is disabled in zoom or pan mode. The
                    zoom or pan button on the navigation toolbar is depressed
                    during this mode and your cursor is no longer an arrow.
                    You can exit from zoom or pan mode by pressing the
                    appropriate button on the navigation toolbar.</i>
                    """ % locals())
                    self.html_frame.Show(False)
                    self.html_frame.Bind(wx.EVT_CLOSE, self.on_help_close)
        
                toolbar = NavigationToolbar2WxAgg(self.panel)
                sizer.Add(toolbar, 0, wx.EXPAND)
                #
                # Make 3 axes
                #
                self.orig_axes = self.figure.add_subplot(2, 2, 1)
                self.orig_axes.set_zorder(1) # preferentially select on click.
                self.orig_axes._adjustable = 'box-forced'
                self.keep_axes = self.figure.add_subplot(
                    2, 2, 2, sharex = self.orig_axes, sharey = self.orig_axes)
                self.remove_axes = self.figure.add_subplot(
                    2, 2, 4, sharex = self.orig_axes, sharey = self.orig_axes)
                for axes in (self.orig_axes, self.keep_axes, self.remove_axes):
                    axes._adjustable = 'box-forced'
                orig_objects_name = self.module.object_name.value
                self.orig_objects_title = "Original: %s" % orig_objects_name
                for axes, title in (
                    (self.orig_axes, 
                     self.orig_objects_title),
                    (self.keep_axes, "Objects to keep"),
                    (self.remove_axes, "Objects to remove")):
                    axes.set_title(title,
                                   fontname=cpprefs.get_title_font_name(),
                                   fontsize=cpprefs.get_title_font_size())
            
                self.info_axes = self.figure.add_subplot(2, 2, 3)
                self.info_axes.set_axis_off()

                sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
                #
                # Need padding on top because tool bar is wonky about its height
                #
                sizer.Add(sub_sizer, 0, wx.EXPAND | wx.TOP, 10)
                        
                #########################################
                #
                # Buttons for keep / remove / toggle
                #
                #########################################
                
                keep_button = wx.Button(self, self.keep_all_id, "Keep all")
                sub_sizer.Add(keep_button, 0, wx.ALIGN_CENTER)
        
                remove_button = wx.Button(self, self.remove_all_id, "Remove all")
                sub_sizer.Add(remove_button,0, wx.ALIGN_CENTER)
        
                toggle_button = wx.Button(self, self.reverse_select, 
                                          "Reverse selection")
                sub_sizer.Add(toggle_button,0, wx.ALIGN_CENTER)
                self.undo_button = wx.Button(self, wx.ID_UNDO)
                self.undo_button.SetToolTipString("Undo last edit")
                self.undo_button.Enable(False)
                sub_sizer.Add(self.undo_button)
                reset_button = wx.Button(self, -1, "Reset")
                reset_button.SetToolTipString(
                    "Undo all editing and restore the original objects")
                sub_sizer.Add(reset_button)
                self.Bind(wx.EVT_BUTTON, self.on_toggle, toggle_button)
                self.Bind(wx.EVT_BUTTON, self.on_keep, keep_button)
                self.Bind(wx.EVT_BUTTON, self.on_remove, remove_button)
                self.Bind(wx.EVT_BUTTON, self.undo, id = wx.ID_UNDO)
                self.Bind(wx.EVT_BUTTON, self.on_reset, reset_button)
                self.figure.canvas.Bind(wx.EVT_PAINT, self.on_paint)
        
                ######################################
                #
                # Buttons for resume and cancel
                #
                ######################################
                button_sizer = wx.StdDialogButtonSizer()
                resume_button = wx.Button(self, self.resume_id, "Done")
                button_sizer.AddButton(resume_button)
                sub_sizer.Add(button_sizer, 0, wx.ALIGN_CENTER)
                def on_resume(event):
                    self.EndModal(wx.OK)
                    self.on_close(event)
                self.Bind(wx.EVT_BUTTON, on_resume, resume_button)
                button_sizer.SetAffirmativeButton(resume_button)
        
                cancel_button = wx.Button(self, self.cancel_id, "Cancel")
                button_sizer.AddButton(cancel_button)
                def on_cancel(event):
                    self.EndModal(wx.CANCEL)
                    self.on_close(event)
                self.Bind(wx.EVT_BUTTON, on_cancel, cancel_button)
                button_sizer.SetNegativeButton(cancel_button)
                button_sizer.AddButton(wx.Button(self, wx.ID_HELP))
                self.Bind(wx.EVT_BUTTON, self.on_help, id= wx.ID_HELP)
                self.Bind(wx.EVT_CLOSE, self.on_close)
                                  
                button_sizer.Realize()
                if self.module.wants_image_display:
                    #
                    # Note: the checkbutton must have a reference or it
                    #       will cease to be checkable.
                    #
                    self.display_image_checkbox = matplotlib.widgets.CheckButtons(
                        self.info_axes, ["Display image"], [True])
                    self.display_image_checkbox.labels[0].set_size("small")
                    r = self.display_image_checkbox.rectangles[0]
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
                    l1, l2 = self.display_image_checkbox.lines[0]
                    l1.set_data((np.array((new_rx, new_rx+new_rwidth)),
                                 np.array((new_ry, new_ry+new_rheight))))
                    l2.set_data((np.array((new_rx, new_rx+new_rwidth)),
                                 np.array((new_ry + new_rheight, new_ry))))
                    
                    self.display_image_checkbox.on_clicked(
                        self.on_display_image_clicked)
                self.figure.canvas.mpl_connect('button_press_event', 
                                               self.on_click)
                self.figure.canvas.mpl_connect('draw_event', self.draw_callback)
                self.figure.canvas.mpl_connect('button_release_event',
                                               self.on_mouse_button_up)
                self.figure.canvas.mpl_connect('motion_notify_event',
                                               self.on_mouse_moved)
                self.figure.canvas.mpl_connect('key_press_event',
                                               self.on_key_down)
                self.figure.canvas.mpl_connect('key_release_event',
                                               self.on_key_up)
                
            def on_display_image_clicked(self, event):
                self.wants_image_display = not self.wants_image_display
                self.display()
                
            def init_labels(self):
                #########################################
                #
                # Construct a stable label index transform
                # and a color display image.
                #
                #########################################
                
                nlabels = len(self.to_keep) - 1
                label_map = np.zeros(nlabels + 1, self.labels[0].dtype)
                lstart = 0
                self.oi = np.zeros(0, int)
                self.oj = np.zeros(0, int)
                self.ol = np.zeros(0, int)
                for label in self.labels:
                    # drive each successive matrix's labels away
                    # from all others.
                    distinct_label_count = np.sum(np.unique(label) != 0)
                    clabels = renumber_labels_for_display(label)
                    clabels[clabels != 0] += lstart
                    lstart += distinct_label_count
                    label_map[label.flatten()] = clabels.flatten()
                    outlines = outline(clabels)
                    oi, oj = np.argwhere(outlines != 0).transpose()
                    ol = label[oi, oj]
                    self.oi = np.hstack((self.oi, oi))
                    self.oj = np.hstack((self.oj, oj))
                    self.ol = np.hstack((self.ol, ol))
                cm = matplotlib.cm.get_cmap(cpprefs.get_default_colormap())
                cm.set_bad((0,0,0))
            
                mappable = matplotlib.cm.ScalarMappable(cmap=cm)
                mappable.set_clim(1, nlabels+1)
                self.colormap = mappable.to_rgba(np.arange(nlabels + 1))[:, :3]
                self.colormap = self.colormap[label_map, :]
                self.oc = self.colormap[self.ol, :]
                
            def on_close(self, event):
                '''Fix up the labels as we close'''
                if self.GetReturnCode() == wx.OK:
                    open_labels = set([d[self.K_LABEL] for d in self.artists.values()])
                    for l in open_labels:
                        self.close_label(l, False)
                    for idx in np.where(~self.to_keep):
                        if idx > 0:
                            self.remove_label(idx)
                
            def remove_label(self, object_number):
                for l in self.labels:
                    l[l == object_number] = 0
                
            def replace_label(self, mask, object_number):
                self.remove_label(object_number)
                self.labels.append(mask.astype(self.labels[0].dtype) * object_number)
                self.restructure_labels()
                
            def restructure_labels(self):
                '''Convert the labels into ijv and back to get the colors right'''
                
                ii = []
                jj = []
                vv = []
                i, j = np.mgrid[0:self.shape[0], 0:self.shape[1]]
                for l in self.labels:
                    mask = l != 0
                    ii.append(i[mask])
                    jj.append(j[mask])
                    vv.append(l[mask])
                temp = cpo.Objects()
                temp.ijv = np.column_stack(
                    [np.hstack(x) for x in (ii, jj, vv)])
                self.labels = [l for l,c in temp.get_labels(self.shape)]
                
            def add_label(self, mask):
                object_number = len(self.to_keep)
                temp = np.ones(self.to_keep.shape[0] + 1, bool)
                temp[:-1] = self.to_keep
                self.to_keep = temp
                self.labels.append(mask.astype(self.labels[0].dtype) * object_number)
                self.restructure_labels()
                
            ################### d i s p l a y #######
            #
            # The following is a function that we can call to refresh
            # the figure's appearance based on the mask and the original labels
            #
            ##########################################
            
            def display(self):
                orig_objects_name = self.module.object_name.value
                if len(self.orig_axes.images) > 0:
                    # Save zoom and scale if coming through here a second time
                    x0, x1 = self.orig_axes.get_xlim()
                    y0, y1 = self.orig_axes.get_ylim()
                    set_lim = True
                else:
                    set_lim = False
                orig_to_show = np.ones(len(self.to_keep), bool)
                for d in self.artists.values():
                    object_number = d[self.K_LABEL]
                    if object_number < len(orig_to_show):
                        orig_to_show[object_number] = False
                for axes, keep in (
                    (self.orig_axes, orig_to_show),
                    (self.keep_axes, self.to_keep),
                    (self.remove_axes, ~ self.to_keep)):
                    
                    assert isinstance(axes, matplotlib.axes.Axes)
                    axes.clear()
                    if self.wants_image_display and self.guide_image is not None:
                        image, _ = cpo.size_similarly(self.orig_labels[0], 
                                                      self.guide_image)
                        if image.ndim == 2:
                            image = np.dstack((image, image, image))
                        cimage = image.copy()
                    else:
                        cimage = np.zeros(
                            (self.shape[0],
                             self.shape[1],
                             3), np.float)
                    if len(keep) > 1:
                        kmask = keep[self.ol]
                        if np.any(kmask):
                            cimage[self.oi[kmask], self.oj[kmask], :] = \
                                self.oc[kmask, :]
                    axes.imshow(cimage)
                self.set_orig_axes_title()
                self.keep_axes.set_title("Objects to keep",
                                         fontname=cpprefs.get_title_font_name(),
                                         fontsize=cpprefs.get_title_font_size())
                self.remove_axes.set_title("Objects to remove",
                                           fontname=cpprefs.get_title_font_name(),
                                           fontsize=cpprefs.get_title_font_size())
                if set_lim:
                    self.orig_axes.set_xlim((x0, x1))
                    self.orig_axes.set_ylim((y0, y1))
                for artist in self.artists:
                    self.orig_axes.add_line(artist)
                if self.split_artist is not None:
                    self.orig_axes.add_line(self.split_artist)
                self.background = None
                self.Refresh()
                
            def on_paint(self, event):
                dc = wx.PaintDC(self.panel)
                if self.background == None or tuple(self.Size) != self.draw_size:
                    self.panel.draw(dc)
                else:
                    self.panel.gui_repaint(dc)
                dc.Destroy()
                event.Skip()
                
            def draw_callback(self, event):
                '''Decorate the drawing with the animated artists'''
                self.background = self.figure.canvas.copy_from_bbox(self.orig_axes.bbox)
                for artist in self.artists:
                    self.orig_axes.draw_artist(artist)
                if self.split_artist is not None:
                    self.orig_axes.draw_artist(self.split_artist)
                if (self.mode == self.FREEHAND_DRAW_MODE and 
                    self.active_artist is not None):
                    self.orig_axes.draw_artist(self.active_artist)
                self.figure.canvas.blit(self.orig_axes.bbox)
                self.draw_size = tuple(self.Size)
                
            def get_control_point(self, event):
                '''Find the artist and control point under the cursor
                
                returns tuple of artist, and index of control point or None, None
                '''
                best_d = np.inf
                best_artist = None
                best_index = None
                for artist in self.artists:
                    data = artist.get_xydata()[:-1, :]
                    xy = artist.get_transform().transform(data)
                    x, y = xy.transpose()
                    d = np.sqrt((x-event.x)**2 + (y-event.y)**2)
                    idx = np.atleast_1d(np.argmin(d)).flatten()[0]
                    d = d[idx]
                    if d < self.epsilon and d < best_d:
                        best_d = d
                        best_artist = artist
                        best_index = idx
                return best_artist, best_index
                    
            def on_click(self, event):
                if event.inaxes not in (
                    self.orig_axes, self.keep_axes, self.remove_axes):
                    return
                if event.inaxes.get_navigate_mode() is not None:
                    return
                if self.mode == self.SPLIT_PICK_FIRST_MODE:
                    self.on_split_first_click(event)
                    return
                elif self.mode == self.SPLIT_PICK_SECOND_MODE:
                    self.on_split_second_click(event)
                    return
                elif self.mode == self.FREEHAND_DRAW_MODE:
                    self.on_freehand_draw_click(event)
                    return
                if event.inaxes == self.orig_axes and event.button == 1:
                    best_artist, best_index = self.get_control_point(event)
                    if best_artist is not None:
                        self.active_artist = best_artist
                        self.active_index = best_index
                        return
                elif event.inaxes == self.orig_axes and event.button == 3:
                    for artist in self.artists:
                        path = Path(artist.get_xydata())
                        if path.contains_point((event.xdata, event.ydata)):
                            self.close_label(self.artists[artist][self.K_LABEL])
                            self.record_undo()
                            return
                x = int(event.xdata)
                y = int(event.ydata)
                if (x < 0 or x >= self.shape[1] or
                    y < 0 or y >= self.shape[0]):
                    return
                for labels in self.labels:
                    lnum = labels[y,x]
                    if lnum != 0:
                        break
                if lnum == 0:
                    return
                if event.button == 1:
                    # Move object into / out of working set
                    if event.inaxes == self.orig_axes:
                        self.to_keep[lnum] = not self.to_keep[lnum]
                    elif event.inaxes == self.keep_axes:
                        self.to_keep[lnum] = False
                    else:
                        self.to_keep[lnum] = True
                    self.display()
                elif event.button == 3:
                    self.make_control_points(lnum)
                    self.display()
            
            def on_key_down(self, event):
                self.pressed_keys.add(event.key)
                if event.key == "1":
                    self.toggle_single_panel(event)
                    return
                if event.key == "f1":
                    self.on_help(event)
                if self.mode == self.NORMAL_MODE:
                    if event.key == "j":
                        self.join_objects(event)
                    elif event.key == "c":
                        self.convex_hull(event)
                    elif event.key == "a":
                        self.add_control_point(event)
                    elif event.key == "d":
                        self.delete_control_point(event)
                    elif event.key == "f":
                        self.enter_freehand_draw_mode(event)
                    elif event.key == "n":
                        self.new_object(event)
                    elif event.key == "s":
                        self.enter_split_mode(event)
                    elif event.key =="z":
                        if len(self.undo_stack) > 0:
                            self.undo()
                    elif event.key == "escape":
                        self.remove_artists(event)
                elif self.mode in (self.SPLIT_PICK_FIRST_MODE, 
                                   self.SPLIT_PICK_SECOND_MODE):
                    if event.key == "escape":
                        self.exit_split_mode(event)
                elif self.mode == self.FREEHAND_DRAW_MODE:
                    self.exit_freehand_draw_mode(event)
            
            def on_key_up(self, event):
                if event.key in self.pressed_keys:
                    self.pressed_keys.remove(event.key)
            
            def on_mouse_button_up(self, event):
                if (event.inaxes is not None and 
                    event.inaxes.get_navigate_mode() is not None):
                    return
                if self.mode == self.FREEHAND_DRAW_MODE:
                    self.on_mouse_button_up_freehand_draw_mode(event)
                else:
                    self.active_artist = None
                    self.active_index = None
                
            def on_mouse_moved(self, event):
                if self.mode == self.FREEHAND_DRAW_MODE:
                    self.handle_mouse_moved_freehand_draw_mode(event)
                elif self.active_artist is not None:
                    self.handle_mouse_moved_active_mode(event)
                elif self.mode == self.SPLIT_PICK_SECOND_MODE:
                    self.handle_mouse_moved_pick_second_mode(event)
                    
            def handle_mouse_moved_active_mode(self, event):
                if event.inaxes != self.orig_axes:
                    return
                #
                # Don't let the user make any lines that cross other lines
                # in this object.
                #
                object_number = self.artists[self.active_artist][self.K_LABEL]
                data = [d[:-1] for d in self.active_artist.get_data()]
                n_points = len(data[0])
                before_index = (n_points - 1 + self.active_index) % n_points
                after_index = (self.active_index + 1) % n_points
                before_pt, after_pt = [
                    np.array([data[0][idx], data[1][idx]]) 
                             for idx in (before_index, after_index)]
                new_pt = np.array([event.xdata, event.ydata], int)
                path = Path(np.array((before_pt, new_pt, after_pt)))
                eps = np.finfo(np.float32).eps
                for artist in self.artists:
                    if (self.module.allow_overlap and 
                        self.artists[artist][self.K_LABEL] != object_number):
                        continue
                    if artist == self.active_artist:
                        if n_points <= 4:
                            continue
                        # Exclude the lines -2 and 2 before and after ours.
                        #
                        xx, yy = [np.hstack((d[self.active_index:],
                                             d[:(self.active_index+1)]))
                                  for d in data]
                        xx, yy = xx[2:-2], yy[2:-2]
                        xydata = np.column_stack((xx, yy))
                    else:
                        xydata = artist.get_xydata()
                    other_path = Path(xydata)
                    
                    l0 = xydata[:-1, :]
                    l1 = xydata[1:, :]
                    neww_pt = np.ones(l0.shape) * new_pt[np.newaxis, :]
                    d = distance2_to_line(neww_pt, l0, l1)
                    different_sign = (np.sign(neww_pt - l0) != 
                                      np.sign(neww_pt - l1))
                    on_segment = ((d < eps) & different_sign[:, 0] & 
                                  different_sign[:, 1])
                        
                    if any(on_segment):
                        # it's ok if the point is on the line.
                        continue
                    if path.intersects_path(other_path, filled = False):
                        return
                 
                data = self.active_artist.get_data()   
                data[0][self.active_index] = event.xdata
                data[1][self.active_index] = event.ydata
                
                #
                # Handle moving the first point which is the
                # same as the last and they need to be moved together.
                # The last should never be moved.
                #
                if self.active_index == 0:
                    data[0][-1] = event.xdata
                    data[1][-1] = event.ydata
                self.active_artist.set_data(data)
                self.artists[self.active_artist]['edited'] = True
                self.update_artists()
                
            def update_artists(self):
                self.figure.canvas.restore_region(self.background)
                for artist in self.artists:
                    self.orig_axes.draw_artist(artist)
                if self.split_artist is not None:
                    self.orig_axes.draw_artist(self.split_artist)
                if (self.mode == self.FREEHAND_DRAW_MODE and 
                    self.active_artist is not None):
                    self.orig_axes.draw_artist(self.active_artist)
                    old = self.panel.IsShownOnScreen
                #
                # Need to keep "blit" from drawing on the screen.
                #
                # On Mac:
                #     Blit makes a new ClientDC
                #     Blit calls gui_repaint
                #     if IsShownOnScreen:
                #        ClientDC.EndDrawing is called
                #        ClientDC.EndDrawing processes queued GUI events
                #        If there are two mouse motion events queued,
                #        the mouse event handler is called recursively.
                #        Blit is called a second time.
                #        A second ClientDC is created which, on the Mac,
                #        throws an exception.
                #
                # It's not my fault that the Mac can't deal with two
                # client dcs being created - not an impossible problem for
                # them to solve.
                #
                # It's not my fault that WX decides to process all pending
                # events in the queue.
                #
                # It's not my fault that Blit is called without an optional
                # dc argument that could be used instead of creating a client
                # DC.
                #
                old = self.panel.IsShownOnScreen
                self.panel.IsShownOnScreen = lambda *args: False
                try:
                    self.figure.canvas.blit(self.orig_axes.bbox)
                finally:
                    self.panel.IsShownOnScreen = old
                self.panel.Refresh()
                
            def toggle_single_panel(self, event):
                for ax in (self.keep_axes, self.info_axes, self.remove_axes):
                    ax.set_visible(not ax.get_visible())
                if self.keep_axes.get_visible():
                    self.orig_axes.change_geometry(2,2,1)
                else:
                    self.orig_axes.change_geometry(1,1,1)
                self.figure.canvas.draw()
                
            def join_objects(self, event):
                all_labels = np.unique([
                    v[self.K_LABEL] for v in self.artists.values()])
                if len(all_labels) < 2:
                    return
                assert all_labels[0] == np.min(all_labels)
                object_number = all_labels[0]
                for label in all_labels:
                    self.close_label(label, display=False)
                
                to_join = np.zeros(len(self.to_keep), bool)
                to_join[all_labels] = True
                #
                # Copy all labels to join to the mask and erase.
                #
                mask = np.zeros(self.shape, bool)
                for label in self.labels:
                    mask |= to_join[label]
                    label[to_join[label]] = 0
                self.labels.append(
                    mask.astype(self.labels[0].dtype) * object_number)
                    
                self.restructure_labels()
                self.init_labels()
                self.make_control_points(object_number)
                self.display()
                self.record_undo()
                return all_labels[0]
                
            def convex_hull(self, event):
                if len(self.artists) == 0:
                    return
                
                all_labels = np.unique([
                    v[self.K_LABEL] for v in self.artists.values()])
                for label in all_labels:
                    self.close_label(label, display=False)
                object_number = all_labels[0]
                mask = np.zeros(self.shape, bool)
                for label in self.labels:
                    for n in all_labels:
                        mask |= label == n
                        
                for n in all_labels:
                    self.remove_label(n)
                if len(all_labels) > 1:
                    keep_to_keep = np.ones(len(self.to_keep), bool)
                    keep_to_keep[all_labels[1:]] = False
                    self.to_keep = self.to_keep[keep_to_keep]
                    
                mask = convex_hull_image(mask)
                self.replace_label(mask, object_number)
                self.init_labels()
                self.make_control_points(object_number)
                self.display()
                self.record_undo()
            
            def add_control_point(self, event):
                if len(self.artists) == 0:
                    return
                pt_i, pt_j = event.ydata, event.xdata
                best_artist = None
                best_index = None
                best_distance = np.inf
                new_pt = None
                for artist in self.artists:
                    l = artist.get_xydata()[:, ::-1]
                    l0 = l[:-1, :]
                    l1 = l[1:, :]
                    llen = np.sqrt(np.sum((l1 - l0) ** 2, 1))
                    # the unit vector
                    v = (l1 - l0) / llen[:, np.newaxis]
                    pt = np.ones(l0.shape, l0.dtype)
                    pt[:, 0] = pt_i
                    pt[:, 1] = pt_j
                    #
                    # Project l0<->pt onto l0<->l1. If the result
                    # is longer than l0<->l1, then the closest point is l1.
                    # If the result is negative, then the closest point is l0.
                    # In either case, don't add.
                    #
                    proj = np.sum(v * (pt - l0), 1)
                    d2 = distance2_to_line(pt, l0, l1)
                    d2[proj <= 0] = np.inf
                    d2[proj >= llen] = np.inf
                    best = np.argmin(d2)
                    if best_distance > d2[best]:
                        best_distance = d2[best]
                        best_artist = artist
                        best_index = best
                        new_pt = (l0[best_index, :] + 
                                  proj[best_index, np.newaxis] * v[best_index, :])
                if best_artist is None:
                    return
                l = best_artist.get_xydata()[:, ::-1]
                l = np.vstack((l[:(best_index+1)], new_pt.reshape(1,2),
                               l[(best_index+1):]))
                best_artist.set_data((l[:, 1], l[:, 0]))
                self.artists[best_artist][self.K_EDITED] = True
                self.update_artists()
                self.record_undo()
            
            def delete_control_point(self, event):
                best_artist, best_index = self.get_control_point(event)
                if best_artist is not None:
                    l = best_artist.get_xydata()
                    if len(l) < 4:
                        object_number = self.artists[best_artist][self.K_LABEL]
                        best_artist.remove()
                        del self.artists[best_artist]
                        if not any([d[self.K_LABEL] == object_number
                                    for d in self.artists.values()]):
                            self.remove_label(object_number)
                            self.init_labels()
                            self.display()
                            self.record_undo()
                            return
                        else:
                            # Mark some other artist as edited.
                            for artist, d in self.artists.iteritems():
                                if d[self.K_LABEL] == object_number:
                                    d[self.K_EDITED] = True
                    else:
                        l = np.vstack((
                            l[:best_index, :], 
                            l[(best_index+1):-1, :]))
                        l = np.vstack((l, l[:1, :]))
                        best_artist.set_data((l[:, 0], l[:, 1]))
                        self.artists[best_artist][self.K_EDITED] = True
                        self.record_undo()
                    self.update_artists()
                    
            def new_object(self, event):
                object_number = len(self.to_keep)
                temp = np.ones(object_number+1, bool)
                temp[:-1] = self.to_keep
                self.to_keep = temp
                angles = np.pi * 2 * np.arange(13) / 12
                x = 20 * np.cos(angles) + event.xdata
                y = 20 * np.sin(angles) + event.ydata
                x[x < 0] = 0
                x[x >= self.shape[1]] = self.shape[1]-1
                y[y >= self.shape[0]] = self.shape[0]-1
                self.init_labels()
                new_artist = Line2D(x, y,
                                    marker='o', markerfacecolor='r',
                                    markersize=6,
                                    color=self.colormap[object_number, :],
                                    animated = True)
                
                self.artists[new_artist] = { self.K_LABEL: object_number,
                                             self.K_EDITED: True,
                                             self.K_OUTSIDE: True}
                self.display()
                self.record_undo()
                
            def remove_artists(self, event):
                for artist in self.artists:
                    artist.remove()
                self.artists = {}
                self.display()
                
            ################################
            #
            # Split mode
            #
            ################################
                
            SPLIT_PICK_FIRST_TITLE = "Pick first point for split or hit Esc to exit"
            SPLIT_PICK_SECOND_TITLE = "Pick second point for split or hit Esc to exit"
            
            def set_orig_axes_title(self):
                if self.mode == self.SPLIT_PICK_FIRST_MODE:
                    title = self.SPLIT_PICK_FIRST_TITLE 
                elif self.mode == self.SPLIT_PICK_SECOND_MODE:
                    title = self.SPLIT_PICK_SECOND_TITLE
                elif self.mode == self.FREEHAND_DRAW_MODE:
                    if self.active_artist is None:
                        title = "Click the mouse to begin to draw or hit Esc"
                    else:
                        title = "Freehand drawing"
                else:
                    title = self.orig_objects_title
                                                
                self.orig_axes.set_title(
                    title,
                    fontname=cpprefs.get_title_font_name(),
                    fontsize=cpprefs.get_title_font_size())
                
            def enter_split_mode(self, event):
                self.mode = self.SPLIT_PICK_FIRST_MODE
                self.set_orig_axes_title()
                self.figure.canvas.draw()
                
            def exit_split_mode(self, event):
                if self.mode == self.SPLIT_PICK_SECOND_MODE:
                    self.split_artist.remove()
                    self.split_artist = None
                    self.update_artists()
                self.mode = self.NORMAL_MODE
                self.set_orig_axes_title()
                self.figure.canvas.draw()
                
            def on_split_first_click(self, event):
                if event.inaxes != self.orig_axes:
                    return
                pick_artist, pick_index = self.get_control_point(event)
                if pick_artist is None:
                    return
                x, y = pick_artist.get_data()
                x, y = x[pick_index], y[pick_index]
                self.split_pick_artist = pick_artist
                self.split_pick_index = pick_index
                self.split_artist = Line2D(np.array((x, x)), 
                                           np.array((y, y)),
                                           color = "blue",
                                           animated = True)
                self.orig_axes.add_line(self.split_artist)
                self.mode = self.SPLIT_PICK_SECOND_MODE
                self.set_orig_axes_title()
                self.figure.canvas.draw()
                
            def handle_mouse_moved_pick_second_mode(self, event):
                if event.inaxes == self.orig_axes:
                    x, y = self.split_artist.get_data()
                    x[1] = event.xdata
                    y[1] = event.ydata
                    self.split_artist.set_data((x, y))
                    pick_artist, pick_index = self.get_control_point(event)
                    if pick_artist is not None and self.ok_to_split(
                        pick_artist, pick_index):
                        self.split_artist.set_color("red")
                    else:
                        self.split_artist.set_color("blue")
                    self.update_artists()
                    
            def ok_to_split(self, pick_artist, pick_index):
                if (self.artists[pick_artist][self.K_LABEL] != 
                    self.artists[self.split_pick_artist][self.K_LABEL]):
                    # Second must be same object as first.
                    return False
                if pick_artist == self.split_pick_artist:
                    min_index, max_index = [
                        fn(pick_index, self.split_pick_index)
                        for fn in (min, max)]
                    if max_index - min_index < 2:
                        # don't allow split of neighbors
                        return False
                    if (len(pick_artist.get_xdata()) - max_index <= 2 and
                        min_index == 0):
                        # don't allow split of last and first
                        return False
                elif (self.artists[pick_artist][self.K_OUTSIDE] ==
                      self.artists[self.split_pick_artist][self.K_OUTSIDE]):
                    # Only allow inter-object split of outside to inside
                    return False
                return True
                
            def on_split_second_click(self, event):
                if event.inaxes != self.orig_axes:
                    return
                pick_artist, pick_index = self.get_control_point(event)
                if pick_artist is None:
                    return
                if not self.ok_to_split(pick_artist, pick_index):
                    return
                if pick_artist == self.split_pick_artist:
                    #
                    # Create two new artists from the former artist.
                    #
                    is_outside = self.artists[pick_artist][self.K_OUTSIDE]
                    old_object_number = self.artists[pick_artist][self.K_LABEL]
                    xy = pick_artist.get_xydata()
                    idx0 = min(pick_index, self.split_pick_index)
                    idx1 = max(pick_index, self.split_pick_index)
                    if is_outside:
                        xy0 = np.vstack((xy[:(idx0+1), :],
                                         xy[idx1:, :]))
                        xy1 = np.vstack((xy[idx0:(idx1+1), :],
                                         xy[idx0:(idx0+1), :]))
                    else:
                        border_pts = np.zeros((2,2,2))
                            
                        border_pts[0, 0, :], border_pts[1, 1, :] = \
                            self.get_split_points(pick_artist, idx0)
                        border_pts[0, 1, :], border_pts[1, 0, :] = \
                            self.get_split_points(pick_artist, idx1)
                        xy0 = np.vstack((xy[:idx0, :],
                                         border_pts[:, 0, :],
                                         xy[(idx1+1):, :]))
                        xy1 = np.vstack((border_pts[:, 1, :],
                                         xy[(idx0+1):idx1, :],
                                         border_pts[:1, 1, :]))
                        
                    pick_artist.set_data((xy0[:, 0], xy0[:, 1]))
                    new_artist = Line2D(xy1[:, 0], xy1[:, 1],
                                        marker='o', markerfacecolor='r',
                                        markersize=6,
                                        color=self.colormap[old_object_number, :],
                                        animated = True)
                    self.orig_axes.add_line(new_artist)
                    if is_outside:
                        new_object_number = len(self.to_keep)
                        self.artists[new_artist] = { 
                            self.K_EDITED: True,
                            self.K_LABEL: new_object_number,
                            self.K_OUTSIDE: is_outside}
                        self.artists[pick_artist][self.K_EDITED] = True
                        temp = np.ones(self.to_keep.shape[0] + 1, bool)
                        temp[:-1] = self.to_keep
                        self.to_keep = temp
                        self.close_label(old_object_number, False)
                        self.close_label(new_object_number, False)
                        self.init_labels()
                        self.make_control_points(old_object_number)
                        self.make_control_points(new_object_number)
                        self.display()
                    else:
                        # Splitting a hole: the two parts are still in
                        # the same object.
                        self.artists[new_artist] = {
                            self.K_EDITED: True,
                            self.K_LABEL: old_object_number,
                            self.K_OUTSIDE: False }
                        self.update_artists()
                else:
                    #
                    # Join head and tail of different objects. The opposite
                    # winding means we don't have to reverse the array.
                    # We figure out which object is inside which and 
                    # combine them to form the outside artist.
                    #
                    xy0 = self.split_pick_artist.get_xydata()
                    xy1 = pick_artist.get_xydata()
                    #
                    # Determine who is inside who by area
                    #
                    a0 = self.get_area(self.split_pick_artist)
                    a1 = self.get_area(pick_artist)
                    if a0 > a1:
                        outside_artist = self.split_pick_artist
                        inside_artist = pick_artist
                        outside_index = self.split_pick_index
                        inside_index = pick_index
                    else:
                        outside_artist = pick_artist
                        inside_artist = self.split_pick_artist
                        outside_index = pick_index
                        inside_index = self.split_pick_index
                        xy0, xy1 = xy1, xy0
                    #
                    # We move the outside and inside points in order to make
                    # a gap. border_pts's first index is 0 for the outside
                    # point and 1 for the inside point. The second index
                    # is 0 for the point to be contributed first and
                    # 1 for the point to be contributed last. 
                    #
                    border_pts = np.zeros((2,2,2))
                        
                    border_pts[0, 0, :], border_pts[1, 1, :] = \
                        self.get_split_points(outside_artist, outside_index)
                    border_pts[0, 1, :], border_pts[1, 0, :] = \
                        self.get_split_points(inside_artist, inside_index)
                        
                    xy = np.vstack((xy0[:outside_index, :], 
                                    border_pts[:, 0, :],
                                    xy1[(inside_index+1):-1, :],
                                    xy1[:inside_index, :],
                                    border_pts[:, 1, :],
                                    xy0[(outside_index+1):, :]))
                    xy[-1, : ] = xy[0, :] # if outside_index == 0
                    
                    outside_artist.set_data((xy[:, 0], xy[:, 1]))
                    del self.artists[inside_artist]
                    inside_artist.remove()
                    object_number = self.artists[outside_artist][self.K_LABEL]
                    self.artists[outside_artist][self.K_EDITED] = True
                    self.close_label(object_number, display=False)
                    self.init_labels()
                    self.make_control_points(object_number)
                    self.display()
                self.record_undo()
                self.exit_split_mode(event)
                
            @staticmethod
            def get_area(artist):
                '''Get the area inside an artist polygon'''
                #
                # Thank you Darel Rex Finley:
                #
                # http://alienryderflex.com/polygon_area/
                #
                # Code is public domain
                #
                x, y = artist.get_data()
                area = abs(np.sum((x[:-1] + x[1:]) * (y[:-1] - y[1:]))) / 2
                return area
                
            @staticmethod
            def get_split_points(artist, idx):
                '''Return the split points on either side of the indexed point
                
                artist - artist in question
                idx - index of the point
                
                returns a point midway between the previous point and the
                point in question and a point midway between the next point
                and the point in question.
                '''
                a = artist.get_xydata().astype(float)
                if idx == 0:
                    idx_left = a.shape[0] - 2
                else:
                    idx_left = idx - 1
                if idx == a.shape[0] - 2:
                    idx_right = 0
                elif idx == a.shape[0] - 1:
                    idx_right = 1
                else:
                    idx_right = idx+1
                return ((a[idx_left, :] + a[idx, :]) / 2,
                        (a[idx_right, :] + a[idx, :]) / 2)
            
            ################################
            #
            # Freehand draw mode
            #
            ################################
            def enter_freehand_draw_mode(self, event):
                self.mode = self.FREEHAND_DRAW_MODE
                self.active_artist = None
                self.set_orig_axes_title()
                self.figure.canvas.draw()
                
            def exit_freehand_draw_mode(self, event):
                if self.active_artist is not None:
                    self.active_artist.remove()
                    self.active_artist = None
                self.mode = self.NORMAL_MODE
                self.set_orig_axes_title()
                self.figure.canvas.draw()
                
            def on_freehand_draw_click(self, event):
                '''Begin drawing on mouse-down'''
                self.active_artist = Line2D([ event.xdata], [event.ydata],
                                            color = "blue",
                                            animated = True)
                self.orig_axes.add_line(self.active_artist)
                self.update_artists()
                
            def handle_mouse_moved_freehand_draw_mode(self, event):
                if event.inaxes != self.orig_axes:
                    return
                if self.active_artist is not None:
                    xdata, ydata = self.active_artist.get_data()
                    self.active_artist.set_data(
                        np.hstack((xdata, [event.xdata])),
                        np.hstack((ydata, [event.ydata])))
                    self.update_artists()
            
            def on_mouse_button_up_freehand_draw_mode(self, event):
                xydata = self.active_artist.get_xydata()
                if event.inaxes == self.orig_axes:
                    xydata = np.vstack((
                        xydata,
                        np.array([[event.xdata, event.ydata]])))
                xydata = np.vstack((
                    xydata,
                    np.array([[xydata[0, 0], xydata[0, 1]]])))
                
                mask = polygon_lines_to_mask(xydata[:-1, 1],
                                             xydata[:-1, 0],
                                             xydata[1:, 1],
                                             xydata[1:, 0],
                                             self.shape)
                self.add_label(mask)
                self.exit_freehand_draw_mode(event)
                self.init_labels()
                self.display()
                self.record_undo()
            
            ################################
            #
            # Functions for keep / remove/ toggle
            #
            ################################
    
            def on_keep(self, event):
                self.to_keep[1:] = True
                self.display()
            
            def on_remove(self, event):
                self.to_keep[1:] = False
                self.display()
            
            def on_toggle(self, event):
                self.to_keep[1:] = ~ self.to_keep[1:]
                self.display()
                
            def on_reset(self, event):
                self.reset()
                
            def reset(self, display=True):
                self.labels = [l.copy() for l in self.orig_labels]
                nlabels = np.max([np.max(l) for l in orig_labels])
                self.to_keep = np.ones(nlabels + 1, bool)
                self.artists = {}
                self.undo_stack = []
                if hasattr(self, "undo_button"):
                    # minor unfortunate hack - reset called before GUI is built
                    self.undo_button.Enable(False)
                self.last_ijv = self.calculate_ijv()
                self.last_artist_save = {}
                if display:
                    self.init_labels()
                    self.display()
                
            def on_help(self, event):
                self.html_frame.Show(True)
                
            def on_help_close(self, event):
                event.Veto()
                self.html_frame.Show(False)
                
            def make_control_points(self, object_number):
                '''Create an artist with control points for editing an object
                
                object_number - # of object to edit
                '''
                #
                # For outside edges, we trace clockwise, conceptually standing 
                # to the left of the outline and putting our right hand on the
                # outline. Inside edges have the opposite winding.
                # We remember the direction we are going and that gives
                # us an order for the points. For instance, if we are going
                # north:
                #
                #  2  3  4
                #  1  x  5
                #  0  7  6
                #
                # If "1" is available, our new direction is southwest:
                #
                #  5  6  7
                #  4  x  0
                #  3  2  1
                #
                #  Take direction 0 to be northeast (i-1, j-1). We look in
                #  this order:
                #
                #  3  4  5
                #  2  x  6
                #  1  0  7
                #
                # The directions are
                #
                #  0  1  2
                #  7     3
                #  6  5  4
                #
                traversal_order = np.array(
                    #   i   j   new direction
                    ((  1,  0,  5 ),
                     (  1, -1,  6 ),
                     (  0, -1,  7 ),
                     ( -1, -1,  0 ),
                     ( -1,  0,  1 ),
                     ( -1,  1,  2 ),
                     (  0,  1,  3 ),
                     (  1,  1,  4 )))
                direction, index, ijd = np.mgrid[0:8, 0:8, 0:3]
                traversal_order = \
                    traversal_order[((direction + index) % 8), ijd]
                #
                # We need to make outlines of both objects and holes.
                # Objects are 8-connected and holes are 4-connected
                #
                for polarity, structure in (
                    (True, np.ones((3,3), bool)),
                    (False, np.array([[0, 1, 0], 
                                      [1, 1, 1], 
                                      [0, 1, 0]], bool))):
                    #
                    # Pad the mask so we don't have to deal with out of bounds
                    #
                    mask = np.zeros((self.shape[0] + 2,
                                     self.shape[1] + 2), bool)
                    for l in self.labels:
                        mask[1:-1, 1:-1] |= l == object_number
                    if not polarity:
                        mask = ~mask
                    labels, count = scipy.ndimage.label(mask, structure)
                    if not polarity:
                        #
                        # The object touching the border is not a hole.
                        # There should only be one because of the padding.
                        #
                        border_object = labels[0,0]
                    for sub_object_number in range(1, count+1):
                        if not polarity and sub_object_number == border_object:
                            continue
                        mask = labels == sub_object_number
                        i, j = np.mgrid[0:mask.shape[0], 0:mask.shape[1]]
                        i, j = i[mask], j[mask]
                        if len(i) < 2:
                            continue
                        topleft = np.argmin(i*i+j*j)
                        chain = []
                        start_i = i[topleft]
                        start_j = j[topleft]
                        #
                        # Pick a direction that points normal and to the right
                        # from the point at the top left.
                        #
                        direction = 2
                        ic = start_i
                        jc = start_j
                        while True:
                            chain.append((ic - 1, jc - 1))
                            hits = mask[ic + traversal_order[direction, :, 0],
                                        jc + traversal_order[direction, :, 1]]
                            t = traversal_order[direction, hits, :][0, :]
                            ic += t[0]
                            jc += t[1]
                            direction = t[2]
                            if ic == start_i and jc == start_j:
                                if len(chain) > 40:
                                    markevery = min(10, int((len(chain)+ 19) / 20))
                                    chain = chain[::markevery]
                                chain.append((ic - 1, jc - 1))
                                if not polarity:
                                    # Reverse the winding order
                                    chain = chain[::-1]
                                break
                        chain = np.array(chain)
                        artist = Line2D(chain[:, 1], chain[:, 0],
                                        marker='o', markerfacecolor='r',
                                        markersize=6,
                                        color=self.colormap[object_number, :],
                                        animated = True)
                        self.orig_axes.add_line(artist)
                        self.artists[artist] = { 
                            self.K_LABEL: object_number, 
                            self.K_EDITED: False,
                            self.K_OUTSIDE: polarity}
                self.update_artists()
            
            def close_label(self, label, display = True):
                '''Close the artists associated with a label
                
                label - label # of label being closed.
                
                If edited, update the labeled pixels.
                '''
                my_artists = [artist for artist, data in self.artists.items()
                              if data[self.K_LABEL] == label]
                if any([self.artists[artist][self.K_EDITED] 
                        for artist in my_artists]):
                    #
                    # Convert polygons to labels. The assumption is that
                    # a polygon within a polygon is a hole.
                    #
                    mask = np.zeros(self.shape, bool)
                    for artist in my_artists:
                        j, i = artist.get_data()
                        m1 = polygon_lines_to_mask(i[:-1], j[:-1],
                                                   i[1:], j[1:],
                                                   self.shape)
                        mask[m1] = ~mask[m1]
                    for artist in my_artists:
                        artist.remove()
                        del self.artists[artist]
                    self.replace_label(mask, label)
                    if display:
                        self.init_labels()
                        self.display()
                    
                else:
                    for artist in my_artists:
                        artist.remove()
                        del self.artists[artist]
                    if display:
                        self.display()
        
        with FilterObjectsDialog(self, guide_image, orig_labels) as dialog_box:
            result = dialog_box.ShowModal()
        if result != wx.OK:
            raise RuntimeError("User cancelled EditObjectsManually")
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
