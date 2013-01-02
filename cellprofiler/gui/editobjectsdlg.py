'''editobjectsdlg.py - a dialog box that lets the user edit objects

'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Copyright (c) 2003-2009 Massachusetts Institute of Technology
# Copyright (c) 2009-2013 Broad Institute
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
import matplotlib
import matplotlib.figure
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg, NavigationToolbar2WxAgg
import numpy as np
import scipy.ndimage
from scipy.ndimage import gaussian_filter, binary_dilation, grey_dilation
import sys
import wx
import wx.html

import cellprofiler.objects as cpo
import cellprofiler.preferences as cpprefs
from cellprofiler.cpmath.outline import outline
from cellprofiler.cpmath.cpmorphology import triangle_areas, distance2_to_line
from cellprofiler.cpmath.cpmorphology import polygon_lines_to_mask
from cellprofiler.cpmath.cpmorphology import get_outline_pts, thicken
from cellprofiler.cpmath.index import Indexes
from cellprofiler.gui.cpfigure_tools import renumber_labels_for_display

class EditObjectsDialog(wx.Dialog):
    '''This dialog can be invoked as an objects editor
    
    EditObjectsDialog takes an optional labels matrix and guide image. If
    no labels matrix is provided, initially, there are no objects. If there
    is no guide image, a black background is displayed.
    
    The resutls of EditObjectsDialog are available in the "labels" attribute
    if the return code is wx.OK.
    '''
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
    def __init__(self, guide_image, orig_labels, allow_overlap, title=None):
        '''Initializer
        
        guide_image - a grayscale or color image to display behind the labels
        
        orig_labels - a sequence of label matrices, such as is available from
                      Objects.get_labels()
                      
        allow_overlap - true to allow objects to overlap
        
        title - title to appear on top of the editing axes
        '''
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
        self.allow_overlap = allow_overlap
        self.title = title
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
        self.orig_axes = self.figure.add_subplot(1, 1, 1)
        self.orig_axes.set_zorder(1) # preferentially select on click.
        self.orig_axes._adjustable = 'box-forced'
        self.orig_axes.set_title(
            self.title,
            fontname=cpprefs.get_title_font_name(),
            fontsize=cpprefs.get_title_font_size())
    
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
        self.Bind(wx.EVT_BUTTON, on_cancel, cancel_button)
        button_sizer.SetNegativeButton(cancel_button)
        button_sizer.AddButton(wx.Button(self, wx.ID_HELP))
        self.Bind(wx.EVT_BUTTON, self.on_help, id= wx.ID_HELP)
        self.Bind(wx.EVT_CLOSE, self.on_close)
                          
        button_sizer.Realize()
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
            idxs = np.unique(label)
            idxs = idxs[idxs!=0]
            distinct_label_count = len(idxs)
            clabels = renumber_labels_for_display(label)
            clabels[clabels != 0] += lstart
            lstart += distinct_label_count
            label_map[label.flatten()] = clabels.flatten()
            if False:
                outlines = outline(clabels)
                oi, oj = np.argwhere(outlines != 0).transpose()
            else:
                l, ct = scipy.ndimage.label(label != 0, 
                                            structure=np.ones((3,3), bool))
                coords, offsets, counts = get_outline_pts(l, np.arange(1, ct+1))
                oi, oj = coords.transpose()
                l, ct = scipy.ndimage.label(label == 0) # 4-connected
                #
                # Have to remove the label that touches the edge, if any
                #
                ledge = np.hstack([l[0, :][label[0, :] == 0],
                                   l[-1, :][label[-1, :] == 0],
                                   l[:, 0][label[:, 0] == 0],
                                   l[:, -1][label[:, -1] == 0]])
                if len(ledge) > 0:
                    l[l == ledge[0]] = 0

                coords, offsets, counts = get_outline_pts(l, np.arange(1, ct+1))
                if coords.shape[0] > 0:
                    oi, oj = [np.hstack((o, coords[:,i]))
                              for i, o in enumerate((oi, oj))]
                
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
            for idx in np.argwhere(~self.to_keep).flatten():
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
        orig_objects_name = self.title
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
        self.orig_axes.clear()
        if self.guide_image is not None:
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
        if len(self.to_keep) > 1:
            i, j = np.mgrid[0:self.shape[0], 0:self.shape[1]]
            for k, stipple in ((self.to_keep, False), (~self.to_keep, True)):
                k = k.copy()
                # Don't show outlines for labels being edited
                for d in self.artists.values():
                    k[d[self.K_LABEL]] = False
                if not np.any(k):
                    continue
                mask = k[self.ol]
                intensity = np.zeros(self.shape, float)
                intensity[self.oi[mask], self.oj[mask]] = 1
                color = np.zeros((self.shape[0], self.shape[1], 3), float)
                if stipple:
                    # Make dashed outlines by throwing away the first 4
                    # border pixels and keeping the next 4. This also makes
                    # small objects disappear when clicked-on.
                    lmap = np.zeros(len(k), int)
                    lmap[k] = np.arange(np.sum(k))
                    counts = np.bincount(lmap[self.ol[mask]])
                    indexer = Indexes((counts,))
                    e = 1 + 3 * (counts[indexer.rev_idx] >= 16)
                    dash_mask = (indexer.idx[0] & (2**e - 1)) >= 2**(e-1)
                    color[self.oi[mask], self.oj[mask]] = \
                        self.oc[mask] * dash_mask[:, np.newaxis]
                else:
                    color[self.oi[mask], self.oj[mask]] = self.oc[mask]
                sigma = 1
                intensity = gaussian_filter(intensity, sigma)
                eps = intensity > np.finfo(intensity.dtype).eps
                color = gaussian_filter(color, (sigma, sigma,0))[eps, :]
                intensity = intensity[eps]
                cimage[eps, :] = \
                    cimage[eps, :] * (1 - intensity[:, np.newaxis]) + color
                
        self.orig_axes.imshow(cimage)
        self.set_orig_axes_title()
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
        if self.background == None:
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
        if event.inaxes != self.orig_axes:
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
        x = int(event.xdata + .5)
        y = int(event.ydata + .5)
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
            self.display()
        elif event.button == 3:
            self.make_control_points(lnum)
            self.display()
    
    def on_key_down(self, event):
        self.pressed_keys.add(event.key)
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
        ydata, xdata = [
            min(self.shape[i]-1, max(yx, 0)) 
            for i, yx in enumerate((event.ydata, event.xdata))]
        new_pt = np.array([xdata, ydata], int)
        path = Path(np.array((before_pt, new_pt, after_pt)))
        eps = np.finfo(np.float32).eps
        for artist in self.artists:
            if (self.allow_overlap and 
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
        data[0][self.active_index] = xdata
        data[1][self.active_index] = ydata
        
        #
        # Handle moving the first point which is the
        # same as the last and they need to be moved together.
        # The last should never be moved.
        #
        if self.active_index == 0:
            data[0][-1] = xdata
            data[1][-1] = ydata
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
            title = self.title
                                        
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
                #
                # Find all points within holes in the old object
                #
                hmask = np.zeros(self.shape, bool)
                for artist, attrs in list(self.artists.items()):
                    if (not attrs[self.K_OUTSIDE] and
                        attrs[self.K_LABEL] == old_object_number):
                        hx, hy = artist.get_data()
                        hmask = hmask | polygon_lines_to_mask(
                            hy[:-1], hx[:-1], hy[1:], hx[1:], 
                            self.shape)
                temp = np.ones(self.to_keep.shape[0] + 1, bool)
                temp[:-1] = self.to_keep
                self.to_keep = temp
                self.close_label(old_object_number, False)
                self.close_label(new_object_number, False)
                #
                # Remove hole points from both objects
                #
                for label in self.labels:
                    label[hmask & ((label == old_object_number) |
                                    (label == new_object_number))] = 0
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
            ydata, xdata = [
                np.minimum(self.shape[i]-1, np.maximum(yx, 0)) 
                for i, yx in enumerate((ydata, xdata))]
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
        nlabels = np.max([np.max(l) for l in self.orig_labels])
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
                border_object = labels[0,0]
                labels[labels == labels[0,0]] = 0
            sub_object_numbers = [
                n for n in range(1, count+1)
                if polarity or n != border_object]
            coords, offsets, counts = get_outline_pts(labels, sub_object_numbers)
            for i, sub_object_number in enumerate(sub_object_numbers):
                chain = coords[offsets[i]:(offsets[i] + counts[i]), :]
                if not polarity:
                    chain = chain[::-1]
                chain = np.vstack((chain, chain[:1, :])).astype(float)
                #
                # Start with the first point and a midpoint in the
                # chain and keep adding points until the maximum
                # error caused by leaving a point out is 4
                #
                minarea = 10
                if len(chain) > 10:
                    accepted = np.zeros(len(chain), bool)
                    accepted[0] = True
                    accepted[-1] = True
                    accepted[int(len(chain)/2)] = True
                    while True:
                        idx1 = np.cumsum(accepted[:-1])
                        idx0 = idx1 - 1
                        ca = chain[accepted]
                        aidx = np.argwhere(accepted).flatten()
                        a = triangle_areas(ca[idx0],
                                           ca[idx1],
                                           chain[:-1])
                        idxmax = np.argmax(a)
                        if a[idxmax] < 4:
                            break
                        # Pick a point halfway in-between
                        idx = int((aidx[idx0[idxmax]] + 
                                   aidx[idx1[idxmax]]) / 2)
                        accepted[idx] = True
                    chain = chain[accepted]
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

if __name__== "__main__":
    import libtiff
    
    f = libtiff.TIFFfile(sys.argv[1])
    a = f.get_tiff_array(0)
    labels = [np.array(a[i, :, :], int) for i in range(a.shape[0])]
    f.close()
    f = libtiff.TIFFfile(sys.argv[2])
    a = f.get_tiff_array(0)
    if a.shape[0] == 1:
        img = np.array(a[0, :, :], float)
    else:
        img = np.array(a[:, :, :], float)
    img = img / np.max(img)
    f.close()
    app = wx.PySimpleApp(True)
    dlg = EditObjectsDialog(img, labels, True, "Hello, world")
    dlg.ShowModal()