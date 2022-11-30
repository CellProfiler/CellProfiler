# coding=utf-8
"""editobjectsdlg.py - a dialog box that lets the user edit objects

"""

import logging
import sys

import centrosome.cpmorphology
import centrosome.index
import matplotlib
import matplotlib.backend_bases
import matplotlib.backends.backend_wxagg
import matplotlib.cm
import matplotlib.figure
import matplotlib.lines
import matplotlib.path
import numpy
import scipy.ndimage
import scipy.sparse
import wx
import wx.adv
import wx.html
from cellprofiler_core.object import Objects
from cellprofiler_core.preferences import (
    get_interpolation_mode,
    get_title_font_name,
    get_default_colormap,
    IM_NEAREST,
    IM_BILINEAR,
    IM_BICUBIC,
    get_title_font_size,
)
from cellprofiler_core.utilities.core.object import size_similarly

import cellprofiler.gui.figure
from cellprofiler.gui.figure import NavigationToolbar
import cellprofiler.gui.tools

LOGGER = logging.getLogger(__name__)


class EditObjectsDialog(wx.Dialog):
    """This dialog can be invoked as an objects editor

    EditObjectsDialog takes an optional labels matrix and guide image. If
    no labels matrix is provided, initially, there are no objects. If there
    is no guide image, a black background is displayed.

    The results of EditObjectsDialog are available in the "labels" attribute
    if the return code is wx.OK.
    """

    resume_id = wx.NewId()
    cancel_id = wx.NewId()
    epsilon = 5  # maximum pixel distance to a vertex for hit test
    FREEHAND_DRAW_MODE = "freehanddrawmode"
    SPLIT_PICK_FIRST_MODE = "split1"
    SPLIT_PICK_SECOND_MODE = "split2"
    NORMAL_MODE = "normal"
    DELETE_MODE = "delete"
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
    #
    # Intensity scaling mode
    #
    SM_RAW = "Raw"
    SM_NORMALIZED = "Normalized"
    SM_LOG_NORMALIZED = "Log normalized"
    #
    # Show image / hide image button labels
    #
    D_SHOW_IMAGE = "Show image"
    D_HIDE_IMAGE = "Hide image"

    ID_CONTRAST_RAW = wx.NewId()
    ID_CONTRAST_NORMALIZED = wx.NewId()
    ID_CONTRAST_LOG_NORMALIZED = wx.NewId()

    ID_INTERPOLATION_NEAREST = wx.NewId()
    ID_INTERPOLATION_BILINEAR = wx.NewId()
    ID_INTERPOLATION_BICUBIC = wx.NewId()

    ID_LABELS_OUTLINES = wx.NewId()
    ID_LABELS_FILL = wx.NewId()

    ID_DISPLAY_IMAGE = wx.NewId()

    ID_MODE_FREEHAND = wx.NewId()
    ID_MODE_SPLIT = wx.NewId()
    ID_MODE_DELETE = wx.NewId()

    ID_ACTION_KEEP = wx.NewId()
    ID_ACTION_REMOVE = wx.NewId()
    ID_ACTION_TOGGLE = wx.NewId()
    ID_ACTION_RESET = wx.NewId()
    ID_ACTION_JOIN = wx.NewId()
    ID_ACTION_CONVEX_HULL = wx.NewId()
    ID_ACTION_DELETE = wx.NewId()

    def __init__(self, guide_image, orig_labels, allow_overlap, title=None):
        """Initializer

        guide_image - a grayscale or color image to display behind the labels

        orig_labels - a sequence of label matrices, such as is available from
                      Objects.get_labels()

        allow_overlap - true to allow objects to overlap

        title - title to appear on top of the editing axes
        """
        #
        # Get the labels matrix and make a mask of objects to keep from it
        #
        #
        # Display a UI for choosing objects
        #
        frame_size = wx.GetDisplaySize()
        frame_size = [max(frame_size[0], frame_size[1]) / 2] * 2
        style = wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.MAXIMIZE_BOX
        wx.Dialog.__init__(self, None, -1, "", size=frame_size, style=style)
        self.allow_overlap = allow_overlap
        self.title = title
        self.guide_image = guide_image
        self.orig_labels = orig_labels
        self.shape = self.orig_labels[0].shape
        self.background = None  # background = None if full repaint needed
        self.reset(display=False)
        self.active_artist = None
        self.active_index = None
        self.mode = self.NORMAL_MODE
        self.scaling_mode = self.SM_NORMALIZED
        self.interpolation_mode = get_interpolation_mode()
        self.label_display_mode = self.ID_LABELS_OUTLINES
        self.skip_right_button_up = False
        self.split_artist = None
        self.delete_mode_artist = None
        self.delete_mode_rect_artist = None
        self.wants_image_display = guide_image is not None
        self.pressed_keys = set()
        self.build_ui()
        self.init_labels()
        self.display()
        self.Layout()
        self.layout_sash()
        self.Raise()
        self.panel.SetFocus()

    def record_undo(self):
        """Push an undo record onto the undo stack"""
        #
        # The undo record is a diff between the last ijv and
        # the current, plus the current state of the artists.
        #
        ijv = self.calculate_ijv()
        if ijv.shape[0] == 0:
            ijvx = numpy.zeros((0, 4), int)
        else:
            #
            # Sort the current and last ijv together, adding
            # an old_new_indicator.
            #
            ijvx = numpy.vstack(
                (
                    numpy.column_stack((ijv, numpy.zeros(ijv.shape[0], ijv.dtype))),
                    numpy.column_stack(
                        (self.last_ijv, numpy.ones(self.last_ijv.shape[0], ijv.dtype))
                    ),
                )
            )
            order = numpy.lexsort((ijvx[:, 3], ijvx[:, 2], ijvx[:, 1], ijvx[:, 0]))
            ijvx = ijvx[order, :]
            #
            # Then mark all prev and next where i,j,v match (in both sets)
            #
            matches = numpy.hstack(
                (
                    (
                        numpy.all(ijvx[:-1, :3] == ijvx[1:, :3], 1)
                        & (ijvx[:-1, 3] == 0)
                        & (ijvx[1:, 3] == 1)
                    ),
                    [False],
                )
            )
            matches[1:] = matches[1:] | matches[:-1]
            ijvx = ijvx[~matches, :]
        artist_save = [(a.get_data(), self.artists[a].copy()) for a in self.artists]
        self.undo_stack.append((ijvx, self.last_artist_save, self.last_to_keep))
        self.last_artist_save = artist_save
        self.last_ijv = ijv
        self.last_to_keep = self.to_keep
        self.undo_button.Enable(True)

    def undo(self, event=None):
        """Pop an entry from the undo stack and apply"""
        #
        # Mix what's on the undo ijv with what's in self.last_ijv
        # and remove any 0/1 pairs.
        #
        ijvx, artist_save, self.to_keep = self.undo_stack.pop()
        ijvx = numpy.vstack(
            (
                ijvx,
                numpy.column_stack(
                    (
                        self.last_ijv,
                        numpy.ones(self.last_ijv.shape[0], self.last_ijv.dtype),
                    )
                ),
            )
        )
        order = numpy.lexsort((ijvx[:, 3], ijvx[:, 2], ijvx[:, 1], ijvx[:, 0]))
        ijvx = ijvx[order, :]
        #
        # Then mark all prev and next where i,j,v match (in both sets)
        #
        matches = numpy.hstack((numpy.all(ijvx[:-1, :3] == ijvx[1:, :3], 1), [False]))
        matches[1:] = matches[1:] | matches[:-1]
        ijvx = ijvx[~matches, :]
        self.last_ijv = ijvx[:, :3]
        self.last_artist_save = artist_save
        self.last_to_keep = self.to_keep
        temp = Objects()
        temp.set_ijv(self.last_ijv, shape=self.shape)
        self.labels = [l for l, c in temp.get_labels()]
        self.init_labels()
        #
        # replace the artists
        #
        for artist in self.artists:
            artist.remove()
        self.artists = {}
        for (x, y), d in artist_save:
            object_number = d[self.K_LABEL]
            artist = matplotlib.lines.Line2D(
                x,
                y,
                marker="o",
                markerfacecolor="r",
                markersize=6,
                color=self.colormap[object_number, :],
                animated=True,
            )
            self.artists[artist] = d
            self.orig_axes.add_line(artist)
        self.display()
        if len(self.undo_stack) == 0:
            self.undo_button.Enable(False)

    def calculate_ijv(self):
        """Return the current IJV representation of the labels"""
        i, j = numpy.mgrid[0 : self.shape[0], 0 : self.shape[1]]
        ijv = numpy.zeros((0, 3), int)
        for l in self.labels:
            ijv = numpy.vstack(
                (ijv, numpy.column_stack([i[l != 0], j[l != 0], l[l != 0]]))
            )
        return ijv

    def build_ui(self):
        sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(sizer)

        self.figure = matplotlib.figure.Figure()
        #
        # The following is a patch that tells the draw event handler not
        # to do anything during printing (causes segfault on Windows)
        #
        self.inside_print = False

        class CanvasPatch(matplotlib.backends.backend_wxagg.FigureCanvasWxAgg):
            def __init__(self, parent, id, figure):
                matplotlib.backends.backend_wxagg.FigureCanvasWxAgg.__init__(
                    self, parent, id, figure
                )

            def print_figure(self, *args, **kwargs):
                self.Parent.inside_print = True
                try:
                    super(CanvasPatch, self).print_figure(*args, **kwargs)
                finally:
                    self.Parent.inside_print = False

        self.panel = CanvasPatch(self, -1, self.figure)
        self.toolbar = NavigationToolbar(self.panel)
        self.toolbar.set_message = self.discard_message
        self.sash_parent = wx.Panel(self)
        #
        # Need to reparent the canvas after instantiating the toolbar so
        # the toolbar gets parented to the frame and the panel gets
        # parented to some window that can be shared with the *^$# sash
        # window.
        #
        self.panel.Reparent(self.sash_parent)
        sizer.Add(self.toolbar, 0, wx.EXPAND)
        sizer.Add(self.sash_parent, 1, wx.EXPAND)
        #
        # Make 3 axes
        #
        self.orig_axes = self.figure.add_subplot(1, 1, 1)
        self.orig_axes.set_zorder(1)  # preferentially select on click.
        self.orig_axes._adjustable = "box"
        self.orig_axes.set_title(
            self.title, fontname=get_title_font_name(), fontsize=get_title_font_size(),
        )

        ########################################
        #
        # The help sash
        #
        ########################################
        self.help_sash = wx.adv.SashLayoutWindow(self.sash_parent)
        self.help_sash.Bind(wx.adv.EVT_SASH_DRAGGED, self.on_help_sash_drag)
        self.help_sash.SetOrientation(wx.adv.LAYOUT_HORIZONTAL)
        self.help_sash.SetAlignment(wx.adv.LAYOUT_BOTTOM)
        self.help_sash.SetDefaultBorderSize(4)
        self.help_sash.SetSashVisible(wx.adv.SASH_TOP, True)
        self.html_panel = wx.html.HtmlWindow(self.help_sash)
        if sys.platform == "darwin":
            LEFT_MOUSE = "mouse"
            LEFT_MOUSE_BUTTON = "mouse button"
            RIGHT_MOUSE = "[control] + mouse"
        else:
            LEFT_MOUSE = "left mouse button"
            LEFT_MOUSE_BUTTON = LEFT_MOUSE
            RIGHT_MOUSE = "right mouse button"
        self.html_panel.SetPage(
            """<h1>Editing help</h1>
                <p>The editing user interface lets you create, remove and
                edit objects. The interface will show the image that you selected
                as the guiding image, overlaid with colored outlines of the selected
                objects. Buttons are available at the bottom of the panel
                for various commands. A number of operations are available with
                the mouse:
                <ul>
                <li><i>Remove an object:</i> Click on the object
                with the %(LEFT_MOUSE)s. The colored object outline will switch from
                a solid line to a dashed line.</li>
                <li><i>Restore a removed object:</i> Click on the object that was previously removed.
                The colored object outline will switch from
                a dashed line back to a solid line. </li>
                <li><i>Edit objects:</i> Select the object
                with the %(RIGHT_MOUSE)s to edit it. The outline will switch
                from a colored outline to a series of points (the <i>control points</i>)
                along the object boundary connected by solid lines. You can place any
                number of objects into this mode for editing simultaneously; the control
                point outline will indicate which objects have been selected.
                Move object control points
                by dragging them while holding the %(LEFT_MOUSE_BUTTON)s
                down. Please note the following editing limitations:
                <ul>
                <li>You cannot move a control point across the interior boundary
                of the object you are editing.</li>
                <li>You cannot move the
                edges on either side of a control point across another control point.</li>
                </ul>
                <li><i>Finish editing an object:</i> Click on the object again with
                the %(RIGHT_MOUSE)s to save changes.</li>
                <li><i>Abandon changes to an object:</i> Hit the <i>Esc</i> key.</li>
                </ul>
                You can always reset your edits to the original state
                before editing by pressing the <i>Reset</i> key.</p>

                <p>Once finished editing all your desired objects, press the <i>Done</i> button to
                save your edits. At that point, the editing user interface will be replaced
                by the module display window showing the original and the edited object set.</p>

                <h2>Editing commands</h2>
                The following keys perform editing
                commands when pressed (the keys are not case-sensitive):
                <ul>
                <li><b>A</b>: Add a control point to the line nearest the
                mouse cursor</li>
                <li><b>C</b>: Join all selected objects into one that forms a
                convex hull around them all. The convex hull is the smallest
                shape that has no indentations and encloses all of the
                objects. You can use this to combine several pieces into
                one round object.</li>
                <li><b>D</b>: Delete the control point nearest to the
                cursor.</li>
                <li><b>F</b>: Freehand draw. Press down on the %(LEFT_MOUSE)s
                to draw a new object outline, then release to complete
                the outline and return to normal editing.</li>
                <li><b>J</b>: Join all the selected objects into one object.</li>
                <li><b>M</b>: Display the context menu to select a command.</li>
                <li><b>N</b>: Create a new object under the cursor. A new set
                of control points is produced which you can then start
                manipulating with the mouse.</li>
                <li><b>S</b>: Split an object. Pressing <b>S</b> puts
                the user interface into <i>Split Mode</i>. The user interface
                will prompt you to select a first and second point for the
                split. Two types of splits are allowed:
                <ul>
                <li>A split between two points on the same contour. This split
                creates two separate objects.</li>
                <li>A split between the inside and the outside of an object that
                has a hole in it. This split creates a channel from the hole
                to the outside of the object.</li>
                </ul>
                <li><b>T</b>: Toggle between fill and outline mode for objects.</li>
                <li><b>X</b>: Delete mode. Press down on the %(LEFT_MOUSE)s to
                start defining the delete region. Drag to define a rectangle. All
                control points within the rectangle (shown as white circles) will be
                deleted when you release the %(LEFT_MOUSE)s. Press the escape key
                to cancel.
                </li>
                </ul>
                <p><b>Note:</b> Editing is disabled in zoom or pan mode. The
                zoom or pan button on the navigation toolbar is depressed
                during this mode and your cursor is no longer an arrow.
                You can exit from zoom or pan mode by pressing the
                appropriate button on the navigation toolbar.</p>
                """
            % locals()
        )
        self.help_sash.Hide()

        ###################################################
        #
        # The buttons on the bottom
        #
        ###################################################
        sub_sizer = wx.WrapSizer(wx.HORIZONTAL)
        #
        # Need padding on top because tool bar is wonky about its height
        #
        sizer.Add(sub_sizer, 0, wx.ALIGN_CENTER)

        #########################################
        #
        # Buttons for keep / remove / toggle
        #
        #########################################

        keep_button = wx.Button(self, self.ID_ACTION_KEEP, "Keep all")
        sub_sizer.Add(keep_button, 0, wx.ALIGN_CENTER)

        remove_button = wx.Button(self, self.ID_ACTION_REMOVE, "Remove all")
        sub_sizer.Add(remove_button, 0, wx.ALIGN_CENTER)

        toggle_button = wx.Button(self, self.ID_ACTION_TOGGLE, "Reverse selection")
        sub_sizer.Add(toggle_button, 0, wx.ALIGN_CENTER)
        self.undo_button = wx.Button(self, wx.ID_UNDO)
        self.undo_button.SetToolTip("Undo last edit")
        self.undo_button.Enable(False)
        sub_sizer.Add(self.undo_button)
        reset_button = wx.Button(self, -1, "Reset")
        reset_button.SetToolTip("Undo all editing and restore the original objects")
        sub_sizer.Add(reset_button)
        self.display_image_button = wx.Button(
            self, self.ID_DISPLAY_IMAGE, label=self.D_HIDE_IMAGE
        )
        if self.guide_image is None:
            self.display_image_button.Show(False)
        else:
            sub_sizer.Add(self.display_image_button)
        self.help_button = wx.Button(self, wx.ID_HELP, label="Show Help")
        sub_sizer.Add(self.help_button)
        self.Bind(wx.EVT_BUTTON, self.on_toggle, toggle_button)
        self.Bind(wx.EVT_MENU, self.on_toggle, id=self.ID_ACTION_TOGGLE)
        self.Bind(wx.EVT_BUTTON, self.on_keep, keep_button)
        self.Bind(wx.EVT_MENU, self.on_keep, id=self.ID_ACTION_KEEP)
        self.Bind(wx.EVT_BUTTON, self.on_remove, remove_button)
        self.Bind(wx.EVT_MENU, self.on_remove, id=self.ID_ACTION_REMOVE)
        self.Bind(wx.EVT_BUTTON, self.undo, id=wx.ID_UNDO)
        self.Bind(wx.EVT_BUTTON, self.on_reset, reset_button)
        self.display_image_button.Bind(wx.EVT_BUTTON, self.on_display_image)
        self.help_button.Bind(wx.EVT_BUTTON, self.on_help)
        self.Bind(wx.EVT_MENU, self.on_reset, id=self.ID_ACTION_RESET)
        self.Bind(wx.EVT_MENU, self.join_objects, id=self.ID_ACTION_JOIN)
        self.Bind(wx.EVT_MENU, self.convex_hull, id=self.ID_ACTION_CONVEX_HULL)
        self.Bind(wx.EVT_MENU, self.on_display_image, id=self.ID_DISPLAY_IMAGE)
        self.Bind(wx.EVT_MENU, self.on_raw_contrast, id=self.ID_CONTRAST_RAW)
        self.Bind(
            wx.EVT_MENU, self.on_normalized_contrast, id=self.ID_CONTRAST_NORMALIZED
        )
        self.Bind(
            wx.EVT_MENU,
            self.on_log_normalized_contrast,
            id=self.ID_CONTRAST_LOG_NORMALIZED,
        )
        self.Bind(
            wx.EVT_MENU,
            self.on_nearest_neighbor_interpolation,
            id=self.ID_INTERPOLATION_NEAREST,
        )
        self.Bind(
            wx.EVT_MENU,
            self.on_bilinear_interpolation,
            id=self.ID_INTERPOLATION_BILINEAR,
        )
        self.Bind(
            wx.EVT_MENU, self.on_bicubic_interpolation, id=self.ID_INTERPOLATION_BICUBIC
        )
        self.Bind(wx.EVT_MENU, self.enter_freehand_draw_mode, id=self.ID_MODE_FREEHAND)
        self.Bind(wx.EVT_MENU, self.enter_split_mode, id=self.ID_MODE_SPLIT)
        self.Bind(wx.EVT_MENU, self.enter_delete_mode, id=self.ID_MODE_DELETE)
        self.Bind(
            wx.EVT_MENU,
            (lambda event: self.set_label_display_mode(self.ID_LABELS_OUTLINES)),
            id=self.ID_LABELS_OUTLINES,
        )
        self.Bind(
            wx.EVT_MENU,
            (lambda event: self.set_label_display_mode(self.ID_LABELS_FILL)),
            id=self.ID_LABELS_FILL,
        )
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
            self.on_close(event, wx.OK)

        self.Bind(wx.EVT_BUTTON, on_resume, resume_button)
        button_sizer.SetAffirmativeButton(resume_button)

        cancel_button = wx.Button(self, self.cancel_id, "Cancel")
        button_sizer.AddButton(cancel_button)

        def on_cancel(event):
            self.Close()

        self.Bind(wx.EVT_BUTTON, on_cancel, cancel_button)
        button_sizer.SetNegativeButton(cancel_button)
        self.Bind(wx.EVT_MENU, self.on_help, id=wx.ID_HELP)
        self.Bind(wx.EVT_CLOSE, lambda event: self.on_close(event, wx.CANCEL))
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.figure.canvas.Bind(wx.EVT_WINDOW_DESTROY, self.on_destroy)
        self.toolbar.Bind(wx.EVT_WINDOW_DESTROY, self.on_destroy)
        self.help_sash.Bind(wx.EVT_WINDOW_DESTROY, self.on_destroy)

        button_sizer.Realize()
        self.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.figure.canvas.mpl_connect("draw_event", self.draw_callback)
        self.figure.canvas.mpl_connect("button_release_event", self.on_mouse_button_up)
        self.figure.canvas.mpl_connect("motion_notify_event", self.on_mouse_moved)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key_down)
        self.figure.canvas.mpl_connect("key_release_event", self.on_key_up)

    def init_labels(self):
        #########################################
        #
        # Construct a stable label index transform
        # and a color display image.
        #
        #########################################

        nlabels = len(self.to_keep) - 1
        label_map = numpy.zeros(nlabels + 1, self.labels[0].dtype)
        lstart = 0
        self.oi = numpy.zeros(0, int)
        self.oj = numpy.zeros(0, int)
        self.ol = numpy.zeros(0, int)
        self.li = numpy.zeros(0, int)
        self.lj = numpy.zeros(0, int)
        self.ll = numpy.zeros(0, int)
        for label_and_touching in self.labels:
            colored_labels = centrosome.cpmorphology.color_labels(label_and_touching)
            for color in range(1, numpy.max(colored_labels) + 1):
                label = numpy.zeros(label_and_touching.shape, label_and_touching.dtype)
                label[colored_labels == color] = label_and_touching[
                    colored_labels == color
                ]
                # drive each successive matrix's labels away
                # from all others.
                idxs = numpy.unique(label)
                idxs = idxs[idxs != 0]
                distinct_label_count = len(idxs)
                clabels = cellprofiler.gui.tools.renumber_labels_for_display(label)
                clabels[clabels != 0] += lstart
                lstart += distinct_label_count
                label_map[label.flatten()] = clabels.flatten()
                l, ct = scipy.ndimage.label(
                    label != 0, structure=numpy.ones((3, 3), bool)
                )
                coords, offsets, counts = centrosome.cpmorphology.get_outline_pts(
                    l, numpy.arange(1, ct + 1)
                )
                oi, oj = coords.transpose()
                l, ct = scipy.ndimage.label(label == 0)  # 4-connected
                #
                # Have to remove the label that touches the edge, if any
                #
                ledge = numpy.hstack(
                    [
                        l[0, :][label[0, :] == 0],
                        l[-1, :][label[-1, :] == 0],
                        l[:, 0][label[:, 0] == 0],
                        l[:, -1][label[:, -1] == 0],
                    ]
                )
                if len(ledge) > 0:
                    l[l == ledge[0]] = 0

                coords, offsets, counts = centrosome.cpmorphology.get_outline_pts(
                    l, numpy.arange(1, ct + 1)
                )
                if coords.shape[0] > 0:
                    oi, oj = [
                        numpy.hstack((o, coords[:, i])) for i, o in enumerate((oi, oj))
                    ]

                ol = label[oi, oj]
                self.oi = numpy.hstack((self.oi, oi))
                self.oj = numpy.hstack((self.oj, oj))
                self.ol = numpy.hstack((self.ol, ol))
                #
                # compile the filled labels
                #
                li, lj = numpy.argwhere(label != 0).transpose()
                ll = label[li, lj]
                self.li = numpy.hstack((self.li, li))
                self.lj = numpy.hstack((self.lj, lj))
                self.ll = numpy.hstack((self.ll, ll))
        cm = matplotlib.cm.get_cmap(get_default_colormap())
        cm.set_bad((0, 0, 0))

        mappable = matplotlib.cm.ScalarMappable(cmap=cm)
        mappable.set_clim(1, nlabels + 1)
        self.colormap = mappable.to_rgba(numpy.arange(nlabels + 1))[:, :3]
        self.colormap = self.colormap[label_map, :]
        self.oc = self.colormap[self.ol, :]
        self.lc = self.colormap[self.ll, :]

    def on_close(self, event, return_code):
        """Fix up the labels as we close"""
        if return_code == wx.OK:
            self.EndModal(return_code)
            open_labels = set([d[self.K_LABEL] for d in list(self.artists.values())])
            for l in open_labels:
                self.close_label(l, False)
            for idx in numpy.argwhere(~self.to_keep).flatten():
                if idx > 0:
                    self.remove_label(idx)
        else:
            result = wx.MessageBox(
                "Are you sure you want to cancel editing?\n\n"
                "If you are in analysis mode this will stop the analysis.",
                "Cancel object editor",
                style=wx.YES_NO | wx.ICON_QUESTION,
                parent=self,
            )
            if result != wx.YES:
                event.Skip(False)
                return
            self.EndModal(return_code)

    @staticmethod
    def on_destroy(event):
        # bug in wx / matplotlib. Toolbar will be in a mouse capture
        # mode which is not cancelled as dialog exits. Bad things
        # happen on Mac when you try to release capture on a window
        # that's not there.
        # And the sash too, despite the positively wonderful (sarcasm)
        # feedback it gives on the Mac as it is doing stuff.
        #
        # This loop... yes... on the Mac, you can release capture on a window
        # and the window you just released capture on, it becomes the new
        # capture window.
        #
        while True:
            window = wx.Window.GetCapture()
            if window is None:
                break
            window.ReleaseCapture()

    def remove_label(self, object_number):
        for l in self.labels:
            l[l == object_number] = 0

    def replace_label(self, mask, object_number):
        if self.allow_overlap:
            self.remove_label(object_number)
            self.labels.append(mask.astype(self.labels[0].dtype) * object_number)
            self.restructure_labels()
        else:
            labels = self.labels[0]
            labels[labels == object_number] = 0
            labels[mask] = object_number

    def restructure_labels(self):
        """Convert the labels into ijv and back to get the colors right"""

        ii = []
        jj = []
        vv = []
        i, j = numpy.mgrid[0 : self.shape[0], 0 : self.shape[1]]
        for l in self.labels:
            mask = l != 0
            ii.append(i[mask])
            jj.append(j[mask])
            vv.append(l[mask])
        temp = Objects()
        temp.set_ijv(
            numpy.column_stack([numpy.hstack(x) for x in (ii, jj, vv)]),
            shape=self.shape,
        )
        self.labels = [l for l, c in temp.get_labels()]

    def add_label(self, mask):
        object_number = len(self.to_keep)
        temp = numpy.ones(self.to_keep.shape[0] + 1, bool)
        temp[:-1] = self.to_keep
        self.to_keep = temp
        self.labels.append(mask.astype(self.labels[0].dtype) * object_number)
        self.restructure_labels()

    def set_label_display_mode(self, mode):
        """Set label display to either outlines or fill

        mode - one of ID_LABELS_FILL or ID_LABELS_OUTLINE
        """
        self.label_display_mode = mode
        self.display()

    def toggle_label_display_mode(self):
        """Toggle between fill and outline modes"""
        mode = (
            self.ID_LABELS_OUTLINES
            if self.label_display_mode == self.ID_LABELS_FILL
            else self.ID_LABELS_FILL
        )
        self.set_label_display_mode(mode)

    # The following is a function that we can call to refresh
    # the figure's appearance based on the mask and the original labels
    def display(self):
        orig_objects_name = self.title
        if len(self.orig_axes.images) > 0:
            # Save zoom and scale if coming through here a second time
            x0, x1 = self.orig_axes.get_xlim()
            y0, y1 = self.orig_axes.get_ylim()
            set_lim = True
        else:
            set_lim = False
        orig_to_show = numpy.ones(len(self.to_keep), bool)
        for d in list(self.artists.values()):
            object_number = d[self.K_LABEL]
            if object_number < len(orig_to_show):
                orig_to_show[object_number] = False
        self.orig_axes.clear()
        if self.guide_image is not None and self.wants_image_display:
            image, _ = size_similarly(self.orig_labels[0], self.guide_image)
            if image.ndim == 2:
                image = numpy.dstack((image, image, image))
            if image.dtype == bool:
                image = image.astype(int)
            if self.scaling_mode == self.SM_RAW:
                cimage = image.copy()
            elif self.scaling_mode in (self.SM_NORMALIZED, self.SM_LOG_NORMALIZED):
                min_intensity = numpy.min(image)
                max_intensity = numpy.max(image)
                if min_intensity == max_intensity:
                    cimage = image.copy()
                elif self.scaling_mode == self.SM_NORMALIZED:
                    cimage = (image - min_intensity) / (max_intensity - min_intensity)
                else:
                    #
                    # Scale the image to 1 <= image <= e
                    # and take log to get numbers between 0 and 1
                    #
                    cimage = numpy.log(
                        1
                        + (image - min_intensity)
                        * ((numpy.e - 1) / (max_intensity - min_intensity))
                    )
        else:
            cimage = numpy.zeros((self.shape[0], self.shape[1], 3), float)
        if len(self.to_keep) > 1:
            in_artist = numpy.zeros(len(self.to_keep), bool)
            for d in list(self.artists.values()):
                in_artist[d[self.K_LABEL]] = True
            if self.label_display_mode == self.ID_LABELS_OUTLINES:
                for k, stipple in ((self.to_keep, False), (~self.to_keep, True)):
                    k = k & ~in_artist
                    if not numpy.any(k):
                        continue
                    mask = k[self.ol]
                    if not numpy.any(mask):
                        continue
                    intensity = numpy.zeros(self.shape, float)
                    intensity[self.oi[mask], self.oj[mask]] = 1
                    color = numpy.zeros((self.shape[0], self.shape[1], 3), float)
                    if stipple:
                        # Make dashed outlines by throwing away the first 4
                        # border pixels and keeping the next 4. This also makes
                        # small objects disappear when clicked-on.
                        lmap = numpy.zeros(len(k), int)
                        lmap[k] = numpy.arange(numpy.sum(k))
                        counts = numpy.bincount(lmap[self.ol[mask]])
                        indexer = centrosome.index.Indexes((counts,))
                        e = 1 + 3 * (counts[indexer.rev_idx] >= 16)
                        dash_mask = (indexer.idx[0] & (2 ** e - 1)) >= 2 ** (e - 1)
                        color[self.oi[mask], self.oj[mask]] = (
                            self.oc[mask] * dash_mask[:, numpy.newaxis]
                        )
                    else:
                        color[self.oi[mask], self.oj[mask]] = self.oc[mask]
                    sigma = 1
                    intensity = scipy.ndimage.gaussian_filter(intensity, sigma)
                    eps = intensity > numpy.finfo(intensity.dtype).eps
                    color = scipy.ndimage.gaussian_filter(color, (sigma, sigma, 0))[
                        eps, :
                    ]
                    intensity = intensity[eps]
                    cimage[eps, :] = (
                        cimage[eps, :] * (1 - intensity[:, numpy.newaxis]) + color
                    )
            else:
                # Show all pixels of kept labels not in artists
                # Show every other pixel of not-kept labels
                mask = (~in_artist[self.ll]) & (
                    self.to_keep[self.ll]
                    | (((self.li & 1) == 0) & ((self.lj & 1) == 0))
                )
                npts = numpy.sum(mask)
                has_color = scipy.sparse.coo_matrix(
                    (numpy.ones(npts), (self.li[mask], self.lj[mask])),
                    shape=cimage.shape[:2],
                ).toarray()
                for i in range(3):
                    rgbval = scipy.sparse.coo_matrix(
                        (self.lc[mask, i], (self.li[mask], self.lj[mask])),
                        shape=cimage.shape[:2],
                    ).toarray()
                    cimage[has_color > 0, i] = (
                        rgbval[has_color > 0] / has_color[has_color > 0]
                    )
        self.orig_axes.imshow(cimage, interpolation=self.interpolation_mode)
        self.set_orig_axes_title()
        if set_lim:
            self.orig_axes.set_xlim((x0, x1))
            self.orig_axes.set_ylim((y0, y1))
        for artist in self.artists:
            self.orig_axes.add_line(artist)
        if self.split_artist is not None:
            self.orig_axes.add_line(self.split_artist)
        # Set background to None as we've effectively redrawn the display
        self.background = None
        self.Refresh()

    def on_paint(self, event):
        dc = wx.PaintDC(self.panel)
        if self.background is None:
            self.panel.draw(dc)
        else:
            self.panel.gui_repaint(dc)
        dc.Destroy()
        event.Skip()

    def draw_callback(self, event):
        """Decorate the drawing with the animated artists"""
        if not self.inside_print:
            self.background = self.figure.canvas.copy_from_bbox(self.orig_axes.bbox)
        for artist in self.artists:
            self.orig_axes.draw_artist(artist)
        if self.split_artist is not None:
            self.orig_axes.draw_artist(self.split_artist)
        if self.mode == self.FREEHAND_DRAW_MODE and self.active_artist is not None:
            self.orig_axes.draw_artist(self.active_artist)
        if not self.inside_print:
            self.figure.canvas.blit(self.orig_axes.bbox)

    def get_control_point(self, event):
        """Find the artist and control point under the cursor

        returns tuple of artist, and index of control point or None, None
        """
        best_d = numpy.inf
        best_artist = None
        best_index = None
        for artist in self.artists:
            data = artist.get_xydata()[:-1, :]
            xy = artist.get_transform().transform(data)
            x, y = xy.transpose()
            d = numpy.sqrt((x - event.x) ** 2 + (y - event.y) ** 2)
            idx = numpy.atleast_1d(numpy.argmin(d)).flatten()[0]
            d = d[idx]
            if d < self.epsilon and d < best_d:
                best_d = d
                best_artist = artist
                best_index = idx
        return best_artist, best_index

    def on_click(self, event):
        if event.button == 3:
            self.skip_right_button_up = False
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
        elif self.mode == self.DELETE_MODE:
            self.on_delete_click(event)
            return
        if event.inaxes == self.orig_axes and event.button == 1:
            best_artist, best_index = self.get_control_point(event)
            if best_artist is not None:
                self.active_artist = best_artist
                self.active_index = best_index
                return
        elif event.inaxes == self.orig_axes and event.button == 3:
            for artist in self.artists:
                path = matplotlib.path.Path(artist.get_xydata())
                if path.contains_point((event.xdata, event.ydata)):
                    self.close_label(self.artists[artist][self.K_LABEL])
                    self.record_undo()
                    self.skip_right_button_up = True
                    return
        lnum = self.get_mouse_event_object_number(event)
        if lnum is None:
            if event.button == 3 or wx.GetKeyState(wx.WXK_CONTROL):
                self.on_context_menu(event)
            return
        if event.button == 1:
            # Move object into / out of working set
            if event.inaxes == self.orig_axes:
                self.to_keep[lnum] = not self.to_keep[lnum]
            self.display()
        elif event.button == 3:
            self.make_control_points(lnum)
            self.display()
            self.skip_right_button_up = True

    def get_mouse_event_object_number(self, event):
        """Return the object number of the object under the mouse

        event - a matplotlib mouse event

        returns the object number at the mouse location or None if
        mouse isn't over an object
        """
        x = int(event.xdata + 0.5)
        y = int(event.ydata + 0.5)
        if x < 0 or x >= self.shape[1] or y < 0 or y >= self.shape[0]:
            return
        for labels in self.labels:
            lnum = labels[y, x]
            if lnum != 0:
                break
        if lnum == 0:
            return
        return lnum

    def on_key_down(self, event):
        assert isinstance(event, matplotlib.backend_bases.KeyEvent)
        self.pressed_keys.add(event.key)
        if self.toolbar.mode != "":
            if event.key == "escape":
                event.guiEvent.Skip(False)
                self.Close()
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
            elif event.key == "m":
                self.on_context_menu(event)
            elif event.key == "n":
                self.new_object(event)
            elif event.key == "s":
                self.enter_split_mode(event)
            elif event.key == "t":
                self.toggle_label_display_mode()
            elif event.key == "x":
                self.enter_delete_mode(event)
            elif event.key == "z":
                if len(self.undo_stack) > 0:
                    self.undo()
            elif event.key == "escape":
                self.remove_artists(event)
                event.guiEvent.Skip(False)
        elif self.mode in (self.SPLIT_PICK_FIRST_MODE, self.SPLIT_PICK_SECOND_MODE):
            if event.key == "escape":
                self.exit_split_mode(event)
                event.guiEvent.Skip(False)
        elif self.mode == self.DELETE_MODE:
            if event.key == "escape":
                self.exit_delete_mode(event)
                event.guiEvent.Skip(False)
        elif self.mode == self.FREEHAND_DRAW_MODE:
            self.exit_freehand_draw_mode(event)

    def on_key_up(self, event):
        if event.key in self.pressed_keys:
            self.pressed_keys.remove(event.key)

    def on_context_menu(self, event):
        """Pop up a context menu for the control"""
        if isinstance(event, wx.MouseEvent):
            x, y = self.panel.ScreenToClient(event.GetPosition())
            location_event = matplotlib.backend_bases.LocationEvent(
                "ContextMenu", self.panel, x, y, event
            )
            if location_event.inaxes == self.orig_axes:
                if self.get_mouse_event_object_number(location_event) is not None:
                    event.Skip(False)
                    return
            if self.mode == self.FREEHAND_DRAW_MODE:
                self.exit_freehand_draw_mode(event)
            elif self.mode in (self.SPLIT_PICK_FIRST_MODE, self.SPLIT_PICK_SECOND_MODE):
                self.exit_split_mode(event)
        menu = wx.Menu()
        if self.guide_image is not None:
            check_item = menu.AppendCheckItem(
                self.ID_DISPLAY_IMAGE, "Display image", "Toggle guiding image display"
            )
            check_item.Check(self.wants_image_display)
        contrast_menu = wx.Menu("Contrast")
        for mid, state, help_text in (
            (self.ID_CONTRAST_RAW, self.SM_RAW, "Display raw intensity image"),
            (
                self.ID_CONTRAST_NORMALIZED,
                self.SM_NORMALIZED,
                "Display the image intensity using the full range of gray scales",
            ),
            (
                self.ID_CONTRAST_LOG_NORMALIZED,
                self.SM_LOG_NORMALIZED,
                "Display the image intensity using a logarithmic scale",
            ),
        ):
            contrast_menu.AppendRadioItem(mid, state, help_text)
            if self.scaling_mode == state:
                contrast_menu.Check(mid, True)
        menu.Append(-1, "Contrast", contrast_menu)

        interpolation_menu = wx.Menu("Interpolation")
        for mid, state, help_text in (
            (
                self.ID_INTERPOLATION_NEAREST,
                IM_NEAREST,
                "Display images using the intensity of the nearest pixel (blocky)",
            ),
            (
                self.ID_INTERPOLATION_BILINEAR,
                IM_BILINEAR,
                "Display images by blending the intensities of the four nearest pixels (smoother)",
            ),
            (
                self.ID_INTERPOLATION_BICUBIC,
                IM_BICUBIC,
                "Display images by blending intensities using cubic spline interpolation (smoothest)",
            ),
        ):
            interpolation_menu.AppendRadioItem(mid, state, help_text)
            if self.interpolation_mode == state:
                interpolation_menu.Check(mid, True)
        menu.Append(-1, "Interpolation", interpolation_menu)

        label_menu = wx.Menu("Label appearance")
        for mid, label, help_text in (
            (self.ID_LABELS_OUTLINES, "Outlines", "Show the outlines of objects"),
            (self.ID_LABELS_FILL, "Fill", "Show objects with a solid fill color"),
        ):
            label_menu.AppendRadioItem(mid, label, help_text)
        label_menu.Check(self.label_display_mode, True)
        menu.Append(-1, "Label appearance", label_menu)

        mode_menu = wx.Menu("Mode")
        mode_menu.Append(
            self.ID_MODE_FREEHAND, "Freehand mode", "Enter freehand object-drawing mode"
        )
        mode_menu.Append(
            self.ID_MODE_SPLIT, "Split mode", "Enter object splitting mode"
        )
        mode_menu.Append(self.ID_MODE_DELETE, "Delete mode", "Select points to delete")
        menu.Append(-1, "Mode", mode_menu)

        actions_menu = wx.Menu("Actions")
        actions_menu.Append(
            self.ID_ACTION_KEEP,
            "Keep",
            "Keep all objects (undo all remove object actions)",
        )
        actions_menu.Append(
            self.ID_ACTION_REMOVE, "Remove", "Mark all objects for removal"
        )
        actions_menu.Append(
            self.ID_ACTION_TOGGLE,
            "Toggle",
            "Mark all kept objects for removal and unmark all objects marked for removal",
        )
        actions_menu.Append(
            self.ID_ACTION_RESET,
            "Reset",
            "Reset the objects to their original state before editing",
        )
        actions_menu.AppendSeparator()
        actions_menu.Append(
            self.ID_ACTION_JOIN,
            "Join objects",
            "Join all objects being edited into a single object",
        )
        actions_menu.Append(
            self.ID_ACTION_CONVEX_HULL,
            "Convex hull",
            "Merge all objects being edited into an object which is the smallest convex hull around them all (can be done to a single object).",
        )
        menu.Append(-1, "Actions", actions_menu)
        menu.Append(wx.ID_HELP)
        self.PopupMenu(menu)

    def on_display_image(self, event):
        self.wants_image_display = not self.wants_image_display
        self.display_image_button.Label = (
            self.D_HIDE_IMAGE if self.wants_image_display else self.D_SHOW_IMAGE
        )
        self.display()

    def on_raw_contrast(self, event):
        self.scaling_mode = self.SM_RAW
        self.display()

    def on_normalized_contrast(self, event):
        self.scaling_mode = self.SM_NORMALIZED
        self.display()

    def on_log_normalized_contrast(self, event):
        self.scaling_mode = self.SM_LOG_NORMALIZED
        self.display()

    def on_nearest_neighbor_interpolation(self, event):
        self.interpolation_mode = IM_NEAREST
        self.display()

    def on_bilinear_interpolation(self, event):
        self.interpolation_mode = IM_BILINEAR
        self.display()

    def on_bicubic_interpolation(self, event):
        self.interpolation_mode = IM_BICUBIC
        self.display()

    def on_mouse_button_up(self, event):
        if self.skip_right_button_up and event.button == 3:
            self.skip_right_button_up = False
            event.guiEvent.Skip(False)
            event.guiEvent.StopPropagation()
        if event.inaxes is not None and event.inaxes.get_navigate_mode() is not None:
            return
        if self.mode == self.FREEHAND_DRAW_MODE:
            self.on_mouse_button_up_freehand_draw_mode(event)
        elif self.mode == self.DELETE_MODE:
            self.on_mouse_button_up_delete_mode(event)
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
        elif self.mode == self.DELETE_MODE:
            self.on_mouse_moved_delete_mode(event)

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
            numpy.array([data[0][idx], data[1][idx]])
            for idx in (before_index, after_index)
        ]
        ydata, xdata = [
            min(self.shape[i] - 1, max(yx, 0))
            for i, yx in enumerate((event.ydata, event.xdata))
        ]
        new_pt = numpy.array([xdata, ydata], int)
        path = matplotlib.path.Path(numpy.array((before_pt, new_pt, after_pt)))
        eps = numpy.finfo(numpy.float32).eps
        for artist in self.artists:
            if (
                self.allow_overlap
                and self.artists[artist][self.K_LABEL] != object_number
            ):
                continue
            if artist == self.active_artist:
                if n_points <= 4:
                    continue
                # Exclude the lines -2 and 2 before and after ours.
                #
                xx, yy = [
                    numpy.hstack((d[self.active_index :], d[: (self.active_index + 1)]))
                    for d in data
                ]
                xx, yy = xx[2:-2], yy[2:-2]
                xydata = numpy.column_stack((xx, yy))
            else:
                xydata = artist.get_xydata()
            other_path = matplotlib.path.Path(xydata)

            l0 = xydata[:-1, :]
            l1 = xydata[1:, :]
            neww_pt = numpy.ones(l0.shape) * new_pt[numpy.newaxis, :]
            d = centrosome.cpmorphology.distance2_to_line(neww_pt, l0, l1)
            different_sign = numpy.sign(neww_pt - l0) != numpy.sign(neww_pt - l1)
            on_segment = (d < eps) & different_sign[:, 0] & different_sign[:, 1]

            if any(on_segment):
                # it's ok if the point is on the line.
                continue
            if path.intersects_path(other_path, filled=False):
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
        self.artists[self.active_artist]["edited"] = True
        self.update_artists()

    def update_artists(self):
        # Only restore if background was set, otherwise no need
        if self.background is not None:
            self.figure.canvas.restore_region(self.background)
        else:
            self.background = self.figure.canvas.copy_from_bbox(self.orig_axes.bbox)

        for artist in self.artists:
            self.orig_axes.draw_artist(artist)
        if self.split_artist is not None:
            self.orig_axes.draw_artist(self.split_artist)
        if self.delete_mode_artist is not None:
            self.orig_axes.draw_artist(self.delete_mode_artist)
        if self.delete_mode_rect_artist is not None:
            self.orig_axes.draw_artist(self.delete_mode_rect_artist)
        if self.mode == self.FREEHAND_DRAW_MODE and self.active_artist is not None:
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
        all_labels = numpy.unique(
            [v[self.K_LABEL] for v in list(self.artists.values())]
        )
        if len(all_labels) < 2:
            return
        assert all_labels[0] == numpy.min(all_labels)
        object_number = all_labels[0]
        for label in all_labels:
            self.close_label(label, display=False)

        to_join = numpy.zeros(len(self.to_keep), bool)
        to_join[all_labels] = True
        #
        # Copy all labels to join to the mask and erase.
        #
        mask = numpy.zeros(self.shape, bool)
        for label in self.labels:
            mask |= to_join[label]
            label[to_join[label]] = 0
        self.labels.append(mask.astype(self.labels[0].dtype) * object_number)

        self.restructure_labels()
        self.init_labels()
        self.make_control_points(object_number)
        self.display()
        self.record_undo()
        return all_labels[0]

    def convex_hull(self, event):
        if len(self.artists) == 0:
            return

        all_labels = numpy.unique(
            [v[self.K_LABEL] for v in list(self.artists.values())]
        )
        for label in all_labels:
            self.close_label(label, display=False)
        object_number = all_labels[0]
        mask = numpy.zeros(self.shape, bool)
        for label in self.labels:
            for n in all_labels:
                mask |= label == n

        for n in all_labels:
            self.remove_label(n)

        mask = centrosome.cpmorphology.convex_hull_image(mask)
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
        best_distance = numpy.inf
        new_pt = None
        for artist in self.artists:
            l = artist.get_xydata()[:, ::-1]
            l0 = l[:-1, :]
            l1 = l[1:, :]
            llen = numpy.sqrt(numpy.sum((l1 - l0) ** 2, 1))
            # the unit vector
            v = (l1 - l0) / llen[:, numpy.newaxis]
            pt = numpy.ones(l0.shape, l0.dtype)
            pt[:, 0] = pt_i
            pt[:, 1] = pt_j
            #
            # Project l0<->pt onto l0<->l1. If the result
            # is longer than l0<->l1, then the closest point is l1.
            # If the result is negative, then the closest point is l0.
            # In either case, don't add.
            #
            proj = numpy.sum(v * (pt - l0), 1)
            d2 = centrosome.cpmorphology.distance2_to_line(pt, l0, l1)
            d2[proj <= 0] = numpy.inf
            d2[proj >= llen] = numpy.inf
            best = numpy.argmin(d2)
            if best_distance > d2[best]:
                best_distance = d2[best]
                best_artist = artist
                best_index = best
                new_pt = (
                    l0[best_index, :]
                    + proj[best_index, numpy.newaxis] * v[best_index, :]
                )
        if best_artist is None:
            return
        l = best_artist.get_xydata()[:, ::-1]
        l = numpy.vstack(
            (l[: (best_index + 1)], new_pt.reshape(1, 2), l[(best_index + 1) :])
        )
        best_artist.set_data((l[:, 1], l[:, 0]))
        self.artists[best_artist][self.K_EDITED] = True
        self.update_artists()
        self.record_undo()

    def delete_control_point(self, event):
        best_artist, best_index = self.get_control_point(event)
        if best_artist is not None:
            l = best_artist.get_xydata()
            if len(l) < 4:
                self.delete_artist(best_artist)
            else:
                l = numpy.vstack((l[:best_index, :], l[(best_index + 1) : -1, :]))
                l = numpy.vstack((l, l[:1, :]))
                best_artist.set_data((l[:, 0], l[:, 1]))
                self.artists[best_artist][self.K_EDITED] = True
            self.record_undo()
            self.update_artists()

    def delete_artist(self, artist):
        """Delete an artist and remove its object

        artist to delete
        """
        object_number = self.artists[artist][self.K_LABEL]
        artist.remove()
        del self.artists[artist]
        if not any(
            [d[self.K_LABEL] == object_number for d in list(self.artists.values())]
        ):
            self.remove_label(object_number)
            self.init_labels()
            self.display()
        else:
            # Mark some other artist as edited.
            for artist, d in list(self.artists.items()):
                if d[self.K_LABEL] == object_number:
                    d[self.K_EDITED] = True

    def new_object(self, event):
        object_number = len(self.to_keep)
        temp = numpy.ones(object_number + 1, bool)
        temp[:-1] = self.to_keep
        self.to_keep = temp
        angles = numpy.pi * 2 * numpy.arange(13) / 12
        x = 20 * numpy.cos(angles) + event.xdata
        y = 20 * numpy.sin(angles) + event.ydata
        x[x < 0] = 0
        x[x >= self.shape[1]] = self.shape[1] - 1
        y[y >= self.shape[0]] = self.shape[0] - 1
        lnew = numpy.zeros(self.labels[0].shape, self.labels[0].dtype)
        i, j = numpy.mgrid[0 : lnew.shape[0], 0 : lnew.shape[1]]
        lnew[(i - event.ydata) ** 2 + (j - event.xdata) ** 2 <= 400] = object_number
        self.labels.append(lnew)
        self.restructure_labels()
        self.init_labels()
        new_artist = matplotlib.lines.Line2D(
            x,
            y,
            marker="o",
            markerfacecolor="r",
            markersize=6,
            color=self.colormap[object_number, :],
            animated=True,
        )

        self.artists[new_artist] = {
            self.K_LABEL: object_number,
            self.K_EDITED: True,
            self.K_OUTSIDE: True,
        }
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
        elif self.mode == self.DELETE_MODE:
            title = self.DELETE_MODE_TITLE
        elif self.mode == self.FREEHAND_DRAW_MODE:
            if self.active_artist is None:
                title = "Click the mouse to begin to draw or hit Esc"
            else:
                title = "Freehand drawing"
        else:
            title = self.title

        self.orig_axes.set_title(
            title, fontname=get_title_font_name(), fontsize=get_title_font_size(),
        )

    def enter_split_mode(self, event):
        self.toolbar.cancel_mode()
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
        self.split_artist = matplotlib.lines.Line2D(
            numpy.array((x, x)), numpy.array((y, y)), color="blue", animated=True
        )
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
            if pick_artist is not None and self.ok_to_split(pick_artist, pick_index):
                self.split_artist.set_color("red")
            else:
                self.split_artist.set_color("blue")
            self.update_artists()

    def ok_to_split(self, pick_artist, pick_index):
        if (
            self.artists[pick_artist][self.K_LABEL]
            != self.artists[self.split_pick_artist][self.K_LABEL]
        ):
            # Second must be same object as first.
            return False
        if pick_artist == self.split_pick_artist:
            min_index, max_index = [
                fn(pick_index, self.split_pick_index) for fn in (min, max)
            ]
            if max_index - min_index < 2:
                # don't allow split of neighbors
                return False
            if len(pick_artist.get_xdata()) - max_index <= 2 and min_index == 0:
                # don't allow split of last and first
                return False
        elif (
            self.artists[pick_artist][self.K_OUTSIDE]
            == self.artists[self.split_pick_artist][self.K_OUTSIDE]
        ):
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
                xy0 = numpy.vstack((xy[: (idx0 + 1), :], xy[idx1:, :]))
                xy1 = numpy.vstack((xy[idx0 : (idx1 + 1), :], xy[idx0 : (idx0 + 1), :]))
            else:
                border_pts = numpy.zeros((2, 2, 2))

                border_pts[0, 0, :], border_pts[1, 1, :] = self.get_split_points(
                    pick_artist, idx0
                )
                border_pts[0, 1, :], border_pts[1, 0, :] = self.get_split_points(
                    pick_artist, idx1
                )
                xy0 = numpy.vstack(
                    (xy[:idx0, :], border_pts[:, 0, :], xy[(idx1 + 1) :, :])
                )
                xy1 = numpy.vstack(
                    (
                        border_pts[:, 1, :],
                        xy[(idx0 + 1) : idx1, :],
                        border_pts[:1, 1, :],
                    )
                )

            pick_artist.set_data((xy0[:, 0], xy0[:, 1]))
            new_artist = matplotlib.lines.Line2D(
                xy1[:, 0],
                xy1[:, 1],
                marker="o",
                markerfacecolor="r",
                markersize=6,
                color=self.colormap[old_object_number, :],
                animated=True,
            )
            self.orig_axes.add_line(new_artist)
            if is_outside:
                new_object_number = len(self.to_keep)
                self.artists[new_artist] = {
                    self.K_EDITED: True,
                    self.K_LABEL: new_object_number,
                    self.K_OUTSIDE: is_outside,
                }
                self.artists[pick_artist][self.K_EDITED] = True
                #
                # Find all points within holes in the old object
                #
                hmask = numpy.zeros(self.shape, bool)
                for artist, attrs in list(self.artists.items()):
                    if (
                        not attrs[self.K_OUTSIDE]
                        and attrs[self.K_LABEL] == old_object_number
                    ):
                        hx, hy = artist.get_data()
                        hmask = hmask | centrosome.cpmorphology.polygon_lines_to_mask(
                            hy[:-1], hx[:-1], hy[1:], hx[1:], self.shape
                        )
                temp = numpy.ones(self.to_keep.shape[0] + 1, bool)
                temp[:-1] = self.to_keep
                self.to_keep = temp
                self.close_label(old_object_number, False)
                self.close_label(new_object_number, False)
                #
                # Remove hole points from both objects
                #
                for label in self.labels:
                    label[
                        hmask
                        & ((label == old_object_number) | (label == new_object_number))
                    ] = 0
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
                    self.K_OUTSIDE: False,
                }
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
            border_pts = numpy.zeros((2, 2, 2))

            border_pts[0, 0, :], border_pts[1, 1, :] = self.get_split_points(
                outside_artist, outside_index
            )
            border_pts[0, 1, :], border_pts[1, 0, :] = self.get_split_points(
                inside_artist, inside_index
            )

            xy = numpy.vstack(
                (
                    xy0[:outside_index, :],
                    border_pts[:, 0, :],
                    xy1[(inside_index + 1) : -1, :],
                    xy1[:inside_index, :],
                    border_pts[:, 1, :],
                    xy0[(outside_index + 1) :, :],
                )
            )
            xy[-1, :] = xy[0, :]  # if outside_index == 0

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
        """Get the area inside an artist polygon"""
        #
        # Thank you Darel Rex Finley:
        #
        # http://alienryderflex.com/polygon_area/
        #
        # Code is public domain
        #
        x, y = artist.get_data()
        area = abs(numpy.sum((x[:-1] + x[1:]) * (y[:-1] - y[1:]))) / 2
        return area

    @staticmethod
    def get_split_points(artist, idx):
        """Return the split points on either side of the indexed point

        artist - artist in question
        idx - index of the point

        returns a point midway between the previous point and the
        point in question and a point midway between the next point
        and the point in question.
        """
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
            idx_right = idx + 1
        return (a[idx_left, :] + a[idx, :]) / 2, (a[idx_right, :] + a[idx, :]) / 2

    ################################
    #
    # Freehand draw mode
    #
    ################################
    def enter_freehand_draw_mode(self, event):
        self.toolbar.cancel_mode()
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
        """Begin drawing on mouse-down"""
        self.active_artist = matplotlib.lines.Line2D(
            [event.xdata], [event.ydata], color="blue", animated=True
        )
        self.orig_axes.add_line(self.active_artist)
        self.update_artists()

    def handle_mouse_moved_freehand_draw_mode(self, event):
        if event.inaxes != self.orig_axes:
            return
        if self.active_artist is not None:
            xdata, ydata = self.active_artist.get_data()
            ydata, xdata = [
                numpy.minimum(self.shape[i] - 1, numpy.maximum(yx, 0))
                for i, yx in enumerate((ydata, xdata))
            ]
            self.active_artist.set_data(
                numpy.hstack((xdata, [event.xdata])),
                numpy.hstack((ydata, [event.ydata])),
            )
            self.update_artists()

    def on_mouse_button_up_freehand_draw_mode(self, event):
        xydata = self.active_artist.get_xydata()
        if event.inaxes == self.orig_axes:
            xydata = numpy.vstack((xydata, numpy.array([[event.xdata, event.ydata]])))
        xydata = numpy.vstack((xydata, numpy.array([[xydata[0, 0], xydata[0, 1]]])))

        mask = centrosome.cpmorphology.polygon_lines_to_mask(
            xydata[:-1, 1], xydata[:-1, 0], xydata[1:, 1], xydata[1:, 0], self.shape
        )
        if not self.allow_overlap:
            self.labels[0][mask] = 0
        self.add_label(mask)

        self.exit_freehand_draw_mode(event)
        self.init_labels()
        self.display()
        self.record_undo()

    ################################
    #
    # Delete mode
    #
    ################################

    DELETE_MODE_TITLE = (
        "Press the mouse button and drag to select points to delete.\n"
        "Hit Esc to exit without deleting.\n"
    )

    def enter_delete_mode(self, event):
        self.toolbar.cancel_mode()
        self.mode = self.DELETE_MODE
        self.delete_mode_start = None
        self.delete_mode_filter = None
        self.toolbar.set_cursor(matplotlib.backend_bases.cursors.SELECT_REGION)
        self.set_orig_axes_title()
        self.figure.canvas.draw()

    def on_delete_click(self, event):
        if event.button == 1:
            if event.inaxes == self.orig_axes:
                self.delete_mode_start = (event.xdata, event.ydata)
                self.delete_mode_rect_artist = matplotlib.lines.Line2D(
                    [event.xdata] * 5,
                    [event.ydata] * 5,
                    linestyle="-",
                    color="w",
                    marker=" ",
                )
                self.orig_axes.add_line(self.delete_mode_rect_artist)
                LOGGER.info("Starting delete at %f, %f" % self.delete_mode_start)
            else:
                self.exit_delete_mode(event)

    def on_mouse_moved_delete_mode(self, event):
        if self.delete_mode_start is not None and event.inaxes == self.orig_axes:
            (xmin, xmax), (ymin, ymax) = [
                [fn(a, b) for fn in (min, max)]
                for a, b in (
                    (self.delete_mode_start[0], event.xdata),
                    (self.delete_mode_start[1], event.ydata),
                )
            ]
            self.delete_mode_rect_artist.set_data(
                [[xmin, xmin, xmax, xmax, xmin], [ymin, ymax, ymax, ymin, ymin]]
            )

            def delete_mode_filter(x, y):
                return (x >= xmin) & (x < xmax) & (y >= ymin) & (y < ymax)

            self.delete_mode_filter = delete_mode_filter
            points = []
            for artist in self.artists:
                x, y = artist.get_data()
                f = delete_mode_filter(x, y)
                if numpy.any(f):
                    points += [numpy.column_stack((x[f], y[f]))]

            if len(points) > 0:
                points = numpy.vstack(points)
                if self.delete_mode_artist is None:
                    self.delete_mode_artist = matplotlib.lines.Line2D(
                        points[:, 0],
                        points[:, 1],
                        marker="o",
                        markeredgecolor="black",
                        markerfacecolor="white",
                        linestyle=" ",
                    )
                    self.orig_axes.add_line(self.delete_mode_artist)
                else:
                    old_points = self.delete_mode_artist.get_xydata()
                    if len(old_points) != len(points) or numpy.any(
                        old_points != points
                    ):
                        self.delete_mode_artist.set_data(points.transpose())
            self.update_artists()

    def on_mouse_button_up_delete_mode(self, event):
        to_delete = []
        if self.delete_mode_filter is not None:
            for artist in self.artists:
                x, y = artist.get_data()
                f = ~self.delete_mode_filter(x, y)
                if numpy.all(f):
                    continue
                xy = numpy.column_stack((x[f], y[f]))
                if len(xy) < 3 or len(xy) == 3 and numpy.all(xy[0] == xy[-1]):
                    to_delete.append(artist)
                else:
                    if numpy.all(xy[0] != xy[-1]):
                        xy = numpy.vstack((xy, xy[:1]))
                    artist.set_data(xy.transpose())
                    self.artists[artist][self.K_EDITED] = True
        object_numbers = set()
        for artist in to_delete:
            object_numbers.add(self.artists[artist][self.K_LABEL])
            artist.remove()
            del self.artists[artist]
        for artist, d in list(self.artists.items()):
            if d[self.K_LABEL] in object_numbers:
                d[self.K_EDITED] = True
                object_numbers.remove(d[self.K_LABEL])
        if len(object_numbers) > 0:
            # Exit delete mode here because self.display() wipes artists
            self.exit_delete_mode(event)
            for object_number in object_numbers:
                self.remove_label(object_number)
            self.init_labels()
            self.display()
        else:
            self.exit_delete_mode(event)
        self.record_undo()

    def exit_delete_mode(self, event):
        LOGGER.info("Exiting delete mode")
        if self.delete_mode_artist is not None:
            self.delete_mode_artist.remove()
            self.delete_mode_artist = None
        if self.delete_mode_rect_artist is not None:
            self.delete_mode_rect_artist.remove()
            self.delete_mode_rect_artist = None
        self.delete_mode_start = None
        self.mode = self.NORMAL_MODE
        self.set_orig_axes_title()
        self.update_artists()
        self.toolbar.set_cursor(matplotlib.backend_bases.cursors.POINTER)
        self.figure.canvas.draw()

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
        self.to_keep[1:] = ~self.to_keep[1:]
        self.display()

    def on_reset(self, event):
        self.reset()

    def reset(self, display=True):
        self.labels = [l.copy() for l in self.orig_labels]
        nlabels = numpy.max([numpy.max(l) for l in self.orig_labels])
        self.to_keep = numpy.ones(nlabels + 1, bool)
        self.artists = {}
        self.undo_stack = []
        if hasattr(self, "undo_button"):
            # minor unfortunate hack - reset called before GUI is built
            self.undo_button.Enable(False)
        self.last_ijv = self.calculate_ijv()
        self.last_artist_save = {}
        self.last_to_keep = self.to_keep
        if display:
            self.init_labels()
            self.display()

    def on_help(self, event):
        do_show = not self.help_sash.IsShown()
        self.help_button.Label = "Hide help" if do_show else "Show help"
        if not do_show:
            self.help_sash.Show(False)
        else:
            self.help_sash.Show(True)
            w, h = self.GetClientSize()
            self.help_sash.SetDefaultSize((w, h / 3))
        self.layout_sash()

    def on_help_sash_drag(self, event):
        height = event.GetDragRect().height
        w, h = self.GetClientSize()
        self.help_sash.SetDefaultSize((w, height))
        self.layout_sash()

    def layout_sash(self):
        wx.adv.LayoutAlgorithm().LayoutWindow(self.sash_parent, self.panel)
        if self.help_sash.IsShown():
            self.help_sash.Layout()
            self.help_sash.Refresh()
        self.panel.draw()
        self.panel.Refresh()

    def on_size(self, event):
        self.Layout()
        self.layout_sash()

    def make_control_points(self, object_number):
        """Create an artist with control points for editing an object

        object_number - # of object to edit
        """
        #
        # We need to make outlines of both objects and holes.
        # Objects are 8-connected and holes are 4-connected
        #
        for polarity, structure in (
            (True, numpy.ones((3, 3), bool)),
            (False, numpy.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)),
        ):
            #
            # Pad the mask so we don't have to deal with out of bounds
            #
            mask = numpy.zeros((self.shape[0] + 2, self.shape[1] + 2), bool)
            for l in self.labels:
                mask[1:-1, 1:-1] |= l == object_number
            if not polarity:
                mask = ~mask
            labels, count = scipy.ndimage.label(mask, structure)
            if not polarity:
                border_object = labels[0, 0]
                labels[labels == labels[0, 0]] = 0
            sub_object_numbers = [
                n for n in range(1, count + 1) if polarity or n != border_object
            ]
            coords, offsets, counts = centrosome.cpmorphology.get_outline_pts(
                labels, sub_object_numbers
            )
            coords -= 1  # account for mask padding
            for i, sub_object_number in enumerate(sub_object_numbers):
                chain = coords[offsets[i] : (offsets[i] + counts[i]), :]
                if not polarity:
                    chain = chain[::-1]
                chain = numpy.vstack((chain, chain[:1, :])).astype(float)
                #
                # Start with the first point and a midpoint in the
                # chain and keep adding points until the maximum
                # error caused by leaving a point out is 4
                #
                minarea = 10
                if len(chain) > 10:
                    accepted = numpy.zeros(len(chain), bool)
                    accepted[0] = True
                    accepted[-1] = True
                    accepted[int(len(chain) / 2)] = True
                    while True:
                        idx1 = numpy.cumsum(accepted[:-1])
                        idx0 = idx1 - 1
                        ca = chain[accepted]
                        aidx = numpy.argwhere(accepted).flatten()
                        a = centrosome.cpmorphology.triangle_areas(
                            ca[idx0], ca[idx1], chain[:-1]
                        )
                        idxmax = numpy.argmax(a)
                        if a[idxmax] < 4:
                            break
                        # Pick a point halfway in-between
                        idx = int((aidx[idx0[idxmax]] + aidx[idx1[idxmax]]) / 2)
                        accepted[idx] = True
                    chain = chain[accepted]
                artist = matplotlib.lines.Line2D(
                    chain[:, 1],
                    chain[:, 0],
                    marker="o",
                    markerfacecolor="r",
                    markersize=6,
                    color=self.colormap[object_number, :],
                    animated=True,
                )
                self.orig_axes.add_line(artist)
                self.artists[artist] = {
                    self.K_LABEL: object_number,
                    self.K_EDITED: False,
                    self.K_OUTSIDE: polarity,
                }
        self.update_artists()

    def close_label(self, label, display=True):
        """Close the artists associated with a label

        label - label # of label being closed.

        If edited, update the labeled pixels.
        """
        my_artists = [
            artist
            for artist, data in list(self.artists.items())
            if data[self.K_LABEL] == label
        ]
        if any([self.artists[artist][self.K_EDITED] for artist in my_artists]):
            #
            # Convert polygons to labels. The assumption is that
            # a polygon within a polygon is a hole.
            #
            mask = numpy.zeros(self.shape, bool)
            for artist in my_artists:
                j, i = artist.get_data()
                m1 = centrosome.cpmorphology.polygon_lines_to_mask(
                    i[:-1], j[:-1], i[1:], j[1:], self.shape
                )
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

    def discard_message(self, message):
        # Matplotlib wants to update the status bar, but this window lacks one.
        # So we ignore it. Matplotlib plan to remove this behaviour eventually.
        return
