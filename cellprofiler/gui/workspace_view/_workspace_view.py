import matplotlib
import numpy
import wx
import wx.lib.scrolledpanel
from cellprofiler_core.constants.measurement import EXPERIMENT
from cellprofiler_core.constants.measurement import IMAGE
from cellprofiler_core.constants.measurement import M_LOCATION_CENTER_X
from cellprofiler_core.constants.measurement import M_LOCATION_CENTER_Y
from cellprofiler_core.measurement import Measurements
from cellprofiler_core.preferences import IM_BILINEAR
from cellprofiler_core.preferences import IM_NEAREST
from cellprofiler_core.preferences import get_interpolation_mode

import cellprofiler.gui.artist
import cellprofiler.gui.figure
import cellprofiler.gui.help
import cellprofiler.gui.help.content
import cellprofiler.gui.htmldialog
from ._workspace_view_figure import WorkspaceViewFigure
from ._workspace_view_image_row import WorkspaceViewImageRow
from ._workspace_view_mask_row import WorkspaceViewMaskRow
from ._workspace_view_measurement_row import WorkspaceViewMeasurementRow
from ._workspace_view_objects_row import WorkspaceViewObjectsRow

__the_workspace_viewer = None


class WorkspaceView(object):
    C_CHOOSER = 0
    C_COLOR = 1
    C_SHOW = 2
    C_REMOVE = 3

    def __init__(self, parent, workspace):
        self.frame = WorkspaceViewFigure(
            parent,
            title="CellProfiler Workspace",
            secret_panel_class=wx.lib.scrolledpanel.ScrolledPanel,
            help_menu_items=cellprofiler.gui.help.content.FIGURE_HELP,
        )
        self.workspace = workspace
        self.ignore_redraw = False
        self.image_rows = []
        self.object_rows = []
        self.mask_rows = []
        self.measurement_rows = []
        self.frame.set_subplots((1, 1))
        self.axes = self.frame.subplot(0, 0)
        self.axes.invert_yaxis()
        interpolation = get_interpolation_mode()
        if interpolation == IM_NEAREST:
            interpolation = cellprofiler.gui.artist.INTERPOLATION_NEAREST
        elif interpolation == IM_BILINEAR:
            interpolation = cellprofiler.gui.artist.INTERPOLATION_BILINEAR
        else:
            interpolation = cellprofiler.gui.artist.INTERPOLATION_BICUBIC
        self.image = cellprofiler.gui.artist.CPImageArtist(interpolation=interpolation)
        assert isinstance(self.axes, matplotlib.axes.Axes)
        self.axes.add_artist(self.image)
        self.axes.set_aspect("equal")
        self.__axes_scale = None

        panel = self.frame.secret_panel
        panel.Sizer = wx.BoxSizer(wx.VERTICAL)
        self.panel = panel
        #
        # Make a grid of image controls
        #
        panel.Sizer.AddSpacer(4)
        self.image_grid = wx.GridBagSizer(vgap=3, hgap=3)
        sub_sizer = wx.BoxSizer(wx.VERTICAL)
        panel.Sizer.Add(sub_sizer, 0, wx.ALIGN_LEFT | wx.LEFT | wx.RIGHT, 3)
        sub_sizer.Add(self.image_grid, 0, wx.ALIGN_LEFT)
        self.image_grid.Add(
            wx.StaticText(panel, label="Images"),
            (0, self.C_CHOOSER),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM,
        )
        self.image_grid.Add(
            wx.StaticText(panel, label="Color"),
            (0, self.C_COLOR),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM,
        )
        self.image_grid.Add(
            wx.StaticText(panel, label="Show"),
            (0, self.C_SHOW),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM,
        )
        self.image_grid.Add(
            wx.StaticText(panel, label="Remove"),
            (0, self.C_REMOVE),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM,
        )
        self.add_image_row(can_delete=False)
        add_image_button = wx.Button(panel, label="Add Image")
        sub_sizer.Add(add_image_button, 0, wx.ALIGN_RIGHT)
        add_image_button.Bind(wx.EVT_BUTTON, lambda event: self.add_image_row())
        panel.Sizer.AddSpacer(4)
        panel.Sizer.Add(wx.StaticLine(panel, style=wx.LI_HORIZONTAL), 0, wx.EXPAND)
        panel.Sizer.AddSpacer(4)
        #
        # Make a grid of object controls
        #
        self.object_grid = wx.GridBagSizer(vgap=3, hgap=3)
        sub_sizer = wx.BoxSizer(wx.VERTICAL)
        panel.Sizer.Add(sub_sizer, 0, wx.ALIGN_LEFT | wx.LEFT | wx.RIGHT, 3)
        sub_sizer.Add(self.object_grid, 0, wx.ALIGN_LEFT)
        self.object_grid.Add(
            wx.StaticText(panel, label="Objects"),
            (0, self.C_CHOOSER),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM,
        )
        self.object_grid.Add(
            wx.StaticText(panel, label="Color"),
            (0, self.C_COLOR),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM,
        )
        self.object_grid.Add(
            wx.StaticText(panel, label="Show"),
            (0, self.C_SHOW),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM,
        )
        self.object_grid.Add(
            wx.StaticText(panel, label="Remove"),
            (0, self.C_REMOVE),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM,
        )
        self.add_objects_row(can_delete=False)
        add_object_button = wx.Button(panel, label="Add Objects")
        sub_sizer.Add(add_object_button, 0, wx.ALIGN_RIGHT)
        add_object_button.Bind(wx.EVT_BUTTON, lambda event: self.add_objects_row())
        panel.Sizer.AddSpacer(4)
        panel.Sizer.Add(wx.StaticLine(panel, style=wx.LI_HORIZONTAL), 0, wx.EXPAND)
        panel.Sizer.AddSpacer(4)
        #
        # Make a grid of mask controls
        #
        self.mask_grid = wx.GridBagSizer(vgap=3, hgap=3)
        sub_sizer = wx.BoxSizer(wx.VERTICAL)
        panel.Sizer.Add(sub_sizer, 0, wx.ALIGN_LEFT | wx.LEFT | wx.RIGHT, 3)
        sub_sizer.Add(self.mask_grid, 0, wx.ALIGN_LEFT)
        self.mask_grid.Add(
            wx.StaticText(panel, label="Masks"),
            (0, self.C_CHOOSER),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM,
        )
        self.mask_grid.Add(
            wx.StaticText(panel, label="Color"),
            (0, self.C_COLOR),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM,
        )
        self.mask_grid.Add(
            wx.StaticText(panel, label="Show"),
            (0, self.C_SHOW),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM,
        )
        self.mask_grid.Add(
            wx.StaticText(panel, label="Remove"),
            (0, self.C_REMOVE),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM,
        )
        self.add_mask_row(can_delete=False)
        add_mask_button = wx.Button(panel, label="Add Mask")
        sub_sizer.Add(add_mask_button, 0, wx.ALIGN_RIGHT)
        add_mask_button.Bind(wx.EVT_BUTTON, lambda event: self.add_mask_row())
        panel.Sizer.AddSpacer(4)
        panel.Sizer.Add(wx.StaticLine(panel, style=wx.LI_HORIZONTAL), 0, wx.EXPAND)
        panel.Sizer.AddSpacer(4)
        #
        # Make a grid of measurements to display
        #
        self.m_grid = wx.GridBagSizer(vgap=3, hgap=3)
        sub_sizer = wx.BoxSizer(wx.VERTICAL)
        panel.Sizer.Add(sub_sizer, 0, wx.ALIGN_LEFT | wx.LEFT | wx.RIGHT, 3)
        sub_sizer.Add(self.m_grid, 0, wx.ALIGN_LEFT)
        self.m_grid.Add(
            wx.StaticText(panel, label="Measurement"),
            (0, self.C_CHOOSER),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM,
        )
        self.m_grid.Add(
            wx.StaticText(panel, label="Font"),
            (0, self.C_COLOR),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM,
        )
        self.m_grid.Add(
            wx.StaticText(panel, label="Show"),
            (0, self.C_SHOW),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM,
        )
        self.m_grid.Add(
            wx.StaticText(panel, label="Remove"),
            (0, self.C_REMOVE),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_BOTTOM,
        )
        self.add_measurement_row(can_delete=False)
        add_measurement_button = wx.Button(panel, label="Add Measurement")
        sub_sizer.Add(add_measurement_button, 0, wx.ALIGN_RIGHT)
        add_measurement_button.Bind(wx.EVT_BUTTON, self.on_add_measurement_row)
        panel.Sizer.AddSpacer(6)
        help_button = wx.Button(panel, wx.ID_HELP)
        panel.Sizer.Add(help_button, 0, wx.ALIGN_RIGHT)

        def on_help(event):
            cellprofiler.gui.htmldialog.HTMLDialog(
                panel,
                "Workspace viewer help",
                cellprofiler.gui.help.WORKSPACE_VIEWER_HELP,
            ).Show()

        help_button.Bind(wx.EVT_BUTTON, on_help)
        self.image.add_to_menu(self.frame, self.frame.menu_subplots)
        self.frame.Bind(wx.EVT_CONTEXT_MENU, self.on_context_menu)
        self.frame.Bind(wx.EVT_CLOSE, self.on_frame_close)
        self.set_workspace(workspace)
        self.frame.secret_panel.Show()
        w, h = self.frame.GetSize()
        w += self.frame.secret_panel.GetMinWidth()
        self.frame.SetSize(wx.Size(w, h))

    def scale_axes(self):
        """Set the axes limits appropriate to the images we have"""
        ax = self.image.axes
        if self.frame.navtoolbar.is_home():
            max_x = max_y = 0
            for image_row in self.image_rows:
                if image_row.data.mode != cellprofiler.gui.artist.MODE_HIDE:
                    shape = image_row.data.pixel_data.shape
                    max_x = max(shape[1], max_x)
                    max_y = max(shape[0], max_y)
            if max_x > 0 and max_y > 0:
                max_x -= 0.5
                max_y -= 0.5
                ax.set_xlim(-0.5, max_x)
                ax.set_ylim(-0.5, max_y)
                ax.invert_yaxis()
                self.__axes_scale = (max_y, max_x)
                self.frame.navtoolbar.reset()

    def layout(self):
        self.panel.SetMinSize(
            (self.panel.GetVirtualSize()[0], self.panel.GetMinHeight())
        )
        self.panel.Layout()
        self.frame.secret_panel.Layout()
        self.panel.SetupScrolling()
        for child in self.panel.GetChildren():
            child.Refresh()

    def on_frame_close(self, event):
        assert isinstance(event, wx.CloseEvent)
        if event.CanVeto():
            self.frame.Hide()
            event.Veto()

    def add_image_row(self, can_delete=True):
        self.add_row(
            WorkspaceViewImageRow, self.image_rows, self.image_grid, can_delete
        )

    def add_objects_row(self, can_delete=True):
        self.add_row(
            WorkspaceViewObjectsRow, self.object_rows, self.object_grid, can_delete
        )

    def add_mask_row(self, can_delete=True):
        self.add_row(WorkspaceViewMaskRow, self.mask_rows, self.mask_grid, can_delete)

    def add_row(self, row_class, rows, grid_sizer, can_delete):
        row = len(rows) + 1
        color = (
            wx.RED
            if row == 1
            else wx.GREEN
            if row == 2
            else wx.BLUE
            if row == 3
            else wx.WHITE
        )
        vw_row = row_class(self, color, can_delete)

        grid_sizer.Add(vw_row.chooser, (row, self.C_CHOOSER), flag=wx.EXPAND)
        grid_sizer.Add(vw_row.color_ctrl, (row, self.C_COLOR), flag=wx.EXPAND)
        grid_sizer.Add(vw_row.show_check, (row, self.C_SHOW), flag=wx.ALIGN_CENTER)
        grid_sizer.Add(vw_row.remove_button, (row, self.C_REMOVE), flag=wx.ALIGN_CENTER)
        rows.append(vw_row)
        if can_delete:

            def remove_this_row(
                event,
                rows=rows,
                grid_sizer=grid_sizer,
                remove_button=vw_row.remove_button,
            ):
                self.remove_row(rows, grid_sizer, remove_button)

            vw_row.remove_button.Bind(wx.EVT_BUTTON, remove_this_row)
        self.update_menu(self.frame.menu_subplots)
        self.layout()

    def remove_row(self, rows, grid_sizer, remove_button):
        for i, vw_row in enumerate(rows):
            if vw_row.remove_button == remove_button:
                break
        else:
            return
        for control in (
            vw_row.chooser,
            vw_row.color_ctrl,
            vw_row.show_check,
            vw_row.remove_button,
        ):
            grid_sizer.Remove(control)
            control.Destroy()
        self.image.remove(vw_row.data)
        rows.remove(vw_row)
        for ii in range(i, len(rows)):
            vw_row = rows[ii]
            for j, control in enumerate(
                (
                    vw_row.chooser,
                    vw_row.color_ctrl,
                    vw_row.show_check,
                    vw_row.remove_button,
                )
            ):
                grid_sizer.SetItemPosition(control, (ii + 1, j))
        self.update_menu(self.frame.menu_subplots)
        self.layout()
        self.redraw()

    def on_add_measurement_row(self, event):
        self.add_measurement_row()
        self.layout()
        self.redraw()

    def add_measurement_row(self, can_delete=True):
        row_idx = len(self.measurement_rows) + 1
        mr = []
        panel = self.panel
        row = WorkspaceViewMeasurementRow(
            panel, self.m_grid, row_idx, lambda: self.on_measurement_changed(mr[0])
        )
        mr.append(row)
        self.measurement_rows.append(row)
        bitmap = wx.ArtProvider.GetBitmap(wx.ART_DELETE, wx.ART_TOOLBAR, (16, 16))

        remove_button = wx.BitmapButton(panel, bitmap=bitmap)
        self.m_grid.Add(
            remove_button,
            (row_idx, self.C_REMOVE),
            flag=wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_TOP,
        )
        remove_button.Bind(
            wx.EVT_BUTTON, lambda event: self.remove_measurement_row(row, remove_button)
        )
        if not can_delete:
            remove_button.Hide()
        row.update(self.workspace)

    def remove_measurement_row(self, measurement_row, remove_button):
        if measurement_row in self.measurement_rows:
            idx = self.measurement_rows.index(measurement_row)
            measurement_row.destroy(self.m_grid)
            self.m_grid.Remove(remove_button)
            remove_button.Destroy()
            self.measurement_rows.remove(measurement_row)
            for ii in range(idx, len(self.measurement_rows)):
                for j in (self.C_CHOOSER, self.C_COLOR, self.C_SHOW, self.C_REMOVE):
                    item = self.m_grid.FindItemAtPosition(wx.GBPosition(ii + 1, j))
                    self.m_grid.SetItemPosition(item, (ii, j))
            self.layout()
            self.redraw()

    def on_measurement_changed(self, measurement_row):
        assert isinstance(measurement_row, WorkspaceViewMeasurementRow)
        measurement_row.update(self.workspace)
        self.layout()
        self.redraw()

    def set_workspace(self, workspace):
        """Rebuild the workspace control panel"""
        self.workspace = workspace
        self.ignore_redraw = True
        try:
            self.update_menu(self.frame.menu_subplots)
            self.update_choices(self.image_rows)
            self.update_choices(self.object_rows)
            self.update_choices(self.mask_rows)
            for measurement_row in self.measurement_rows:
                measurement_row.update(workspace)
        finally:
            self.ignore_redraw = False
        self.redraw()

    def update_menu(self, menu):
        event = wx.CommandEvent(wx.EVT_MENU_OPEN.evtType[0], self.frame.GetId())
        event.SetEventObject(self.frame)
        self.image.on_update_menu(event, menu)

    @staticmethod
    def update_choices(rows):
        for row in rows:
            row.update_chooser()

    def on_context_menu(self, event):
        menu = wx.Menu()
        try:
            self.image.add_to_menu(self.frame, menu)
            self.update_menu(menu)
            self.frame.PopupMenu(menu)
        finally:
            menu.Destroy()

    def redraw(self, event=None):
        if self.ignore_redraw:
            return
        for vw_row in self.image_rows + self.object_rows + self.mask_rows:
            vw_row.update()
        if not self.frame.figure.canvas.IsShown():
            self.frame.figure.canvas.Show()
            self.layout()
        #
        # Remove all the old text labels
        #
        to_remove = []
        for artist in list(self.axes.texts):
            artist.remove()

        m = self.workspace.measurements
        assert isinstance(m, Measurements)
        title_lines = []
        object_values = {}
        for measurement_row in self.measurement_rows:
            assert isinstance(measurement_row, WorkspaceViewMeasurementRow)
            if not measurement_row.should_show():
                continue
            object_name = measurement_row.get_object_name()
            if object_name is None or object_name not in m.get_object_names():
                continue
            feature = measurement_row.get_measurement_name()
            if feature is None or not m.has_feature(object_name, feature):
                continue

            value = m[object_name, feature]
            if object_name in (IMAGE, EXPERIMENT,):
                if isinstance(value, int):
                    fmt = "%s: %d"
                elif isinstance(value, float):
                    fmt = "%s: %.4f"
                else:
                    fmt = "%s: %s"
                title_lines.append(fmt % (feature, value))
            else:
                if object_name not in object_values:
                    if any(
                        [
                            not m.has_feature(object_name, lf)
                            for lf in (M_LOCATION_CENTER_X, M_LOCATION_CENTER_Y,)
                        ]
                    ):
                        continue
                    object_values[object_name] = []
                object_values[object_name].append((value, measurement_row))
        if len(title_lines) > 0:
            self.axes.set_title("\n".join(title_lines))
        else:
            self.axes.set_title("Image set # %d" % m.image_number)
        for object_name, value_rows in list(object_values.items()):
            values = [vr[0] for vr in value_rows]
            measurement_rows = [vr[1] for vr in value_rows]
            x, y = [
                m[object_name, ftr]
                for ftr in (M_LOCATION_CENTER_X, M_LOCATION_CENTER_Y,)
            ]
            for i in range(len(x)):
                xi, yi = x[i], y[i]
                if numpy.isnan(xi) or numpy.isnan(yi):
                    continue
                height = 0
                for j, measurement_row in enumerate(measurement_rows):
                    if len(values[j]) <= i or numpy.isnan(values[j][i]):
                        continue
                    value = values[j][i]
                    font = measurement_row.font
                    if font.GetStyle() == wx.ITALIC:
                        fontstyle = "italic"
                    else:
                        fontstyle = "normal"
                    color = measurement_row.foreground_color
                    fontcolor, backgroundcolor = [
                        tuple([float(c) / 255 for c in color][:3])
                        for color in (
                            measurement_row.foreground_color,
                            measurement_row.background_color,
                        )
                    ]

                    fmt = "%%.%df" % measurement_row.precision
                    a = self.axes.annotate(
                        fmt % value,
                        (xi, yi),
                        xytext=(0, -height),
                        textcoords="offset points",
                        ha="center",
                        va="center",
                        bbox={
                            "boxstyle": measurement_row.box_style,
                            "fc": backgroundcolor,
                            "alpha": measurement_row.background_alpha,
                        },
                        color=fontcolor,
                        family=font.GetFaceName(),
                        fontsize=font.GetPointSize(),
                        fontstyle=fontstyle,
                        weight=font.GetWeight(),
                    )
                    height += font.GetPointSize() + 1

        self.scale_axes()
        self.frame.figure.canvas.draw()
