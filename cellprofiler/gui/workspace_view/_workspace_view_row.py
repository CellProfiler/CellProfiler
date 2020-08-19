import wx
import wx.lib.colourselect

from ..artist import MODE_HIDE


class WorkspaceViewRow:
    """A row of controls and a data item"""

    def __init__(self, workspace_view, color, can_delete):
        self.workspace_view = workspace_view
        panel = workspace_view.panel
        self.chooser = wx.Choice(panel)
        self.color_ctrl = wx.lib.colourselect.ColourSelect(panel, colour=color)
        self.show_check = wx.CheckBox(panel)
        bitmap = wx.ArtProvider.GetBitmap(wx.ART_DELETE, wx.ART_TOOLBAR, (16, 16))
        self.remove_button = wx.BitmapButton(panel, bitmap=bitmap)
        if not can_delete:
            self.remove_button.Hide()
        self.chooser.Bind(wx.EVT_CHOICE, self.on_choice)
        self.color_ctrl.Bind(wx.lib.colourselect.EVT_COLOURSELECT, self.on_color_change)
        self.show_check.Bind(wx.EVT_CHECKBOX, self.on_check_change)
        self.update_chooser(first=True)
        self.last_mode = None

    @property
    def color(self):
        """The color control's current color scaled for matplotlib"""
        return tuple([float(x) / 255 for x in self.color_ctrl.GetColour()])

    def on_choice(self, event):
        self.data.name = self.chooser.GetStringSelection()
        self.show_check.SetValue(True)
        self.workspace_view.redraw()

    def on_color_change(self, event):
        self.data.color = tuple([float(c) / 255.0 for c in self.color_ctrl.GetColour()])
        self.workspace_view.redraw()

    def on_check_change(self, event):
        self.workspace_view.redraw()

    def update(self):
        name = self.chooser.GetStringSelection()
        names = sorted(self.get_names())
        image_set = self.workspace_view.workspace.image_set
        if self.show_check.IsChecked() and name in names:
            self.data.name = name
            self.update_data(name)
            if self.data.mode == MODE_HIDE:
                self.data.mode = self.last_mode
        elif self.data.mode != MODE_HIDE:
            self.last_mode = self.data.get_raw_mode()
            self.data.mode = MODE_HIDE
        self.update_chooser()

    def update_chooser(self, first=False):
        """Update the chooser with the given list of names"""
        name = self.chooser.GetStringSelection()
        names = self.get_names()
        current_names = sorted(self.chooser.GetItems())
        if tuple(current_names) != tuple(names):
            if name not in names:
                names = sorted(list(names) + [name])
            self.chooser.SetItems(names)
            self.chooser.SetStringSelection(name)
        if first and len(names) > 0:
            name = names[0]
            self.chooser.SetStringSelection(name)

    def get_names(self):
        return []

    def data(self):
        pass

    def update_data(self, name):
        pass
