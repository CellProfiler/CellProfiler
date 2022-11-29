import logging
import textwrap

import wx
import wx.lib.scrolledpanel
from wx.lib.agw.infobar import AutoWrapStaticText

from cellprofiler_core.preferences import config_write_typed, config_read_typed

from cellprofiler_core.reader import activate_readers

from cellprofiler_core.constants.reader import ALL_READERS, BAD_READERS, AVAILABLE_READERS
from ..html.utils import rst_to_html_fragment
from ..htmldialog import HTMLDialog


class ReadersDialog(wx.Dialog):
    """
    Display a dialog for configuring readers

    The dialog handles fetching current defaults and setting the
    defaults when the user hits OK.
    """

    def __init__(
        self,
        parent=None,
        ID=-1,
        title="CellProfiler Image Readers",
        size=wx.DefaultSize,
        pos=wx.DefaultPosition,
        style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER,
        name=wx.DialogNameStr,
    ):
        wx.Dialog.__init__(self, parent, ID, title, pos, size, style, name)
        self.changed_settings = {}
        self.SetExtraStyle(wx.WS_EX_VALIDATE_RECURSIVELY)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.SetMinSize((800, 600))
        self.SetSizer(main_sizer)
        h_splitter = wx.BoxSizer(wx.HORIZONTAL)
        main_sizer.Add(h_splitter, 1, wx.EXPAND)
        v_splitter = wx.BoxSizer(wx.VERTICAL)
        self.active_readers = wx.ListBox(self)
        self.inactive_readers = wx.ListBox(self)
        active_header = wx.StaticText(self, label="Active Readers")
        inactive_header = wx.StaticText(self, label="Inactive Readers")
        self.header_font = active_header.GetFont().MakeBold()
        active_header.SetFont(self.header_font)
        inactive_header.SetFont(self.header_font)
        v_splitter.Add(active_header)
        v_splitter.Add(self.active_readers, 1, wx.EXPAND)
        v_splitter.AddSpacer(10)
        v_splitter.Add(inactive_header)
        v_splitter.Add(self.inactive_readers, 1, wx.EXPAND)

        self.control_panel = wx.lib.scrolledpanel.ScrolledPanel(self)
        self.control_panel.SetupScrolling(scroll_x=False)

        self.control_sizer = wx.BoxSizer(wx.VERTICAL)
        self.control_panel.SetSizer(self.control_sizer)
        h_splitter.Add(v_splitter, proportion=1, flag=wx.EXPAND | wx.ALL, border=10)
        h_splitter.Add(self.control_panel, proportion=3, flag=wx.EXPAND | wx.ALL, border=10)

        control_frame_header = wx.StaticText(self.control_panel, label="Reader Properties")
        control_frame_header.SetFont(self.header_font)
        self.control_sizer.Add(control_frame_header, flag=wx.ALL)

        default_text = wx.StaticText(self.control_panel, label="Select a reader to configure")
        self.control_sizer.AddSpacer(20)
        self.control_sizer.Add(default_text, 0, wx.EXPAND)

        self.active_readers.Bind(wx.EVT_LISTBOX, self.on_select_reader)
        self.inactive_readers.Bind(wx.EVT_LISTBOX, self.on_select_reader)

        btnsizer = wx.StdDialogButtonSizer()
        btnSave = wx.Button(self, wx.ID_SAVE)
        btnCancel = wx.Button(self, wx.ID_CANCEL)
        btnsizer.SetAffirmativeButton(btnSave)
        btnsizer.SetCancelButton(btnCancel)
        self.Bind(wx.EVT_BUTTON, self.save_reader_config, id=wx.ID_SAVE)
        btnsizer.Realize()
        main_sizer.Add(btnsizer, 0, wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 5)
        self.populate_readers()
        self.Fit()

    def populate_readers(self):
        disabled = set(ALL_READERS) - set(AVAILABLE_READERS)
        self.active_readers.SetItems(list(AVAILABLE_READERS))
        self.inactive_readers.SetItems(list(disabled) + list(BAD_READERS))

    def on_select_reader(self, event):
        if event.EventObject is self.active_readers:
            self.inactive_readers.SetSelection(-1)
            selected = event.EventObject.GetStringSelection()
            reader_class = ALL_READERS[selected]
            self.display_functional_reader(reader_class)
        elif event.EventObject is self.inactive_readers:
            self.active_readers.SetSelection(-1)
            selected = event.EventObject.GetStringSelection()
            if selected in ALL_READERS:
                reader_class = ALL_READERS[selected]
                self.display_functional_reader(reader_class)
            else:
                self.display_nonfunctional_reader(selected)
        else:
            logging.warning(f"Unknown selection {event}")
            return
        self.Layout()
        self.Refresh()

    def display_functional_reader(self, reader_class):
        self.reset_control_frame()
        reader_name = reader_class.reader_name
        reader_description = reader_class.__doc__
        nm = wx.StaticText(self.control_panel, label=reader_name)

        nm.SetFont(self.header_font)
        self.control_sizer.Add(nm, 0, wx.ALIGN_CENTER_HORIZONTAL)
        self.control_sizer.AddSpacer(10)
        text_chunk = AutoWrapStaticText(self.control_panel, label=reader_description)
        self.control_sizer.Add(text_chunk, 0, wx.EXPAND)
        self.control_sizer.AddSpacer(5)
        enabled_checkbox = wx.CheckBox(self.control_panel, label="Enabled:", style=wx.ALIGN_RIGHT)
        enabled_checkbox.SetFont(self.header_font)
        enabled_key = f"Reader.{reader_name}.enabled"
        is_enabled = self.fetch_preference(enabled_key, bool)
        if is_enabled or is_enabled is None:
            is_enabled = True
        else:
            is_enabled = False
        enabled_checkbox.SetValue(is_enabled)

        def toggle_callback(event):
            enabled = enabled_checkbox.GetValue()
            self.changed_settings[enabled_key] = enabled
            if enabled:
                self.active_readers.Insert(reader_name, 0)
                self.active_readers.SetSelection(0)
                self.active_readers.SetFocus()
                tgt = self.inactive_readers.FindString(reader_name)
                self.inactive_readers.Delete(tgt)
                self.inactive_readers.SetSelection(-1)
            else:
                self.inactive_readers.Insert(reader_name, 0)
                self.inactive_readers.SetSelection(0)
                self.inactive_readers.SetFocus()
                tgt = self.active_readers.FindString(reader_name)
                self.active_readers.Delete(tgt)
                self.active_readers.SetSelection(-1)

        self.Bind(wx.EVT_CHECKBOX, toggle_callback, enabled_checkbox)
        self.control_sizer.Add(enabled_checkbox)

        self.control_sizer.AddSpacer(5)
        ver = wx.StaticText(self.control_panel)
        ver.SetLabelMarkup(f"<b>Version:</b> {reader_class.variable_revision_number}")
        self.control_sizer.Add(ver)

        self.control_sizer.AddSpacer(5)
        ext_desc = f"<b>Supported extensions:</b> {', '.join(reader_class.supported_filetypes)}"
        ext_desc = (textwrap.fill(ext_desc, self.control_sizer.GetSize().GetWidth() // 6))
        ext_st = wx.StaticText(self.control_panel, label='')
        ext_st.SetLabelMarkup(ext_desc)
        self.control_sizer.Add(ext_st, 0, wx.EXPAND | wx.TE_RICH)

        self.control_sizer.AddSpacer(5)
        scheme_desc = f"<b>Supported schemes:</b> {', '.join(reader_class.supported_schemes)}"
        scheme_desc = (textwrap.fill(scheme_desc, self.control_sizer.GetSize().GetWidth() // 6))
        scheme_st = wx.StaticText(self.control_panel, label='')
        scheme_st.SetLabelMarkup(scheme_desc)
        self.control_sizer.Add(scheme_st, 0, wx.EXPAND | wx.TE_RICH)

        settings = reader_class.get_settings()
        if settings:
            self.control_sizer.AddSpacer(5)
            settings_head = wx.StaticText(self.control_panel, label="Settings:")
            settings_head.SetFont(self.header_font)
            self.control_sizer.Add(settings_head)
            for key, name, desc, setting_type, default in settings:
                ident = wx.NewId()
                self.control_sizer.AddSpacer(5)
                setting_sizer = wx.BoxSizer(orient=wx.HORIZONTAL)
                self.control_sizer.Add(setting_sizer, flag=wx.EXPAND, border=5)
                setting_sizer.AddSpacer(10)
                setting_sizer.Add(wx.StaticText(self.control_panel, label=f"{name}:  "))
                widget_key = f"Reader.{reader_name}.{key}"
                current_value = self.fetch_preference(widget_key, setting_type)
                if current_value is None:
                    current_value = default

                if setting_type == bool:
                    widget = wx.CheckBox(self.control_panel, id=ident)
                    widget.SetValue(bool(current_value))

                    def callback(event, obj=widget, config_key=widget_key):
                        self.changed_settings[config_key] = obj.GetValue()

                    self.Bind(wx.EVT_CHECKBOX, callback, widget, id=ident)
                elif setting_type == int:
                    widget = wx.SpinCtrl(self.control_panel, min=-65536, max=65536,
                                         id=ident)
                    widget.SetValue(int(current_value))

                    def callback(event, obj=widget, config_key=widget_key):
                        self.changed_settings[config_key] = obj.GetValue()

                    self.Bind(wx.EVT_SPINCTRL, callback, widget, id=ident)
                elif setting_type == float:
                    widget = wx.SpinCtrlDouble(self.control_panel, id=ident,
                                               min=-65536, max=65536)
                    widget.SetValue(float(current_value))

                    def callback(event, obj=widget, config_key=widget_key):
                        self.changed_settings[config_key] = obj.GetValue()

                    self.Bind(wx.EVT_SPINCTRLDOUBLE, callback, widget, id=ident)
                else:
                    widget = wx.TextCtrl(self.control_panel, value=current_value, id=ident)

                    def callback(event, obj=widget, config_key=widget_key):
                        self.changed_settings[config_key] = obj.GetValue()

                    self.Bind(wx.EVT_TEXT, callback, widget, id=ident)
                widget.Refresh()
                setting_sizer.Add(widget, proportion=1)

                help_button = wx.Button(self.control_panel, -1, "?", (0, 0), (30, -1))

                def on_help(event, help_text=desc):
                    dlg = HTMLDialog(
                        self, "Preferences help", rst_to_html_fragment(help_text),
                    )
                    dlg.Show()

                setting_sizer.AddStretchSpacer()
                setting_sizer.Add(help_button)
                self.Bind(wx.EVT_BUTTON, on_help, help_button)

        self.control_sizer.Layout()

    def display_nonfunctional_reader(self, selected):
        self.reset_control_frame()
        reader_name = selected
        error = BAD_READERS[reader_name]
        nm = wx.StaticText(self.control_panel, label=reader_name)
        nm.SetFont(self.header_font)
        self.control_sizer.Add(nm, 0, wx.ALIGN_CENTER_HORIZONTAL)
        self.control_sizer.AddSpacer(10)
        text_chunk = AutoWrapStaticText(self.control_panel, label="Reader failed to load")
        text_chunk.SetForegroundColour((255, 0, 0))
        text_chunk.SetFont(self.header_font)
        self.control_sizer.Add(text_chunk, 0, wx.EXPAND)
        self.control_sizer.AddSpacer(5)

        err_header = wx.StaticText(self.control_panel)
        err_header.SetLabelMarkup("<b>Error Trace:</b>")
        self.control_sizer.Add(err_header)
        self.control_sizer.AddSpacer(5)
        trace_text = AutoWrapStaticText(self.control_panel, label=error)
        self.control_sizer.Add(trace_text)

        def copy_handler(event):
            if wx.TheClipboard.Open():
                wx.TheClipboard.SetData(wx.TextDataObject(error))
                wx.TheClipboard.Close()
            else:
                logging.error("Failed to copy to clipboard")

        copy_button = wx.Button(self.control_panel, -1, "Copy to clipboard")
        self.Bind(wx.EVT_BUTTON, copy_handler, copy_button)
        self.control_sizer.Add(copy_button, 0, wx.ALIGN_RIGHT, 5)

        self.control_sizer.Layout()

    def reset_control_frame(self):
        self.control_sizer.Clear(True)
        control_frame_header = wx.StaticText(self.control_panel, label="Reader Properties")
        control_frame_header.SetFont(self.header_font)
        self.control_sizer.Add(control_frame_header, flag=wx.ALL)

        self.control_sizer.AddSpacer(20)

    def fetch_preference(self, key, key_type):
        return self.changed_settings.get(key, config_read_typed(key, key_type))

    def save_reader_config(self, event):
        event.Skip()
        changed_active_readers = False
        for key, value in self.changed_settings.items():
            if key.endswith('.enabled'):
                changed_active_readers = True
            config_write_typed(key, value)
        if changed_active_readers:
            activate_readers()
        self.Close()
        if not AVAILABLE_READERS:
            wx.MessageBox(
                'No image readers are enabled!\nCellProfiler will be unable to load data!', caption="Reader config"
            )
