# # coding=utf-8
# #TODO: this file uselss until CellProfiler/CellProfiler#4684 is resolved
# """omerologin - dialog box to capture login credentials for Omero
# """


# from cellprofiler_core.bioformats import formatreader
# import wx
# from cellprofiler_core.preferences import (
#     get_omero_server,
#     get_omero_port,
#     get_omero_user,
#     set_omero_server,
#     set_omero_port,
#     set_omero_user,
# )


# class OmeroLoginDlg(wx.Dialog):
#     SERVER_LABEL = "Server:"
#     PORT_LABEL = "Port:"
#     USER_LABEL = "User:"
#     PASSWORD_LABEL = "Password:"

#     def __init__(self, *args, **kwargs):
#         super(self.__class__, self).__init__(*args, **kwargs)

#         self.server = get_omero_server() or ""
#         self.port = get_omero_port()
#         self.user = get_omero_user() or ""
#         self.session_id = None
#         self.SetSizer(wx.BoxSizer(wx.VERTICAL))
#         sizer = wx.BoxSizer(wx.VERTICAL)
#         self.Sizer.Add(sizer, 1, wx.EXPAND | wx.ALL, 6)
#         sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
#         sizer.Add(sub_sizer, 0, wx.EXPAND)

#         max_width = 0
#         max_height = 0
#         for label in (
#             self.SERVER_LABEL,
#             self.PORT_LABEL,
#             self.USER_LABEL,
#             self.PASSWORD_LABEL,
#         ):
#             w, h, _, _ = self.GetFullTextExtent(label)
#             max_width = max(w, max_width)
#             max_height = max(h, max_height)

#         # Add extra padding
#         lsize = wx.Size(max_width + 5, max_height)
#         sub_sizer.Add(
#             wx.StaticText(self, label="Server:", size=lsize),
#             0, wx.ALIGN_CENTER_VERTICAL)
#         self.omero_server_ctrl = wx.TextCtrl(self, value=self.server)
#         sub_sizer.Add(self.omero_server_ctrl, 1, wx.EXPAND)

#         sizer.AddSpacer(5)
#         sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
#         sizer.Add(sub_sizer, 0, wx.EXPAND)
#         sub_sizer.Add(
#             wx.StaticText(self, label="Port:", size=lsize),
#             0, wx.ALIGN_CENTER_VERTICAL)
#         self.omero_port_ctrl = wx.TextCtrl(self, value=str(self.port))
#         sub_sizer.Add(self.omero_port_ctrl, 1, wx.EXPAND)

#         sizer.AddSpacer(5)
#         sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
#         sizer.Add(sub_sizer, 0, wx.EXPAND)
#         sub_sizer.Add(
#             wx.StaticText(self, label="User:", size=lsize),
#             0, wx.ALIGN_CENTER_VERTICAL)
#         self.omero_user_ctrl = wx.TextCtrl(self, value=self.user)
#         sub_sizer.Add(self.omero_user_ctrl, 1, wx.EXPAND)

#         sizer.AddSpacer(5)
#         sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
#         sizer.Add(sub_sizer, 0, wx.EXPAND)
#         sub_sizer.Add(
#             wx.StaticText(self, label="Password:", size=lsize),
#             0, wx.ALIGN_CENTER_VERTICAL)
#         self.omero_password_ctrl = wx.TextCtrl(self, value="", style=wx.TE_PASSWORD)
#         sub_sizer.Add(self.omero_password_ctrl, 1, wx.EXPAND)

#         sizer.AddSpacer(5)
#         sub_sizer = wx.BoxSizer(wx.HORIZONTAL)
#         sizer.Add(sub_sizer, 0, wx.EXPAND)
#         connect_button = wx.Button(self, label="Connect")
#         connect_button.Bind(wx.EVT_BUTTON, self.on_connect_pressed)
#         sub_sizer.Add(connect_button, 0, wx.EXPAND)
#         sub_sizer.AddSpacer(5)

#         self.message_ctrl = wx.StaticText(self, label="Not connected")
#         sub_sizer.Add(self.message_ctrl, 1, wx.EXPAND)

#         button_sizer = wx.StdDialogButtonSizer()
#         self.Sizer.Add(button_sizer, 0, wx.EXPAND)

#         cancel_button = wx.Button(self, wx.ID_CANCEL)
#         button_sizer.AddButton(cancel_button)
#         cancel_button.Bind(wx.EVT_BUTTON, self.on_cancel)

#         self.ok_button = wx.Button(self, wx.ID_OK)
#         button_sizer.AddButton(self.ok_button)
#         self.ok_button.Bind(wx.EVT_BUTTON, self.on_ok)
#         self.ok_button.Enable(False)
#         button_sizer.Realize()

#         self.omero_password_ctrl.Bind(wx.EVT_TEXT, self.mark_dirty)
#         self.omero_port_ctrl.Bind(wx.EVT_TEXT, self.mark_dirty)
#         self.omero_server_ctrl.Bind(wx.EVT_TEXT, self.mark_dirty)
#         self.omero_user_ctrl.Bind(wx.EVT_TEXT, self.mark_dirty)
#         self.Layout()

#     def mark_dirty(self, event):
#         if self.ok_button.IsEnabled():
#             self.ok_button.Enable(False)
#             self.message_ctrl.Label = "Please connect with your new credentials"
#             self.message_ctrl.ForegroundColour = "black"

#     def on_connect_pressed(self, event):
#         self.connect()

#     def connect(self):
#         try:
#             server = self.omero_server_ctrl.GetValue()
#             port = int(self.omero_port_ctrl.GetValue())
#             user = self.omero_user_ctrl.GetValue()
#         except:
#             self.message_ctrl.Label = (
#                 "The port number must be an integer between 0 and 65535 (try 4064)"
#             )
#             self.message_ctrl.ForegroundColour = "red"
#             self.message_ctrl.Refresh()
#             return False
#         try:
#             self.session_id = formatreader.set_omero_credentials(
#                 server, port, user, self.omero_password_ctrl.GetValue()
#             )
#             self.message_ctrl.Label = "Connected"
#             self.message_ctrl.ForegroundColour = "green"
#             self.message_ctrl.Refresh()
#             self.server = server
#             self.port = port
#             self.user = user
#             set_omero_server(server)
#             set_omero_port(port)
#             set_omero_user(user)
#             self.ok_button.Enable(True)
#             return True
#         except:
#             self.message_ctrl.Label = "Failed to log onto server"
#             self.message_ctrl.ForegroundColour = "red"
#             self.message_ctrl.Refresh()
#             return False

#     def on_cancel(self, event):
#         self.EndModal(wx.CANCEL)

#     def on_ok(self, event):
#         self.EndModal(wx.OK)
