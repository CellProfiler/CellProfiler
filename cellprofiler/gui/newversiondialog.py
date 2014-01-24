""" A new version dialog with buttons for going to the website and
turning off new version checking.

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import wx
import wx.html

class NewVersionDialog(wx.Dialog):
    def __init__(self, parent, title, contents, url, check_pref, check_pref_callback, skip_callback):
        super(NewVersionDialog, self).__init__(parent, -1, title, 
                                               style=(wx.DEFAULT_DIALOG_STYLE | 
                                                      wx.RESIZE_BORDER))
        html = wx.html.HtmlWindow(parent=self)
        html.SetPage(contents)
        
        self.url = url
        self.check_pref_callback = check_pref_callback
        self.check_pref = check_pref
        self.skip_callback = skip_callback

        check_pref_later = self.check_pref_later = wx.CheckBox(self, -1, 'Check for updates on startup?  ')
        check_pref_later.SetValue(check_pref)

        buttons_sizer = self.CreateStdDialogButtonSizer(wx.YES | wx.NO | wx.CANCEL)
        go_button = buttons_sizer.GetAffirmativeButton()
        go_button.SetLabel('Download new version')
        skip_button = buttons_sizer.GetNegativeButton()
        skip_button.SetLabel('Skip this version')
        remind_button = buttons_sizer.GetCancelButton()
        remind_button.SetLabel('Remind me later')

        # Layout
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(html, 1, wx.EXPAND | wx.ALL, 5)
        sizer.Add(check_pref_later, 0, wx.ALIGN_RIGHT)
        sizer.AddSpacer(5)
        sizer.Add(buttons_sizer, flag=wx.ALIGN_RIGHT|wx.EXPAND)

        border = wx.BoxSizer()
        border.Add(sizer, 1, wx.EXPAND | wx.ALL, 5)
        self.SetSizer(border)
        self.Layout()

        # events
        self.Bind(wx.EVT_BUTTON, self.check_pref_changed)
        self.Bind(wx.EVT_BUTTON, self.go_to_url, go_button)
        self.Bind(wx.EVT_BUTTON, self.skip_version, skip_button)

        html.SetFocus()
        self.SetMinSize(border.GetMinSize())
        self.Centre()

    def check_pref_changed(self, evt):
        if bool(self.check_pref_later.Value) != bool(self.check_pref):
            self.check_pref_callback(bool(self.check_pref_later.Value))
        evt.Skip()

    def skip_version(self, evt):
        self.skip_callback()
        self.Close()
        evt.Skip()

    def go_to_url(self, evt):
        if not wx.LaunchDefaultBrowser(self.url):
            wx.MessageBox("Could not open default browser (%s)"%(self.url), "Can't open browser", wx.ICON_EXCLAMATION)
            

if __name__ == "__main__":
    def cb(new_pref):
        print "Pref changed to", new_pref
        
    def sk():
        print "skip this version"

    app = wx.PySimpleApp()
    dialog = NewVersionDialog(None, "New version available", 
                              "<h1>NEW REVISION: TEH AWESOME</h1>awesome new features!<br>get it now!", 
                              "http://cellprofiler.org/", 
                              True, cb, sk)
    dialog.ShowModal()
    dialog.Destroy()
    app.MainLoop()

    
