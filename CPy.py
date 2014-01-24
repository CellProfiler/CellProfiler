#!/usr/bin/python
"""
CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""

import wx

class MyFrame(wx.Frame):
    def __init__(self, parent, id, title):
        wx.Frame.__init__(self, parent, id, title)
        panel = wx.Panel(self, -1)

        menuBar = wx.MenuBar()
        menu = wx.Menu()
        menu.Append(99,  "Show Image", "Show Image")
        menu.Append(100,  "Show Data on Image", "Show Data on Image")
        menuBar.Append(menu, "Image Tools")
        self.SetMenuBar(menuBar)

        self.CreateStatusBar()

        # top level:
        #       logo and pipeline | Module info
        #       --------------------------------------
        #       file list         | paths and settings
        toplevel_sizer = wx.FlexGridSizer(2, 2, 1, 1)
        logopipe_panel = wx.Panel(self, -1, style=wx.SIMPLE_BORDER)
        filelist_panel = wx.Panel(self, -1, style=wx.SIMPLE_BORDER)
        module_panel = wx.Panel(self, -1, style=wx.SIMPLE_BORDER)
        paths_analyze_panel = wx.Panel(self, -1, style=wx.SIMPLE_BORDER)

        filelist_text = wx.StaticText(filelist_panel, -1, 'File list', style=wx.ALIGN_CENTRE)
        self.module_text = wx.StaticText(module_panel, -1, 'Module text', style=wx.ALIGN_CENTRE)
        
        toplevel_sizer.AddMany([(logopipe_panel, 0, wx.EXPAND),
                                (module_panel, 1, wx.EXPAND),
                                (filelist_panel, 0, wx.EXPAND),
                                (paths_analyze_panel, 0, wx.EXPAND)])
        toplevel_sizer.AddGrowableCol(1)
        toplevel_sizer.AddGrowableRow(0)
        
        self.SetSizer(toplevel_sizer)


        PIPELINE_ID = 100
        # logopipe panel: logo and pipeline
        logopipe_box = wx.BoxSizer(wx.VERTICAL)
        logo_panel = wx.Panel(logopipe_panel, -1, style=wx.NO_BORDER)
        self.pipeline_list = wx.ListBox(logopipe_panel, PIPELINE_ID, style=wx.LB_SINGLE)
        module_list = ['LoadImage', 'IdentifyPrimaryAutomatic ', 'MeasureEverything']
        self.pipeline_list.InsertItems(module_list, 0)
        pipeline_controls = wx.Panel(logopipe_panel, -1, style=wx.SIMPLE_BORDER)
        logopipe_box.Add(logo_panel, 0,  wx.EXPAND | wx.ALL, 1)
        logopipe_box.Add(self.pipeline_list, 3, wx.EXPAND | wx.ALL, 1)
        logopipe_box.Add(pipeline_controls, 0, wx.EXPAND | wx.ALL, 1)
        logopipe_panel.SetSizer(logopipe_box)

        # pipeline controls
        wx.StaticText(pipeline_controls, -1, 'Pipeline Controls', style=wx.ALIGN_LEFT)

        # logo panel: logo and text
        logo_box = wx.BoxSizer(wx.HORIZONTAL)
        logo = wx.Bitmap('/Users/thouis/CPlogo.png')
        logoimg_panel = wx.Panel(logo_panel, -1, style=wx.NO_BORDER)
        logo_bitmap = wx.StaticBitmap(logoimg_panel)
        logo_bitmap.SetFocus()
        logo_bitmap.SetBitmap(logo)
        cp_text = wx.StaticText(logo_panel, -1, 'Pipeline:', style=wx.ALIGN_LEFT)
        logo_box.Add(cp_text, 4, wx.ALL | wx.ALIGN_BOTTOM, 1)
        logo_box.Add((1,1), 1)
        logo_box.Add(logoimg_panel, 4, wx.EXPAND | wx.ALL | wx.ALIGN_RIGHT, 1)
        logo_panel.SetSizer(logo_box)

        # Directories, Settings, and Analyze Images button
        paths_analyze_box = wx.BoxSizer(wx.VERTICAL)
        folders_panel = wx.Panel(paths_analyze_panel, -1, style=wx.NO_BORDER)
        analyze_panel = wx.Panel(paths_analyze_panel, -1)
        paths_analyze_box.Add(folders_panel, 1, wx.EXPAND | wx.ALIGN_CENTER, 1)
        paths_analyze_box.Add(analyze_panel, 0, wx.ALIGN_RIGHT | wx.ALL, 1)
        paths_analyze_panel.SetSizer(paths_analyze_box)

        # default input folder and default output folder

        #
        # ? | Default input folder | ___________ | Browse...
        # ? | Default output folder | ___________ | Browse...
        folders_sizer = wx.FlexGridSizer(2, 4, 1, 1)

        image_folder_help = wx.Button(folders_panel, -1, '?', style=wx.BU_EXACTFIT)
        output_folder_help = wx.Button(folders_panel, -1, '?', style=wx.BU_EXACTFIT)

        sz = image_folder_help.GetBestSize()
        image_folder_help.SetMinSize((1.5*sz[1], sz[1]))
        output_folder_help.SetMinSize((1.5*sz[1], sz[1]))

        image_folder_text = wx.StaticText(folders_panel, -1, 'Default Input\nFolder:', style=wx.ALIGN_RIGHT)
        output_folder_text = wx.StaticText(folders_panel, -1, 'Default Output\nFolder:', style=wx.ALIGN_RIGHT)

        image_folder_panel = wx.Panel(folders_panel, -1, style=wx.NO_BORDER)
        output_folder_panel = wx.Panel(folders_panel, -1)
        
        image_folder_browse = wx.Button(folders_panel, -1, 'Browse...')
        output_folder_browse = wx.Button(folders_panel, -1, 'Browse...')

        folders_sizer.AddMany([
            # top row
            (image_folder_help, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT),
            (image_folder_text, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT),
            (image_folder_panel, 1, wx.EXPAND),
            (image_folder_browse, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT),
            # bottom row
            (output_folder_help, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_LEFT),
            (output_folder_text, 0, wx.EXPAND),
            (output_folder_panel, 1, wx.EXPAND),
            (output_folder_browse, 0, wx.ALIGN_CENTER_VERTICAL | wx.ALIGN_RIGHT)])
                               
        folders_panel.SetSizer(folders_sizer)
        folders_sizer.AddGrowableCol(2)

        def center_in_panel(panel, object, direction):
            sizer = wx.BoxSizer(direction)
            sizer.Add((1,1), 1)
            sizer.Add(object, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 3)
            sizer.Add((1,1), 1)
            panel.SetSizer(sizer)

        image_folder_entry = wx.TextCtrl(image_folder_panel, -1)
        center_in_panel(image_folder_panel, image_folder_entry, wx.VERTICAL)

        output_folder_entry = wx.TextCtrl(output_folder_panel, -1)
        center_in_panel(output_folder_panel, output_folder_entry, wx.VERTICAL)

        # analyze image button
        ab_sizer = wx.BoxSizer(wx.VERTICAL)
        ab_sizer.Add((1,1), 1)
        ab_sizer.Add(wx.Button(analyze_panel, -1, 'Analyze Images'))
        ab_sizer.Add((1,1), 1)
        analyze_panel.SetSizer(ab_sizer)

        
        # Bind pipeline list to handler
        self.Bind(wx.EVT_LISTBOX, self.SelectModule, id=PIPELINE_ID)

        self.SetSize((750, 400))

        self.Centre()

        # Make sure the Listbox doesn't shrink too much
        self.pipeline_list.SetMinSize((self.pipeline_list.GetSize()[0]+10, 10))        
        self.pipeline_list.SetMaxSize((self.pipeline_list.GetSize()[0], 1000))        
        


    def SelectModule(self, event):
        idx = event.GetSelection()
        name = self.pipeline_list.GetString(idx)
        print "selected", name
        self.module_text.SetLabel('settings for module #%d (%s)'%(idx, name))
        # self.pipeline_list.InsertItems(['IdentifyPrimaryAutomatic2322'], 0)

#     def OnPaint(self, event):
#         dc = wx.PaintDC(self.logo_panel)
#         sz = dc.GetSize()
#         lsz = self.logo.GetSize()
#         dc.DrawBitmap(self.logo, (sz[0] - lsz[0]) / 2, (sz[1] - lsz[1]) / 2)




class MyApp(wx.App):
     def OnInit(self):
         frame = MyFrame(None, -1, 'CellProfiler')
         frame.Show(True)
         self.SetAppName('CellProfiler.py')
         return True

app = MyApp(0)
app.MainLoop()
