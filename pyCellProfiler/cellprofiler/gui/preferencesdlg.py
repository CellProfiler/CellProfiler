'''preferencesdlg.py Edit global preferences

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import wx
import matplotlib.cm

import cellprofiler.preferences as cpprefs

DIRBROWSE = "Browse"
FONT = "Font"
COLOR = "Color"

class PreferencesDlg(wx.Dialog):
    '''Display a dialog for setting preferences
    
    The dialog handles fetching current defaults and setting the
    defaults when the user hits OK.
    '''
    def __init__(self, parent=None, ID=-1, title="CellProfiler preferences",
                 size=wx.DefaultSize, pos=wx.DefaultPosition, 
                 style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER | wx.THICK_FRAME,
                 name=wx.DialogNameStr):
        wx.Dialog.__init__(self, parent, ID, title, pos, size, style,name)
        p = self.get_preferences()
        top_sizer = wx.BoxSizer(wx.VERTICAL)
        sizer = wx.GridBagSizer(len(p),3)
        sizer.SetFlexibleDirection(wx.HORIZONTAL)
        sizer.AddGrowableCol(1,1)
        top_sizer.Add(sizer,1, wx.EXPAND|wx.ALL, 5)
        index = 0
        controls = []
        for text, getter, setter, ui_info in p:
            text_ctl = wx.StaticText(self, label=text)
            sizer.Add(text_ctl,(index,0))
            if getattr(ui_info,"__getitem__",False) and not isinstance(ui_info,str):
                ctl = wx.ComboBox(self, -1, 
                                  choices=ui_info, style=wx.CB_READONLY)
                ctl.SetStringSelection(getter())
            elif ui_info == COLOR:
                ctl = wx.Panel(self, -1, style=wx.BORDER_SUNKEN)
                ctl.BackgroundColour = getter()
            else:
                ctl = wx.TextCtrl(self, -1, getter())
                min_height = ctl.GetMinHeight()
                min_width  = ctl.GetTextExtent("Make sure the window can display this")[0]
                ctl.SetMinSize((min_width, min_height))
            controls.append(ctl)
            sizer.Add(ctl,(index,1),flag=wx.EXPAND)
            if ui_info == DIRBROWSE:
                def on_press(event, ctl=ctl, parent=self):
                    dlg = wx.DirDialog(parent)
                    dlg.Path = ctl.Value
                    if dlg.ShowModal() == wx.ID_OK:
                        ctl.Value = dlg.Path
            elif ui_info == FONT:
                def on_press(event, ctl=ctl, parent=self):
                    name, size = ctl.Value.split(",")
                    fd = wx.FontData()
                    fd.SetInitialFont(wx.FFont(float(size),
                                              wx.FONTFAMILY_DEFAULT,
                                              face=name))
                    dlg = wx.FontDialog(parent, fd)
                    if dlg.ShowModal() == wx.ID_OK:
                        fd = dlg.GetFontData()
                        font = fd.GetChosenFont()
                        name = font.GetFaceName()
                        size = font.GetPointSize()
                        ctl.Value = "%s, %f"%(name,size) 
            elif ui_info == COLOR:
                def on_press(event, ctl=ctl, parent=self):
                    color = wx.GetColourFromUser(self, ctl.BackgroundColour)
                    if any([x != -1 for x in color.asTuple()]):
                        ctl.BackgroundColour = color
                        ctl.Refresh()
            else:
                on_press = None
            if not on_press is None:
                id = wx.NewId()
                button = wx.Button(self,id,ui_info)
                self.Bind(wx.EVT_BUTTON, on_press, button,id)
                sizer.Add(button, (index, 2))
            index += 1
        top_sizer.Add(wx.StaticLine(self), 0, wx.EXPAND | wx.ALL, 2)
        btnsizer = wx.StdDialogButtonSizer()
        
        btn = wx.Button(self, wx.ID_OK)
        btn.SetDefault()
        btnsizer.AddButton(btn)

        btn = wx.Button(self, wx.ID_CANCEL)
        btnsizer.AddButton(btn)
        btnsizer.Realize()

        top_sizer.Add(btnsizer, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALL, 5)

        self.SetSizer(top_sizer)
        top_sizer.Fit(self)
        self.controls = controls
    
    
    def show_modal(self):
        if self.ShowModal() == wx.ID_OK:
            p = self.get_preferences()
            for control, (text, getter, setter, ui_info) in zip(self.controls, p):
                if ui_info == COLOR:
                    setter(control.BackgroundColour)
                else:
                    setter(control.Value)
    
    def get_preferences(self):
        '''Get the list of preferences.
        
        Each row in the list has the following form:
        Title - the text that appears to the right of the edit box
        get_function - retrieves the persistent preference value
        set_function - sets the preference value
        display - If this is a list, it represents the valid choices.
                  If it is "dirbrowse", put a directory browse button
                  to the right of the edit box.
        '''
        cmaps = list(matplotlib.cm.cmapnames)
        cmaps.sort()
        return [["Matlab module directory",cpprefs.module_directory, cpprefs.set_module_directory,DIRBROWSE],
                ["Default image directory",cpprefs.get_default_image_directory, cpprefs.set_default_image_directory,DIRBROWSE],
                ["Default output directory",cpprefs.get_default_output_directory, cpprefs.set_default_output_directory,DIRBROWSE],
                ["Title font", self.get_title_font, self.set_title_font, FONT],
                ["Table font", self.get_table_font, self.set_table_font, FONT],
                ["Default colormap", cpprefs.get_default_colormap, cpprefs.set_default_colormap, cmaps],
                ["Window background", cpprefs.get_background_color, cpprefs.set_background_color, COLOR]
                ]
    
    def get_title_font(self):
        return "%s,%f"%(cpprefs.get_title_font_name(),
                         cpprefs.get_title_font_size())
    
    def set_title_font(self, font):
        name, size = font.split(",")
        cpprefs.set_title_font_name(name)
        cpprefs.set_title_font_size(float(size))
    
    def get_table_font(self):
        return "%s,%f"%(cpprefs.get_table_font_name(),
                         cpprefs.get_table_font_size())
    
    def set_table_font(self, font):
        name, size = font.split(",")
        cpprefs.set_table_font_name(name)
        cpprefs.set_table_font_size(float(size))

if __name__=='__main__':
    class MyApp(wx.App):
        def OnInit(self):
            wx.InitAllImageHandlers()
            dlg = PreferencesDlg()
            dlg.show_modal()
            return 1
    app = MyApp(0)
 
    