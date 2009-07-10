"""regexp_editor - give a user feedback on their regular expression

"""
__version__="$Revision$"

import re
import wx

def edit_regexp(parent, regexp, test_text):
    frame = RegexpDialog(parent, size=(300,200),
                         style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
    frame.value = regexp
    frame.test_text = test_text
    if frame.ShowModal():
        return frame.value
    return None

class RegexpDialog(wx.Dialog):
    def __init__(self, *args,**varargs):
        varargs["title"] = "Regular expression editor"
        super(RegexpDialog,self).__init__(*args,**varargs)
        self.__value = "Not initialized"
        self.__test_text = "Not initialized"
        self.font = wx.SystemSettings.GetFont(wx.SYS_ANSI_FIXED_FONT) 
        temp = wx.ClientDC(self)
        temp.Font = self.font
        edit_size = temp.GetTextExtent("                                        ")
        temp.Destroy()

        sizer = wx.BoxSizer(wx.VERTICAL)
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self,label="Regex:"),0,wx.ALIGN_CENTER|wx.ALL, 5)
        self.editor = wx.TextCtrl(self,value=self.value)
        self.editor.SetFont(self.font)
        self.editor.SetMinSize((edit_size[0],self.editor.GetMinHeight()))
        hsizer.Add(self.editor,1,wx.ALIGN_CENTER|wx.ALL,  5)
        sizer.Add(hsizer,0,wx.GROW|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)
        
        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.Add(wx.StaticText(self,label="Test text:"),0,wx.ALIGN_CENTER|wx.ALL,5)
        self.test_text_ctl = wx.TextCtrl(self,value=self.__test_text)
        self.test_text_ctl.Font = self.font
        hsizer.Add(self.test_text_ctl,1,wx.ALIGN_CENTER|wx.ALL, 5)
        sizer.Add(hsizer,0, wx.GROW|wx.ALIGN_CENTER_VERTICAL|wx.ALL, 5)

        self.feedback = wx.StaticBitmap(self, bitmap=self.get_bitmap())
        sizer.Add(self.feedback,0,wx.ALIGN_CENTER_VERTICAL | wx.ALL, 5)

        line = wx.StaticLine(self, -1, size=(20,-1), style=wx.LI_HORIZONTAL)
        sizer.Add(line, 0, wx.GROW|wx.ALIGN_CENTER_VERTICAL|wx.RIGHT|wx.LEFT, 5)

        hsizer = wx.StdDialogButtonSizer()
        ok_button = wx.Button(self,label="OK")
        ok_button.SetDefault()
        hsizer.Add(ok_button,0,wx.ALIGN_RIGHT)
        cancel_button = wx.Button(self, label="Cancel")
        hsizer.Add(cancel_button,0,wx.ALIGN_RIGHT|wx.LEFT,5)
        hsizer.Realize()
        sizer.Add(hsizer,0,wx.ALIGN_RIGHT|wx.ALL,5)
        
        self.Bind(wx.EVT_BUTTON,self.on_ok_button, ok_button)
        self.Bind(wx.EVT_BUTTON,self.on_cancel_button, cancel_button)
        self.Bind(wx.EVT_TEXT, self.on_editor_text_change, self.editor)
        self.Bind(wx.EVT_TEXT, self.on_test_text_text_change, self.test_text_ctl)
        self.SetSizer(sizer)
        self.Fit()
        
    def get_bitmap(self):
        color_db = ["BLACK", "RED", "GREEN", "BLUE", "CYAN","MAGENTA","SIENNA","PURPLE"]
        color_db = [wx.TheColourDatabase.FindColour(x) for x in color_db]
        try:
            match = re.search(self.value, self.test_text)
            if match:
                text = self.test_text
                colors = [wx.Colour(128,128,128,255)
                           for i in range(len(text))]
                group_idx = list(range(len(match.groups())+1))
                for i in group_idx:
                    for j in range(match.start(i),match.end(i)):
                        colors[j] = color_db[i % len(color_db)]
            else:
                text = "Regular expression did not match"
                colors = [wx.RED for i in range(len(text))]
        except:
            text = "Regular expression is not valid"
            colors = [wx.RED for i in range(len(text))]
        temp = wx.ClientDC(self)
        temp.Font = self.font
        width, height = temp.GetTextExtent(text)
        temp.Destroy()
        bitmap = wx.EmptyBitmap(width, height)
        dc = wx.MemoryDC(bitmap)
        dc.Background = wx.WHITE_BRUSH
        dc.Clear()
        dc.Font = self.font
        for i in range(len(text)):
            dc.SetTextForeground(colors[i])
            dc.DrawText(text[i],i*width/len(text),0)
        dc.SelectObject(wx.NullBitmap)
        return bitmap

    def on_ok_button(self, event):
        self.EndModal(1)
    
    def on_cancel_button(self, event):
        self.__value = None
        self.EndModal(0)  
          
    def on_editor_text_change(self, event):
        self.__value = self.editor.Value
        self.refresh_bitmap()
    
    def on_test_text_text_change(self, event):
        self.__test_text = self.test_text_ctl.Value
        self.refresh_bitmap()
        
    def refresh_bitmap(self):
        self.feedback.SetBitmap(self.get_bitmap())
        self.Refresh()
        
    def get_value(self):
        return self.__value
    
    def set_value(self, value):
        self.__value = value
        self.editor.Value = value
        self.refresh_bitmap()
    value = property(get_value, set_value)
    
    def get_test_text(self):
        return self.__test_text
    
    def set_test_text(self, test_text):
        self.__test_text = test_text
        self.test_text_ctl.Value = test_text
        self.refresh_bitmap()

    test_text = property(get_test_text, set_test_text)

if __name__== "__main__":
    import wx.lib.inspection
    class MyApp(wx.App):
        def OnInit(self):
            wx.InitAllImageHandlers()
            return True

    app = MyApp(0)
    edit_regexp(None, "(?P<foo>foo)", "Where is the food?")

