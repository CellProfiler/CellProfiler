'''metadatadlg.py - dialog for editing an expression that might contain metadata

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2010 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__ = "$Revision$"

import re

import wx
import wx.lib.masked 

import cellprofiler.measurements as cpmeas
from cellprofiler.preferences import get_primary_outline_color

__choice_ids = []
def get_choice_id(index):
    global __choice_ids
    while len(__choice_ids) <= index:
        __choice_ids.append(wx.NewId())
    return __choice_ids[index]

class MetadataControl(wx.PyControl):
    class MetadataToken(object):
        def __init__(self):
            self.value = u""
            
    def __init__(self, pipeline, module, *args, **kwargs):
        '''Initialize the field
        
        pipeline - the pipeline being run
        module - the module containing the setting
        value (optional) - initial value for control
        padding (optional) - padding around text in pixels
        '''
        kwargs = kwargs.copy()
        style = kwargs.get("style", wx.BORDER_DEFAULT)
        value = kwargs.pop("value", "")
        self.padding = kwargs.pop("padding", 1)
        self.offset = 0
        if (style & wx.BORDER_MASK) == wx.BORDER_DEFAULT:
            self.native_border = True
            self.padding += 2
            style = (style & ~ wx.BORDER_MASK) | wx.BORDER_NONE
        else:
            self.native_border = False
        kwargs["style"] = style | wx.WANTS_CHARS
            
        super(MetadataControl, self).__init__(*args, **kwargs)
        columns = pipeline.get_measurement_columns(module)
        choices = [feature[(len(cpmeas.C_METADATA)+1):]
                   for object_name, feature, coltype in columns
                   if object_name == cpmeas.IMAGE and
                      feature.startswith(cpmeas.C_METADATA)]
        self.__metadata_choices = choices
        self.SetValue(value)
        self.__cursor_pos = len(self.__tokens)
        self.__caret = None
        self.__metadata_choice_ids = {}
        self.__metadata_choice_dict = {}
        for i, choice in enumerate(sorted(choices)):
            choice_id = get_choice_id(i)
            self.__metadata_choice_ids[choice_id] = choice
            self.__metadata_choice_dict[choice] = choice_id
            self.Bind(wx.EVT_MENU, self.select_value, id=choice_id)
        
        def on_focus(event):
            if self.__caret is None:
                self.make_caret()
            self.show_caret()
            
        def on_lose_focus(event):
            if self.__caret is not None:
                self.__caret.Hide()
                del self.__caret
                self.__caret = None
            
        def on_show(event):
            if event.GetShow():
                self.make_caret()
            else:
                del self.__caret
                self.__caret = None
            
        self.Bind(wx.EVT_SET_FOCUS, on_focus)
        self.Bind(wx.EVT_KILL_FOCUS, on_lose_focus)
        self.Bind(wx.EVT_KEY_DOWN, self.on_key_down)
        self.Bind(wx.EVT_CHAR, self.on_char)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_RIGHT_DOWN, self.on_right_down)
        self.Bind(wx.EVT_CONTEXT_MENU, self.on_context_menu)
        self.Bind(wx.EVT_SHOW, on_show)
        self.Bind(wx.EVT_SIZE, self.OnSize)
        
    @property
    def xoffset(self):
        return self.offset + self.padding
    
    def make_caret(self):
        self.__caret = wx.Caret(self, wx.Size(1, self.Size[1] - 2*self.padding))
        
    def SetValue(self, value):
        #
        # Scan through the value, switching state
        #
        STATE_INITIAL = 0    # Not in a tag
        STATE_BACKSLASH = 1  # At first backslash
        STATE_PRE = 2 # Found either \? or \g
        STATE_INSIDE = 3 # Inside a metadata tag
        
        self.__tokens = []
        state = STATE_INITIAL
        index = 0
        while index < len(value):
            if state == STATE_INITIAL:
                if value[index] == '\\':
                    state = STATE_BACKSLASH
                else:
                    self.__tokens.append(value[index])
            elif state == STATE_BACKSLASH:
                if value[index] in ('g', '?'):
                    state = STATE_PRE
                else:
                    self.__tokens.append(unicode(value[index]))
                    state = STATE_INITIAL
            elif state == STATE_PRE:
                if value[index] != '<':
                    # WTF? bad input, output last 3 tokens
                    self.__tokens += value[(index-2):(index+1)]
                    state = STATE_INITIAL
                else:
                    self.__tokens += [ self.MetadataToken()]
                    state = STATE_INSIDE
            else:
                assert state == STATE_INSIDE
                if value[index] != '>':
                    self.__tokens[-1].value += value[index]
                else:
                    state = STATE_INITIAL
            index += 1
        self.SetMinSize(self.DoGetBestSize())
        self.__cursor_pos = len(self.__tokens)
        self.Cursor = wx.StockCursor(wx.CURSOR_IBEAM)
        
    def GetValue(self):
        '''The setting value underlying the text representation'''
        value = ""
        for token in self.__tokens:
            if isinstance(token, self.MetadataToken):
                value += "\\g<" + token.value + ">"
            elif token == "\\":
                value += "\\\\"
            else:
                value += token
        return value

    value = property(GetValue, SetValue)
    Value = property(GetValue, SetValue)
    
    def adjust_scroll(self):
        '''Scroll the cursor position into view'''

        rawpos = 0
        for i in range(self.__cursor_pos):
            rawpos += self.GetTextExtent(self.get_text(i,i+1))[0]
        xsize = self.Size[0] - self.padding * 2
        
        pos = self.xoffset + rawpos
        slop = pos - xsize
        if slop > 0:
            slop += self.GetTextExtent("M")[0]
            self.offset -= slop
            pos -= slop
            self.Refresh()
        elif pos < self.padding:
            self.offset = rawpos
            pos = self.padding
            self.Refresh()
        elif rawpos + self.GetTextExtent("M")[0] < xsize:
            self.offset = 0
            self.Refresh()
        if self.__caret is not None:
            pos = rawpos + self.xoffset
            self.__caret.MoveXY(pos, self.padding)

    def show_caret(self):
        if (self.__caret is not None and
            self.FindFocus() == self):
            self.adjust_scroll()
            self.__caret.Show()
    
    def on_key_down(self, event):
        keycode = event.GetKeyCode()
        if keycode in (wx.WXK_LEFT, wx.WXK_NUMPAD_LEFT):
            if self.__cursor_pos > 0:
                self.__cursor_pos -= 1
                self.show_caret()
        elif keycode in (wx.WXK_RIGHT, wx.WXK_NUMPAD_RIGHT):
            if self.__cursor_pos < len(self.__tokens):
                self.__cursor_pos += 1
                self.show_caret()
        elif keycode in (wx.WXK_DOWN, wx.WXK_NUMPAD_DOWN):
            pos = self.__cursor_pos
            if (pos < len(self.__tokens) and 
                isinstance(self.__tokens[pos], self.MetadataToken)):
                token = self.__tokens[pos]
                try:
                    idx = self.__metadata_choices.index(token.value) + 1
                    idx = idx % len(self.__metadata_choices)
                except ValueError:
                    idx = 0
                token.value = self.__metadata_choices[idx]
                self.on_token_change()
                
        elif keycode in (wx.WXK_UP, wx.WXK_NUMPAD_UP):
            pos = self.__cursor_pos
            if pos < len(self.__tokens) and isinstance(self.__tokens[pos], self.MetadataToken):
                token = self.__tokens[pos]
                try:
                    idx = self.__metadata_choices.index(token.value) - 1
                    if idx < 0:
                        idx = len(self.__metadata_choices)-1
                except ValueError:
                    idx = 0
                token.value = self.__metadata_choices[idx]
                self.on_token_change()
                
        elif keycode in (wx.WXK_INSERT, wx.WXK_NUMPAD_INSERT):
            pos = self.__cursor_pos
            token = self.MetadataToken()
            token.value = self.__metadata_choices[0]
            self.__tokens.insert(pos, token)
            self.on_token_change()
            
        elif keycode == wx.WXK_BACK:
            pos = self.__cursor_pos
            if pos > 0:
                pos -= 1
                self.__cursor_pos = pos
                del self.__tokens[pos]
                self.on_token_change()
                
        elif keycode in (wx.WXK_DELETE, wx.WXK_NUMPAD_DELETE):
            pos = self.__cursor_pos
            if pos < len(self.__tokens):
                del self.__tokens[pos]
                self.on_token_change()
                
        elif keycode in (wx.WXK_HOME, wx.WXK_NUMPAD_HOME):
            self.__cursor_pos = 0
            self.show_caret()
        elif keycode in (wx.WXK_END, wx.WXK_NUMPAD_END):
            self.__cursor_pos = len(self.__tokens)
            self.show_caret()
        elif keycode == wx.WXK_WINDOWS_MENU:
            self.on_context_menu(event)
        elif keycode == wx.WXK_TAB:
            #
            # Code adapted from wx.lib.calendar: author Lorne White
            #
            forward = not event.ShiftDown()
            ne = wx.NavigationKeyEvent()
            ne.SetDirection(forward)
            ne.SetCurrentFocus(self)
            ne.SetEventObject(self)
            self.GetParent().GetEventHandler().ProcessEvent(ne)
            #
            # Seems to be confused about the focus
            #
            if self.FindFocus() != self and self.__caret is not None:
                self.__caret.Hide()
                del self.__caret
                self.__caret = None
        else:
            event.Skip()
    
    def on_char(self, event):
        c = unichr(event.GetUnicodeKey())
        self.__tokens.insert(self.__cursor_pos, c)
        self.__cursor_pos += 1
        self.on_token_change()
        
    def on_token_change(self):
        event = wx.CommandEvent(wx.wxEVT_COMMAND_TEXT_UPDATED, self.GetId())
        self.GetEventHandler().ProcessEvent(event)
        self.show_caret()
        self.SetMinSize(self.DoGetBestSize())
        self.Refresh()
    
    def get_text(self, start_idx = 0, end_idx = None):
        '''Return the text representation of the tokens between the given indices
        
        start_idx - index of first token in string
        end_idx - index of last token or -1 for all
        '''
        value = ""
        if end_idx is None:
            end_idx = len(self.__tokens)
        for token in self.__tokens[start_idx:end_idx]:
            if isinstance(token, self.MetadataToken):
                value += token.value
            else:
                value += token
        return value
        
    def AcceptsFocus(self):
        return True
    
    def AcceptsFocusFromKeyboard(self):
        return True
    
    def DoGetBestSize(self):
        size = self.GetTextExtent(self.get_text() + "M")
        size = wx.Size(size[0] + self.padding * 2, 
                       size[1] + self.padding * 2)
        return self.ClientToWindowSize(size)
    
    def OnSize(self, event):
        self.offset = 0
        self.adjust_scroll()
        event.Skip()
    
    def hit_test(self, pos):
        last = 0
        x = self.xoffset
        for i, token in enumerate(self.__tokens):
            text = self.get_text(i, i+1)
            x += self.GetTextExtent(text)[0]
            if x > pos:
                if isinstance(token, self.MetadataToken):
                    return i
                if pos - last > x-pos:
                    return i+1
                else:
                    return i
        return len(self.__tokens)
    
    def OnLeftDown(self, event):
        if self.HitTest(event.GetPosition()) == wx.HT_WINDOW_INSIDE:
            self.__cursor_pos = self.hit_test(event.GetPositionTuple()[0])
            if self.FindFocus() == self:
                self.show_caret()
            event.Skip()

    def on_right_down(self, event):
        if self.HitTest(event.GetPosition()) == wx.HT_WINDOW_INSIDE:
            index = self.hit_test(event.GetPositionTuple()[0])
            self.__cursor_pos = index
            self.SetFocus()
            self.show_caret()
            event.Skip()
        
    def on_context_menu(self, event):
        menu = wx.Menu()
        index = self.__cursor_pos
        if (index < len(self.__tokens) and
            isinstance(self.__tokens[index], self.MetadataToken)):
            heading = "Change tag"
        else:
            heading = "Insert tag"
        item = menu.Append(-1, heading)
        item.Enable(False)
        for choice in sorted(self.__metadata_choice_dict.keys()):
            menu.Append(self.__metadata_choice_dict[choice], choice)
        self.PopupMenu(menu)
        menu.Destroy()
        
    def select_value(self, event):
        choice = self.__metadata_choice_ids[event.GetId()]
        index = self.__cursor_pos
        if (index < len(self.__tokens) and 
            isinstance(self.__tokens[index], self.MetadataToken)):
            self.__tokens[index].value = choice
        else:
            token = self.MetadataToken()
            token.value = choice
            self.__tokens.insert(index, token)
        self.on_token_change()
    
    def OnPaint(self, event):
        dc = wx.BufferedPaintDC(self)
        try:
            dc.BackgroundMode = wx.SOLID
            background_color = wx.SystemSettings_GetColour(wx.SYS_COLOUR_WINDOW)
            metadata_color = get_primary_outline_color()
            dc.Background = wx.Brush(background_color)
            dc.Font = self.Font
            dc.Clear()
            if self.native_border:
                renderer = wx.RendererNative.Get()
                style = 0
                if self.FindFocus() == self:
                    style |= wx.CONTROL_FOCUSED | wx.CONTROL_CURRENT
                if not self.Enabled:
                    style |= wx.CONTROL_DISABLED
                renderer.DrawTextCtrl(self, dc, (0, 0, 
                                                 self.ClientSize[0], 
                                                 self.ClientSize[1]),
                                      style)
                dc.SetClippingRect((self.padding, self.padding, 
                                    self.ClientSize[0] - 2*self.padding,
                                    self.ClientSize[1] - 2*self.padding))
            loc = self.xoffset
            for i, token in enumerate(self.__tokens):
                if isinstance(token, self.MetadataToken):
                    dc.TextBackground = metadata_color
                    text = token.value
                else:
                    text = self.get_text(i, i+1)
                    dc.TextBackground = background_color
                dc.DrawText(text, loc, self.padding)
                loc += self.GetTextExtent(text)[0]
        finally:
            dc.Destroy()
    
            
if __name__ == "__main__":
    import cellprofiler.pipeline as cpp
    import sys
    
    class MetadataDialog(wx.Dialog):
        '''A dialog that graphically displays metadata tags.
        
        To use:
        
        dlg = MetadataDialog(pipeline, module)
        dlg.value = setting.value # the setting to be edited
        if dlg.ShowModal() == wx.ID_OK:
            setting.value = dlg.value
        '''
        def __init__(self, pipeline, module, *args, **kwargs):
            '''Class initializer
            
            pipeline - pipeline being edited
            module - module containing setting to be edited. Only metadata previous
                     to this module will be available.
            (from wx.Dialog)
            parent - parent window
            id - window ID for this window
            title - Title in caption bar of window
            pos - initial window position
            size - initial window size
            style - dialog style
            name - window name
            '''
            super(MetadataDialog, self).__init__(*args, **kwargs)
            self.value = ""
            columns = pipeline.get_measurement_columns(module)
            choices = [feature[(len(cpmeas.C_METADATA)+1):]
                       for object_name, feature, coltype in columns
                       if object_name == cpmeas.IMAGE and
                          feature.startswith(cpmeas.C_METADATA)]
            
            sizer = wx.GridBagSizer(3,2)
            self.SetSizer(sizer)
            sizer.AddGrowableCol(1)
            sizer.Add(wx.StaticText(self, -1, "Expression:"), (0,0),
                      flag = wx.ALIGN_CENTER | wx.ALL, border = 2)
            
            self.expression_ctrl = MetadataControl(pipeline, module, self)
            sizer.Add(self.expression_ctrl, (0,1), (1,2),
                      flag = wx.ALIGN_CENTER | wx.ALL, border = 2)
            
            buttons = wx.StdDialogButtonSizer()
            sizer.Add(buttons, (1,0), (1,3),
                      flag = wx.EXPAND | wx.ALL, border = 2)
            buttons.AddButton(wx.Button(self, wx.ID_OK))
            buttons.AddButton(wx.Button(self, wx.ID_CANCEL))
            buttons.Realize()
            self.Layout()
            
    class MyApp(wx.App):
        def OnInit(self):
            p = cpp.Pipeline()
            p.load(sys.argv[1])
            dlg = MetadataDialog(p, p.modules()[-1], None)
            dlg.expression_ctrl.SetValue("Hello \\g<PLATE> giraffe platypus!")
            dlg.ShowModal()
            return 1
    
    my_app = MyApp()
    my_app.MainLoop()
    
