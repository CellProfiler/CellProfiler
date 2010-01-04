"""test_moduleview - test layout of settings in the moduleview

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"

import time
import unittest

import wx

import cellprofiler.cellprofilerapp
import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps
import cellprofiler.gui.moduleview as cpmv

class TestModuleView(unittest.TestCase):
    
    def set_pipeline(self, module):
        app = cellprofiler.cellprofilerapp.CellProfilerApp(redirect=False)
        pipeline = app.frame.pipeline 
        while(len(pipeline.modules())):
            pipeline.remove_module(1)
        module.module_num = 1
        pipeline.add_module(module)
        app.frame.module_view.set_selection(1)
        app.ProcessPendingEvents()
        return app

    def set_setting(self, v):
        class TestModule(cpm.CPModule):
            def __init__(self):
                super(TestModule,self).__init__()
                self.vv = [v]

            def visible_settings(self):
                return self.vv
        
        test_module = TestModule()
        app = self.set_pipeline(test_module)
        module_panel = app.frame.module_view.module_panel
        return (app,
                self.get_text_control(app,v), 
                self.get_edit_control(app,v))
    
    def get_edit_control(self,app,v):
        module_panel = app.frame.module_view.module_panel
        edit_control = module_panel.FindWindowByName(cpmv.edit_control_name(v))
        return edit_control
    
    def get_text_control(self,app,v):
        module_panel = app.frame.module_view.module_panel
        text_control = module_panel.FindWindowByName(cpmv.text_control_name(v))
        return text_control
    
    def get_min_control(self,app,v):
        module_panel = app.frame.module_view.module_panel
        return module_panel.FindWindowByName(cpmv.min_control_name(v))
    
    def get_max_control(self,app,v):
        module_panel = app.frame.module_view.module_panel
        return module_panel.FindWindowByName(cpmv.max_control_name(v))
    
    def test_01_01_display_text_setting(self):
        v=cps.Text("text","value")
        app,text_control,edit_control = self.set_setting(v)
        self.assertEqual(text_control.Label,"text")
        self.assertTrue(isinstance(text_control, wx.StaticText))
        self.assertEqual(edit_control.Value,"value")
        self.assertTrue(isinstance(edit_control,wx.TextCtrl))
        event = wx.CommandEvent(wx.wxEVT_COMMAND_TEXT_UPDATED,edit_control.Id)
        event.SetString("abc")
        edit_control.Command(event)
        app.ProcessPendingEvents()
        self.assertEqual(v.value,"abc")
        app.frame.Close(True)
        app.ProcessPendingEvents()
        app.ProcessIdle()
    
    def test_01_02_display_binary_setting(self):
        v=cps.Binary("text",True)
        app,text_control,checkbox = self.set_setting(v)
        self.assertTrue(checkbox.Value)
        self.assertTrue(isinstance(checkbox,wx.CheckBox))
        event = wx.CommandEvent(wx.wxEVT_COMMAND_CHECKBOX_CLICKED,checkbox.Id)
        checkbox.Command(event)
        app.ProcessPendingEvents()
        checkbox = self.get_edit_control(app,v)
        self.assertFalse(checkbox.Value)
        app.frame.Close(True)
        app.ProcessPendingEvents()
        app.ProcessIdle()
    
    def test_01_03_display_choice_setting(self):
        v = cps.Choice("text",['foo','bar'])
        app,text_control,combobox = self.set_setting(v)
        self.assertTrue(combobox.Value,'foo')
        self.assertTrue(isinstance(combobox, wx.ComboBox))
        self.assertEqual(len(combobox.GetItems()),2)
        self.assertEqual(combobox.GetItems()[0],'foo')
        self.assertEqual(combobox.GetItems()[1],'bar')
        event = wx.CommandEvent(wx.wxEVT_COMMAND_COMBOBOX_SELECTED,combobox.Id)
        event.SetInt(1)
        combobox.Command(event)
        app.ProcessPendingEvents()
        combobox = self.get_edit_control(app,v)
        self.assertEqual(combobox.Value,'bar')
        self.assertTrue(v=='bar')
        app.frame.Close(True)
        app.ProcessPendingEvents()
        app.ProcessIdle()

    def test_01_04_display_integer_setting(self):
        v = cps.Integer("text",1)
        app,text_control,edit_control = self.set_setting(v)
        self.assertTrue(edit_control.Value,'1')
        self.assertTrue(isinstance(edit_control, wx.TextCtrl))
        event = wx.CommandEvent(wx.wxEVT_COMMAND_TEXT_UPDATED,edit_control.Id)
        event.SetString('2')
        edit_control.Command(event)
        app.ProcessPendingEvents()
        edit_control = self.get_edit_control(app,v)
        self.assertEqual(edit_control.Value,'2')
        self.assertTrue(v==2)
        app.frame.Close(True)
        app.ProcessPendingEvents()
        app.ProcessIdle()
    
    def test_01_05_display_integer_range(self):
        v = cps.IntegerRange("text",value=(1,2))
        app,text_control,panel = self.set_setting(v)
        min_control = self.get_min_control(app,v)
        self.assertEqual(min_control.Value,"1")
        max_control = self.get_max_control(app,v)
        self.assertEqual(max_control.Value,"2")
        event = wx.CommandEvent(wx.wxEVT_COMMAND_TEXT_UPDATED,min_control.Id)
        event.SetString('0')
        min_control.Command(event)
        app.ProcessPendingEvents()
        self.assertEqual(v.min,0)
        max_control = self.get_max_control(app,v)
        event = wx.CommandEvent(wx.wxEVT_COMMAND_TEXT_UPDATED,max_control.Id)
        event.SetString('3')
        max_control.Command(event)
        app.ProcessPendingEvents()
        self.assertEqual(v.max,3)
        app.frame.Close(True)
        app.ProcessPendingEvents()
        app.ProcessIdle()

    def test_01_06_display_float_setting(self):
        v = cps.Float("text",1.5)
        app,text_control,edit_control = self.set_setting(v)
        self.assertAlmostEqual(float(edit_control.Value),1.5)
        self.assertTrue(isinstance(edit_control, wx.TextCtrl))
        event = wx.CommandEvent(wx.wxEVT_COMMAND_TEXT_UPDATED,edit_control.Id)
        event.SetString('2.5')
        edit_control.Command(event)
        app.ProcessPendingEvents()
        edit_control = self.get_edit_control(app,v)
        self.assertAlmostEqual(float(edit_control.Value),2.5)
        self.assertAlmostEqual(v.value,2.5)
        app.frame.Close(True)
        app.ProcessPendingEvents()
        app.ProcessIdle()
    
    def test_01_07_display_float_range(self):
        v = cps.FloatRange("text",value=(1.5,2.5))
        app,text_control,panel = self.set_setting(v)
        min_control = self.get_min_control(app,v)
        self.assertAlmostEqual(float(min_control.Value),1.5)
        max_control = self.get_max_control(app,v)
        self.assertAlmostEqual(float(max_control.Value),2.5)
        event = wx.CommandEvent(wx.wxEVT_COMMAND_TEXT_UPDATED,min_control.Id)
        event.SetString('0.5')
        min_control.Command(event)
        app.ProcessPendingEvents()
        self.assertAlmostEqual(v.min,0.5)
        max_control = self.get_max_control(app,v)
        event = wx.CommandEvent(wx.wxEVT_COMMAND_TEXT_UPDATED,max_control.Id)
        event.SetString('3.5')
        max_control.Command(event)
        app.ProcessPendingEvents()
        self.assertAlmostEqual(v.max,3.5)
        app.frame.Close(True)
        app.ProcessPendingEvents()
        app.ProcessIdle()
    
    def test_01_08_display_name_provider(self):
        v = cps.NameProvider("text",group="mygroup",value="value")
        app,text_control,edit_control = self.set_setting(v)
        self.assertEqual(text_control.Label,"text")
        self.assertTrue(isinstance(text_control, wx.StaticText))
        self.assertEqual(edit_control.Value,"value")
        self.assertTrue(isinstance(edit_control,wx.TextCtrl))
        event = wx.CommandEvent(wx.wxEVT_COMMAND_TEXT_UPDATED,edit_control.Id)
        event.SetString("abc")
        edit_control.Command(event)
        app.ProcessPendingEvents()
        self.assertEqual(v.value,"abc")
        app.frame.Close(True)
        app.ProcessPendingEvents()
        app.ProcessIdle()

    def test_02_01_bad_integer_value(self):
        v = cps.Integer("text",1)
        app,text_control,edit_control = self.set_setting(v)
        event = wx.CommandEvent(wx.wxEVT_COMMAND_TEXT_UPDATED,edit_control.Id)
        event.SetString('bad')
        edit_control.Command(event)
        app.ProcessPendingEvents()
        app.frame.module_view.on_idle(None)
        text_control = self.get_text_control(app,v)
        self.assertEqual(text_control.ForegroundColour,wx.RED)
        app.frame.Close(True)
        app.ProcessPendingEvents()
        app.ProcessIdle()
