"""test_moduleview - test layout of variables in the moduleview

"""
__version__="$Revision$"

import time
import unittest

import wx

import cellprofiler.cellprofilerapp
import cellprofiler.cpmodule as cpm
import cellprofiler.variable as cpv
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

    def set_variable(self, v):
        class TestModule(cpm.CPModule):
            def __init__(self):
                super(TestModule,self).__init__()
                self.vv = [v]

            def visible_variables(self):
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
    
    def test_01_01_display_text_variable(self):
        v=cpv.Text("text","value")
        app,text_control,edit_control = self.set_variable(v)
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
    
    def test_01_02_display_binary_variable(self):
        v=cpv.Binary("text",True)
        app,text_control,checkbox = self.set_variable(v)
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
    
    def test_01_03_display_choice_variable(self):
        v = cpv.Choice("text",['foo','bar'])
        app,text_control,combobox = self.set_variable(v)
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

    def test_01_04_display_integer_variable(self):
        v = cpv.Integer("text",1)
        app,text_control,edit_control = self.set_variable(v)
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
        v = cpv.IntegerRange("text",value=(1,2))
        app,text_control,panel = self.set_variable(v)
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

    def test_01_06_display_float_variable(self):
        v = cpv.Float("text",1.5)
        app,text_control,edit_control = self.set_variable(v)
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
        v = cpv.FloatRange("text",value=(1.5,2.5))
        app,text_control,panel = self.set_variable(v)
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
        v = cpv.NameProvider("text",group="mygroup",value="value")
        app,text_control,edit_control = self.set_variable(v)
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
        v = cpv.Integer("text",1)
        app,text_control,edit_control = self.set_variable(v)
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
