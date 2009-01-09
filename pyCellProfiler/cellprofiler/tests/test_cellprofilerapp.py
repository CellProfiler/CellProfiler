import os
import unittest
import base64
import tempfile

import cellprofiler.cellprofilerapp
import cellprofiler.gui.moduleview as mv
import cellprofiler.variable as vvv
import cellprofiler.modules.platonicmodule
import cellprofiler.modules.identifyprimautomatic
import cellprofiler.gui.addmoduleframe

__version__ = "$Revision: 1$"
my_pipeline = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBGcmkgRGVjIDA1IDA4OjU5OjQzIDIwMDggICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAAbQMAAHic7VlPT9swFHdCKbAh1A2N7rYeQYOqgR2YNDHYH1ilQStAbNPEkNuY4imNq8QBumn3fYdJ+xa777bDtO+z4+I2aRPT1iG4JdKwZFnP9fv5vZ+fn2N3BgCw/gCAtNtOulUF7TLuyUqgMnkPUYrNmj0OUuC+1//TrQfQwrBioANoOMgGneL3F81jst9sdH7aJrpjoB1YDw52y45TryDLLh37it7PZXyOjD38CYFw8YftolNsY2J6+h4+39uZl1Bu3hm3vs10eVA4Hlj/bKCfjV8H3fGpHrxlAuMzXt1H53Tp5Tms0lwd0uoJw1kV4ExyOEwuWbj2zKV6FPppTj/dWqeqgXCbB5H+BKc/4c2/i/RI+iL7Reswxekz2daOzrT8fnHT62c4JwKcVQ6HyR/mnz4pG5CitfzDhSNXeIMMY5ecrb3fWCofLngdz4nh1M2194Wlx4eftcXlLwsyeBf5fYvTZ/ILkjMJzTk26vpdEOAoIRylNT4JcePbL8IZ53CYrBUWH/l+XBePmuT5RTj99sEytw+i+jMWwhsD79ycmoS4vE4+WXxuWQiZMfhUQ3gq2CFy/Ynq17DjbSWh8Sabn6uea1Htuc3hMLlEbSe3ZZAKNKT7JRtPtl1x83Dc/TksXqL6w++LwqLW0664/KxcUj8V0k+BQr6gDbJHNl8ivF55qWhSZNqYNgfY6bevBPh3OHwmY1PHp1h3oJHDdVjr3D5GafdV4zuOfRsOJe5FB1cl2MfHZR6M1r44+DULNu0qNFAPvLjnncjOqPZedl8lbb5h8yfrPPDbr1OD3xeCeeOqvLaSTM0iTmP0OL2+R0jlI6rSLlAS/eLbOHk+4GfOzfmoEcCTxZtsPBFOr3esLn9tN+Pkg+vyt1+bVHtFODfrc7M+SVgfvg2en2luPr/486kBPZGdcfmUZd9NO9xWdO7eBeH1ZzJxqIFNdOHgHTTPX6X/95gCwu/Xcb8zXhOoFwMXvrjxvYnb/5ZtIwp1SGEUnu5xOEwu6sik+LhZtnA9ePeJgjfH4TF5G0HbsVCplSY2LAT3TmADRcPrZZ+H16IseLeVYl8AkOEF3+f4/c+vv+rJGXV2MjUR7X7Q750pzrwpZXpMVS/Gj0if2TXt1l/Z79lv2R9zf+Z+Z4HP5yXjfx70H++X/3X8P4BzuRE='

class Test_CellProfilerApp(unittest.TestCase):
    def get_app(self):
        """Get an instance of CellProfilerApp prepared to assert if there's an error"""
        app = cellprofiler.cellprofilerapp.CellProfilerApp(redirect=False)
        def blowup(message,error):
            self.assertTrue(False,message)
        app.frame.add_error_listener(blowup)
        return app
        
    def load_pipeline_in_app(self):
        (matfd,matpath) = tempfile.mkstemp('.mat')
        matfh = os.fdopen(matfd,'wb')
        data = base64.b64decode(my_pipeline)
        matfh.write(data)
        matfh.flush()
        app = self.get_app()
        app.frame.pipeline_controller.do_load_pipeline(matpath)
        matfh.close()
        return app
    
    def get_platonic_loader(self):
        def loader(module_num):
            module = cellprofiler.modules.platonicmodule.LoadImages()
            module.set_module_num(module_num)
            return module
        return loader
    
    def get_identify_prim_automatic_loader(self):
        def loader(module_num):
            module = cellprofiler.modules.identifyprimautomatic.IdentifyPrimAutomatic()
            module.set_module_num(module_num)
            return module
        return loader
        
    def test_00_00_Init(self):
        """Start the GUI and exit"""
        app = cellprofiler.cellprofilerapp.CellProfilerApp()
        app.Exit()
    
    def test_01_01_Load(self):
        """Test loading a pipeline from a file
        
        Load a pipeline, select the first module in the GUI
        and check to make sure the controls and text labels
        match what was loaded.
        """
        app = self.load_pipeline_in_app()
        app.frame.module_view.set_selection(1)
        app.ProcessPendingEvents()
        module_panel = app.frame.module_view.module_panel
        module = app.frame.pipeline.module(1)
        #
        # Module 1 is Matlab LoadImages
        #
        vv = module.visible_variables()
        self.assertTrue(isinstance(vv[0],vvv.Choice))
        self.assertTrue(isinstance(vv[2],vvv.Text))
        self.assertTrue(isinstance(vv[3],vvv.Binary))
        for v,i in zip(vv,range(len(vv))):
            text_name = mv.text_control_name(v)
            text_control = module_panel.FindWindowByName(text_name)
            self.assertTrue(text_control)
            self.assertEqual(v.text.replace('\n',' '),text_control.LabelText.replace('\n',' '))
            control_name = mv.edit_control_name(v)
            edit_control = module_panel.FindWindowByName(control_name)
            self.assertTrue(edit_control)
            if not isinstance(v,cellprofiler.variable.DoSomething):
                self.assertTrue(v == edit_control.Value,"variable number %d: %s != %s"%(i,v.value,edit_control.Value))
    
    def test_01_02_Subscriber(self):
        """Test provide/subscribe for images
        
        Module 1 provides three images, Module 2's first control
        subscribes to images and has a value, when loaded, of the
        "OrigBlue" image. Make sure that the gui reflects this
        when Module 2 is loaded
        """
        app = self.load_pipeline_in_app()
        #
        # The second module is FileNameMetadata. The first field asks for a prior image
        #
        app.frame.module_view.set_selection(2)
        app.ProcessPendingEvents()
        module_panel = app.frame.module_view.module_panel
        module = app.frame.pipeline.module(2)
        v = module.visible_variables()[0]
        self.assertTrue(isinstance(v,vvv.NameSubscriber))
        control = module_panel.FindWindowByName(mv.edit_control_name(v))
        self.assertTrue(control)
        self.assertTrue(v==control.Value)
        items = control.Items
        self.assertEqual(len(items),3)
        self.assertEqual(items[0],"OrigRed")
        self.assertEqual(items[1],"OrigGreen")
        self.assertEqual(items[2],"OrigBlue")
    
    def test_02_01_move_up(self):
        """Move the second module up one using the pipeline controller"""
        app = self.load_pipeline_in_app()
        app.frame.pipeline_list_view.select_module(2)
        module_names = [app.frame.pipeline.module(1).module_name,app.frame.pipeline.module(2).module_name, app.frame.pipeline.module(3).module_name]
        app.ProcessPendingEvents()
        app.frame.pipeline_controller.on_module_up(None)
        self.assertEqual(module_names[0],app.frame.pipeline.module(2).module_name)
        self.assertEqual(module_names[1],app.frame.pipeline.module(1).module_name)
        self.assertEqual(module_names[2],app.frame.pipeline.module(3).module_name)

    def test_02_02_move_down(self):
        """Move the second module up one using the pipeline controller"""
        app = self.load_pipeline_in_app()
        app.frame.pipeline_list_view.select_module(2)
        module_names = [app.frame.pipeline.module(1).module_name,app.frame.pipeline.module(2).module_name, app.frame.pipeline.module(3).module_name]
        app.ProcessPendingEvents()
        app.frame.pipeline_controller.on_module_down(None)
        self.assertEqual(module_names[0],app.frame.pipeline.module(1).module_name)
        self.assertEqual(module_names[1],app.frame.pipeline.module(3).module_name)
        self.assertEqual(module_names[2],app.frame.pipeline.module(2).module_name)
    
    def test_02_03_insert(self):
        """Insert a module into an empty pipeline"""
        app = self.get_app()
        event = cellprofiler.gui.addmoduleframe.AddToPipelineEvent("LoadImages",self.get_platonic_loader())
        app.frame.pipeline_controller.on_add_to_pipeline(self,event)
        self.assertEqual(len(app.frame.pipeline.modules()),1)
        self.assertEqual(app.frame.pipeline.module(1).module_name, "LoadImages")
    
    def test_02_04_insert_two(self):
        """Insert two modules and check to see if the second was inserted after
        
        This is a regression test - users expect latter modules to be inserted
        at the end of the list, even if nothing is selected
        """
        app = self.get_app()
        event = cellprofiler.gui.addmoduleframe.AddToPipelineEvent("LoadImages",self.get_platonic_loader())
        app.frame.pipeline_controller.on_add_to_pipeline(self,event)
        event = cellprofiler.gui.addmoduleframe.AddToPipelineEvent("IdentifyPrimAutomatic",self.get_identify_prim_automatic_loader())
        app.frame.pipeline_controller.on_add_to_pipeline(self,event)
        self.assertEqual(len(app.frame.pipeline.modules()),2)
        self.assertEqual(app.frame.pipeline.module(1).module_name, "LoadImages")
        self.assertEqual(app.frame.pipeline.module(2).module_name, "IdentifyPrimAutomatic")
       

    