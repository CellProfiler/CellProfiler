"""test_cellprofilerapp.py

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"
import os
import unittest
import base64
import tempfile

import cellprofiler.cellprofilerapp
import cellprofiler.gui.moduleview as mv
import cellprofiler.settings as vvv
import cellprofiler.modules.identifyprimautomatic
import cellprofiler.gui.addmoduleframe

__version__ = "$Revision$"
my_pipeline = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBUaHUgTWFyIDI2IDE1OjUwOjMwIDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAAgAMAAHic7FnbTtswGHbTA7AhxEGbJnGTS9CgamAXTJrY2AFWadAKENOEGHIbUzylcZU4QDftfm+xR9meYq+wV9jl4jZpE5PWaXBLN2FkWb/r//P3H2zHZgYAMKsCkHPbSbcqoF2ynpwKVCYfIEqxWbOzIAMeef0/3XoELQwrBjqChoNs0Cl+f9E8I4fNRuenXaI7BtqD9eBgt+w59Qqy7NKZr+j9XMZXyDjAnxEIF3/YPrrANiamp+/h872deQnl5p1x69Jc1w8pzg+sfyHQz8a/AN3xmQi/zQbGz3r1EF3R1TdXsErVOqTVc4azIcCZ5HCYXLJw7aXr6lHo5zj9XCtOVQPhth9E+hOc/oQ3/z7SY+mL+IviMMXpM9nWTi+1/GFx2+tnOOcCnA0Oh8kfl54/KxuQos384+VTV3iPDGOfXG4eb62WT5a9jlfEcOrm5nFh9enJF21l7euyDL+L7L7H6TP5NVFNQlXHRl27CwKcVAgn1Ro/Dnkjy34RjyyHw2StsPLE94MsHoPGQZM8vwin1zpa49ZRXHvSIbw0+ODuyeOQ17fpT5bfOxZCZgJ/KiE8BewRufbEtWvY+bY+pvkm2z83PRfj8rnP4TC5RG1H3TFIBRrS7ZKNJ5tX0n046focll/i2sOvi8KKFskrqX/WB9TPhPQzoJAvaP34yPaXCC9qXyqaFJk2ps0+PP32rQB/jsNnMjZ1fIF1BxoqrsNa5/YySt43ze8k/LYcStyLEq5K4MfnZR6Mll8S/JoFm3YVGigCL+l514vfsOM76Dr8l/nlOHy/+PhKQC+ufYPGc1D8uOeT336b6v9eEtzHbhqH1qZXs4jTGD1O1PcRqXxCVdoFGke7+DbJuROwU3XPINQI4Mnym2w8EU7Uu1zXf20zk+zft2Vvr3Zc+Ypw7uJzF59xiA/fJj3fRTyT+lMWv7t2uK3o3J0H4fgzmTjUwCa6dvD2m+dPqvf3WAqE3+OTfme8I1AvBi6gSfN7G7f/+7eLKNQhhXH89IDDYXJRRybFZ82yhevBu1gcvIccHpN3EbQdC5Va28SWheDBOWygeHhR/Dy8lsuCd20p/AKADC/4Xsivfz7+iifPpxcmMxPx7ge93r2SzJtNT6cV5Xr+iPQZr2m3fl/8vfhj8VfrD/j+HDD/l0Dv8X7538f/BQAA//9jZWBg4GBAAEYoDQBAcr/7'

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
            module = cellprofiler.modules.loadimages.LoadImages()
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
        vv = module.visible_settings()
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
            if not isinstance(v,cellprofiler.settings.DoSomething):
                self.assertTrue(v == edit_control.Value,"setting number %d: %s != %s"%(i,v.value,edit_control.Value))
    
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
        v = module.visible_settings()[0]
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
        """Move the second module down one using the pipeline controller"""
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
       

    
