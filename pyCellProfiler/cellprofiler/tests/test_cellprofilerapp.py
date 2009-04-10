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
my_pipeline = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBGcmkgQXByIDEwIDE2OjM3OjU4IDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAAlQMAAHic7FlbT9swFHZD2sKGEBdtmrQ+5BEEVA3sgUkTG7vAKg1aAWKaGENuY4qnNK4SB+imve9n7VdNe1zcJm1q0joNbtdJWLKs4/p8/s7FduzOAQDKGgAZr532qgLaJe3LqVBl8hGiFFs1Jw1U8MTv/+XVE2hjWDHRCTRd5IBOCfqL1gU5bjY6P+0TwzXRAayHB3vlwK1XkO2ULgJF/+cyvkHmEf6GQG8Jhh2iK+xgYvn6Pj7f25mXUG7eOa9eLnT9kOL8wPqXQv1s/CvQHa9G+G0+NH7er8fohq6/u4FVqtUhrV4ynC0BzjSHw+SSjWuvPVePQz/D6WdacaqaCLf9INLPcvpZf/5DZMTSF/EXxWGG02eyo59f6/nj4q7f34q/AGeLw2Hyl+WXL8ompGg7v7py7gkfkWkekuvt05318tmK3/GGmG7d2j4trD8/+66vbfxYkeF3kd0POH0mvyWaRajmOqhrd0GAk+rBSbXGT0LeyLJfxCPN4TBZL6w9C/wgi8ewcdAlzy/C6beONrh1NC5/TPXgTIFP3p4+CetC1vwinKh4sPWxZyNkgS5OXHuUHjwFHBC59sS1a9T5usnla1z/jDrfZPvnrudqXD4PORwml6jjansmqUBTul2y8WTzSrqPJ12fo/JLXHv4dVFY0yN5JfXP5pD6ao++Cgr5gj6Ij2x/ifCi9qWiRZHlYNocwDNo3wvwFzh8JmPLwFfYcKGp4TqsdW4/4+R91/xOwm/HpcS7aOGqBH58XubBePklwa/ZsOlUoYki8JKed/34jTq+w67D/5lfhsMPSoCvhPTi2jdsPIfFj3s+Be3PmcHvLeF97K5xaG16NZu4jfHjRH0fkcpXVKVdoEm0i2+TnDshOzXvDEKNEJ4sv8nGE+FEvet1/dc2M8n+/a/s7ddOKl8Rzn187uMzCfHh26Tnu4hnUn/K4nffjrYVnbuLoDf+TCYuNbGFbh28g+b5k+r/PZYCve/5Sb8zPhBoFEMX0KT5vYvb/x7uIwoNSGEcPz3icJhcNJBF8UWzbON6+C4WB+8xh8fkfQQd10al1jaxYyN4dAkbKB5eFD8fr+Wy8F1bCr8QIMMLvxfy65+Pv+LLi+rStJqNdz/o9+6VZN60OjulKLfzR6TPeM169XOO5Eq51dzvp6s5EPhzyPxfBv3HB2XSx/8FAAD//2NjYGDgAGJGBghghfJhACbOB8QaQEysegBnzcI3'

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
       

    
