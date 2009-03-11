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
my_pipeline = 'TUFUTEFCIDUuMCBNQVQtZmlsZSwgUGxhdGZvcm06IFBDV0lOLCBDcmVhdGVkIG9uOiBXZWQgRmViIDE4IDA5OjQyOjM2IDIwMDkgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIAABSU0PAAAAfwMAAHic7FnNT9swFHfTD2BDiA9tYj3lCBpULezApImNfcAqDVoBYpoQQ25jiqc0rhIH6Kbdd+eyP2aH/Sn7M3bYYXGbtIlJ6jSkpdNqybKe6/fze7/3bMfuDLCKDEDGaiatKoF2SdtywlWZfIAoxVrNSIMUWLT7f1r1COoYVlR0BFUTGaBTnP6idkYOm43OT7tEMVW0B+vuwVbZM+sVpBulM0fR/rmMr5B6gD8j4C3OsH10gQ1MNFvfxud7O/MSys07Y9XFuS4PCY4H1r/g6mfjX4Du+JQPb7Ou8bN2PURXdPXNFaxSuQ5p9ZzhbAhwJjkcJpd0XHtpUT0M/Qynn2nFqaoi3OZBpD/B6U/Y8+8jJZS+yH5RHKY4fSYbhdPLQu6wuG33M5xzAc4Gh8Pkj0vPn5VVSNFm7vHyqSW8R6q6Ty43j7dWyyfLdscropp1bfM4v/r05EthZe3rchy8i/y+x+kz+TWRNUJl00Bdv/MCnIQHJ9EaPwp5I7Jb8uhLYI+EmzfNzcvkQn7lieP3XfFeiHl+EU7Qulnj1k1Yf5IevCT4YO3Bo5DHd8kny+cdHSEtAp9B+R2XP2H9GnS+rY9ovsXNz23PwbD23OdwmFyihinvqKQC1dj9ihsvbrui7sNR1+egeAnrD78u8isFX7ui8rPep37Ko58C+Vy+0MueuPkS4fntS0WNIs3AtNnDTqd9K8Cf4/CZjDUFX2DFhKqM67DWua0M0+7b5ncU+7ZMSqyLEa7GYB+flzkwXPui4Nd02DSqUEU+eFHPuyD7Bh3fftfhv2xfhsN3ioMvufTC+tdvPPvFD3s+Oe23qd7vI+597LZxaG16NZ2YjeHj+H0fkconVKVdoFH0i2+jnDsuP2XrDEINF15cvMWNJ8Lxe4fr8td2M8r+fVf+BrWjaq8IZxyfcXxGIT58G/V8F9kZlc+47Bu3g21F5+488MafycSkKtbQjYO31zy/E8HfYwngfX+P+p3xjkCl6LqARs3vbdz+t28XUahACsPw9IDDYXJRQRrFZ82yjuvuu1gYvIccHpN3ETRMHZVa28SWjuDBOWygcHh+9tl4Lcrcd+1Y7HMBMjz3eyG//vn4S7Y8n1yYTE2Eux8EvXtFmTctTScl6Wb+iPSZXdNW/ZP9lf2Rvc5+f3SdBQ6ffeb/Egge75T/bfxfAAAA//9jZIAAAFhQvG4='

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
       

    
