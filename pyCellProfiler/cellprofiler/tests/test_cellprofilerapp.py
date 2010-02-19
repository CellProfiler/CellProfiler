"""test_cellprofilerapp.py

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2010

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__="$Revision$"
import os
import unittest
import base64
import tempfile
import zlib

import cellprofiler.cellprofilerapp
import cellprofiler.gui.moduleview as mv
import cellprofiler.settings as vvv
import cellprofiler.modules.identifyprimaryobjects
import cellprofiler.gui.cpframe
import cellprofiler.gui.addmoduleframe

__version__ = "$Revision$"
my_pipeline = ('eJztWt1u0zAUTtoOsQ3BdjUufYO0wVqlhUlQoa1lZaKCbtVaQGjix2vc1cix'
               'q8Rh69AkHoXH4FF4DB4BO0ub1rRLW9L9SIlktefE3/nOObZPajeVYv1N8QXY'
               'yBigUqynm5ggUCWQN5lt5QHl62DbRpAjEzCaBxVGwV6DA2MDZI38Ri5vGCBn'
               'GM+06S69XLkrPqoPNO2W+LwtWsK/NefLel+Tcg1xjumRM6eltPu+/rdo76CN'
               '4SFB7yBxkRNQdPVl2mT1Trt3q8JMl6BdaPV3Fteuax0i29lrdoH+7So+QaSG'
               'T5ESQrfbPvqGHcyoj/ftq9oeL+MKb63Fjnds4Y5iX+bHWA7yoyv5SYq20qeX'
               '/V9pQf/UkHwu9/Vf8mVMTfwNmy4kAFvwqOedxx9iLzlgL6mVdoserhCCu6f4'
               'IeUdgtttZO7Z+OiFGMmx7CwpdmSroxOefnkCxWS1IG+0ovQnLB+JATsJbZeN'
               'l0d9AKdrj/38h/m9qPgt5RIDlHHQFO57+nH4UwN2UsJvisbhX1D4FwJ+1/En'
               '9Dh25hU7Un7rIGCxnpl/4ril2OleXTvzI3BRjts4uEn8DFu//et9yZdLqAld'
               'wkFZLl5QwjZqcGZ3xsr7HcWelMu9ekA6w/I3bd6jHC+17hjrxqXyfRBVUuKe'
               'huDmtMH8StlYzxrGjPMzal4WQnDD5kOL2fiUUR71fJhl/Ywyv9cZV9Auzsuw'
               'urrdgpQikk1fk3gn+V1x3fxU5102Qr4o/Rz1fL9sP3+E+PlaG5yvUv60ulV9'
               'LjcmaDPzaO2zlN4jQvbZ8eZBMV39uNbVbDPiWnTzwEg/+/g9u547O+9cwwLp'
               'Kdf+2/9pca2QuJ8qcUtZ+v4BQdsP6MnZWlqqxEaMt3xdzteVYCfQ3LA6lLvK'
               'OjTN87vYENvQK/J30nqUu6Z+xvUorkezjO/nvcnOTWZV94btj71DliObue3o'
               '7cwqjmHnLAE/wNRE7auuM1HGO+xchR1+FTvrIOE3Kd4YF+NiXIyLcfHzMcbF'
               'uBgXr/8YF+OuGtfWA5y6T1f/B5D9v2gXr8OH2uA6lHIDEdK2mXy/xc5Y3ksY'
               'ToYwaJ6/7ZB5I76W+158GIfHUHiMUTzyX3hITZtxyFFGvlJQpOa+J3nxh/AU'
               'FJ7CKB4LQce1kRcSphxRB/NOpnKu9aIrd7XqeM0P4e3Pe0JIKwvJC8dZHd9g'
               '3P9sTcOXSuj/nM8uhuBSfT7JS+J/aZPNr9UL+ndjvMz+k+ZN13XtLwmLkVg=')
        
class Test_CellProfilerApp(unittest.TestCase):
    def get_app(self):
        """Get an instance of CellProfilerApp prepared to assert if there's an error"""
        app = cellprofiler.cellprofilerapp.CellProfilerApp(redirect=False, check_for_new_version=False)
        def blowup(message,error):
            self.assertTrue(False,message)
        app.frame.add_error_listener(blowup)
        return app
        
    def load_pipeline_in_app(self):
        (matfd,matpath) = tempfile.mkstemp('.mat')
        matfh = os.fdopen(matfd,'wb')
        data = zlib.decompress(base64.b64decode(my_pipeline))
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
        app = cellprofiler.cellprofilerapp.CellProfilerApp(check_for_new_version=False)
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
        self.assertTrue(isinstance(vv[6],vvv.Text))
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
        # The second module is FlipAndRotate. The first field asks for a prior image
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
        self.assertEqual(len(items),2)
        self.assertEqual(items[0],"Actin")
        self.assertEqual(items[1],"DNA")
    
    def test_02_01_move_up(self):
        """Move the second module up one using the pipeline controller"""
        app = self.load_pipeline_in_app()
        app.frame.pipeline_list_view.select_one_module(2)
        module_names = [app.frame.pipeline.module(1).module_name,
                        app.frame.pipeline.module(2).module_name,
                        app.frame.pipeline.module(3).module_name]
        app.ProcessPendingEvents()
        app.frame.pipeline_controller.on_module_up(None)
        self.assertEqual(module_names[0],app.frame.pipeline.module(2).module_name)
        self.assertEqual(module_names[1],app.frame.pipeline.module(1).module_name)
        self.assertEqual(module_names[2],app.frame.pipeline.module(3).module_name)

    def test_02_02_move_down(self):
        """Move the second module down one using the pipeline controller"""
        app = self.load_pipeline_in_app()
        app.frame.pipeline_list_view.select_one_module(2)
        module_names = [app.frame.pipeline.module(1).module_name,
                        app.frame.pipeline.module(2).module_name,
                        app.frame.pipeline.module(3).module_name]
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
       

    
