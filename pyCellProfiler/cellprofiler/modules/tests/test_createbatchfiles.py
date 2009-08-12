'''test_createbatchfiles - test the CreateBatchFiles module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Developed by the Broad Institute
Copyright 2003-2009

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''
__version__="$Revision$"

import base64
import numpy as np
import os
import tempfile
import unittest
from StringIO import StringIO
import zlib

import cellprofiler.workspace as cpw
import cellprofiler.pipeline as cpp
import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.settings as cps
import cellprofiler.preferences as cpprefs

import cellprofiler.modules.loadimages as LI
import cellprofiler.modules.createbatchfiles as C
import cellprofiler.modules.tests as T

class TestCreateBatchFiles(unittest.TestCase):
    def test_01_01_load_matlab(self):
        '''Load a matlab pipeline containing a single CreateBatchFiles module'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUggpTVVwLE1XMDRUMLC0Mja0MjVQMDIwsFQgGTAwevryMzAwaDAy'
                'MFTMmRu00e+wgcDeJXlXV2pJMzMrCzdO5WxsOeXgKBLKw7OjQHeazPIuWZff'
                'pTp/xP2Sl/eFqmbOL+l6/2fe+e+PGBl6OBz8639bOWlcMK184DLPP0BTTWwT'
                'N0uf79/w41OW9Zyz8NM8H+DEGb1f/POT1K062+Umne95POOYXmTynKPGPk8f'
                '3+E6+034t6NdYnaLls5byf0zUzoNqor3xev+Vjv12vpCoVROLUfZD4c5Vef2'
                'N+jOv/JjWvKK8Oe6pXO71jG/Xr29N+/KAutVL7pskgxjX8iLBTr+9DKy/qr4'
                'XUDw7ot7C++pl+ft9z//ODqi7sWLadXfdubNSH/xc/Lx3K6Qh7yh8vHlPvu3'
                'Lfv9q9Z5r5Lu3y/3D6dy15d9fS2/QDnqZ3nQi7jvX76uactPfP2/tNF3IQBj'
                'pbBe')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))        
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, C.CreateBatchFiles))
        self.assertTrue(module.wants_default_output_directory.value)
        self.assertEqual(len(module.mappings), 1)
        self.assertEqual(module.mappings[0].local_directory.value, 'z:\\')
        self.assertEqual(module.mappings[0].remote_directory.value, '/imaging/analysis')
        self.assertFalse(module.remote_host_is_windows.value)
        self.assertFalse(module.batch_mode.value)
    
    def test_01_02_load_v1(self):
        '''Load a version 1 pipeline'''
        data = ('eJztVdtOwjAY7pCDxMTond710gsdgwSiuzHDxEgiSGQhmpCYAgWadCvZwYhP'
                '4aWP4SP4SDyCLWwwmoUBemmzZvsP3/+1X5t/dcO8N6qwrGqwbpgXA0IxbFLk'
                'DZhj6dD2zuGNg5GH+5DZOjR9DA1/CItFWNT0UkUvlWFJ067AbkOp1Q/56yML'
                'AH/APp+pIJQJbCUyhd3CnkfsoZsBaXAS+L/5bCOHoC7FbUR97C4pQn/NHjBz'
                'Ml6E6qzvU9xAVjSZj4ZvdbHjPgxCYBBukjdMW+QdS1sI0x7xK3EJswN8UF/2'
                'LniZJ/EKHabKUgclRodcxC/yNbDMT8fk70Xyj7j1zDkF7jIBl1vBze2e3vGw'
                'Nd6IN7WCT4EG22y9m+CyEi4cIS4f0aeZwHcq7VPYnQ5hfWLjDrHQkF+0F2Qj'
                'OnGJG6l7l1D3WKor7EJQrxBT7+sX576NHv+4v8V9gvXnpoDVcxuB9femAlbv'
                'jbB7mNKxw0RfdlRr1jxctTdryV3k9UYi4KrzHl0Vjlsy61nyvvIxfNH1pfhX'
                'LkEPWYelPtPrXfiUGL6DBFw6+EMI3NOW+p+tyQdS/g+XWIS0')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, C.CreateBatchFiles))
        self.assertTrue(module.wants_default_output_directory.value)
        self.assertEqual(len(module.mappings), 1)
        self.assertEqual(module.mappings[0].local_directory.value, '\\\\iodine\\imaging_analysis')
        self.assertEqual(module.mappings[0].remote_directory.value, '/imaging/analysis')
        self.assertFalse(module.remote_host_is_windows.value)
        self.assertFalse(module.batch_mode.value)
    
    def test_02_01_module_must_be_last(self):
        '''Make sure that the pipeline is invalid if CreateBatchFiles is not last'''
        #
        # First, make sure that a naked CPModule tests valid
        #
        pipeline = cpp.Pipeline()
        module = cpm.CPModule()
        module.module_num = 1
        pipeline.add_module(module)
        pipeline.test_valid()
        #
        # Make sure that CreateBatchFiles on its own tests valid
        #
        pipeline = cpp.Pipeline()
        module = C.CreateBatchFiles()
        module.module_num = 1
        pipeline.add_module(module)
        pipeline.test_valid()
        
        module = cpm.CPModule()
        module.module_num = 2
        pipeline.add_module(module)
        self.assertRaises(cps.ValidationError, pipeline.test_valid)
    
    def test_03_01_save_and_load(self):
        '''Save a pipeline to batch data, open it to check and load it'''
        data = ('eJztWW1PGkEQXhC1WtPYTzb9tB+llROoGiWNgi9NSYUSIbZGbbvCApvu7ZJ7'
                'UWlj0o/9Wf1J/QndxTs4tsoBRS3JHbkcMzfPPDOzs8uxl8uU9jPbcFWLw1ym'
                'FKsSimGBIqvKDT0FmbUEdwyMLFyBnKVgycYwY9dgIgET8dTqRmolCZPx+AYY'
                '7ghlc0/EJf4cgClxfSTOsHNr0pFDnlPKRWxZhNXMSRABzxz9L3EeIoOgM4oP'
                'EbWx2aFw9VlW5aVmo30rxys2xXmke43Fkbf1M2yY76su0LldIJeYFsk3rKTg'
                'mh3gc2ISzhy841/Vtnm5pfDKOhTmOnUIKXWQdVnw6KX9W9Cxj9xQt6ce+3lH'
                'JqxCzknFRhQSHdXaUbTGwcffRJe/CXAk0BKX9sHNK3HIs4QvrdjeJSpbUEdW'
                'uS79rPv4mVb8SLmcOrGw3ugr/lAXPgRe9Zl3uAsXBnkO+sp7VolXyrscMm5B'
                '28T91/02/lHgphSce7i4GdCJ0y/fOSVfKe9RE1/UsYE1TXP91H38rCh+pCzG'
                'uYwpbRhcLlHGiWXY7OuJaCC9ISZ3q5NdqbhdzLZb+yH4hpmXy3I2ioVtGTFE'
                'myYZxbwcdpzvu68eku8+cGmf/GZAdz9IeaeOGMM0ERtx3P30z24+c6d86jqc'
                'uOP8Il18EdE/DP8L3w8fvnegezyl/Glxq/BaPljhTe1l9LOUPoj15YBfbB5n'
                'YoXTqKvZ4dTW2eZxPLZx+j2xlLy6Ni4SgWwpozfmPUj8fuvhuhK/lGUMRxgZ'
                'TmArV9GYVOU4s+qOLunodlGzo3mgeZMcxbwZir9p8QZFpj4C/kHnUfKO+YJ5'
                'NJ7z6OPsYP8rxuV3NcAFuAAX4P43XNqD63c/pLUZUzO43YCEVXBjnPINcOON'
                'S4OgXwPc8DipvO35Ut2/kfZfQO9+ewG6+03K3s04TW9topsa5ahyvYut7Yuv'
                'Wc+Gdj/P52sKz9ptPOXWK5AzuU8tb5ja9TuRbal4Q1rvCNT6zdzA561DWHwW'
                'pnvXXa13Zxx+bw3DFwn9zffYBxdxKidxP8Fg47zYw97NbVj7P/nFW+E=')
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        for windows_mode in (False, True):
            pipeline = cpp.Pipeline()
            pipeline.add_listener(callback)
            pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
            ipath = os.path.join(T.example_images_directory(),'ExampleSBSImages')
            bpath = tempfile.mkdtemp()
            bfile = os.path.join(bpath,C.F_BATCH_DATA)
            try:
                li = pipeline.modules()[0]
                self.assertTrue(isinstance(li, LI.LoadImages))
                module = pipeline.modules()[1]
                self.assertTrue(isinstance(module, C.CreateBatchFiles))
                li.location.value = LI.DIR_OTHER
                li.location_other.value = ipath
                module.wants_default_output_directory.value = False
                module.custom_output_directory.value = bpath
                module.remote_host_is_windows.value = windows_mode
                self.assertFalse(pipeline.in_batch_mode())
                image_set_list = pipeline.prepare_run(None)
                self.assertFalse(pipeline.in_batch_mode())
                self.assertTrue(image_set_list is None)
                self.assertFalse(module.batch_mode.value)
                self.assertTrue(os.path.exists(bfile))
                pipeline = cpp.Pipeline()
                pipeline.add_listener(callback)
                fd = open(bfile,'rb')
                try:
                    pipeline.load(fd)
                finally:
                    fd.close()
                image_set_list = pipeline.prepare_run(None)
                self.assertTrue(pipeline.in_batch_mode())
                module = pipeline.modules()[1]
                self.assertTrue(isinstance(module, C.CreateBatchFiles))
                self.assertTrue(module.batch_mode.value)
                self.assertTrue(isinstance(image_set_list, cpi.ImageSetList))
                self.assertEqual(image_set_list.count(), 96)
                pipeline.prepare_group(image_set_list, {}, range(1,97))
                for i in range(96):
                    image_set = image_set_list.get_image_set(i)
                    for image_name in ('DNA', 'Cytoplasm'):
                        provider = image_set.get_image_provider(image_name)
                        self.assertEqual(provider.get_pathname(), 
                                         '\\imaging\\analysis' if windows_mode
                                         else '/imaging/analysis')
                        
            finally:
                if os.path.exists(bfile):
                    os.unlink(bfile)
                os.rmdir(bpath)