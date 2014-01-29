'''test_createwebpage - Test the CreateWebPage module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''


import base64
import numpy as np
import os
from scipy.misc import imsave
import shutil
from StringIO import StringIO
import tempfile
import unittest
from urllib2 import urlopen
import xml.dom.minidom as DOM
import zipfile
import zlib

import cellprofiler.preferences as cpprefs
cpprefs.set_headless()

import cellprofiler.workspace as cpw
import cellprofiler.cpgridinfo as cpg
import cellprofiler.cpimage as cpi
import cellprofiler.cpmodule as cpm
import cellprofiler.objects as cpo
import cellprofiler.measurements as cpmeas
import cellprofiler.pipeline as cpp
import cellprofiler.modules.createwebpage as C
from cellprofiler.modules.loadimages import C_FILE_NAME, C_PATH_NAME, C_URL
from cellprofiler.modules.loadimages import pathname2url, url2pathname
IMAGE_NAME = "image"
THUMB_NAME = "thumb"
DEFAULT_HTML_FILE = "default.html"
ZIPFILE_NAME = "zipfile.zip"

class TestCreateWebPage(unittest.TestCase):
    def setUp(self):
        #
        # Make a temporary directory structure
        #
        cpprefs.set_headless()
        directory = self.directory = tempfile.mkdtemp()
        for i in range(3):
            os.mkdir(os.path.join(directory, str(i)))
            for j in range(3):
                os.mkdir(os.path.join(directory, str(i), str(j)))
                for k in range(3):
                    os.mkdir(os.path.join(directory, str(i), str(j), str(k)))
        cpprefs.set_default_image_directory(os.path.join(self.directory, "1"))
        self.alt_directory = tempfile.mkdtemp()
        
    def tearDown(self):
        shutil.rmtree(self.directory)
        shutil.rmtree(self.alt_directory)
        
    def test_00_00_remember_to_put_new_text_in_the_dictionary(self):
        '''Make sure people use TRANSLATION_DICTIONARY'''
        self.assertTrue(C.DIR_ABOVE in C.TRANSLATION_DICTIONARY)
        self.assertTrue(C.DIR_SAME in C.TRANSLATION_DICTIONARY)
        self.assertTrue("One level over the images" in C.TRANSLATION_DICTIONARY)
        self.assertTrue("Same as the images" in C.TRANSLATION_DICTIONARY)
        
        self.assertTrue(C.OPEN_ONCE in C.TRANSLATION_DICTIONARY)
        self.assertTrue(C.OPEN_EACH in C.TRANSLATION_DICTIONARY)
        self.assertTrue(C.OPEN_NO in C.TRANSLATION_DICTIONARY)
        
        self.assertTrue("Once only" in C.TRANSLATION_DICTIONARY)
        self.assertTrue("For each image" in C.TRANSLATION_DICTIONARY)
        self.assertTrue("No" in C.TRANSLATION_DICTIONARY)
        
        self.assertEqual(len(C.TRANSLATION_DICTIONARY), 5, 
                         "Please update this test to include your newly entered translation")

    def test_01_01_load_matlab(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUvDNz1PwTSxSMLBQMDSxMjKwMjRVMDIwNFAgGTAwevryMzAw2DEx'
                'MFTMeRrmmH/ZQKTu5aVVzmuknh5i7JTZaXBNcaHsxVMsU92iL2Z2LL69b9k0'
                'md1ec1Y93aJs3/m4TsrMe9aREu/JZY8blLWrzv3e9/3nZtmdjAzLeRtmVV0P'
                'XX8gd7XbxqmvrOQWTJmy21E0p0Kw9/6KOtcarzae9Rzchg2h3ceta5T0pm++'
                'ulw/ScbojZD+3DVR5l/aXrDv9eTpsfv+kulmwz+HE9ss3mqfy3Rg7f8guN9y'
                '1yrtJYHhkQzXvx5hP/3B8fcln97Xf/gnPLmdXWPt81Te7a+Q+KezP5t6ljwW'
                'tJyVNe+yzLxdBxc/nud/1zi1NLKm2svC/dqtFLnfbHX2q/6fdH+iaS9XJWfK'
                'vP6x8RwO+dPX+2ssbh5pX84Wd/yKsh7b2sUrv0S8Obzuz585hy+bVJTYBKif'
                'UmxUXv7iTpb3j08ltoY/Sv4EWsR4qPMUFMee55WT3T5x4z59tSkL35cIJe6+'
                'UjrncHJ6Rc6OBXJ2abXah5MP7naftz/yf4X33crna7NeXCvsv2j573Hl/ANp'
                'v3/MDIw/PqX/5Zw+50NGx7O1ZSt3eLGFN6678cj+sPsLjnPTol/YNd1dFz+z'
                'coddJJvyoUe/HRYfqPlmVFP9IXZtkuCf2NeB2+zqvY1+uD79V33CMcum9/Tn'
                'wrtVf24eWad/6Obhj08/rtmtrmrTZ1Vk4f1aa9Xn+uf6v/6xlr+3L/4//a66'
                'eFOPNet7dXa2/r7bdkKd9wtuhMd8+P5z9dMX61csv6nxXT5n6prwrV2Pc77u'
                'qS/9HPS4zmnj18pa/7o7p579Z9NbdcMfAGQ1SXU=')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, C.CreateWebPage))
        self.assertEqual(module.orig_image_name, "OrigBlue")
        self.assertTrue(module.wants_thumbnails)
        self.assertEqual(module.thumbnail_image_name, "OrigGreen")
        self.assertEqual(module.web_page_file_name, "images1.html")
        self.assertEqual(module.directory_choice.dir_choice, C.DIR_SAME)
        self.assertEqual(module.title, "CellProfiler Images")
        self.assertEqual(module.background_color, "Lime")
        self.assertEqual(module.columns, 3)
        self.assertEqual(module.table_border_width, 0)
        self.assertEqual(module.table_border_color, "Olive")
        self.assertEqual(module.image_spacing, 2)
        self.assertEqual(module.image_border_width, 2)
        self.assertEqual(module.create_new_window, C.OPEN_ONCE)
        self.assertFalse(module.wants_zip_file)
        
    def test_01_02_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9462

CreateWebPage:[module_num:1|svn_version:\'9401\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Full-size image name\x3A:ColorImage
    Use thumbnail images?:Yes
    Thumbnail image name\x3A:ColorThumbnail
    Web page file name\x3A:sbsimages\\g<Controls>.html
    Folder for the .html file\x3A:Same as the images
    Webpage title\x3A:SBS Images\x3A Controls=\\g<Controls>
    Webpage background color\x3A:light grey
    # of columns\x3A:12
    Table border width\x3A:1
    Table border color\x3A:blue
    Image spacing\x3A:1
    Image border width\x3A:1
    Open new window when viewing full image?:Once only
    Make a zipfile containing the full-size images?:No
    Zipfile name\x3A:Images.zip

CreateWebPage:[module_num:2|svn_version:\'9401\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Full-size image name\x3A:IllumDNA
    Use thumbnail images?:No
    Thumbnail image name\x3A:IllumGFP
    Web page file name\x3A:sbsimages\\g<Controls>.html
    Folder for the .html file\x3A:One level over the images
    Webpage title\x3A:SBS Images\x3A Controls=\\g<Controls>
    Webpage background color\x3A:light grey
    # of columns\x3A:1
    Table border width\x3A:1
    Table border color\x3A:blue
    Image spacing\x3A:1
    Image border width\x3A:1
    Open new window when viewing full image?:For each image
    Make a zipfile containing the full-size images?:Yes
    Zipfile name\x3A:Images.zip

CreateWebPage:[module_num:3|svn_version:\'9401\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Full-size image name\x3A:IllumDNA
    Use thumbnail images?:Yes
    Thumbnail image name\x3A:IllumGFP
    Web page file name\x3A:sbsimages.html
    Folder for the .html file\x3A:One level over the images
    Webpage title\x3A:SBS Images
    Webpage background color\x3A:light grey
    # of columns\x3A:5
    Table border width\x3A:4
    Table border color\x3A:blue
    Image spacing\x3A:3
    Image border width\x3A:2
    Open new window when viewing full image?:No
    Make a zipfile containing the full-size images?:Yes
    Zipfile name\x3A:Images.zip
"""
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, C.CreateWebPage))
        self.assertEqual(module.orig_image_name, "ColorImage")
        self.assertTrue(module.wants_thumbnails)
        self.assertEqual(module.thumbnail_image_name, "ColorThumbnail")
        self.assertEqual(module.web_page_file_name, "sbsimages\\g<Controls>.html")
        self.assertEqual(module.directory_choice.dir_choice, C.DIR_SAME)
        self.assertEqual(module.title, "SBS Images\x3A Controls=\\g<Controls>")
        self.assertEqual(module.background_color, "light grey")
        self.assertEqual(module.columns, 12)
        self.assertEqual(module.table_border_width, 1)
        self.assertEqual(module.table_border_color, "blue")
        self.assertEqual(module.image_spacing, 1)
        self.assertEqual(module.image_border_width, 1)
        self.assertEqual(module.create_new_window, C.OPEN_ONCE)
        self.assertFalse(module.wants_zip_file)
        self.assertEqual(module.zipfile_name, "Images.zip")
        
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, C.CreateWebPage))
        self.assertEqual(module.orig_image_name, "IllumDNA")
        self.assertFalse(module.wants_thumbnails)
        self.assertEqual(module.thumbnail_image_name, "IllumGFP")
        self.assertEqual(module.web_page_file_name, "sbsimages\\g<Controls>.html")
        self.assertEqual(module.directory_choice.dir_choice, C.DIR_ABOVE)
        self.assertEqual(module.title, "SBS Images: Controls=\\g<Controls>")
        self.assertEqual(module.background_color, "light grey")
        self.assertEqual(module.columns, 1)
        self.assertEqual(module.table_border_width, 1)
        self.assertEqual(module.table_border_color, "blue")
        self.assertEqual(module.image_spacing, 1)
        self.assertEqual(module.image_border_width, 1)
        self.assertEqual(module.create_new_window, C.OPEN_EACH)
        self.assertTrue(module.wants_zip_file)
        self.assertEqual(module.zipfile_name, "Images.zip")
        
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, C.CreateWebPage))
        self.assertEqual(module.orig_image_name, "IllumDNA")
        self.assertTrue(module.wants_thumbnails)
        self.assertEqual(module.thumbnail_image_name, "IllumGFP")
        self.assertEqual(module.web_page_file_name, "sbsimages.html")
        self.assertEqual(module.directory_choice.dir_choice, C.DIR_ABOVE)
        self.assertEqual(module.title, "SBS Images")
        self.assertEqual(module.background_color, "light grey")
        self.assertEqual(module.columns, 5)
        self.assertEqual(module.table_border_width, 4)
        self.assertEqual(module.table_border_color, "blue")
        self.assertEqual(module.image_spacing, 3)
        self.assertEqual(module.image_border_width, 2)
        self.assertEqual(module.create_new_window, C.OPEN_NO)
        self.assertTrue(module.wants_zip_file)
        self.assertEqual(module.zipfile_name, "Images.zip")
        
    def test_01_03_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9462

CreateWebPage:[module_num:1|svn_version:\'9401\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Full-size image name\x3A:ColorImage
    Use thumbnail images?:Yes
    Thumbnail image name\x3A:ColorThumbnail
    Web page file name\x3A:sbsimages\\g<Controls>.html
    Folder for the .html file\x3A:Elsewhere...|/imaging/analysis
    Webpage title\x3A:SBS Images\x3A Controls=\\g<Controls>
    Webpage background color\x3A:light grey
    # of columns\x3A:12
    Table border width\x3A:1
    Table border color\x3A:blue
    Image spacing\x3A:1
    Image border width\x3A:1
    Open new window when viewing full image?:Once only
    Make a zipfile containing the full-size images?:No
    Zipfile name\x3A:Images.zip
"""
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, C.CreateWebPage))
        self.assertEqual(module.orig_image_name, "ColorImage")
        self.assertTrue(module.wants_thumbnails)
        self.assertEqual(module.thumbnail_image_name, "ColorThumbnail")
        self.assertEqual(module.web_page_file_name, "sbsimages\\g<Controls>.html")
        self.assertEqual(module.directory_choice.dir_choice, 
                         C.ABSOLUTE_FOLDER_NAME)
        self.assertEqual(module.directory_choice.custom_path, 
                         "/imaging/analysis")
        self.assertEqual(module.title, "SBS Images\x3A Controls=\\g<Controls>")
        self.assertEqual(module.background_color, "light grey")
        self.assertEqual(module.columns, 12)
        self.assertEqual(module.table_border_width, 1)
        self.assertEqual(module.table_border_color, "blue")
        self.assertEqual(module.image_spacing, 1)
        self.assertEqual(module.image_border_width, 1)
        self.assertEqual(module.create_new_window, C.OPEN_ONCE)
        self.assertFalse(module.wants_zip_file)
        self.assertEqual(module.zipfile_name, "Images.zip")
    
    def run_create_webpage(self, image_paths, thumb_paths = None, 
                           metadata = None, alter_fn = None):
        '''Run the create_webpage module, returning the resulting HTML document
        
        image_paths - list of path / filename tuples. The function will
                      write an image to each of these and put images and
                      measurements into the workspace for each.
        thumb_paths - if present a list of path / filename tuples. Same as above
        metadata    - a dictionary of feature / string values
        alter_fn    - function taking a CreateWebPage module, for you to
                      alter the module's settings
        '''
        
        np.random.seed(0)
        module = C.CreateWebPage()
        module.module_num = 1
        module.orig_image_name.value = IMAGE_NAME
        module.web_page_file_name.value = DEFAULT_HTML_FILE
        if alter_fn is not None:
            alter_fn(module)
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.add_module(module)
        
        images = [ (IMAGE_NAME, image_paths)]
        if thumb_paths:
            images += [ (THUMB_NAME, thumb_paths)]
            self.assertEqual(len(image_paths), len(thumb_paths))
            module.wants_thumbnails.value = True
            module.thumbnail_image_name.value = THUMB_NAME
        else:
            module.wants_thumbnails.value = False
            
        measurements = cpmeas.Measurements()
        
        workspace = cpw.Workspace(pipeline, module, 
                                  measurements, None, measurements, 
                                  None, None)
        for i in range(len(image_paths)):
            image_number = i+1
            if metadata is not None:
                for key in metadata.keys():
                    values = metadata[key]
                    feature = cpmeas.C_METADATA + "_" + key
                    measurements[cpmeas.IMAGE, feature, image_number] = values[i]
                    
            for image_name, paths in images:
                pixel_data = np.random.uniform(size=(10,13))
                path_name, file_name = paths[i]
                if path_name is None:
                    path_name = cpprefs.get_default_image_directory()
                    is_file = True
                elif path_name.lower().startswith("http"):
                    is_file = False
                    url = path_name + "/" + file_name
                    if "?" in file_name:
                        file_name = file_name.split("?", 1)[0]
                if is_file:
                    full_path = os.path.abspath(os.path.join(
                        self.directory, path_name, file_name))
                    url = pathname2url(full_path)
                    path = os.path.split(full_path)[0]
                else:
                    full_path = url
                    path = path_name
                if is_file:
                    imsave(full_path, pixel_data)
                path_feature = '_'.join((C_PATH_NAME, image_name))
                file_feature = '_'.join((C_FILE_NAME, image_name))
                url_feature = '_'.join((C_URL, image_name))
                measurements[cpmeas.IMAGE, path_feature, image_number] = \
                    path
                measurements[cpmeas.IMAGE, file_feature, image_number] =\
                    file_name
                measurements[cpmeas.IMAGE, url_feature, image_number] = url
                
        module.post_run(workspace)
        return measurements
    
    def read_html(self, html_path = None):
        '''Read html file, assuming the default location
        
        returns a DOM
        '''
        if html_path is None:
            html_path = os.path.join(cpprefs.get_default_image_directory(), 
                                     DEFAULT_HTML_FILE)
        fd = open(html_path, 'r')
        try:
            data = fd.read()
            return DOM.parseString(data)
        finally:
            fd.close()
    
    def ap(self, path):
        '''Get the absolute path to the file'''
        path = os.path.join(cpprefs.get_default_image_directory(), path)
        return os.path.abspath(path)
    
    def test_02_01_one_image_file(self):
        '''Test an image set with one image file'''
        self.run_create_webpage([(None, 'A01.png')])
        dom = self.read_html()
        self.assertTrue(dom.documentElement.tagName.lower(), "html")
        html_children = [n for n in dom.documentElement.childNodes
                         if n.nodeType == dom.ELEMENT_NODE]
        self.assertEqual(len(html_children), 2)
        body = html_children[1]
        self.assertEqual(body.tagName.lower(), "body")
        tables = body.getElementsByTagName("table")
        self.assertEqual(len(tables), 1)
        table = tables[0]
        #
        # Get the <tr> nodes
        #
        table_children = [n for n in table.childNodes
                         if n.nodeType == dom.ELEMENT_NODE]
        self.assertEqual(len(table_children), 1)
        tr = table_children[0]
        self.assertEqual(tr.tagName, "tr")
        #
        # Get the <td> node
        #
        tr_children = [n for n in tr.childNodes
                       if n.nodeType == dom.ELEMENT_NODE]
        self.assertEqual(len(tr_children), 1)
        td = tr_children[0]
        self.assertEqual(td.tagName, "td")
        #
        # Get the image node
        #
        td_children = [n for n in td.childNodes
                       if n.nodeType == dom.ELEMENT_NODE]
        self.assertEqual(len(td_children), 1)
        img = td_children[0]
        self.assertEqual(img.tagName, "img")
        self.assertTrue(img.hasAttribute("src"))
        self.assertEqual(img.getAttribute("src"), "A01.png")
        
    def test_02_02_title(self):
        TITLE = "My Title"
        def alter_fn(module):
            self.assertTrue(isinstance(module, C.CreateWebPage))
            module.title.value = TITLE
        
        self.run_create_webpage([(None, 'A01.png')], alter_fn = alter_fn)
        
        dom = self.read_html()
        title_elements = dom.getElementsByTagName('title')
        self.assertEqual(len(title_elements), 1)
        text_children = [x.data for x in title_elements[0].childNodes
                         if x.nodeType == dom.TEXT_NODE]
        text = ''.join(text_children).strip()
        self.assertEqual(text, TITLE)
        
    def test_02_03_title_with_metadata(self):
        expected = "Lee's Title"
        def alter_fn(module):
            self.assertTrue(isinstance(module, C.CreateWebPage))
            module.title.value = "\\g<BelongsToMe> Title"
        
        self.run_create_webpage([(None, 'A01.png')], alter_fn = alter_fn,
                                metadata={ "BelongsToMe":["Lee's"] })
        
        dom = self.read_html()
        title_elements = dom.getElementsByTagName('title')
        self.assertEqual(len(title_elements), 1)
        text_children = [x.data for x in title_elements[0].childNodes
                         if x.nodeType == dom.TEXT_NODE]
        text = ''.join(text_children).strip()
        self.assertEqual(text, expected)
        
    def test_02_04_bg_color(self):
        COLOR = "hazelnut"
        def alter_fn(module):
            self.assertTrue(isinstance(module, C.CreateWebPage))
            module.background_color.value = COLOR
        self.run_create_webpage([(None, 'A01.png')], alter_fn = alter_fn)
        dom = self.read_html()
        bodies = dom.getElementsByTagName("body")
        self.assertEqual(len(bodies), 1)
        body = bodies[0]
        self.assertTrue(body.hasAttribute("bgcolor"))
        self.assertEqual(body.getAttribute("bgcolor"), COLOR)
        
    def test_02_05_table_border_width(self):
        BORDERWIDTH = 15
        def alter_fn(module):
            self.assertTrue(isinstance(module, C.CreateWebPage))
            module.table_border_width.value = BORDERWIDTH
        self.run_create_webpage([(None, 'A01.png')], alter_fn = alter_fn)
        dom = self.read_html()
        tables = dom.getElementsByTagName("table")
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertTrue(table.hasAttribute("border"))
        self.assertEqual(table.getAttribute("border"), str(BORDERWIDTH))
        
    def test_02_06_table_border_color(self):
        COLOR = "corvetteyellow"
        def alter_fn(module):
            self.assertTrue(isinstance(module, C.CreateWebPage))
            module.table_border_color.value = COLOR
        self.run_create_webpage([(None, 'A01.png')], alter_fn = alter_fn)
        dom = self.read_html()
        tables = dom.getElementsByTagName("table")
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertTrue(table.hasAttribute("bordercolor"))
        self.assertEqual(table.getAttribute("bordercolor"), COLOR)
        
    def test_02_07_table_cell_spacing(self):
        CELL_SPACING = 11
        def alter_fn(module):
            self.assertTrue(isinstance(module, C.CreateWebPage))
            module.image_spacing.value = CELL_SPACING
        self.run_create_webpage([(None, 'A01.png')], alter_fn = alter_fn)
        dom = self.read_html()
        tables = dom.getElementsByTagName("table")
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertTrue(table.hasAttribute("cellspacing"))
        self.assertEqual(table.getAttribute("cellspacing"), str(CELL_SPACING))
    
    def test_02_08_image_border_width(self):
        IMAGE_BORDER_WIDTH = 23
        def alter_fn(module):
            self.assertTrue(isinstance(module, C.CreateWebPage))
            module.image_border_width.value = IMAGE_BORDER_WIDTH
        self.run_create_webpage([(None, 'A01.png')], alter_fn = alter_fn)
        dom = self.read_html()
        imgs = dom.getElementsByTagName("img")
        self.assertEqual(len(imgs), 1)
        img = imgs[0]
        self.assertTrue(img.hasAttribute("border"))
        self.assertEqual(img.getAttribute("border"), str(IMAGE_BORDER_WIDTH))
        
    def test_02_09_columns(self):
        def alter_fn(module):
            self.assertTrue(isinstance(module, C.CreateWebPage))
            module.columns.value = 3
        self.run_create_webpage(
            [(None, 'A01.png'), (None, 'A02.png'), (None, 'A03.png'),
             (None, 'B01.png'), (None, 'B02.png'), (None, 'B03.png')], 
            alter_fn = alter_fn)
        dom = self.read_html()
        tables = dom.getElementsByTagName("table")
        self.assertEqual(len(tables), 1)
        table = tables[0]
        trs = table.getElementsByTagName("tr")
        self.assertEqual(len(trs), 2)
        for col, tr in zip(("A","B"), trs):
            tds = tr.getElementsByTagName("td")
            self.assertEqual(len(tds), 3)
            for i, td in enumerate(tds):
                imgs = td.getElementsByTagName("img")
                self.assertEqual(len(imgs), 1)
                img = imgs[0]
                self.assertTrue(img.hasAttribute("src"))
                self.assertEqual(img.getAttribute("src"), 
                                 "%s0%d.png" % (col, i+1))

    def test_02_10_partial_columns(self):
        def alter_fn(module):
            self.assertTrue(isinstance(module, C.CreateWebPage))
            module.columns.value = 3
        self.run_create_webpage(
            [(None, 'A01.png'), (None, 'A02.png'), (None, 'A03.png'),
             (None, 'B01.png'), (None, 'B02.png')], 
            alter_fn = alter_fn)
        dom = self.read_html()
        tables = dom.getElementsByTagName("table")
        self.assertEqual(len(tables), 1)
        table = tables[0]
        trs = table.getElementsByTagName("tr")
        self.assertEqual(len(trs), 2)
        for col, colcount, tr in zip(("A","B"), (3,2), trs):
            tds = tr.getElementsByTagName("td")
            self.assertEqual(len(tds), colcount)
            for i, td in enumerate(tds):
                imgs = td.getElementsByTagName("img")
                self.assertEqual(len(imgs), 1)
                img = imgs[0]
                self.assertTrue(img.hasAttribute("src"))
                self.assertEqual(img.getAttribute("src"), 
                                 "%s0%d.png" % (col, i+1))
    
    def test_03_01_thumb(self):
        self.run_create_webpage([(None, 'A01.png')],
                                [(None, 'A01_thumb.png')])
        dom = self.read_html()
        tds = dom.getElementsByTagName('td')
        self.assertEqual(len(tds), 1)
        td = tds[0]
        links = td.getElementsByTagName('a')
        self.assertEqual(len(links), 1)
        link = links[0]
        self.assertTrue(link.hasAttribute("href"))
        self.assertEqual(self.ap(link.getAttribute("href")),
                         self.ap("A01.png"))
        imgs = link.getElementsByTagName('img')
        self.assertEqual(len(imgs), 1)
        img = imgs[0]
        self.assertTrue(img.hasAttribute("src"))
        self.assertEqual(self.ap(img.getAttribute("src")), 
                         self.ap("A01_thumb.png"))
        
    def test_03_02_open_once(self):
        def alter_fn(module):
            self.assertTrue(isinstance(module, C.CreateWebPage))
            module.create_new_window.value = C.OPEN_ONCE
        self.run_create_webpage([(None, 'A01.png')],
                                [(None, 'A01_thumb.png')], alter_fn = alter_fn)
        dom = self.read_html()
        links = dom.getElementsByTagName("a")
        self.assertEqual(len(links), 1)
        link = links[0]
        self.assertTrue(link.hasAttribute("target"))
        self.assertEqual(link.getAttribute("target"), "_CPNewWindow")
        
    def test_03_03_open_each(self):
        def alter_fn(module):
            self.assertTrue(isinstance(module, C.CreateWebPage))
            module.create_new_window.value = C.OPEN_EACH
        self.run_create_webpage([(None, 'A01.png')],
                                [(None, 'A01_thumb.png')], alter_fn = alter_fn)
        dom = self.read_html()
        links = dom.getElementsByTagName("a")
        self.assertEqual(len(links), 1)
        link = links[0]
        self.assertTrue(link.hasAttribute("target"))
        self.assertEqual(link.getAttribute("target"), "_blank")

    def test_03_03_open_no(self):
        def alter_fn(module):
            self.assertTrue(isinstance(module, C.CreateWebPage))
            module.create_new_window.value = C.OPEN_NO
        self.run_create_webpage([(None, 'A01.png')],
                                [(None, 'A01_thumb.png')], alter_fn = alter_fn)
        dom = self.read_html()
        links = dom.getElementsByTagName("a")
        self.assertEqual(len(links), 1)
        link = links[0]
        self.assertFalse(link.hasAttribute("target"))
        
    def test_04_01_above_image(self):
        '''Make the HTML file in the directory above the image'''
        def alter_fn(module):
            self.assertTrue(isinstance(module, C.CreateWebPage))
            module.directory_choice.value = C.DIR_ABOVE
        self.run_create_webpage([(None, 'A01.png')], alter_fn = alter_fn)
        dom = self.read_html(os.path.join(self.directory, DEFAULT_HTML_FILE))
        imgs = dom.getElementsByTagName("img")
        self.assertEqual(len(imgs), 1)
        img = imgs[0]
        self.assertTrue(img.hasAttribute("src"))
        self.assertEqual(img.getAttribute("src"), "1/A01.png")
        
    def test_04_02_thumb_in_other_dir(self):
        '''Put the image and thumbnail in different directories'''
        self.run_create_webpage([(None, 'A01.png')],
                                [(os.path.join(self.directory, "2"), 'A01_thumb.png')])
        dom = self.read_html()
        imgs = dom.getElementsByTagName("img")
        self.assertEqual(len(imgs), 1)
        img = imgs[0]
        self.assertTrue(img.hasAttribute("src"))
        self.assertEqual(img.getAttribute("src"), "../2/A01_thumb.png")
        
    def test_04_03_metadata_filename(self):
        '''Make two different webpages using metadata'''
        def alter_fn(module):
            self.assertTrue(isinstance(module, C.CreateWebPage))
            module.web_page_file_name.value = '\\g<FileName>'
        self.run_create_webpage([(None, 'A01.png'),(None, 'A02.png')],
                                metadata={'FileName':['foo','bar']},
                                alter_fn = alter_fn)
        for file_name, image_name in (('foo.html', 'A01.png'),
                                      ('bar.html', 'A02.png')):
            path = os.path.join(cpprefs.get_default_image_directory(), file_name)
            dom = self.read_html(path)
            imgs = dom.getElementsByTagName("img")
            self.assertEqual(len(imgs), 1)
            img = imgs[0]
            self.assertTrue(img.hasAttribute("src"))
            self.assertEqual(img.getAttribute("src"), image_name)
            
    def test_04_04_abspath(self):
        # Specify an absolute path for the images.
        def alter_fn(module):
            self.assertTrue(isinstance(module, C.CreateWebPage))
            module.directory_choice.dir_choice = C.ABSOLUTE_FOLDER_NAME
            module.directory_choice.custom_path = self.alt_directory
        filenames = [(None, 'A%02d.png' % i) for i in range(1, 3)]
        self.run_create_webpage(filenames, alter_fn=alter_fn)
        dom = self.read_html(os.path.join(self.alt_directory, DEFAULT_HTML_FILE))
        imgs = dom.getElementsByTagName("img")
        self.assertEqual(len(imgs), 2)
        for img in imgs:
            self.assertTrue(img.hasAttribute("src"))
            image_name = str(img.getAttribute("src"))
            path = url2pathname(image_name)
            self.assertTrue(os.path.exists(path))
            
    def test_05_01_zipfiles(self):
        # Test the zipfile function
        def alter_fn(module):
            self.assertTrue(isinstance(module, C.CreateWebPage))
            module.wants_zip_file.value = True
            module.zipfile_name.value = ZIPFILE_NAME
        filenames = ['A%02d.png' % i for i in range(1, 3)]
        self.run_create_webpage([(None, fn) for fn in filenames],
                                alter_fn=alter_fn)
        
        zpath = os.path.join(cpprefs.get_default_image_directory(), ZIPFILE_NAME)
        with zipfile.ZipFile(zpath, "r") as zfile:
            assert isinstance(zfile, zipfile.ZipFile)
            for filename in filenames:
                fpath = os.path.join(cpprefs.get_default_image_directory(), 
                                     filename)
                with open(fpath, "rb") as fd:
                    with zfile.open(filename, "r") as zfd:
                        self.assertEqual(fd.read(), zfd.read())
                        
    def test_05_02_zipfile_and_metadata(self):
        # Test the zipfile function with metadata substitution
        def alter_fn(module):
            self.assertTrue(isinstance(module, C.CreateWebPage))
            module.wants_zip_file.value = True
            module.zipfile_name.value = '\\g<FileName>'
        filenames = ['A%02d.png' % i for i in range(1, 3)]
        zipfiles = ['A%02d' % i for i in range(1, 3)]
        self.run_create_webpage(
            [(None, fn) for fn in filenames],
            metadata=dict(FileName=zipfiles),
            alter_fn=alter_fn)
        
        for filename, zname in zip(filenames, zipfiles):
            zpath = os.path.join(cpprefs.get_default_image_directory(), zname)
            zpath += ".zip"
            fpath = os.path.join(cpprefs.get_default_image_directory(), 
                                 filename)
            with zipfile.ZipFile(zpath, "r") as zfile:
                with open(fpath, "rb") as fd:
                    with zfile.open(filename, "r") as zfd:
                        self.assertEqual(fd.read(), zfd.read())
                        
    def test_05_03_http_image_zipfile(self):
        # Make a zipfile using files accessed from the web
        def alter_fn(module):
            self.assertTrue(isinstance(module, C.CreateWebPage))
            module.wants_zip_file.value = True
            module.zipfile_name.value = ZIPFILE_NAME
            module.directory_choice.dir_choice = C.ABSOLUTE_FOLDER_NAME
            module.directory_choice.custom_path = cpprefs.get_default_image_directory()
            
        url_root = "https://svn.broadinstitute.org/CellProfiler/trunk/ExampleImages/ExampleSBSImages/"
        url_query = "?r=11710"
        filenames = [ (url_root,  fn + url_query) for fn in
                      ("Channel1-01-A-01.tif", "Channel2-01-A-01.tif",
                       "Channel1-02-A-02.tif", "Channel2-02-A-02.tif")]
        self.run_create_webpage(filenames, alter_fn = alter_fn)
        zpath = os.path.join(cpprefs.get_default_image_directory(),
                             ZIPFILE_NAME)
        with zipfile.ZipFile(zpath, "r") as zfile:
            for _, filename in filenames:
                fn = filename.split("?", 1)[0]
                url = url_root + "/" + filename
                svn_fd = urlopen(url)
                with zfile.open(fn, "r") as zip_fd:
                    data = zip_fd.read()
                    offset = 0
                    while offset < len(data):
                        udata = svn_fd.read(len(data) - offset)
                        self.assertEqual(
                            udata, data[offset:(offset + len(udata))])
                        offset += len(udata)
                svn_fd.close()
        

