import StringIO
import os
import shutil
import tempfile
import unittest
import urllib
import urllib2
import xml.dom.minidom
import zipfile

import bioformats
import bioformats.formatwriter
import cellprofiler.grid
import cellprofiler.image
import cellprofiler.measurement
import cellprofiler.module
import cellprofiler.modules.createwebpage
import cellprofiler.modules.loadimages
import cellprofiler.modules.loadimages
import cellprofiler.pipeline
import cellprofiler.preferences
import cellprofiler.region
import cellprofiler.workspace
import numpy

cellprofiler.preferences.set_headless()

IMAGE_NAME = "image"
THUMB_NAME = "thumb"
DEFAULT_HTML_FILE = "default.html"
ZIPFILE_NAME = "zipfile.zip"


class TestCreateWebPage(unittest.TestCase):
    def setUp(self):
        #
        # Make a temporary directory structure
        #
        cellprofiler.preferences.set_headless()
        directory = self.directory = tempfile.mkdtemp()
        for i in range(3):
            os.mkdir(os.path.join(directory, str(i)))
            for j in range(3):
                os.mkdir(os.path.join(directory, str(i), str(j)))
                for k in range(3):
                    os.mkdir(os.path.join(directory, str(i), str(j), str(k)))
        cellprofiler.preferences.set_default_image_directory(os.path.join(self.directory, "1"))
        self.alt_directory = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.directory)
        shutil.rmtree(self.alt_directory)

    def test_00_00_remember_to_put_new_text_in_the_dictionary(self):
        """Make sure people use TRANSLATION_DICTIONARY"""
        self.assertTrue(cellprofiler.modules.createwebpage.DIR_ABOVE in cellprofiler.modules.createwebpage.TRANSLATION_DICTIONARY)
        self.assertTrue(cellprofiler.modules.createwebpage.DIR_SAME in cellprofiler.modules.createwebpage.TRANSLATION_DICTIONARY)
        self.assertTrue("One level over the images" in cellprofiler.modules.createwebpage.TRANSLATION_DICTIONARY)
        self.assertTrue("Same as the images" in cellprofiler.modules.createwebpage.TRANSLATION_DICTIONARY)

        self.assertTrue(cellprofiler.modules.createwebpage.OPEN_ONCE in cellprofiler.modules.createwebpage.TRANSLATION_DICTIONARY)
        self.assertTrue(cellprofiler.modules.createwebpage.OPEN_EACH in cellprofiler.modules.createwebpage.TRANSLATION_DICTIONARY)
        self.assertTrue(cellprofiler.modules.createwebpage.OPEN_NO in cellprofiler.modules.createwebpage.TRANSLATION_DICTIONARY)

        self.assertTrue("Once only" in cellprofiler.modules.createwebpage.TRANSLATION_DICTIONARY)
        self.assertTrue("For each image" in cellprofiler.modules.createwebpage.TRANSLATION_DICTIONARY)
        self.assertTrue("No" in cellprofiler.modules.createwebpage.TRANSLATION_DICTIONARY)

        self.assertEqual(len(cellprofiler.modules.createwebpage.TRANSLATION_DICTIONARY), 5,
                         "Please update this test to include your newly entered translation")

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
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
        self.assertEqual(module.orig_image_name, "ColorImage")
        self.assertTrue(module.wants_thumbnails)
        self.assertEqual(module.thumbnail_image_name, "ColorThumbnail")
        self.assertEqual(module.web_page_file_name, "sbsimages\\g<Controls>.html")
        self.assertEqual(module.directory_choice.dir_choice, cellprofiler.modules.createwebpage.DIR_SAME)
        self.assertEqual(module.title, "SBS Images\x3A Controls=\\g<Controls>")
        self.assertEqual(module.background_color, "light grey")
        self.assertEqual(module.columns, 12)
        self.assertEqual(module.table_border_width, 1)
        self.assertEqual(module.table_border_color, "blue")
        self.assertEqual(module.image_spacing, 1)
        self.assertEqual(module.image_border_width, 1)
        self.assertEqual(module.create_new_window, cellprofiler.modules.createwebpage.OPEN_ONCE)
        self.assertFalse(module.wants_zip_file)
        self.assertEqual(module.zipfile_name, "Images.zip")

        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
        self.assertEqual(module.orig_image_name, "IllumDNA")
        self.assertFalse(module.wants_thumbnails)
        self.assertEqual(module.thumbnail_image_name, "IllumGFP")
        self.assertEqual(module.web_page_file_name, "sbsimages\\g<Controls>.html")
        self.assertEqual(module.directory_choice.dir_choice, cellprofiler.modules.createwebpage.DIR_ABOVE)
        self.assertEqual(module.title, "SBS Images: Controls=\\g<Controls>")
        self.assertEqual(module.background_color, "light grey")
        self.assertEqual(module.columns, 1)
        self.assertEqual(module.table_border_width, 1)
        self.assertEqual(module.table_border_color, "blue")
        self.assertEqual(module.image_spacing, 1)
        self.assertEqual(module.image_border_width, 1)
        self.assertEqual(module.create_new_window, cellprofiler.modules.createwebpage.OPEN_EACH)
        self.assertTrue(module.wants_zip_file)
        self.assertEqual(module.zipfile_name, "Images.zip")

        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
        self.assertEqual(module.orig_image_name, "IllumDNA")
        self.assertTrue(module.wants_thumbnails)
        self.assertEqual(module.thumbnail_image_name, "IllumGFP")
        self.assertEqual(module.web_page_file_name, "sbsimages.html")
        self.assertEqual(module.directory_choice.dir_choice, cellprofiler.modules.createwebpage.DIR_ABOVE)
        self.assertEqual(module.title, "SBS Images")
        self.assertEqual(module.background_color, "light grey")
        self.assertEqual(module.columns, 5)
        self.assertEqual(module.table_border_width, 4)
        self.assertEqual(module.table_border_color, "blue")
        self.assertEqual(module.image_spacing, 3)
        self.assertEqual(module.image_border_width, 2)
        self.assertEqual(module.create_new_window, cellprofiler.modules.createwebpage.OPEN_NO)
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
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO.StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
        self.assertEqual(module.orig_image_name, "ColorImage")
        self.assertTrue(module.wants_thumbnails)
        self.assertEqual(module.thumbnail_image_name, "ColorThumbnail")
        self.assertEqual(module.web_page_file_name, "sbsimages\\g<Controls>.html")
        self.assertEqual(module.directory_choice.dir_choice,
                         cellprofiler.modules.createwebpage.ABSOLUTE_FOLDER_NAME)
        self.assertEqual(module.directory_choice.custom_path,
                         "/imaging/analysis")
        self.assertEqual(module.title, "SBS Images\x3A Controls=\\g<Controls>")
        self.assertEqual(module.background_color, "light grey")
        self.assertEqual(module.columns, 12)
        self.assertEqual(module.table_border_width, 1)
        self.assertEqual(module.table_border_color, "blue")
        self.assertEqual(module.image_spacing, 1)
        self.assertEqual(module.image_border_width, 1)
        self.assertEqual(module.create_new_window, cellprofiler.modules.createwebpage.OPEN_ONCE)
        self.assertFalse(module.wants_zip_file)
        self.assertEqual(module.zipfile_name, "Images.zip")

    def run_create_webpage(self, image_paths, thumb_paths=None, metadata=None, alter_fn=None):
        """Run the create_webpage module, returning the resulting HTML document

        image_paths - list of path / filename tuples. The function will
                      write an image to each of these and put images and
                      measurements into the workspace for each.
        thumb_paths - if present a list of path / filename tuples. Same as above
        metadata    - a dictionary of feature / string values
        alter_fn    - function taking a CreateWebPage module, for you to
                      alter the module's settings
        """

        numpy.random.seed(0)
        module = cellprofiler.modules.createwebpage.CreateWebPage()
        module.module_num = 1
        module.orig_image_name.value = IMAGE_NAME
        module.web_page_file_name.value = DEFAULT_HTML_FILE
        if alter_fn is not None:
            alter_fn(module)
        pipeline = cellprofiler.pipeline.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cellprofiler.pipeline.RunExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.add_module(module)

        images = [(IMAGE_NAME, image_paths)]
        if thumb_paths:
            images += [(THUMB_NAME, thumb_paths)]
            self.assertEqual(len(image_paths), len(thumb_paths))
            module.wants_thumbnails.value = True
            module.thumbnail_image_name.value = THUMB_NAME
        else:
            module.wants_thumbnails.value = False

        measurements = cellprofiler.measurement.Measurements()

        workspace = cellprofiler.workspace.Workspace(pipeline, module,
                                                     measurements, None, measurements,
                                                     None, None)
        for i in range(len(image_paths)):
            image_number = i + 1
            if metadata is not None:
                for key in metadata.keys():
                    values = metadata[key]
                    feature = cellprofiler.measurement.C_METADATA + "_" + key
                    measurements[cellprofiler.measurement.IMAGE, feature, image_number] = values[i]

            for image_name, paths in images:
                pixel_data = numpy.random.uniform(size=(10, 13))
                path_name, file_name = paths[i]
                if path_name is None:
                    path_name = cellprofiler.preferences.get_default_image_directory()
                    is_file = True
                elif path_name.lower().startswith("http"):
                    is_file = False
                    url = path_name + "/" + file_name
                    if "?" in file_name:
                        file_name = file_name.split("?", 1)[0]
                if is_file:
                    full_path = os.path.abspath(os.path.join(
                            self.directory, path_name, file_name))
                    url = cellprofiler.modules.loadimages.pathname2url(full_path)
                    path = os.path.split(full_path)[0]
                else:
                    full_path = url
                    path = path_name
                if is_file:
                    bioformats.formatwriter.write_image(full_path, pixel_data, bioformats.PT_UINT8)
                path_feature = '_'.join((cellprofiler.modules.loadimages.C_PATH_NAME, image_name))
                file_feature = '_'.join((cellprofiler.modules.loadimages.C_FILE_NAME, image_name))
                url_feature = '_'.join((cellprofiler.modules.loadimages.C_URL, image_name))
                measurements[cellprofiler.measurement.IMAGE, path_feature, image_number] = \
                    path
                measurements[cellprofiler.measurement.IMAGE, file_feature, image_number] = \
                    file_name
                measurements[cellprofiler.measurement.IMAGE, url_feature, image_number] = url

        module.post_run(workspace)
        return measurements

    def read_html(self, html_path=None):
        """Read html file, assuming the default location

        returns a DOM
        """
        if html_path is None:
            html_path = os.path.join(cellprofiler.preferences.get_default_image_directory(),
                                     DEFAULT_HTML_FILE)
        fd = open(html_path, 'r')
        try:
            data = fd.read()
            return xml.dom.minidom.parseString(data)
        finally:
            fd.close()

    def ap(self, path):
        """Get the absolute path to the file"""
        path = os.path.join(cellprofiler.preferences.get_default_image_directory(), path)
        return os.path.abspath(path)

    def test_02_01_one_image_file(self):
        """Test an image set with one image file"""
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
            self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
            module.title.value = TITLE

        self.run_create_webpage([(None, 'A01.png')], alter_fn=alter_fn)

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
            self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
            module.title.value = "\\g<BelongsToMe> Title"

        self.run_create_webpage([(None, 'A01.png')], alter_fn=alter_fn,
                                metadata={"BelongsToMe": ["Lee's"]})

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
            self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
            module.background_color.value = COLOR

        self.run_create_webpage([(None, 'A01.png')], alter_fn=alter_fn)
        dom = self.read_html()
        bodies = dom.getElementsByTagName("body")
        self.assertEqual(len(bodies), 1)
        body = bodies[0]
        self.assertTrue(body.hasAttribute("bgcolor"))
        self.assertEqual(body.getAttribute("bgcolor"), COLOR)

    def test_02_05_table_border_width(self):
        BORDERWIDTH = 15

        def alter_fn(module):
            self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
            module.table_border_width.value = BORDERWIDTH

        self.run_create_webpage([(None, 'A01.png')], alter_fn=alter_fn)
        dom = self.read_html()
        tables = dom.getElementsByTagName("table")
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertTrue(table.hasAttribute("border"))
        self.assertEqual(table.getAttribute("border"), str(BORDERWIDTH))

    def test_02_06_table_border_color(self):
        COLOR = "corvetteyellow"

        def alter_fn(module):
            self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
            module.table_border_color.value = COLOR

        self.run_create_webpage([(None, 'A01.png')], alter_fn=alter_fn)
        dom = self.read_html()
        tables = dom.getElementsByTagName("table")
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertTrue(table.hasAttribute("bordercolor"))
        self.assertEqual(table.getAttribute("bordercolor"), COLOR)

    def test_02_07_table_cell_spacing(self):
        CELL_SPACING = 11

        def alter_fn(module):
            self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
            module.image_spacing.value = CELL_SPACING

        self.run_create_webpage([(None, 'A01.png')], alter_fn=alter_fn)
        dom = self.read_html()
        tables = dom.getElementsByTagName("table")
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertTrue(table.hasAttribute("cellspacing"))
        self.assertEqual(table.getAttribute("cellspacing"), str(CELL_SPACING))

    def test_02_08_image_border_width(self):
        IMAGE_BORDER_WIDTH = 23

        def alter_fn(module):
            self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
            module.image_border_width.value = IMAGE_BORDER_WIDTH

        self.run_create_webpage([(None, 'A01.png')], alter_fn=alter_fn)
        dom = self.read_html()
        imgs = dom.getElementsByTagName("img")
        self.assertEqual(len(imgs), 1)
        img = imgs[0]
        self.assertTrue(img.hasAttribute("border"))
        self.assertEqual(img.getAttribute("border"), str(IMAGE_BORDER_WIDTH))

    def test_02_09_columns(self):
        def alter_fn(module):
            self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
            module.columns.value = 3

        self.run_create_webpage(
                [(None, 'A01.png'), (None, 'A02.png'), (None, 'A03.png'),
                 (None, 'B01.png'), (None, 'B02.png'), (None, 'B03.png')],
                alter_fn=alter_fn)
        dom = self.read_html()
        tables = dom.getElementsByTagName("table")
        self.assertEqual(len(tables), 1)
        table = tables[0]
        trs = table.getElementsByTagName("tr")
        self.assertEqual(len(trs), 2)
        for col, tr in zip(("A", "B"), trs):
            tds = tr.getElementsByTagName("td")
            self.assertEqual(len(tds), 3)
            for i, td in enumerate(tds):
                imgs = td.getElementsByTagName("img")
                self.assertEqual(len(imgs), 1)
                img = imgs[0]
                self.assertTrue(img.hasAttribute("src"))
                self.assertEqual(img.getAttribute("src"),
                                 "%s0%d.png" % (col, i + 1))

    def test_02_10_partial_columns(self):
        def alter_fn(module):
            self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
            module.columns.value = 3

        self.run_create_webpage(
                [(None, 'A01.png'), (None, 'A02.png'), (None, 'A03.png'),
                 (None, 'B01.png'), (None, 'B02.png')],
                alter_fn=alter_fn)
        dom = self.read_html()
        tables = dom.getElementsByTagName("table")
        self.assertEqual(len(tables), 1)
        table = tables[0]
        trs = table.getElementsByTagName("tr")
        self.assertEqual(len(trs), 2)
        for col, colcount, tr in zip(("A", "B"), (3, 2), trs):
            tds = tr.getElementsByTagName("td")
            self.assertEqual(len(tds), colcount)
            for i, td in enumerate(tds):
                imgs = td.getElementsByTagName("img")
                self.assertEqual(len(imgs), 1)
                img = imgs[0]
                self.assertTrue(img.hasAttribute("src"))
                self.assertEqual(img.getAttribute("src"),
                                 "%s0%d.png" % (col, i + 1))

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
            self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
            module.create_new_window.value = cellprofiler.modules.createwebpage.OPEN_ONCE

        self.run_create_webpage([(None, 'A01.png')],
                                [(None, 'A01_thumb.png')], alter_fn=alter_fn)
        dom = self.read_html()
        links = dom.getElementsByTagName("a")
        self.assertEqual(len(links), 1)
        link = links[0]
        self.assertTrue(link.hasAttribute("target"))
        self.assertEqual(link.getAttribute("target"), "_CPNewWindow")

    def test_03_03_open_each(self):
        def alter_fn(module):
            self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
            module.create_new_window.value = cellprofiler.modules.createwebpage.OPEN_EACH

        self.run_create_webpage([(None, 'A01.png')],
                                [(None, 'A01_thumb.png')], alter_fn=alter_fn)
        dom = self.read_html()
        links = dom.getElementsByTagName("a")
        self.assertEqual(len(links), 1)
        link = links[0]
        self.assertTrue(link.hasAttribute("target"))
        self.assertEqual(link.getAttribute("target"), "_blank")

    def test_03_03_open_no(self):
        def alter_fn(module):
            self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
            module.create_new_window.value = cellprofiler.modules.createwebpage.OPEN_NO

        self.run_create_webpage([(None, 'A01.png')],
                                [(None, 'A01_thumb.png')], alter_fn=alter_fn)
        dom = self.read_html()
        links = dom.getElementsByTagName("a")
        self.assertEqual(len(links), 1)
        link = links[0]
        self.assertFalse(link.hasAttribute("target"))

    def test_04_01_above_image(self):
        """Make the HTML file in the directory above the image"""

        def alter_fn(module):
            self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
            module.directory_choice.value = cellprofiler.modules.createwebpage.DIR_ABOVE

        self.run_create_webpage([(None, 'A01.png')], alter_fn=alter_fn)
        dom = self.read_html(os.path.join(self.directory, DEFAULT_HTML_FILE))
        imgs = dom.getElementsByTagName("img")
        self.assertEqual(len(imgs), 1)
        img = imgs[0]
        self.assertTrue(img.hasAttribute("src"))
        self.assertEqual(img.getAttribute("src"), "1/A01.png")

    def test_04_02_thumb_in_other_dir(self):
        """Put the image and thumbnail in different directories"""
        self.run_create_webpage([(None, 'A01.png')],
                                [(os.path.join(self.directory, "2"), 'A01_thumb.png')])
        dom = self.read_html()
        imgs = dom.getElementsByTagName("img")
        self.assertEqual(len(imgs), 1)
        img = imgs[0]
        self.assertTrue(img.hasAttribute("src"))
        self.assertEqual(img.getAttribute("src"), "../2/A01_thumb.png")

    def test_04_03_metadata_filename(self):
        """Make two different webpages using metadata"""

        def alter_fn(module):
            self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
            module.web_page_file_name.value = '\\g<FileName>'

        self.run_create_webpage([(None, 'A01.png'), (None, 'A02.png')],
                                metadata={'FileName': ['foo', 'bar']},
                                alter_fn=alter_fn)
        for file_name, image_name in (('foo.html', 'A01.png'),
                                      ('bar.html', 'A02.png')):
            path = os.path.join(cellprofiler.preferences.get_default_image_directory(), file_name)
            dom = self.read_html(path)
            imgs = dom.getElementsByTagName("img")
            self.assertEqual(len(imgs), 1)
            img = imgs[0]
            self.assertTrue(img.hasAttribute("src"))
            self.assertEqual(img.getAttribute("src"), image_name)

    def test_04_04_abspath(self):
        # Specify an absolute path for the images.
        def alter_fn(module):
            self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
            module.directory_choice.dir_choice = cellprofiler.modules.createwebpage.ABSOLUTE_FOLDER_NAME
            module.directory_choice.custom_path = self.alt_directory

        filenames = [(None, 'A%02d.png' % i) for i in range(1, 3)]
        self.run_create_webpage(filenames, alter_fn=alter_fn)
        dom = self.read_html(os.path.join(self.alt_directory, DEFAULT_HTML_FILE))
        imgs = dom.getElementsByTagName("img")
        self.assertEqual(len(imgs), 2)
        for img in imgs:
            self.assertTrue(img.hasAttribute("src"))
            image_name = str(img.getAttribute("src"))
            path = cellprofiler.modules.loadimages.url2pathname(image_name)
            self.assertTrue(os.path.exists(path))

    def test_05_01_zipfiles(self):
        # Test the zipfile function
        def alter_fn(module):
            self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
            module.wants_zip_file.value = True
            module.zipfile_name.value = ZIPFILE_NAME

        filenames = ['A%02d.png' % i for i in range(1, 3)]
        self.run_create_webpage([(None, fn) for fn in filenames],
                                alter_fn=alter_fn)

        zpath = os.path.join(cellprofiler.preferences.get_default_image_directory(), ZIPFILE_NAME)
        with zipfile.ZipFile(zpath, "r") as zfile:
            assert isinstance(zfile, zipfile.ZipFile)
            for filename in filenames:
                fpath = os.path.join(cellprofiler.preferences.get_default_image_directory(),
                                     filename)
                with open(fpath, "rb") as fd:
                    with zfile.open(filename, "r") as zfd:
                        self.assertEqual(fd.read(), zfd.read())

    def test_05_02_zipfile_and_metadata(self):
        # Test the zipfile function with metadata substitution
        def alter_fn(module):
            self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
            module.wants_zip_file.value = True
            module.zipfile_name.value = '\\g<FileName>'

        filenames = ['A%02d.png' % i for i in range(1, 3)]
        zipfiles = ['A%02d' % i for i in range(1, 3)]
        self.run_create_webpage(
                [(None, fn) for fn in filenames],
                metadata=dict(FileName=zipfiles),
                alter_fn=alter_fn)

        for filename, zname in zip(filenames, zipfiles):
            zpath = os.path.join(cellprofiler.preferences.get_default_image_directory(), zname)
            zpath += ".zip"
            fpath = os.path.join(cellprofiler.preferences.get_default_image_directory(),
                                 filename)
            with zipfile.ZipFile(zpath, "r") as zfile:
                with open(fpath, "rb") as fd:
                    with zfile.open(filename, "r") as zfd:
                        self.assertEqual(fd.read(), zfd.read())

    def test_05_03_http_image_zipfile(self):
        # Make a zipfile using files accessed from the web
        def alter_fn(module):
            self.assertTrue(isinstance(module, cellprofiler.modules.createwebpage.CreateWebPage))
            module.wants_zip_file.value = True
            module.zipfile_name.value = ZIPFILE_NAME
            module.directory_choice.dir_choice = cellprofiler.modules.createwebpage.ABSOLUTE_FOLDER_NAME
            module.directory_choice.custom_path = cellprofiler.preferences.get_default_image_directory()

        url_root = "http://cellprofiler.org/svnmirror/ExampleImages/ExampleSBSImages/"
        url_query = "?r=11710"
        filenames = [(url_root, fn + url_query) for fn in
                     ("Channel1-01-A-01.tif", "Channel2-01-A-01.tif",
                      "Channel1-02-A-02.tif", "Channel2-02-A-02.tif")]
        #
        # Make sure URLs are accessible
        #
        try:
            for filename in filenames:
                urllib.URLopener().open("".join(filename)).close()
        except IOError, e:
            def bad_url(e=e):
                raise e

            unittest.expectedFailure(bad_url)()

        self.run_create_webpage(filenames, alter_fn=alter_fn)
        zpath = os.path.join(cellprofiler.preferences.get_default_image_directory(),
                             ZIPFILE_NAME)
        with zipfile.ZipFile(zpath, "r") as zfile:
            for _, filename in filenames:
                fn = filename.split("?", 1)[0]
                url = url_root + "/" + filename
                svn_fd = urllib2.urlopen(url)
                with zfile.open(fn, "r") as zip_fd:
                    data = zip_fd.read()
                    offset = 0
                    while offset < len(data):
                        udata = svn_fd.read(len(data) - offset)
                        self.assertEqual(
                                udata, data[offset:(offset + len(udata))])
                        offset += len(udata)
                svn_fd.close()
