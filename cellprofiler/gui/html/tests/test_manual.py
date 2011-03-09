'''Tests for cellprofiler.gui.html.manual'''

import os
import re
import unittest
import tempfile
import traceback
import cellprofiler.gui.html.manual as M
from cellprofiler.modules import get_module_names

class TestManual(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        for root, dirnames, filenames in os.walk(self.temp_dir, False):
            for x in filenames:
                os.remove(os.path.join(root, x))
            for x in dirnames:
                os.remove(os.path.join(root, x))
        os.rmdir(self.temp_dir)
    
    @unittest.expectedFailure
    def test_01_01_output_module_html(self):
        M.output_module_html(self.temp_dir)
        for module_name in sorted(get_module_names()):
            fd = None
            try:
                fd = open(os.path.join(self.temp_dir, module_name + ".html"))
            except:
                traceback.print_exc()
                self.assert_("Failed to open %s.html" %module_name)
            data = fd.read()
            fd.close()
            
            #
            # Make sure that some nesting rules are obeyed.
            #
            tags_we_care_about =  ("i","b","ul","li","table","tr","td","td",
                                   "h1","h2","h3","html","head", "body")
            pattern = r"<\s*([a-zA-Z0-9]+).[^>]*>"
            anti_pattern = r"</\s*([a-zA-Z0-9]+)[^>]*>"
            d = {}
            anti_d = {}
            for p, dd in ((pattern, d),
                          (anti_pattern, anti_d)):
                pos = 0
                while(True):
                    m = re.search(p, data[pos:])
                    if m is None:
                        break
                    tag = m.groups()[0]
                    if not dd.has_key(tag):
                        dd[tag] = 0
                    dd[tag] += 1
                    pos = pos + m.start(1)+1
            for tag in tags_we_care_about:
                if d.has_key(tag):
                    self.assertTrue(anti_d.has_key(tag),
                                    "Missing closing </%s> tag in %s" % 
                                    (tag, module_name))
                    self.assertEqual(
                        d[tag], anti_d[tag],
                        "Found %d <%s>, != %d </%s> in %s" %
                        (d[tag], tag, anti_d[tag], tag, module_name))
                else:
                    self.assertFalse(anti_d.has_key(tag),
                                     "Missing opening <%s> tag in %s" %
                                     (tag, module_name))
                