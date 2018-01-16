'''Test check_for_updates
'''

import os
import tempfile
import unittest
import urllib

import cellprofiler.utilities.check_for_updates as cfu

FAKE_VERSION = 1000
FAKE_RELEASENOTES = """Blahblahblah
More blahblahblah.
More blahblahblah.
"""

class TestCheckForUpdates(unittest.TestCase):
    def setUp(self):
        fd, self.name = tempfile.mkstemp(suffix=".txt", text=False)
        self.f = os.fdopen(fd, "wb")
        self.f.write("%d\n%s" % (FAKE_VERSION, FAKE_RELEASENOTES))
        self.f.flush()

    def tearDown(self):
        self.f.close()
        os.remove(self.name)

    def test_01_01_dont_update(self):
        url = "file:" + urllib.pathname2url(self.name)
        def fn(new_version, info):
            self.fail("We should not update")
        vc = cfu.VersionChecker(url, FAKE_VERSION + 1, fn,
                                'CellProfiler_unit_test')
        vc.start()
        vc.join()

    def test_01_02_update(self):
        url = "file:" + urllib.pathname2url(self.name)
        we_updated = [False]
        def fn(new_version, info, we_updated = we_updated):
            self.assertEqual(new_version, FAKE_VERSION)
            self.assertEqual(info, FAKE_RELEASENOTES)
            we_updated[0] = True
        vc = cfu.VersionChecker(url, FAKE_VERSION - 1, fn,
                                'CellProfiler_unit_test')
        vc.start()
        vc.join()
        self.assertTrue(we_updated[0])

    @unittest.skip(
        'Update infrastructure removed in CellProfiler/CellProfiler#1878'
    )
    def test_01_03_website(self):
        url = 'http://cellprofiler.org/CPupdate.html'
        we_updated = [False]
        def fn(new_version, info, we_updated = we_updated):
            we_updated[0] = True
        vc = cfu.VersionChecker(url, 0, fn,
                                'CellProfiler_unit_test')
        vc.start()
        vc.join()
        self.assertTrue(we_updated[0])
