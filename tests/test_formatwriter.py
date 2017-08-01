# Python-bioformats is distributed under the GNU General Public
# License, but this file is licensed under the more permissive BSD
# license.  See the accompanying file LICENSE for details.
#
# Copyright (c) 2009-2014 Broad Institute
# All rights reserved.

'''test_formatwriter.py - test the Bioformats format reader wrapper

'''

from __future__ import absolute_import, unicode_literals

import numpy as np
import os
import tempfile
import unittest

import bioformats.formatwriter as W
from bioformats.formatreader import load_using_bioformats, get_omexml_metadata
import bioformats.omexml as OME

class TestFormatWriter(unittest.TestCase):
    def setUp(self):
        self.path = tempfile.mkdtemp()
        self.files = []

    def get_tempfilename(self, suffix):
        fd, name = tempfile.mkstemp(suffix, self.path)
        self.files.append(name)
        os.close(fd)
        return name

    def tearDown(self):
        for filename in self.files:
            os.remove(filename)
        os.rmdir(self.path)

    def test_01_01_write_monochrome_8_bit_tif(self):
        r = np.random.RandomState()
        r.seed(101)
        img = r.randint(0, 256, (11, 33)).astype(np.uint8)
        path = self.get_tempfilename(".tif")
        W.write_image(path, img, OME.PT_UINT8)
        result = load_using_bioformats(path, rescale=False)
        np.testing.assert_array_equal(img, result)

    def test_01_02_write_monchrome_16_bit__tiff(self):
        r = np.random.RandomState()
        r.seed(102)
        img = r.randint(0, 4096, size=(21, 24))
        path = self.get_tempfilename(".tif")
        W.write_image(path, img, OME.PT_UINT16)
        result = load_using_bioformats(path, rescale=False)
        np.testing.assert_array_equal(img, result)

    def test_01_03_write_color_tiff(self):
        r = np.random.RandomState()
        r.seed(103)
        img = r.randint(0, 256, (9, 11, 3))
        path = self.get_tempfilename(".tif")
        W.write_image(path, img, OME.PT_UINT8)
        result = load_using_bioformats(path, rescale = False)
        np.testing.assert_array_equal(img, result)

    def test_02_01_write_movie(self):
        r = np.random.RandomState()
        r.seed(103)
        img = r.randint(0, 256, (7, 23, 11))
        path = self.get_tempfilename(".tif")
        for i in range(img.shape[0]):
            W.write_image(path, img[i], OME.PT_UINT8, t=i, size_t = img.shape[0])
        for i in range(img.shape[0]):
            result = load_using_bioformats(path, t=i, rescale = False)
            np.testing.assert_array_equal(img[i], result)

