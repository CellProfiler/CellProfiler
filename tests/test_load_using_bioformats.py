# Python-bioformats is distributed under the GNU General Public
# License, but this file is licensed under the more permissive BSD
# license.  See the accompanying file LICENSE for details.
#
# Copyright (c) 2009-2014 Broad Institute
# All rights reserved.

from __future__ import absolute_import, print_function, unicode_literals

import os
import unittest

import javabridge
import bioformats

import bioformats.formatreader as formatreader
import bioformats.metadatatools as metadatatools
from bioformats import load_image, load_image_url

import sys
if sys.version_info.major == 3:
    from urllib.request import pathname2url
else:
    from urllib import pathname2url

class TestLoadUsingBioformats(unittest.TestCase):

    def setUp(self):
        javabridge.attach()
        bioformats.init_logger()

    def tearDown(self):
        javabridge.detach()

    def test_load_using_bioformats(self):
        path = os.path.join(os.path.dirname(__file__), 'Channel1-01-A-01.tif')
        image, scale = load_image(path, rescale=False,
                                  wants_max_intensity=True)
        print(image.shape)

    def test_file_not_found(self):
        # Regression test of issue #6
        path = os.path.join(os.path.dirname(__file__), 'Channel5-01-A-01.tif')
        self.assertRaises(IOError, lambda :load_image(path))

class TestLoadUsingBioformatsURL(unittest.TestCase):
    def setUp(self):
        javabridge.attach()
        bioformats.init_logger()

    def tearDown(self):
        javabridge.detach()

    def test_01_01_open_file(self):
        path = os.path.join(os.path.dirname(__file__), 'Channel1-01-A-01.tif')
        url = "file:" + pathname2url(path)
        image, scale = load_image_url(
            url, rescale=False, wants_max_intensity=True)
        self.assertEqual(image.shape[0], 640)

    def test_01_02_open_http(self):
        url = "https://github.com/CellProfiler/python-bioformats"+\
            "/raw/master/bioformats/tests/Channel1-01-A-01.tif"
        image, scale = load_image_url(
            url, rescale=False, wants_max_intensity=True)
        self.assertEqual(image.shape[0], 640)

    def test_01_03_unicode_url(self):
        #
        # Regression test of issue #17: ensure that this does not
        # raise an exception when converting URL to string
        #
        path = os.path.join(os.path.dirname(__file__), 'Channel1-01-A-01.tif')
        url = "file:" + pathname2url(path)
        image, scale = load_image_url(
            url, rescale=False, wants_max_intensity=True)
        self.assertEqual(image.shape[0], 640)


