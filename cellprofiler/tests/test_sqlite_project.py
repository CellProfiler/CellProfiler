"""test_sqlite_project.py - the sqlite backend for a project

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
Copyright (c) 2011 Institut Curie
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""


import sqlite3
import unittest

import cellprofiler.project as P
import cellprofiler.sqlite_project as S

T_URL1 = "http://www.cellprofiler.org"
T_URL2 = "http://www.broadinstitute.org"
T_URL3 = "http://www.google.com"

T_UNICODE_URL = u"Not a url \x904"
T_URLSET = "my urlset"
T_ALT_URLSET = "alt urlset"

T_PLATE1 = "P-00001"
T_PLATE2 = "P-00002"

T_IMAGESET_NAME = "imageset"

M_PLATE = "Plate"
M_WELL = "Well"
M_CHANNEL = "Channel"

class TestSQLLiteProject(unittest.TestCase):
    def setUp(self):
        self.project = P.open_project(":memory:", S.SQLiteProject)
        
    def tearDown(self):
        if self.project is not None:
            self.project.close()
        
    def test_01_01_open_new(self):
        pass
    
    def test_01_02_add_url(self):
        self.assertEqual(self.project.add_url(T_URL1), 1)
        
    def test_01_03_get_url_image_id(self):
        image_id = self.project.add_url(T_URL1)
        self.assertEqual(self.project.get_url_image_id(T_URL1), image_id)
        
    def test_01_04_get_url(self):
        for url in (T_URL1, T_UNICODE_URL):
            image_id = self.project.add_url(url)
            self.assertEqual(self.project.get_url(image_id), url)
            
    def test_01_05_remove_url(self):
        image_id = self.project.add_url(T_URL1)
        self.project.remove_url(T_URL1)
        self.assertTrue(self.project.get_url(image_id) is None)
        
    def test_01_06_remove_url_by_id(self):
        image_id = self.project.add_url(T_URL1)
        self.project.remove_url_by_id(image_id)
        self.assertTrue(self.project.get_url(image_id) is None)
        
    def test_02_01_add_image_metadata(self):
        image_id = self.project.add_url(T_URL1)
        self.project.add_image_metadata(["Well", "Channel"],
                                        ["A01", "PI"], image_id)
        
    def test_02_02_get_image_metadata(self):
        image_id = self.project.add_url(T_URL1)
        self.project.add_image_metadata(["Well", "Channel"],
                                        ["A01", "PI"], image_id)
        d = self.project.get_image_metadata(image_id)
        self.assertEqual(len(d), 2)
        self.assertEqual(d["Well"], "A01")
        self.assertEqual(d["Channel"], "PI")
        
    def test_02_03_remove_image_metadata(self):
        image_id = self.project.add_url(T_URL1)
        self.project.add_image_metadata(["Well", "Channel"],
                                        ["A01", "PI"], image_id)
        self.project.remove_image_metadata("Well", image_id)
        d = self.project.get_image_metadata(image_id)
        self.assertEqual(len(d), 1)
        self.assertEqual(d["Channel"], "PI")
        
    def test_02_04_get_images_by_metadata_keys(self):
        image_id1 = self.project.add_url(T_URL1)
        image_id2 = self.project.add_url(T_URL2)
        self.project.add_image_metadata(["Well", "Channel"],
                                        ["A01", "PI"], image_id1)
        self.project.add_image_metadata(["Site", "Channel"],
                                        ["1", "PI"], image_id2)
        result = self.project.get_images_by_metadata(["Well", "Channel"])
        self.assertEqual(len(result),1)
        self.assertEqual(len(result[0]), 3)
        self.assertEqual(result[0][0], "A01")
        self.assertEqual(result[0][1], "PI")
        self.assertEqual(result[0][2], image_id1)
        
    def test_02_05_get_images_by_metadata_keys_and_values(self):
        image_id1 = self.project.add_url(T_URL1)
        image_id2 = self.project.add_url(T_URL2)
        self.project.add_image_metadata(["Well", "Channel"],
                                        ["A01", "PI"], image_id1)
        self.project.add_image_metadata(["Well", "Channel"],
                                        ["A02", "PI"], image_id2)
        result = self.project.get_images_by_metadata(
            ["Well", "Channel"], ["A02", "PI"])
        self.assertEqual(len(result),1)
        self.assertEqual(len(result[0]), 3)
        self.assertEqual(result[0][0], "A02")
        self.assertEqual(result[0][1], "PI")
        self.assertEqual(result[0][2], image_id2)
        
    def test_02_06_get_images_in_urlset(self):
        image_id1 = self.project.add_url(T_URL1)
        image_id2 = self.project.add_url(T_URL2)
        self.project.add_image_metadata(["Well", "Channel"],
                                        ["A01", "PI"], image_id1)
        self.project.add_image_metadata(["Well", "Channel"],
                                        ["A02", "PI"], image_id2)
        self.project.make_urlset(T_URLSET)
        self.project.add_images_to_urlset(T_URLSET, [image_id1])
        result = self.project.get_images_by_metadata(
            ["Well", "Channel"], urlset = T_URLSET)
        self.assertEqual(len(result),1)
        self.assertEqual(len(result[0]), 3)
        self.assertEqual(result[0][0], "A01")
        self.assertEqual(result[0][1], "PI")
        self.assertEqual(result[0][2], image_id1)
        result = self.project.get_images_by_metadata(
            ["Well", "Channel"], 
            ["A01", "PI"], urlset = T_URLSET)
        self.assertEqual(len(result),1)
        self.assertEqual(len(result[0]), 3)
        self.assertEqual(result[0][0], "A01")
        self.assertEqual(result[0][1], "PI")
        self.assertEqual(result[0][2], image_id1)
        result = self.project.get_images_by_metadata(
            ["Well", "Channel"], 
            ["A02", "PI"], urlset = T_URLSET)
        self.assertEqual(len(result), 0)

    def test_02_07_get_metadata_keys(self):
        image_id1 = self.project.add_url(T_URL1)
        image_id2 = self.project.add_url(T_URL2)
        self.project.add_image_metadata(["Well", "Channel"],
                                        ["A01", "PI"], image_id1)
        self.project.add_image_metadata(["Well", "Channel"],
                                        ["A02", "PI"], image_id2)
        result = self.project.get_metadata_keys()
        self.assertEqual(len(result), 2)
        self.assertTrue("Well" in result)
        self.assertTrue("Channel" in result)
        
    def test_02_08_get_metadata_values(self):
        image_id1 = self.project.add_url(T_URL1)
        image_id2 = self.project.add_url(T_URL2)
        self.project.add_image_metadata(["Well", "Channel"],
                                        ["A01", "PI"], image_id1)
        self.project.add_image_metadata(["Well", "Channel"],
                                        ["A02", "PI"], image_id2)
        result = self.project.get_metadata_values("Well")
        self.assertEqual(len(result), 2)
        self.assertTrue(all([x in result for x in ("A01", "A02")]))
        
    def test_03_01_get_urlset_names(self):
        self.project.make_urlset(T_URLSET)
        self.project.make_urlset(T_ALT_URLSET)
        result = self.project.get_urlset_names()
        self.assertEqual(len(result), 2)
        self.assertTrue(all([x in result for x in (T_URLSET, T_ALT_URLSET)]))
        
    def test_03_02_get_urlset_members(self):
        self.project.make_urlset(T_URLSET)
        self.project.make_urlset(T_ALT_URLSET)
        image_id1 = self.project.add_url(T_URL1)
        image_id2 = self.project.add_url(T_URL2)
        self.project.add_images_to_urlset(T_URLSET, [image_id1, image_id2])
        self.project.add_images_to_urlset(T_ALT_URLSET, [image_id2])
        result = self.project.get_urlset_members(T_URLSET)
        self.assertEqual(len(result), 2)
        self.assertTrue(all([x in result for x in [image_id1, image_id2]]))
        result = self.project.get_urlset_members(T_ALT_URLSET)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], image_id2)
        
    def test_03_03_remove_urlset(self):
        self.project.make_urlset(T_URLSET)
        self.project.make_urlset(T_ALT_URLSET)
        self.project.remove_urlset(T_URLSET)
        result = self.project.get_urlset_names()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], T_ALT_URLSET)
        
        self.project.make_urlset(T_URLSET)
        image_id1 = self.project.add_url(T_URL1)
        image_id2 = self.project.add_url(T_URL2)
        self.project.add_images_to_urlset(T_URLSET, [image_id1, image_id2])
        self.project.add_images_to_urlset(T_ALT_URLSET, [image_id2])
        self.project.remove_urlset(T_URLSET)
        result = self.project.get_urlset_names()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], T_ALT_URLSET)
        result = self.project.get_urlset_members(T_ALT_URLSET)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], image_id2)
        
    def test_03_04_remove_images_from_urlset(self):
        self.project.make_urlset(T_URLSET)
        self.project.make_urlset(T_ALT_URLSET)
        image_id1 = self.project.add_url(T_URL1)
        image_id2 = self.project.add_url(T_URL2)
        image_id3 = self.project.add_url(T_URL3)
        self.project.add_images_to_urlset(T_URLSET, [image_id1, image_id2, image_id3])
        self.project.add_images_to_urlset(T_ALT_URLSET, [image_id2])
        self.project.remove_images_from_urlset(T_URLSET, [image_id2, image_id1])
        result = self.project.get_urlset_members(T_URLSET)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], image_id3)
        result = self.project.get_urlset_members(T_ALT_URLSET)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], image_id2)

    def add_plate(self, plate_name, nchannels):
        d = {}
        for channel in range(1, nchannels+1):
            channel_name = str(channel)
            d[channel_name] = {}
            for well in ["%s%02d" % (row, col)
                         for row in "ABCDEFGH" for col in range(1,13)]:
                url = "file://imaging/analysis/%s_%s_w%s" % (
                    plate_name, well, channel_name)
                image_id = self.project.add_url(url)
                d[channel_name][well] = image_id
                self.project.add_image_metadata(
                    (M_PLATE, M_WELL, M_CHANNEL),
                    (plate_name, well, channel_name), image_id)
        return d
        
    def test_04_01_create_imageset(self):
        p1 = self.add_plate(T_PLATE1, 3)
        p2 = self.add_plate(T_PLATE2, 3)
        self.project.create_imageset(T_IMAGESET_NAME,
                                     [M_PLATE, M_WELL], M_CHANNEL)
        self.assertEqual(self.project.get_imageset_row_count(T_IMAGESET_NAME), 192)
        for image_index in range(192):
            image_number = image_index+1
            plate_name = T_PLATE1 if image_number <= 96 else T_PLATE2
            well = "%s%02d" % ("ABCDEFGH"[int(image_index / 12) % 8], 
                                          (image_index % 12) + 1)
            metadata = self.project.get_imageset_row_metadata(
                T_IMAGESET_NAME, image_number)
            self.assertEqual(len(metadata), 2)
            self.assertTrue(all([x in metadata.keys() for x in (M_PLATE, M_WELL)]))
            self.assertEqual(metadata[M_PLATE], plate_name)
            self.assertEqual(metadata[M_WELL], well)
            result = self.project.get_imageset_row_images(T_IMAGESET_NAME,
                                                          image_number)
            self.assertEqual(len(result), 3)
            self.assertTrue(all([str(x) in result.keys() for x in range(1,3)]))
            for channel, image_ids in result.items():
                self.assertEqual(len(image_ids), 1)
                p = p1 if image_number <= 96 else p2
                self.assertEqual(p[channel][well], image_ids[0])
                
    def test_04_02_create_imageset_from_urlset(self):
        p1 = self.add_plate(T_PLATE1, 3)
        p2 = self.add_plate(T_PLATE2, 3)
        self.project.make_urlset(T_PLATE1)
        image_ids = sum([p1[k].values() for k in p1.keys()],[])
        self.project.add_images_to_urlset(T_PLATE1, image_ids)
        self.project.create_imageset(T_IMAGESET_NAME,
                                     [M_PLATE, M_WELL], M_CHANNEL,
                                     urlset = T_PLATE1)
        self.assertEqual(self.project.get_imageset_row_count(T_IMAGESET_NAME), 96)
        for image_index in range(96):
            image_number = image_index+1
            plate_name = T_PLATE1
            well = "%s%02d" % ("ABCDEFGH"[int(image_index / 12) % 8], 
                                          (image_index % 12) + 1)
            metadata = self.project.get_imageset_row_metadata(
                T_IMAGESET_NAME, image_number)
            self.assertEqual(len(metadata), 2)
            self.assertTrue(all([x in metadata.keys() for x in (M_PLATE, M_WELL)]))
            self.assertEqual(metadata[M_PLATE], plate_name)
            self.assertEqual(metadata[M_WELL], well)
            result = self.project.get_imageset_row_images(T_IMAGESET_NAME,
                                                          image_number)
            self.assertEqual(len(result), 3)
            self.assertTrue(all([str(x) in result.keys() for x in range(1,3)]))
            for channel, image_ids in result.items():
                self.assertEqual(len(image_ids), 1)
                p = p1 if image_number <= 96 else p2
                self.assertEqual(p[channel][well], image_ids[0])
    
    def test_04_03_create_imageset_from_some_channels(self):
        p1 = self.add_plate(T_PLATE1, 3)
        p2 = self.add_plate(T_PLATE2, 3)
        self.project.create_imageset(T_IMAGESET_NAME,
                                     [M_PLATE, M_WELL], 
                                     M_CHANNEL,
                                     channel_values = ["1","3"])
        self.assertEqual(self.project.get_imageset_row_count(T_IMAGESET_NAME), 192)
        for image_index in range(192):
            image_number = image_index+1
            plate_name = T_PLATE1 if image_number <= 96 else T_PLATE2
            well = "%s%02d" % ("ABCDEFGH"[int(image_index / 12) % 8], 
                                          (image_index % 12) + 1)
            metadata = self.project.get_imageset_row_metadata(
                T_IMAGESET_NAME, image_number)
            self.assertEqual(len(metadata), 2)
            self.assertTrue(all([x in metadata.keys() for x in (M_PLATE, M_WELL)]))
            self.assertEqual(metadata[M_PLATE], plate_name)
            self.assertEqual(metadata[M_WELL], well)
            result = self.project.get_imageset_row_images(T_IMAGESET_NAME,
                                                          image_number)
            self.assertEqual(len(result), 2)
            self.assertTrue(all([x in result.keys() for x in ("1","3")]))
            for channel, image_ids in result.items():
                self.assertEqual(len(image_ids), 1)
                p = p1 if image_number <= 96 else p2
                self.assertEqual(p[channel][well], image_ids[0])
    
    def test_04_04_create_imageset_from_urlset_and_some_channels(self):
        p1 = self.add_plate(T_PLATE1, 3)
        p2 = self.add_plate(T_PLATE2, 3)
        self.project.make_urlset(T_PLATE1)
        image_ids = sum([p1[k].values() for k in p1.keys()],[])
        self.project.add_images_to_urlset(T_PLATE1, image_ids)
        self.project.create_imageset(T_IMAGESET_NAME,
                                     [M_PLATE, M_WELL], 
                                     M_CHANNEL,
                                     channel_values = ["1","3"],
                                     urlset = T_PLATE1)
        self.assertEqual(self.project.get_imageset_row_count(T_IMAGESET_NAME), 96)
        for image_index in range(96):
            image_number = image_index+1
            plate_name = T_PLATE1
            well = "%s%02d" % ("ABCDEFGH"[int(image_index / 12) % 8], 
                                          (image_index % 12) + 1)
            metadata = self.project.get_imageset_row_metadata(
                T_IMAGESET_NAME, image_number)
            self.assertEqual(len(metadata), 2)
            self.assertTrue(all([x in metadata.keys() for x in (M_PLATE, M_WELL)]))
            self.assertEqual(metadata[M_PLATE], plate_name)
            self.assertEqual(metadata[M_WELL], well)
            result = self.project.get_imageset_row_images(T_IMAGESET_NAME,
                                                          image_number)
            self.assertEqual(len(result), 2)
            self.assertTrue(all([x in result.keys() for x in ("1","3")]))
            for channel, image_ids in result.items():
                self.assertEqual(len(image_ids), 1)
                p = p1 if image_number <= 96 else p2
                self.assertEqual(p[channel][well], image_ids[0])
                
    def test_04_05_add_image_to_imageset(self):
        p1 = self.add_plate(T_PLATE1, 3)
        p2 = self.add_plate(T_PLATE2, 3)
        self.project.create_imageset(T_IMAGESET_NAME,
                                     [M_PLATE, M_WELL], M_CHANNEL)
        bogus_id = self.project.add_url(T_URL1)
        self.project.add_image_to_imageset(bogus_id, T_IMAGESET_NAME, 5, "2")
        result = self.project.get_imageset_row_images(T_IMAGESET_NAME, 5)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result["1"]), 1)
        self.assertEqual(len(result["3"]), 1)
        self.assertEqual(len(result["2"]), 2)
        self.assertTrue(bogus_id in result["2"])
        
    def test_04_06_remove_image_from_imageset(self):
        p1 = self.add_plate(T_PLATE1, 3)
        p2 = self.add_plate(T_PLATE2, 3)
        self.project.create_imageset(T_IMAGESET_NAME,
                                     [M_PLATE, M_WELL], M_CHANNEL)
        self.project.remove_image_from_imageset(T_IMAGESET_NAME,
                                                p1["2"]["A05"])
        result = self.project.get_imageset_row_images(T_IMAGESET_NAME, 5)
        self.assertEqual(len(result), 3)
        self.assertEqual(len(result["1"]), 1)
        self.assertEqual(len(result["3"]), 1)
        self.assertEqual(len(result["2"]), 0)
        
    def test_04_07_get_problem_imagesets_missing(self):
        p1 = self.add_plate(T_PLATE1, 3)
        p2 = self.add_plate(T_PLATE2, 3)
        self.project.create_imageset(T_IMAGESET_NAME,
                                     [M_PLATE, M_WELL], M_CHANNEL)
        self.project.remove_image_from_imageset(T_IMAGESET_NAME,
                                                p1["2"]["A05"])
        result = self.project.get_problem_imagesets(T_IMAGESET_NAME)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 5)
        self.assertEqual(result[0][1], "2")
        self.assertEqual(result[0][2], 0)
        
    def test_04_08_get_problem_imagesets_duplicate(self):
        p1 = self.add_plate(T_PLATE1, 3)
        p2 = self.add_plate(T_PLATE2, 3)
        self.project.create_imageset(T_IMAGESET_NAME,
                                     [M_PLATE, M_WELL], M_CHANNEL)
        bogus_id = self.project.add_url(T_URL1)
        self.project.add_image_to_imageset(bogus_id, T_IMAGESET_NAME, 5, "2")
        result = self.project.get_problem_imagesets(T_IMAGESET_NAME)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 5)
        self.assertEqual(result[0][1], "2")
        self.assertEqual(result[0][2], 2)

    def test_04_09_add_urlset_to_imageset(self):
        p1 = self.add_plate(T_PLATE1, 3)
        p2 = self.add_plate(T_PLATE2, 3)
        self.project.create_imageset(T_IMAGESET_NAME,
                                     [M_PLATE, M_WELL], 
                                     M_CHANNEL,
                                     channel_values = ["1","3"])
        self.project.make_urlset("Channel2")
        image_ids = sum([p["2"].values() for p in (p1,p2)],[])
        self.project.add_images_to_urlset("Channel2", image_ids)
        self.project.add_channel_to_imageset(T_IMAGESET_NAME,
                                             [M_PLATE, M_WELL],
                                             "Channel2",
                                             "two")
        for image_index in range(192):
            image_number = image_index+1
            plate_name = T_PLATE1 if image_number <= 96 else T_PLATE2
            well = "%s%02d" % ("ABCDEFGH"[int(image_index / 12) % 8], 
                                          (image_index % 12) + 1)
            result = self.project.get_imageset_row_images(T_IMAGESET_NAME,
                                                          image_number)
            self.assertEqual(len(result), 3)
            self.assertTrue(all([x in result.keys() for x in ("1", "two", "3")]))
            for channel, image_ids in result.items():
                if channel == "two":
                    channel = "2"
                self.assertEqual(len(image_ids), 1)
                p = p1 if image_number <= 96 else p2
                self.assertEqual(p[channel][well], image_ids[0])
                
    def test_04_10_add_channel_to_imageset(self):
        p1 = self.add_plate(T_PLATE1, 3)
        p2 = self.add_plate(T_PLATE2, 3)
        self.project.create_imageset(T_IMAGESET_NAME,
                                     [M_PLATE, M_WELL], 
                                     M_CHANNEL,
                                     channel_values = ["1","3"])
        self.project.add_channel_to_imageset(T_IMAGESET_NAME,
                                             [M_PLATE, M_WELL],
                                             channel_name = "two",
                                             channel_key = M_CHANNEL,
                                             channel_value = "2")
        for image_index in range(192):
            image_number = image_index+1
            plate_name = T_PLATE1 if image_number <= 96 else T_PLATE2
            well = "%s%02d" % ("ABCDEFGH"[int(image_index / 12) % 8], 
                                          (image_index % 12) + 1)
            result = self.project.get_imageset_row_images(T_IMAGESET_NAME,
                                                          image_number)
            self.assertEqual(len(result), 3)
            self.assertTrue(all([x in result.keys() for x in ("1", "two", "3")]))
            for channel, image_ids in result.items():
                if channel == "two":
                    channel = "2"
                self.assertEqual(len(image_ids), 1)
                p = p1 if image_number <= 96 else p2
                self.assertEqual(p[channel][well], image_ids[0])
        
    def test_05_01_add_directory(self):
        self.project.add_directory("http://www.cellprofiler.org")
        result = self.project.get_directories()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "http://www.cellprofiler.org")
        
    def test_05_02_add_subdirectory(self):
        self.project.add_directory("http://www.cellprofiler.org")
        self.project.add_directory("http://www.cellprofiler.org/wiki",
                                   "http://www.cellprofiler.org")
        result = self.project.get_directories()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "http://www.cellprofiler.org")
        self.assertEqual(result[1], "http://www.cellprofiler.org/wiki")

    def test_05_03_add_duplicate_directory(self):
        self.project.add_directory("http://www.cellprofiler.org")
        result = self.project.get_directories()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "http://www.cellprofiler.org")

    def test_05_04_get_root_directories(self):
        self.project.add_directory("http://www.cellprofiler.org")
        self.project.add_directory("http://www.cellprofiler.org/wiki",
                                   "http://www.cellprofiler.org")
        self.project.add_directory("http://svn.broadinstitute.org")
        result = self.project.get_root_directories()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "http://svn.broadinstitute.org")
        self.assertEqual(result[1], "http://www.cellprofiler.org")
        
    def test_05_05_get_subdirectories(self):
        self.project.add_directory("http://www.cellprofiler.org")
        self.project.add_directory("http://www.cellprofiler.org/wiki",
                                   "http://www.cellprofiler.org")
        self.project.add_directory("http://svn.broadinstitute.org")
        result = self.project.get_subdirectories("http://www.cellprofiler.org")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "http://www.cellprofiler.org/wiki")
        
        result = self.project.get_subdirectories("http://svn.broadinstitute.org")
        self.assertEqual(len(result), 0)
        
    def test_05_06_remove_one_directory(self):
        self.project.add_directory("http://www.cellprofiler.org")
        self.project.add_directory("http://www.cellprofiler.org/wiki",
                                   "http://www.cellprofiler.org")
        self.project.add_directory("http://svn.broadinstitute.org")
        self.project.remove_directory("http://svn.broadinstitute.org")
        result = self.project.get_directories()
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "http://www.cellprofiler.org")
        self.assertEqual(result[1], "http://www.cellprofiler.org/wiki")

    def test_05_07_remove_directory_tree(self):
        self.project.add_directory("http://www.cellprofiler.org")
        self.project.add_directory("http://www.cellprofiler.org/wiki",
                                   "http://www.cellprofiler.org")
        self.project.add_directory("http://svn.broadinstitute.org")
        self.project.remove_directory("http://www.cellprofiler.org")
        result = self.project.get_directories()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "http://svn.broadinstitute.org")
