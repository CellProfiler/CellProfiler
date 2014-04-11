'''test_metadata.py - test the Metadata module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

import numpy as np
import os
from cStringIO import StringIO
import tempfile
import unittest
import urllib

import cellprofiler.pipeline as cpp
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas
import cellprofiler.modules.images as I
import cellprofiler.modules.metadata as M
import cellprofiler.workspace as cpw

OME_XML = open(os.path.join(os.path.split(__file__)[0], "omexml.xml"), "r").read()

class TestMetadata(unittest.TestCase):
    def test_01_01_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120112154631
ModuleCount:1
HasImagePlaneDetails:False

Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Extract metadata?:Yes
    Extraction method count:2
    Extraction method:Manual
    Source:From file name
    Regular expression:^Channel(?P<ChannelNumber>\x5B12\x5D)-(?P<Index>\x5B0-9\x5D+)-(?P<WellRow>\x5BA-H\x5D)-(?P<WellColumn>\x5B0-9\x5D{2}).tif$
    Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
    Filter images:All images
    :or (file does contain "Channel2")
    Metadata file location\x3A:
    Match file and image metadata:\x5B\x5D
    Extraction method:Import metadata
    Source:From folder name
    Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)_w(?P<ChannelNumber>\x5B0-9\x5D)
    Regular expression:Example(?P<Project>\x5B^\\\\\\\\\x5D+)Images
    Filter images:Images selected using a filter
    :or (file does contain "")
    Metadata file location\x3A:/imaging/analysis/metadata.csv
    Match file and image metadata:\x5B{\'Image Metadata\'\x3A u\'ChannelNumber\', \'CSV Metadata\'\x3A u\'Wavelength\'}\x5D
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, M.Metadata))
        self.assertTrue(module.wants_metadata)
        self.assertEqual(len(module.extraction_methods), 2)
        em0, em1 = module.extraction_methods
        self.assertEqual(em0.extraction_method, M.X_MANUAL_EXTRACTION)
        self.assertEqual(em0.source, M.XM_FILE_NAME)
        self.assertEqual(em0.file_regexp.value,
                         r"^Channel(?P<ChannelNumber>[12])-(?P<Index>[0-9]+)-(?P<WellRow>[A-H])-(?P<WellColumn>[0-9]{2}).tif$")
        self.assertEqual(em0.folder_regexp.value,
                         r"(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$")
        self.assertEqual(em0.filter_choice, M.F_ALL_IMAGES)
        self.assertEqual(em0.filter, 'or (file does contain "Channel2")')
        self.assertFalse(em0.wants_case_insensitive)
        
        self.assertEqual(em1.extraction_method, M.X_IMPORTED_EXTRACTION)
        self.assertEqual(em1.source, M.XM_FOLDER_NAME)
        self.assertEqual(em1.filter_choice, M.F_FILTERED_IMAGES)
        self.assertEqual(em1.csv_location, "/imaging/analysis/metadata.csv")
        self.assertEqual(em1.csv_joiner.value, "[{'Image Metadata': u'ChannelNumber', 'CSV Metadata': u'Wavelength'}]")
        self.assertFalse(em1.wants_case_insensitive)
        
    def test_01_02_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120112154631
ModuleCount:1
HasImagePlaneDetails:False

Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Extract metadata?:Yes
    Extraction method count:2
    Extraction method:Manual
    Source:From file name
    Regular expression:^Channel(?P<ChannelNumber>\x5B12\x5D)-(?P<Index>\x5B0-9\x5D+)-(?P<WellRow>\x5BA-H\x5D)-(?P<WellColumn>\x5B0-9\x5D{2}).tif$
    Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
    Filter images:All images
    :or (file does contain "Channel2")
    Metadata file location\x3A:
    Match file and image metadata:\x5B\x5D
    Case insensitive matching:No
    Extraction method:Import metadata
    Source:From folder name
    Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)_w(?P<ChannelNumber>\x5B0-9\x5D)
    Regular expression:Example(?P<Project>\x5B^\\\\\\\\\x5D+)Images
    Filter images:Images selected using a filter
    :or (file does contain "")
    Metadata file location\x3A:/imaging/analysis/metadata.csv
    Match file and image metadata:\x5B{\'Image Metadata\'\x3A u\'ChannelNumber\', \'CSV Metadata\'\x3A u\'Wavelength\'}\x5D
    Case insensitive matching:Yes
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, M.Metadata))
        self.assertTrue(module.wants_metadata)
        self.assertEqual(len(module.extraction_methods), 2)
        em0, em1 = module.extraction_methods
        self.assertEqual(em0.extraction_method, M.X_MANUAL_EXTRACTION)
        self.assertEqual(em0.source, M.XM_FILE_NAME)
        self.assertEqual(em0.file_regexp.value,
                         r"^Channel(?P<ChannelNumber>[12])-(?P<Index>[0-9]+)-(?P<WellRow>[A-H])-(?P<WellColumn>[0-9]{2}).tif$")
        self.assertEqual(em0.folder_regexp.value,
                         r"(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$")
        self.assertEqual(em0.filter_choice, M.F_ALL_IMAGES)
        self.assertEqual(em0.filter, 'or (file does contain "Channel2")')
        self.assertFalse(em0.wants_case_insensitive)
        
        self.assertEqual(em1.extraction_method, M.X_IMPORTED_EXTRACTION)
        self.assertEqual(em1.source, M.XM_FOLDER_NAME)
        self.assertEqual(em1.filter_choice, M.F_FILTERED_IMAGES)
        self.assertEqual(em1.csv_location, "/imaging/analysis/metadata.csv")
        self.assertEqual(em1.csv_joiner.value, "[{'Image Metadata': u'ChannelNumber', 'CSV Metadata': u'Wavelength'}]")
        self.assertTrue(em1.wants_case_insensitive)

    def test_01_03_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120112154631
ModuleCount:1
HasImagePlaneDetails:False

Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Extract metadata?:Yes
    Extraction method count:2
    Metadata extraction method:Extract from file/folder names
    Metadata source:File name
    Regular expression:^Channel(?P<ChannelNumber>\x5B12\x5D)-(?P<Index>\x5B0-9\x5D+)-(?P<WellRow>\x5BA-H\x5D)-(?P<WellColumn>\x5B0-9\x5D{2}).tif$
    Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
    Extract metadata from:All images
    Select the filtering criteria:or (file does contain "Channel2")
    Metadata file location:
    Match file and image metadata:\x5B\x5D
    Case insensitive matching:No
    Metadata extraction method:Import from file
    Metadata source:Folder name
    Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)_w(?P<ChannelNumber>\x5B0-9\x5D)
    Regular expression:Example(?P<Project>\x5B^\\\\\\\\\x5D+)Images
    Extract metadata from:Images matching a rule
    Select the filtering criteria:or (file does contain "")
    Metadata file location\x3A:/imaging/analysis/metadata.csv
    Match file and image metadata:\x5B{\'Image Metadata\'\x3A u\'ChannelNumber\', \'CSV Metadata\'\x3A u\'Wavelength\'}\x5D
    Case insensitive matching:Yes
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, M.Metadata))
        self.assertTrue(module.wants_metadata)
        self.assertEqual(module.data_type_choice, M.DTC_TEXT)
        self.assertEqual(len(module.extraction_methods), 2)
        em0, em1 = module.extraction_methods
        self.assertEqual(em0.extraction_method, M.X_MANUAL_EXTRACTION)
        self.assertEqual(em0.source, M.XM_FILE_NAME)
        self.assertEqual(em0.file_regexp.value,
                         r"^Channel(?P<ChannelNumber>[12])-(?P<Index>[0-9]+)-(?P<WellRow>[A-H])-(?P<WellColumn>[0-9]{2}).tif$")
        self.assertEqual(em0.folder_regexp.value,
                         r"(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$")
        self.assertEqual(em0.filter_choice, M.F_ALL_IMAGES)
        self.assertEqual(em0.filter, 'or (file does contain "Channel2")')
        self.assertFalse(em0.wants_case_insensitive)
        
        self.assertEqual(em1.extraction_method, M.X_IMPORTED_EXTRACTION)
        self.assertEqual(em1.source, M.XM_FOLDER_NAME)
        self.assertEqual(em1.filter_choice, M.F_FILTERED_IMAGES)
        self.assertEqual(em1.csv_location, "/imaging/analysis/metadata.csv")
        self.assertEqual(em1.csv_joiner.value, "[{'Image Metadata': u'ChannelNumber', 'CSV Metadata': u'Wavelength'}]")
        self.assertTrue(em1.wants_case_insensitive)
        
    def test_01_04_load_v4(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120112154631
ModuleCount:1
HasImagePlaneDetails:False

Metadata:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Extract metadata?:Yes
    Metadata data type:Choose for each
    Metadata types:{"Index"\x3A "none", "WellRow"\x3A "text", "WellColumn"\x3A "float", "ChannelNumber"\x3A "integer"}
    Extraction method count:2
    Metadata extraction method:Extract from file/folder names
    Metadata source:File name
    Regular expression:^Channel(?P<ChannelNumber>\x5B12\x5D)-(?P<Index>\x5B0-9\x5D+)-(?P<WellRow>\x5BA-H\x5D)-(?P<WellColumn>\x5B0-9\x5D{2}).tif$
    Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
    Extract metadata from:All images
    Select the filtering criteria:or (file does contain "Channel2")
    Metadata file location:
    Match file and image metadata:\x5B\x5D
    Case insensitive matching:No
    Metadata extraction method:Import from file
    Metadata source:Folder name
    Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})_s(?P<Site>\x5B0-9\x5D)_w(?P<ChannelNumber>\x5B0-9\x5D)
    Regular expression:Example(?P<Project>\x5B^\\\\\\\\\x5D+)Images
    Extract metadata from:Images matching a rule
    Select the filtering criteria:or (file does contain "")
    Metadata file location\x3A:/imaging/analysis/metadata.csv
    Match file and image metadata:\x5B{\'Image Metadata\'\x3A u\'ChannelNumber\', \'CSV Metadata\'\x3A u\'Wavelength\'}\x5D
    Case insensitive matching:Yes
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, M.Metadata))
        self.assertTrue(module.wants_metadata)
        self.assertEqual(module.data_type_choice, M.DTC_CHOOSE)
        d = cps.DataTypes.decode_data_types(module.data_types.value_text)
        for k, v in (("Index", cps.DataTypes.DT_NONE),
                     ("WellRow", cps.DataTypes.DT_TEXT),
                     ("WellColumn", cps.DataTypes.DT_FLOAT),
                     ("ChannelNumber", cps.DataTypes.DT_INTEGER)):
            self.assertTrue(k in d)
            self.assertEqual(d[k], v)
        self.assertEqual(len(module.extraction_methods), 2)
        em0, em1 = module.extraction_methods
        self.assertEqual(em0.extraction_method, M.X_MANUAL_EXTRACTION)
        self.assertEqual(em0.source, M.XM_FILE_NAME)
        self.assertEqual(em0.file_regexp.value,
                         r"^Channel(?P<ChannelNumber>[12])-(?P<Index>[0-9]+)-(?P<WellRow>[A-H])-(?P<WellColumn>[0-9]{2}).tif$")
        self.assertEqual(em0.folder_regexp.value,
                         r"(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$")
        self.assertEqual(em0.filter_choice, M.F_ALL_IMAGES)
        self.assertEqual(em0.filter, 'or (file does contain "Channel2")')
        self.assertFalse(em0.wants_case_insensitive)
        
        self.assertEqual(em1.extraction_method, M.X_IMPORTED_EXTRACTION)
        self.assertEqual(em1.source, M.XM_FOLDER_NAME)
        self.assertEqual(em1.filter_choice, M.F_FILTERED_IMAGES)
        self.assertEqual(em1.csv_location, "/imaging/analysis/metadata.csv")
        self.assertEqual(em1.csv_joiner.value, "[{'Image Metadata': u'ChannelNumber', 'CSV Metadata': u'Wavelength'}]")
        self.assertTrue(em1.wants_case_insensitive)
        
    def check(self, module, url, dd, keys=None, xml=None):
        '''Check that running the metadata module on a url generates the expected dictionary'''
        pipeline = cpp.Pipeline()
        imgs = I.Images()
        imgs.filter_choice.value = I.FILTER_CHOICE_NONE
        imgs.module_num = 1
        pipeline.add_module(imgs)
        module.module_num = 2
        pipeline.add_module(module)
        pipeline.add_urls([url])
        m = cpmeas.Measurements()
        workspace = cpw.Workspace(pipeline, module, None, None, m, None)
        file_list = workspace.file_list
        file_list.add_files_to_filelist([url])
        if xml is not None:
            file_list.add_metadata(url, xml)
        ipds = pipeline.get_image_plane_details(workspace)
        self.assertEqual(len(ipds), len(dd))
        for d, ipd in zip(dd, ipds):
            self.assertDictContainsSubset(d, ipd.metadata)
        all_keys = pipeline.get_available_metadata_keys().keys()
        if keys is not None:
            for key in keys:
                self.assertIn(key, all_keys)
        
    def test_02_01_get_metadata_from_filename(self):
        module = M.Metadata()
        module.wants_metadata.value=True
        em = module.extraction_methods[0]
        em.filter_choice.value = M.F_ALL_IMAGES
        em.extraction_method.value = M.X_MANUAL_EXTRACTION
        em.source.value = M.XM_FILE_NAME
        em.file_regexp.value = "^(?P<Plate>[^_]+)_(?P<Well>[A-H][0-9]{2})_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])"
        em.filter_choice.value = M.F_ALL_IMAGES
        url = "file:/imaging/analysis/P-12345_B08_s5_w2.tif"
        self.check(module, url, 
                   [{ "Plate":"P-12345",
                      "Well":"B08",
                      "Site":"5",
                      "Wavelength":"2"}],
                   ("Plate", "Well", "Site", "Wavelength"))
        
    def test_02_02_get_metadata_from_path(self):
        module = M.Metadata()
        module.wants_metadata.value = True
        em = module.extraction_methods[0]
        em.filter_choice.value = M.F_ALL_IMAGES
        em.extraction_method.value = M.X_MANUAL_EXTRACTION
        em.source.value = M.XM_FOLDER_NAME
        em.folder_regexp.value = r".*[/\\](?P<Plate>.+)$"
        em.filter_choice.value = M.F_ALL_IMAGES
        url = "file:/imaging/analysis/P-12345/_B08_s5_w2.tif"
        self.check(module, url, [{ "Plate":"P-12345" }], ("Plate",))
        
    def test_02_03_filter_positive(self):
        module = M.Metadata()
        module.wants_metadata.value=True
        em = module.extraction_methods[0]
        em.filter_choice.value = M.F_FILTERED_IMAGES
        em.filter.value = 'or (file does contain "B08")'
        em.extraction_method.value = M.X_MANUAL_EXTRACTION
        em.source.value = M.XM_FILE_NAME
        em.file_regexp.value = "^(?P<Plate>[^_]+)_(?P<Well>[A-H][0-9]{2})_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])"
        em.filter_choice.value = M.F_ALL_IMAGES
        url = "file:/imaging/analysis/P-12345_B08_s5_w2.tif"
        self.check(module, url, 
                   [{ "Plate":"P-12345",
                     "Well":"B08",
                     "Site":"5",
                     "Wavelength":"2"}])

    def test_02_04_filter_negative(self):
        module = M.Metadata()
        module.wants_metadata.value=True
        em = module.extraction_methods[0]
        em.filter_choice.value = M.F_FILTERED_IMAGES
        em.filter.value = 'or (file doesnot contain "B08")'
        em.extraction_method.value = M.X_MANUAL_EXTRACTION
        em.source.value = M.XM_FILE_NAME
        em.file_regexp.value = "^(?P<Plate>[^_]+)_(?P<Well>[A-H][0-9]{2})_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])"
        em.filter_choice.value = M.F_ALL_IMAGES
        url = "file:/imaging/analysis/P-12345_B08_s5_w2.tif"
        self.check(module, url, 
                   [{ "Plate":"P-12345",
                      "Well":"B08",
                      "Site":"5",
                      "Wavelength":"2"}])
        
    def test_02_05_imported_extraction(self):
        metadata_csv = """WellName,Treatment,Dose,Counter
B08,DMSO,0,1
C10,BRD041618,1.5,2
"""
        filenum, path = tempfile.mkstemp(suffix = ".csv")
        fd = os.fdopen(filenum, "w")
        fd.write(metadata_csv)
        fd.close()
        try:
            module = M.Metadata()
            module.wants_metadata.value = True
            module.data_type_choice.value = M.DTC_CHOOSE
            module.data_types.value=cps.json.dumps(dict(
                Plate=cps.DataTypes.DT_TEXT,
                Well=cps.DataTypes.DT_TEXT,
                WellName=cps.DataTypes.DT_NONE,
                Treatment=cps.DataTypes.DT_TEXT,
                Dose=cps.DataTypes.DT_FLOAT,
                Counter=cps.DataTypes.DT_NONE))
            module.add_extraction_method()
            em = module.extraction_methods[0]
            em.filter_choice.value = M.F_ALL_IMAGES
            em.extraction_method.value = M.X_MANUAL_EXTRACTION
            em.source.value = M.XM_FILE_NAME
            em.file_regexp.value = "^(?P<Plate>[^_]+)_(?P<Well>[A-Ha-h][0-9]{2})_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])"
            em.filter_choice.value = M.F_ALL_IMAGES
            
            em = module.extraction_methods[1]
            em.filter_choice.value = M.F_ALL_IMAGES
            em.extraction_method.value = M.X_IMPORTED_EXTRACTION
            em.csv_location.value = path
            em.csv_joiner.value = '[{"%s":"WellName","%s":"Well"}]' % (
                module.CSV_JOIN_NAME, module.IPD_JOIN_NAME)
            url = "file:/imaging/analysis/P-12345_B08_s5_w2.tif"
            self.check(module, url, 
                       [{ "Plate":"P-12345",
                          "Well":"B08",
                          "Site":"5",
                          "Wavelength":"2",
                          "Treatment":"DMSO",
                          "Dose":"0",
                          "Counter":"1"}])
            url = "file:/imaging/analysis/P-12345_C10_s2_w3.tif"
            self.check(module, url,
                       [{ "Plate":"P-12345",
                          "Well":"C10",
                          "Site":"2",
                          "Wavelength":"3",
                          "Treatment":"BRD041618",
                          "Dose":"1.5",
                          "Counter":"2"}])
            url = "file:/imaging/analysis/P-12345_A01_s2_w3.tif"
            self.check(module, url,
                       [{ "Plate":"P-12345",
                          "Well":"A01",
                          "Site":"2",
                          "Wavelength":"3"}])
            pipeline = cpp.Pipeline()
            imgs = I.Images()
            imgs.filter_choice.value = I.FILTER_CHOICE_NONE
            imgs.module_num = 1
            pipeline.add_module(imgs)
            module.module_num = 2
            pipeline.add_module(module)
            columns = module.get_measurement_columns(pipeline)
            self.assertFalse(any([c[1] == "Counter" for c in columns]))
            for feature_name, data_type in (
                ("Metadata_Treatment", cpmeas.COLTYPE_VARCHAR_FILE_NAME),
                ("Metadata_Dose", cpmeas.COLTYPE_FLOAT)):
                self.assertTrue(any([c[0] == cpmeas.IMAGE and
                                     c[1] == feature_name and
                                     c[2] == data_type for c in columns]))
        finally:
            try:
                os.unlink(path)
            except:
                pass
            
    def test_02_06_imported_extraction_case_insensitive(self):
        metadata_csv = """WellName,Treatment
b08,DMSO
C10,BRD041618
"""
        filenum, path = tempfile.mkstemp(suffix = ".csv")
        fd = os.fdopen(filenum, "w")
        fd.write(metadata_csv)
        fd.close()
        try:
            module = M.Metadata()
            module.wants_metadata.value=True
            module.add_extraction_method()
            em = module.extraction_methods[0]
            em.filter_choice.value = M.F_ALL_IMAGES
            em.extraction_method.value = M.X_MANUAL_EXTRACTION
            em.source.value = M.XM_FILE_NAME
            em.file_regexp.value = "^(?P<Plate>[^_]+)_(?P<Well>[A-Ha-h][0-9]{2})_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])"
            em.filter_choice.value = M.F_ALL_IMAGES
            
            em = module.extraction_methods[1]
            em.filter_choice.value = M.F_ALL_IMAGES
            em.extraction_method.value = M.X_IMPORTED_EXTRACTION
            em.csv_location.value = path
            em.csv_joiner.value = '[{"%s":"WellName","%s":"Well"}]' % (
                module.CSV_JOIN_NAME, module.IPD_JOIN_NAME)
            em.wants_case_insensitive.value = True
            url = "file:/imaging/analysis/P-12345_B08_s5_w2.tif"
            self.check(module, url,
                       [{ "Plate":"P-12345",
                          "Well":"B08",
                          "Site":"5",
                          "Wavelength":"2",
                          "Treatment":"DMSO"}])
            url = "file:/imaging/analysis/P-12345_c10_s2_w3.tif"
            self.check(module, url, 
                       [{ "Plate":"P-12345",
                          "Well":"c10",
                          "Site":"2",
                          "Wavelength":"3",
                          "Treatment":"BRD041618"}])
            url = "file:/imaging/analysis/P-12345_A01_s2_w3.tif"
            self.check(module, url, 
                       [{ "Plate":"P-12345",
                          "Well":"A01",
                          "Site":"2",
                          "Wavelength":"3"}])
        finally:
            try:
                os.unlink(path)
            except:
                pass

    def test_02_07_imported_extraction_case_sensitive(self):
        metadata_csv = """WellName,Treatment
b08,DMSO
C10,BRD041618
"""
        filenum, path = tempfile.mkstemp(suffix = ".csv")
        fd = os.fdopen(filenum, "w")
        fd.write(metadata_csv)
        fd.close()
        try:
            module = M.Metadata()
            module.wants_metadata.value=True
            module.add_extraction_method()
            em = module.extraction_methods[0]
            em.filter_choice.value = M.F_ALL_IMAGES
            em.extraction_method.value = M.X_MANUAL_EXTRACTION
            em.source.value = M.XM_FILE_NAME
            em.file_regexp.value = "^(?P<Plate>[^_]+)_(?P<Well>[A-H][0-9]{2})_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])"
            em.filter_choice.value = M.F_ALL_IMAGES
            
            em = module.extraction_methods[1]
            em.filter_choice.value = M.F_ALL_IMAGES
            em.extraction_method.value = M.X_IMPORTED_EXTRACTION
            em.csv_location.value = path
            em.csv_joiner.value = '[{"%s":"WellName","%s":"Well"}]' % (
                module.CSV_JOIN_NAME, module.IPD_JOIN_NAME)
            em.wants_case_insensitive.value = False
            url = "file:/imaging/analysis/P-12345_B08_s5_w2.tif"
            self.check(module, url,
                       [{ "Plate":"P-12345",
                          "Well":"B08",
                          "Site":"5",
                          "Wavelength":"2"}])
            url = "file:/imaging/analysis/P-12345_C10_s2_w3.tif"
            self.check(module, url,
                       [{ "Plate":"P-12345",
                          "Well":"C10",
                          "Site":"2",
                          "Wavelength":"3",
                          "Treatment":"BRD041618"}])
            url = "file:/imaging/analysis/P-12345_A01_s2_w3.tif"
            self.check(module, url,
                       [{ "Plate":"P-12345",
                          "Well":"A01",
                          "Site":"2",
                          "Wavelength":"3"}])
        finally:
            try:
                os.unlink(path)
            except:
                pass

    def test_02_08_numeric_joining(self):
        # Check that Metadata correctly joins metadata items
        # that are supposed to be extracted as numbers
        metadata_csv = """Site,Treatment
05,DMSO
02,BRD041618
"""
        filenum, path = tempfile.mkstemp(suffix = ".csv")
        fd = os.fdopen(filenum, "w")
        fd.write(metadata_csv)
        fd.close()
        try:
            module = M.Metadata()
            module.wants_metadata.value=True
            module.data_types.value = cps.DataTypes.encode_data_types(
                {"Site":cps.DataTypes.DT_INTEGER})
            module.data_type_choice.value = M.DTC_CHOOSE
            module.add_extraction_method()
            em = module.extraction_methods[0]
            em.filter_choice.value = M.F_ALL_IMAGES
            em.extraction_method.value = M.X_MANUAL_EXTRACTION
            em.source.value = M.XM_FILE_NAME
            em.file_regexp.value = "^(?P<Plate>[^_]+)_(?P<Well>[A-H][0-9]{2})_s(?P<Site>[0-9])_w(?P<Wavelength>[0-9])"
            em.filter_choice.value = M.F_ALL_IMAGES
            
            em = module.extraction_methods[1]
            em.filter_choice.value = M.F_ALL_IMAGES
            em.extraction_method.value = M.X_IMPORTED_EXTRACTION
            em.csv_location.value = path
            em.csv_joiner.value = '[{"%s":"Site","%s":"Site"}]' % (
                module.CSV_JOIN_NAME, module.IPD_JOIN_NAME)
            em.wants_case_insensitive.value = False
            url = "file:/imaging/analysis/P-12345_B08_s5_w2.tif"
            self.check(module, url,
                       [{ "Plate":"P-12345",
                         "Well":"B08",
                         "Site":"5",
                         "Wavelength":"2",
                         "Treatment":"DMSO"}])
            url = "file:/imaging/analysis/P-12345_C10_s2_w3.tif"
            self.check(module, url,
                       [{ "Plate":"P-12345",
                          "Well":"C10",
                          "Site":"2",
                          "Wavelength":"3",
                          "Treatment":"BRD041618"}])
            url = "file:/imaging/analysis/P-12345_A01_s3_w3.tif"
            self.check(module, url,
                       [{ "Plate":"P-12345",
                          "Well":"A01",
                          "Site":"3",
                          "Wavelength":"3"}])
        finally:
            try:
                os.unlink(path)
            except:
                pass
        
    def test_03_01_well_row_column(self):
        # Make sure that Metadata_Well is generated if we have
        # Metadata_Row and Metadata_Column
        #
        for row_tag, column_tag in (
            ("row", "column"),
            ("wellrow", "wellcolumn"),
            ("well_row", "well_column")):
            module = M.Metadata()
            module.wants_metadata.value=True
            em = module.extraction_methods[0]
            em.filter_choice.value = M.F_ALL_IMAGES
            em.extraction_method.value = M.X_MANUAL_EXTRACTION
            em.source.value = M.XM_FILE_NAME
            em.file_regexp.value = (
                "^Channel(?P<Wavelength>[1-2])-"
                "(?P<%(row_tag)s>[A-H])-"
                "(?P<%(column_tag)s>[0-9]{2}).tif$") % locals()
            em.filter_choice.value = M.F_ALL_IMAGES
            url = "file:/imaging/analysis/Channel1-C-05.tif"
            self.check(module, url,
                       [{ "Wavelength":"1",
                          row_tag:"C",
                          column_tag:"05",
                          cpmeas.FTR_WELL:"C05"}])
            pipeline = cpp.Pipeline()
            imgs = I.Images()
            imgs.filter_choice.value = I.FILTER_CHOICE_NONE
            imgs.module_num = 1
            pipeline.add_module(imgs)
            module.module_num = 2
            pipeline.add_module(module)
            self.assertIn(
                cpmeas.M_WELL, 
                [c[1] for c in module.get_measurement_columns(pipeline)])
            
    def test_04_01_ome_metadata(self):
        # Test loading one URL with the humongous stack XML
        # (pat self on back if passes)
        module = M.Metadata()
        module.wants_metadata.value=True
        em = module.extraction_methods[0]
        em.filter_choice.value = M.F_ALL_IMAGES
        em.extraction_method.value = M.X_AUTOMATIC_EXTRACTION
        url = "file:/imaging/analysis/Channel1-C-05.tif"
        metadata = []
        for series in range(4):
            for z in range(36):
                metadata.append(dict(
                    Series=str(series),
                    Frame=str(z),
                    Plate="136570140804 96_Greiner",
                    Well="E11",
                    Site=str(series),
                    ChannelName="Exp1Cam1",
                    SizeX=str(688),
                    SizeY=str(512),
                    SizeZ=str(36),
                    SizeC=str(1),
                    SizeT=str(1),
                    Z=str(z),
                    C=str(0),
                    T=str(0)))
        self.check(module, url, metadata, xml=OME_XML)