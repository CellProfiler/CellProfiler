'''test_namesandtypes.py - test the NamesAndTypes module

'''
#CellProfiler is distributed under the GNU General Public License.
#See the accompanying file LICENSE for details.
#
#Copyright (c) 2003-2009 Massachusetts Institute of Technology
#Copyright (c) 2009-2013 Broad Institute
#All rights reserved.
#
#Please see the AUTHORS file for credits.
#
#Website: http://www.cellprofiler.org

import numpy as np
import os
from cStringIO import StringIO
import tempfile
import unittest
import urllib

import cellprofiler.pipeline as cpp
import cellprofiler.modules.namesandtypes as N
import cellprofiler.measurements as cpmeas
import cellprofiler.workspace as cpw
import cellprofiler.utilities.jutil as J
from cellprofiler.modules.tests import example_images_directory, testimages_directory
from cellprofiler.modules.loadimages import pathname2url, C_MD5_DIGEST, C_WIDTH, C_HEIGHT, C_SCALING

M0, M1, M2, M3, M4, M5, M6 = ["MetadataKey%d" % i for i in range(7)]
C0, C1, C2, C3, C4, C5, C6 = ["Column%d" % i for i in range(7)]

IMAGE_NAME = "imagename"
ALT_IMAGE_NAME = "altimagename"
OBJECTS_NAME = "objectsname"
ALT_OBJECTS_NAME = "altobjectsname"

def md(keys_and_counts):
    '''Generate metadata dictionaries for the given metadata shape
    
    keys_and_counts - a collection of metadata keys and the dimension of
                      their extent. For instance [(M0, 2), (M1, 3)] generates
                      six dictionaries with two unique values of M0 and
                      three for M1
    '''
    keys = [k for k, c in keys_and_counts]
    counts = np.array([c for k, c in keys_and_counts])
    divisors = np.hstack([[1], np.cumprod(counts[:-1])])
    
    return [dict([(k, "k" + str(int(i / d) % c))
                  for k, d, c in zip(keys, divisors, counts)])
            for i in range(np.prod(counts))]
                  

class TestNamesAndTypes(unittest.TestCase):
    def test_00_01_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120213205828
ModuleCount:3
HasImagePlaneDetails:True

Images:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    :{"ShowFiltered"\x3A false}
    Filter based on rules:Yes
    Filter:or (extension does istif)

Metadata:[module_num:2|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Extract metadata?:Yes
    Extraction method count:1
    Extraction method:Manual
    Source:From file name
    Regular expression:^(?P<Plate>.*)_(?P<Well>\x5BA-P\x5D\x5B0-9\x5D{2})f(?P<Site>\x5B0-9\x5D{2})d(?P<ChannelNumber>\x5B0-9\x5D)
    Regular expression:(?P<Date>\x5B0-9\x5D{4}_\x5B0-9\x5D{2}_\x5B0-9\x5D{2})$
    Filter images:All images
    :or (file does contain "")
    Metadata file location\x3A:
    Match file and image metadata:\x5B\x5D

NamesAndTypes:[module_num:3|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Assignment method:Assign images matching rules
    Load as:Color image
    Image name:PI
    :\x5B{u\'Illum\'\x3A u\'Plate\', u\'DNA\'\x3A u\'Plate\', \'Cells\'\x3A u\'Plate\', u\'Actin\'\x3A u\'Plate\', u\'GFP\'\x3A u\'Plate\'}, {u\'Illum\'\x3A u\'Well\', u\'DNA\'\x3A u\'Well\', \'Cells\'\x3A u\'Well\', u\'Actin\'\x3A u\'Well\', u\'GFP\'\x3A u\'Well\'}, {u\'Illum\'\x3A u\'Site\', u\'DNA\'\x3A u\'Site\', \'Cells\'\x3A u\'Site\', u\'Actin\'\x3A u\'Site\', u\'GFP\'\x3A u\'Site\'}\x5D
    Match channels by:Order
    Assignments count:5
    Match this rule:or (metadata does ChannelNumber "0")
    Image name:DNA
    Objects name:Nuclei
    Load as:Grayscale image
    Match this rule:or (image does ismonochrome) (metadata does ChannelNumber "1") (extension does istif)
    Image name:Actin
    Objects name:Cells
    Load as:Color image
    Match this rule:or (metadata does ChannelNumber "2")
    Image name:GFP
    Objects name:Cells
    Load as:Mask
    Match this rule:or (metadata does ChannelNumber "2")
    Image name:Foo
    Objects name:Cells
    Load as:Objects
    Match this rule:or (metadata does ChannelNumber "2")
    Image name:Illum
    Objects name:Cells
    Load as:Illumination function

"Version":"1","PlaneCount":"5"
"URL","Series","Index","Channel","ColorFormat","SizeC","SizeT","SizeZ"
"file:///C:/trunk/ExampleImages/ExampleHT29/AS_09125_050116030001_D03f00d0.tif",,,,"monochrome","1","1","1"
"file:///C:/trunk/ExampleImages/ExampleHT29/AS_09125_050116030001_D03f00d1.tif",,,,"monochrome","1","1","1"
"file:///C:/trunk/ExampleImages/ExampleHT29/AS_09125_050116030001_D03f00d2.tif",,,,"monochrome","1","1","1"
"file:///C:/trunk/ExampleImages/ExampleHT29/ExampleHT29.cp",,,,,,,
"file:///C:/trunk/ExampleImages/ExampleHT29/k27IllumCorrControlv1.mat",,,,,,,
"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[2]
        self.assertTrue(isinstance(module, N.NamesAndTypes))
        self.assertEqual(module.assignment_method, N.ASSIGN_RULES)
        self.assertEqual(module.single_load_as_choice, N.LOAD_AS_COLOR_IMAGE)
        self.assertEqual(module.single_image_provider.value, "PI")
        self.assertEqual(module.matching_choice, N.MATCH_BY_ORDER)
        self.assertEqual(module.assignments_count.value, 5)
        aa = module.assignments
        for assignment, rule, image_name, objects_name, load_as in (
            (aa[0], 'or (metadata does ChannelNumber "0")', "DNA", "Nuclei", N.LOAD_AS_GRAYSCALE_IMAGE),
            (aa[1], 'or (image does ismonochrome) (metadata does ChannelNumber "1") (extension does istif)', "Actin", "Cells", N.LOAD_AS_COLOR_IMAGE),
            (aa[2], 'or (metadata does ChannelNumber "2")', "GFP", "Cells", N.LOAD_AS_MASK),
            (aa[3], 'or (metadata does ChannelNumber "2")', "Foo", "Cells", N.LOAD_AS_OBJECTS),
            (aa[4], 'or (metadata does ChannelNumber "2")', "Illum", "Cells", N.LOAD_AS_ILLUMINATION_FUNCTION)):
            self.assertEqual(assignment.rule_filter.value, rule)
            self.assertEqual(assignment.image_name, image_name)
            self.assertEqual(assignment.object_name, objects_name)
            self.assertEqual(assignment.load_as_choice, load_as)
            
    def do_teest(self, module, channels, expected_tags, expected_metadata, additional=None):
        '''Ensure that NamesAndTypes recreates the column layout when run
        
        module - instance of NamesAndTypes, set up for the test
        
        channels - a dictionary of channel name to list of "ipds" for that
                   channel where "ipd" is a tuple of URL and metadata dictionary
                   Entries may appear multiple times (e.g. illumination function)
        expected_tags - the metadata tags that should have been generated
                   by prepare_run.
        expected_metadata - a sequence of two-tuples of metadata key and
                            the channel from which the metadata is extracted.
        additional - if present, these are added as ImagePlaneDetails in order
                     to create errors. Format is same as a single channel of
                     channels.
        '''
        ipds = []
        urls = set()
        url_root = "file:" + urllib.pathname2url(os.path.abspath(os.path.curdir))
        channels = dict(channels)
        if additional is not None:
            channels["Additional"] = additional
        for channel_name in list(channels):
            channel_data = [(url_root + "/" + path, metadata)
                             for path, metadata in channels[channel_name]]
            channels[channel_name] = channel_data
            for url, metadata in channel_data:
                if url in urls:
                    continue
                urls.add(url)
                ipd = cpp.ImagePlaneDetails(url, None, None, None, **metadata)
                jmetadata = J.make_map(**metadata)
                ipd.jipd = J.run_script("""
                importPackage(Packages.org.cellprofiler.imageset);
                importPackage(Packages.org.cellprofiler.imageset.filter);
                var imageFile=new ImageFile(new java.net.URI(url));
                var imagePlane=new ImagePlane(imageFile);
                new ImagePlaneDetails(imagePlane, metadata);
                """, dict(url=url, metadata=jmetadata))
                ipds.append(ipd)
        if additional is not None:
            del channels["Additional"]
        ipds.sort(key = lambda x: x.url)
        pipeline = cpp.Pipeline()
        del pipeline.image_plane_details[:]
        pipeline.image_plane_details.extend(ipds)
        module.module_num = 1
        pipeline.add_module(module)
        m = cpmeas.Measurements()
        workspace = cpw.Workspace(pipeline, module, m, None, m, None)
        self.assertTrue(module.prepare_run(workspace))
        tags = m.get_metadata_tags()
        self.assertEqual(len(tags), len(expected_tags))
        for tag, expected_tag in zip(tags, expected_tags):
            for et in expected_tag:
                ftr = et if et == cpmeas.IMAGE_NUMBER else \
                    "_".join((cpmeas.C_METADATA, et))
                if ftr == tag:
                    break
            else:
                self.fail("%s not in %s" % (tag, ",".join(expected_tag)))
        iscds = m.get_channel_descriptors()
        self.assertEqual(len(iscds), len(channels))
        for channel_name in channels.keys():
            iscd_match = filter(lambda x:x.name == channel_name, iscds)
            self.assertEquals(len(iscd_match), 1)
            iscd = iscd_match[0]
            assert isinstance(iscd, cpp.Pipeline.ImageSetChannelDescriptor)
            for i, (expected_url, metadata) in enumerate(channels[channel_name]):
                image_number = i+1
                if iscd.channel_type == iscd.CT_OBJECTS:
                    url_ftr = "_".join((cpmeas.C_OBJECTS_URL, channel_name))
                else:
                    url_ftr = "_". join((cpmeas.C_URL, channel_name))
                self.assertEqual(expected_url, m[cpmeas.IMAGE, url_ftr, image_number])
                for key, channel in expected_metadata:
                    if channel != channel_name:
                        continue
                    md_ftr = "_".join((cpmeas.C_METADATA, key))
                    self.assertEqual(metadata[key], 
                                     m[cpmeas.IMAGE, md_ftr, image_number])
            
    def test_01_00_01_all(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_ALL
        n.single_image_provider.value = C0
        data = { C0: [("images/1.jpg", {M0:"1"})] }
        self.do_teest(n, data, [(cpmeas.IMAGE_NUMBER,)], [(M0,C0)])
        
    def test_01_01_one(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_RULES
        n.matching_choice.value = N.MATCH_BY_METADATA
        n.assignments[0].image_name.value = C0
        n.join.build("[{\"%s\":\"%s\"}]" % (C0, M0))
        # It should match by order, even if match by metadata and the joiner
        # are set up.
        data = { C0: [ ("images/1.jpg", {M0:"k1"}),
                       ("images/2.jpg", {M0:"k3"}),
                       ("images/3.jpg", {M0:"k2"})]}
        self.do_teest(n, data, [ (cpmeas.IMAGE_NUMBER,) ], [(M0, C0)])
        
    def test_01_02_match_one_same_key(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_RULES
        n.matching_choice.value = N.MATCH_BY_METADATA
        n.add_assignment()
        n.assignments[0].image_name.value = C0
        n.assignments[1].image_name.value = C1
        n.assignments[0].rule_filter.value = 'file does contain "1"'
        n.assignments[1].rule_filter.value = 'file doesnot contain "1"'
        n.join.build("[{'%s':'%s','%s':'%s'}]" % (C0, M0, C1, M0))
        data = { C0: [ ("images/1.jpg", {M0:"k1"})],
                 C1: [ ("images/2.jpg", {M0:"k1"})]}
        self.do_teest(n, data, [ (M0,) ], [(M0, C0)])
        
    def test_01_03_match_one_different_key(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_RULES
        n.matching_choice.value = N.MATCH_BY_METADATA
        n.add_assignment()
        n.assignments[0].image_name.value = C0
        n.assignments[1].image_name.value = C1
        n.assignments[0].rule_filter.value = 'file does contain "1"'
        n.assignments[1].rule_filter.value = 'file doesnot contain "1"'
        n.join.build("[{'%s':'%s','%s':'%s'}]" % (C0, M0, C1, M1))
        data = { C0: [ ("images/1.jpg", {M0:"k1"})],
                 C1: [ ("images/2.jpg", {M1:"k1"})]}
        self.do_teest(n, data, [ (M0, M1) ], [(M0, C0), (M1, C1)])
        
    def test_01_04_match_two_one_key(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_RULES
        n.matching_choice.value = N.MATCH_BY_METADATA
        n.add_assignment()
        n.assignments[0].image_name.value = C0
        n.assignments[1].image_name.value = C1
        n.assignments[0].rule_filter.value = 'file does contain "%s"' % C0
        n.assignments[1].rule_filter.value = 'file does contain "%s"' % C1
        n.join.build("[{'%s':'%s','%s':'%s'}]" % (C0, M0, C1, M1))
        data = { 
            C0:[("%s%d" % (C0, i), m)
                for i, m in enumerate(md([(M0, 2)]))],
            C1:[("%s%d" % (C1, i), m)
                for i, m in enumerate(md([(M1, 2)]))]}
        self.do_teest(n, data, [(M0,M1)], [(M0, C0), (M1, C1)])
                
    def test_01_05_match_two_and_two(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_RULES
        n.matching_choice.value = N.MATCH_BY_METADATA
        n.add_assignment()
        n.assignments[0].image_name.value = C0
        n.assignments[1].image_name.value = C1
        n.assignments[0].rule_filter.value = 'file does contain "%s"' % C0
        n.assignments[1].rule_filter.value = 'file does contain "%s"' % C1
        n.join.build("[{'%s':'%s','%s':'%s'},{'%s':'%s','%s':'%s'}]" % 
                     (C0, M0, C1, M2, C0, M1, C1, M3))
        data = {
            C0:[("%s%s%s" % (C0, m[M0], m[M1]), m) 
                for i, m in enumerate(md([(M1, 3), (M0, 2)]))],
            C1:[("%s%s%s" % (C1, m[M2], m[M3]), m) 
                for i, m in enumerate(md([(M3, 3), (M2, 2)]))] }
        self.do_teest(n, data, [(M0,M2), (M1, M3)], 
                      [(M0, C0), (M1, C0), (M2, C1), (M3, C1)])
                
    def test_01_06_01_two_with_same_metadata(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_RULES
        n.matching_choice.value = N.MATCH_BY_METADATA
        n.add_assignment()
        n.assignments[0].image_name.value = C0
        n.assignments[1].image_name.value = C1
        n.assignments[0].rule_filter.value = 'file does contain "%s"' % C0
        n.assignments[1].rule_filter.value = 'file does contain "%s"' % C1
        n.join.build("[{'%s':'%s','%s':'%s'},{'%s':'%s','%s':'%s'}]" % 
                     (C0, M0, C1, M2, C0, M1, C1, M3))
        data = {
            C0:[("%s%s%s" % (C0, m[M0], m[M1]), m)
              for i, m in enumerate(md([(M1, 3), (M0, 2)]))],
            C1:[("%s%s%s" % (C1, m[M2], m[M3]), m)
                for i, m in enumerate(md([(M3, 3), (M2, 2)]))] }
        bad_row = 5
        # Steal the bad row's metadata
        additional = [ ("%sBad" %C0, data[C0][bad_row][1]),
                       data[C0][bad_row], data[C1][bad_row]]
        del data[C0][bad_row]
        del data[C1][bad_row]
        self.do_teest(n, data, [(M0,M2), (M1, M3)], 
                      [(M0, C0), (M1, C0), (M2, C1), (M3, C1)], additional)
        
    def test_01_06_02_missing(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_RULES
        n.matching_choice.value = N.MATCH_BY_METADATA
        n.add_assignment()
        n.assignments[0].image_name.value = C0
        n.assignments[1].image_name.value = C1
        n.assignments[0].rule_filter.value = 'file does contain "%s"' % C0
        n.assignments[1].rule_filter.value = 'file does contain "%s"' % C1
        n.join.build("[{'%s':'%s','%s':'%s'},{'%s':'%s','%s':'%s'}]" % 
                     (C0, M0, C1, M2, C0, M1, C1, M3))
        data = {
            C0:[("%s%s%s" % (C0, m[M0], m[M1]), m)
              for i, m in enumerate(md([(M1, 3), (M0, 2)]))],
            C1:[("%s%s%s" % (C1, m[M2], m[M3]), m)
                for i, m in enumerate(md([(M3, 3), (M2, 2)]))] }
        bad_row = 3
        # Steal the bad row's metadata
        additional = [ data[C1][bad_row]]
        del data[C0][bad_row]
        del data[C1][bad_row]
        self.do_teest(n, data, [(M0,M2), (M1, M3)], 
                      [(M0, C0), (M1, C0), (M2, C1), (M3, C1)], additional)        

    def test_01_07_one_against_all(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_RULES
        n.matching_choice.value = N.MATCH_BY_METADATA
        n.add_assignment()
        n.assignments[0].image_name.value = C0
        n.assignments[1].image_name.value = C1
        n.assignments[0].rule_filter.value = 'file does contain "%s"' % C0
        n.assignments[1].rule_filter.value = 'file does contain "%s"' % C1
        n.join.build("[{'%s':None,'%s':'%s'}]" % (C0, C1, M0))
        data = {
            C0:[(C0, {})] * 3,
            C1:[("%s%d" % (C1, i), m) for i, m in enumerate(md([(M0, 3)]))] }
        self.do_teest(n, data, [(M0, )], [(M0, C1)])
        
    def test_01_08_some_against_all(self):
        #
        # Permute both the order of the columns and the order of joins
        #
        
        joins = [{C0:M0, C1:M1},{C0:None, C1:M2}]
        for cA, cB in ((C0, C1), (C1, C0)):
            for j0, j1 in ((0,1),(1,0)):
                n = N.NamesAndTypes()
                n.assignment_method.value = N.ASSIGN_RULES
                n.matching_choice.value = N.MATCH_BY_METADATA
                n.add_assignment()
                n.assignments[0].image_name.value = cA
                n.assignments[1].image_name.value = cB
                n.assignments[0].rule_filter.value = 'file does contain "%s"' % cA
                n.assignments[1].rule_filter.value = 'file does contain "%s"' % cB
                n.join.build(repr([joins[j0], joins[j1]]))
                mA0 = [M0, M3][j0]
                mB0 = [M0, M3][j1]
                mA1 = joins[j0][C1]
                mB1 = joins[j1][C1]
                data = { 
                    C0: [("%s%s" % (C0, m[M0]), m) 
                         for i, m in enumerate(md([(mB0, 2), (mA0, 3)]))],
                    C1: [("%s%s%s" % (C1, m[M1], m[M2]), m)
                         for i, m in enumerate(md([(mB1, 2),(mA1, 3)]))] }
                expected_keys = [[(M0, M1), (M2, )][i] for i in j0, j1]
                self.do_teest(n, data, expected_keys, 
                              [(C0, M0), (C0, M3), (C1, M1), (C1, M2)])
                    
    def test_01_10_by_order(self):
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_RULES
        n.matching_choice.value = N.MATCH_BY_ORDER
        n.add_assignment()
        n.assignments[0].image_name.value = C0
        n.assignments[1].image_name.value = C1
        n.assignments[0].rule_filter.value = 'file does contain "%s"' % C0
        n.assignments[1].rule_filter.value = 'file does contain "%s"' % C1
        data = {
            C0:[("%s%d" % (C0, i+1), m) for i, m in enumerate(md([(M0, 2)]))],
            C1:[("%s%d" % (C1, i+1), m) for i, m in enumerate(md([(M1, 2)]))] }
        self.do_teest(n, data, [ (cpmeas.IMAGE_NUMBER,)], [(C0, M0), (C1, M1)])
                
    def test_01_11_by_order_bad(self):
        # Regression test of issue #392: columns of different lengths
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_RULES
        n.matching_choice.value = N.MATCH_BY_ORDER
        n.add_assignment()
        n.assignments[0].image_name.value = C0
        n.assignments[1].image_name.value = C1
        n.assignments[0].rule_filter.value = 'file does contain "%s"' % C0
        n.assignments[1].rule_filter.value = 'file does contain "%s"' % C1
        n.ipd_columns = \
            [[cpp.ImagePlaneDetails("%s%d" % (C0, (3-i)), None, None, None, **m)
              for i, m in enumerate(md([(M0, 3)]))],
             [cpp.ImagePlaneDetails("%s%d" % (C1, i+1), None, None, None, **m)
                           for i, m in enumerate(md([(M1, 2)]))]]
        data = {
            C0:[("%s%d" % (C0, i+1), m) for i, m in enumerate(md([(M0, 2)]))],
            C1:[("%s%d" % (C1, i+1), m) for i, m in enumerate(md([(M1, 2)]))] }
        additional = [("%sBad" % C0, {})]
        self.do_teest(n, data, [ (cpmeas.IMAGE_NUMBER,)], [(C0, M0), (C1, M1)])
                
    def test_02_01_prepare_to_create_batch_single(self):
        n = N.NamesAndTypes()
        n.module_num = 1
        n.assignment_method.value = N.ASSIGN_ALL
        n.single_image_provider.value = IMAGE_NAME
        m = cpmeas.Measurements(mode="memory")
        pathnames = ["foo", "fuu"]
        expected_pathnames = ["bar", "fuu"]
        filenames = ["boo", "foobar"]
        expected_filenames = ["boo", "barbar"]
        urlnames = ["file:/foo/bar", "http://foo/bar"]
        expected_urlnames = ["file:/bar/bar", "http://foo/bar"]
        
        m.add_all_measurements(cpmeas.IMAGE,
                               cpmeas.C_FILE_NAME + "_" + IMAGE_NAME,
                               filenames)
        m.add_all_measurements(cpmeas.IMAGE,
                               cpmeas.C_PATH_NAME + "_" + IMAGE_NAME,
                               pathnames)
        m.add_all_measurements(cpmeas.IMAGE,
                               cpmeas.C_URL + "_" + IMAGE_NAME,
                               urlnames)
        pipeline = cpp.Pipeline()
        pipeline.add_module(n)
        workspace = cpw.Workspace(pipeline, n, m, None, m, None)
        n.prepare_to_create_batch(workspace, lambda x: x.replace("foo", "bar"))
        for feature, expected in ((cpmeas.C_FILE_NAME, expected_filenames),
                                  (cpmeas.C_PATH_NAME, expected_pathnames),
                                  (cpmeas.C_URL, expected_urlnames)):
            values = m.get_measurement(cpmeas.IMAGE, 
                                       feature + "_" + IMAGE_NAME,
                                       np.arange(len(expected)) + 1)
            self.assertSequenceEqual(expected, list(values))
            
    def test_02_02_prepare_to_create_batch_multiple(self):
        n = N.NamesAndTypes()
        n.module_num = 1
        n.assignment_method.value = N.ASSIGN_RULES
        n.add_assignment()
        n.assignments[0].load_as_choice.value = N.LOAD_AS_GRAYSCALE_IMAGE
        n.assignments[0].image_name.value = IMAGE_NAME
        n.assignments[1].load_as_choice.value = N.LOAD_AS_OBJECTS
        n.assignments[1].object_name.value = OBJECTS_NAME
        m = cpmeas.Measurements(mode="memory")
        pathnames = ["foo", "fuu"]
        expected_pathnames = ["bar", "fuu"]
        filenames = ["boo", "foobar"]
        expected_filenames = ["boo", "barbar"]
        urlnames = ["file:/foo/bar", "http://foo/bar"]
        expected_urlnames = ["file:/bar/bar", "http://foo/bar"]
        
        for feature, name, values in (
            (cpmeas.C_FILE_NAME, IMAGE_NAME, filenames),
            (cpmeas.C_OBJECTS_FILE_NAME, OBJECTS_NAME, reversed(filenames)),
            (cpmeas.C_PATH_NAME, IMAGE_NAME, pathnames),
            (cpmeas.C_OBJECTS_PATH_NAME, OBJECTS_NAME, reversed(pathnames)),
            (cpmeas.C_URL, IMAGE_NAME, urlnames),
            (cpmeas.C_OBJECTS_URL, OBJECTS_NAME, reversed(urlnames))):
            m.add_all_measurements(cpmeas.IMAGE,
                                   feature + "_" + name,
                                   values)
        pipeline = cpp.Pipeline()
        pipeline.add_module(n)
        workspace = cpw.Workspace(pipeline, n, m, None, m, None)
        n.prepare_to_create_batch(workspace, lambda x: x.replace("foo", "bar"))
        for feature, name, expected in (
            (cpmeas.C_FILE_NAME, IMAGE_NAME, expected_filenames),
            (cpmeas.C_OBJECTS_FILE_NAME, OBJECTS_NAME, reversed(expected_filenames)),
            (cpmeas.C_PATH_NAME, IMAGE_NAME, expected_pathnames),
            (cpmeas.C_OBJECTS_PATH_NAME, OBJECTS_NAME, reversed(expected_pathnames)),
            (cpmeas.C_URL, IMAGE_NAME, expected_urlnames),
            (cpmeas.C_OBJECTS_URL, OBJECTS_NAME, reversed(expected_urlnames))):
            values = m.get_measurement(cpmeas.IMAGE, 
                                       feature + "_" + name,
                                       np.arange(1, 3))
            self.assertSequenceEqual(list(expected), list(values))
        
    def run_workspace(self, path, load_as_type, 
                      series = None, index = None, channel = None,
                      single=False):
        '''Run a workspace to load a file
        
        path - path to the file
        load_as_type - one of the LOAD_AS... constants
        series, index, channel - pick a plane from within a file
        
        returns the workspace after running
        '''
        n = N.NamesAndTypes()
        n.assignment_method.value = N.ASSIGN_ALL if single else N.ASSIGN_RULES
        n.single_image_provider.value = IMAGE_NAME
        n.single_load_as_choice.value = load_as_type
        n.assignments[0].image_name.value = IMAGE_NAME
        n.assignments[0].object_name.value = OBJECTS_NAME
        n.assignments[0].load_as_choice.value = load_as_type
        n.module_num = 1
        pipeline = cpp.Pipeline()
        pipeline.add_module(n)
        url = pathname2url(path)
        pathname, filename = os.path.split(path)
        m = cpmeas.Measurements(mode="memory")
        if load_as_type == N.LOAD_AS_OBJECTS:
            url_feature = cpmeas.C_OBJECTS_URL + "_" + OBJECTS_NAME
            path_feature = cpmeas.C_OBJECTS_PATH_NAME + "_" + OBJECTS_NAME
            file_feature = cpmeas.C_OBJECTS_FILE_NAME + "_" + OBJECTS_NAME
            series_feature = cpmeas.C_OBJECTS_SERIES + "_" + OBJECTS_NAME
            frame_feature = cpmeas.C_OBJECTS_FRAME + "_" + OBJECTS_NAME
            channel_feature = cpmeas.C_OBJECTS_CHANNEL + "_" + OBJECTS_NAME
        else:
            url_feature = cpmeas.C_URL + "_" + IMAGE_NAME
            path_feature = cpmeas.C_PATH_NAME + "_" + IMAGE_NAME
            file_feature = cpmeas.C_FILE_NAME + "_" + IMAGE_NAME
            series_feature = cpmeas.C_SERIES + "_" + IMAGE_NAME
            frame_feature = cpmeas.C_FRAME + "_" + IMAGE_NAME
            channel_feature = cpmeas.C_CHANNEL + "_" + IMAGE_NAME
            
        m.image_set_number = 1
        m.add_measurement(cpmeas.IMAGE, url_feature, url)
        m.add_measurement(cpmeas.IMAGE, path_feature, pathname)
        m.add_measurement(cpmeas.IMAGE, file_feature, filename)
        if series is not None:
            m.add_measurement(cpmeas.IMAGE, series_feature, series)
        if index is not None:
            m.add_measurement(cpmeas.IMAGE, frame_feature, index)
        if channel is not None:
            m.add_measurement(cpmeas.IMAGE, channel_feature, channel)
        m.add_measurement(cpmeas.IMAGE, cpmeas.GROUP_NUMBER, 1)
        m.add_measurement(cpmeas.IMAGE, cpmeas.GROUP_INDEX, 1)
        
        workspace = cpw.Workspace(pipeline, n, m,
                                  N.cpo.ObjectSet(),
                                  m, None)
        n.run(workspace)
        return workspace
        
    def test_03_01_load_color(self):
        path = os.path.join(example_images_directory(), 
                            "ExampleColorToGray",
                            "AS_09125_050116030001_D03f00_color.tif")
        workspace = self.run_workspace(path, N.LOAD_AS_COLOR_IMAGE)
        image = workspace.image_set.get_image(IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertSequenceEqual(pixel_data.shape, (512, 512, 3))
        self.assertTrue(np.all(pixel_data >= 0))
        self.assertTrue(np.all(pixel_data <= 1))
        m = workspace.measurements
        self.assertEqual(m[cpmeas.IMAGE, C_MD5_DIGEST + "_" + IMAGE_NAME],
                         "16729de931dc40f5ca19621598a3e7d6")
        self.assertEqual(m[cpmeas.IMAGE, C_HEIGHT + "_" + IMAGE_NAME], 512)
        self.assertEqual(m[cpmeas.IMAGE, C_WIDTH + "_" + IMAGE_NAME], 512)
        
        
    def test_03_02_load_monochrome_as_color(self):
        path = os.path.join(example_images_directory(),
                            "ExampleGrayToColor",
                            "AS_09125_050116030001_D03f00d0.tif")
        workspace = self.run_workspace(path, N.LOAD_AS_COLOR_IMAGE)
        image = workspace.image_set.get_image(IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertSequenceEqual(pixel_data.shape, (512, 512, 3))
        self.assertTrue(np.all(pixel_data >= 0))
        self.assertTrue(np.all(pixel_data <= 1))
        np.testing.assert_equal(pixel_data[:, :, 0], pixel_data[:, :, 1])
        np.testing.assert_equal(pixel_data[:, :, 0], pixel_data[:, :, 2])
        
    def test_03_03_load_color_frame(self):
        path = os.path.join(testimages_directory(),
                            "DrosophilaEmbryo_GFPHistone.avi")
        workspace = self.run_workspace(path, N.LOAD_AS_COLOR_IMAGE,
                                       index = 3)
        image = workspace.image_set.get_image(IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertSequenceEqual(pixel_data.shape, (264, 542, 3))
        self.assertTrue(np.all(pixel_data >= 0))
        self.assertTrue(np.all(pixel_data <= 1))
        self.assertTrue(np.any(pixel_data[:, :, 0] != pixel_data[:, :, 1]))
        self.assertTrue(np.any(pixel_data[:, :, 0] != pixel_data[:, :, 2]))
        
    def test_03_04_load_monochrome(self):
        path = os.path.join(example_images_directory(),
                            "ExampleGrayToColor",
                            "AS_09125_050116030001_D03f00d0.tif")
        workspace = self.run_workspace(path, N.LOAD_AS_GRAYSCALE_IMAGE)
        image = workspace.image_set.get_image(IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertSequenceEqual(pixel_data.shape, (512, 512))
        self.assertTrue(np.all(pixel_data >= 0))
        self.assertTrue(np.all(pixel_data <= 1))

    def test_03_05_load_color_as_monochrome(self):
        path = os.path.join(example_images_directory(),
                            "ExampleGrayToColor",
                            "AS_09125_050116030001_D03f00d0.tif")
        workspace = self.run_workspace(path, N.LOAD_AS_GRAYSCALE_IMAGE)
        image = workspace.image_set.get_image(IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertSequenceEqual(pixel_data.shape, (512, 512))
        self.assertTrue(np.all(pixel_data >= 0))
        self.assertTrue(np.all(pixel_data <= 1))
        
    def test_03_06_load_monochrome_plane(self):
        path = os.path.join(testimages_directory(), "5channel.tif")
        
        for i in range(5):
            workspace = self.run_workspace(path, N.LOAD_AS_GRAYSCALE_IMAGE,
                                           index=i)
            image = workspace.image_set.get_image(IMAGE_NAME)
            pixel_data = image.pixel_data
            self.assertSequenceEqual(pixel_data.shape, (64, 64))
            if i == 0:
                plane_0 = pixel_data.copy()
            else:
                self.assertTrue(np.any(pixel_data != plane_0))
                
    def test_03_07_load_raw(self):
        path = os.path.join(example_images_directory(),
                            "ExampleSpecklesImages",
                            "1-162hrh2ax2.tif")
        workspace = self.run_workspace(path, N.LOAD_AS_ILLUMINATION_FUNCTION)
        image = workspace.image_set.get_image(IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertSequenceEqual(pixel_data.shape, (1000, 1200))
        self.assertTrue(np.all(pixel_data >= 0))
        self.assertTrue(np.all(pixel_data <= 1. / 16.))
        
    def test_03_08_load_mask(self):
        path = os.path.join(example_images_directory(),
                            "ExampleSBSImages",
                            "Channel2-01-A-01.tif")
        workspace = self.run_workspace(path, N.LOAD_AS_MASK)
        image = workspace.image_set.get_image(IMAGE_NAME)
        pixel_data = image.pixel_data
        self.assertSequenceEqual(pixel_data.shape, (640, 640))
        self.assertEqual(np.sum(~pixel_data), 627)
        
    def test_03_09_load_objects(self):
        path = os.path.join(example_images_directory(),
                            "ExampleSBSImages",
                            "Channel2-01-A-01.tif")
        workspace = self.run_workspace(path, N.LOAD_AS_OBJECTS)
        o = workspace.object_set.get_objects(OBJECTS_NAME)
        assert isinstance(o, N.cpo.Objects)
        areas = o.areas
        self.assertEqual(areas[0], 9)
        self.assertEqual(areas[1], 321)
        self.assertEqual(areas[2], 2655)
        m = workspace.measurements
        self.assertEqual(m[cpmeas.IMAGE, C_MD5_DIGEST + "_" + OBJECTS_NAME],
                         "67880f6269fbf438d4b9c92256aa1d8f")
        self.assertEqual(m[cpmeas.IMAGE, C_WIDTH + "_" + OBJECTS_NAME], 640)
        
    def test_03_10_load_overlapped_objects(self):
        from .test_loadimages import overlapped_objects_data
        from .test_loadimages import overlapped_objects_data_masks
        fd, path = tempfile.mkstemp(".tif")
        f = os.fdopen(fd, "wb")
        f.write(overlapped_objects_data)
        f.close()
        try:
            workspace = self.run_workspace(path, N.LOAD_AS_OBJECTS)
            o = workspace.object_set.get_objects(OBJECTS_NAME)
            assert isinstance(o, N.cpo.Objects)
            self.assertEqual(o.count, 2)
            mask = np.zeros(overlapped_objects_data_masks[0].shape, bool)
            expected_mask = (overlapped_objects_data_masks[0] |
                             overlapped_objects_data_masks[1])
            for i in range(2):
                expected = overlapped_objects_data_masks[i]
                i, j = o.ijv[o.ijv[:, 2] == i+1, :2].transpose()
                self.assertTrue(np.all(expected[i, j]))
                mask[i, j] = True
            self.assertFalse(np.any(mask[~ expected_mask]))
        finally:
            try:
                os.unlink(path)
            except:
                pass
        