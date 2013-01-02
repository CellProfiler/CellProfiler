'''test_groups.py - test the Groups module

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
from cStringIO import StringIO
import unittest

import cellprofiler.pipeline as cpp
import cellprofiler.modules.groups as G

class TestGroups(unittest.TestCase):
    def test_01_01_load_v1(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120213205828
ModuleCount:1
HasImagePlaneDetails:False

Groups:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Do you want to group your images?:Yes
    grouping metadata count:1
    Image name:DNA
    Metadata category:Plate

"""
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, G.Groups))
        self.assertTrue(module.wants_groups)
        self.assertEqual(module.grouping_metadata_count.value, 1)
        g0 = module.grouping_metadata[0]
        self.assertEqual(g0.image_name, "DNA")
        self.assertEqual(g0.metadata_choice, "Plate")
        
    def make_image_sets(self, key_metadata, channel_metadata):
        '''Make image sets by permuting key and channel metadata
        
        key_metadata: collection of 2 tuples. 2 tuple is key + collection of
                      values, for instance, [("Well", ("A01", "A02", "A03")),
                      ("Site", ("1","2","3","4"))]
                      
        channel_metadata: collection of tuple of channel name, channel metadata key,
                          channel metadata value and image type
                          
        returns image set channel descriptors, image set key names and image sets
        '''
        iscds = [cpp.Pipeline.ImageSetChannelDescriptor(channel_name, 
                                                        image_type)
                 for channel_name, metadata_key, metadata_value, image_type
                 in channel_metadata]
        image_set_key_names = [key for key, values in key_metadata]
        
        slices = tuple([slice(0,len(values)) for key, values in key_metadata])
        ijkl = np.mgrid[slices]
        ijkl = ijkl.reshape(ijkl.shape[0], np.prod(ijkl.shape[1:]))
        image_set_values = [values for key, values in key_metadata]
        image_set_keys = [tuple([image_set_values[i][ijkl[i, j]]
                                 for i in range(ijkl.shape[0])])
                          for j in range(ijkl.shape[1])]
        #
        # The image set dictionary is a dictionary of image set key
        # to a collection of ipds. We make up a file name that consists
        # of the image set keys strung together with the channel metadata key.
        # Likewise, the IPD metadata consists of the image set keys and
        # the channel metadata key/value.
        # Construct the dictionary by passing its constructor a collection
        # of image_set_key / ipd collection tuples.
        #
        image_sets = dict(
            [(k, [cpp.ImagePlaneDetails(
                "file:/images/" + "_".join(list(k) + [channel_metadata_value])+".tif", 
                None, None, None,
                **dict([(kn, kv) for kn, kv in 
                        zip(image_set_key_names + [channel_metadata_key], 
                            list(k) + [channel_metadata_value])]))
                  for _, channel_metadata_key, channel_metadata_value, _
                  in channel_metadata])
             for k in image_set_keys])
        return iscds, image_set_key_names, image_sets
                               
        
    def test_02_00_compute_no_groups(self):
        groups = G.Groups()
        groups.wants_groups.value = False
        iscds, image_set_key_names, image_sets = self.make_image_sets(
            (("Plate", ("P-12345", "P-23456")),
             ("Well", ("A01", "A02", "A03")),
             ("Site", ("1", "2", "3", "4"))),
            (("DNA", "Wavelength", "1", cpp.Pipeline.ImageSetChannelDescriptor.CT_GRAYSCALE),
             ("GFP", "Wavelength", "1", cpp.Pipeline.ImageSetChannelDescriptor.CT_GRAYSCALE)))
        key_list, groupings = groups.compute_groups(
            iscds, image_set_key_names, image_sets)
        self.assertEqual(len(key_list), 0)
        self.assertEqual(len(groupings), 1)
        self.assertEqual(groupings.keys()[0], ())
        image_set_list = groupings.values()[0]
        self.assertEqual(len(image_set_list), 2 * 3 * 4)
        for image_set_key, expected_image_set_key in zip(
            image_set_list, sorted(image_sets.keys())):
            self.assertEqual(image_set_key, expected_image_set_key)
            
    def test_02_01_group_on_one(self):
        groups = G.Groups()
        groups.wants_groups.value = True
        groups.grouping_metadata[0].image_name.value = "DNA"
        groups.grouping_metadata[0].metadata_choice.value = "Plate"
        iscds, image_set_key_names, image_sets = self.make_image_sets(
            (("Plate", ("P-12345", "P-23456")),
             ("Well", ("A01", "A02", "A03")),
             ("Site", ("1", "2", "3", "4"))),
            (("DNA", "Wavelength", "1", cpp.Pipeline.ImageSetChannelDescriptor.CT_GRAYSCALE),
             ("GFP", "Wavelength", "1", cpp.Pipeline.ImageSetChannelDescriptor.CT_GRAYSCALE)))
        key_list, groupings = groups.compute_groups(
            iscds, image_set_key_names, image_sets)
        self.assertEqual(len(key_list), 1)
        self.assertEqual(key_list[0], (0, "Plate"))
        self.assertEqual(len(groupings), 2)
        for plate in ("P-12345", "P-23456"):
            self.assertTrue(groupings.has_key((plate,)))
            image_set_list = groupings[(plate,)]
            self.assertEqual(len(image_set_list), 3 * 4)
            expected_image_set_list = \
                sorted(filter(lambda x: x[0] == plate, image_sets.keys()))
            for image_set_key, expected_image_set_key in zip(
                image_set_list, expected_image_set_list):
                self.assertEqual(image_set_key, expected_image_set_key)
                
    def test_02_01_group_on_two(self):
        groups = G.Groups()
        groups.wants_groups.value = True
        groups.grouping_metadata[0].image_name.value = "GFP"
        groups.grouping_metadata[0].metadata_choice.value = "Plate"
        groups.add_grouping_metadata()
        groups.grouping_metadata[1].image_name.value = "DNA"
        groups.grouping_metadata[1].metadata_choice.value = "Site"
        iscds, image_set_key_names, image_sets = self.make_image_sets(
            (("Plate", ("P-12345", "P-23456")),
             ("Well", ("A01", "A02", "A03")),
             ("Site", ("1", "2", "3", "4"))),
            (("DNA", "Wavelength", "1", cpp.Pipeline.ImageSetChannelDescriptor.CT_GRAYSCALE),
             ("GFP", "Wavelength", "1", cpp.Pipeline.ImageSetChannelDescriptor.CT_GRAYSCALE)))
        key_list, groupings = groups.compute_groups(
            iscds, image_set_key_names, image_sets)
        self.assertEqual(len(key_list), 2)
        self.assertEqual(key_list[0], (1, "Plate"))
        self.assertEqual(key_list[1], (0, "Site"))
        self.assertEqual(len(groupings), 2 * 4)
        for plate in ("P-12345", "P-23456"):
            for site in ("1", "2", "3", "4"):
                self.assertTrue(groupings.has_key((plate, site)))
                image_set_list = groupings[(plate, site)]
                self.assertEqual(len(image_set_list), 3)
                expected_image_set_list = \
                    sorted(filter(lambda x: x[0] == plate and x[2] == site, 
                                  image_sets.keys()))
                for image_set_key, expected_image_set_key in zip(
                    image_set_list, expected_image_set_list):
                    self.assertEqual(image_set_key, expected_image_set_key)
                
        
        