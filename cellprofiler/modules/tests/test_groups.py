'''test_groups.py - test the Groups module
'''

import os
import unittest
from cStringIO import StringIO

import numpy as np

import cellprofiler.measurement as cpmeas
import cellprofiler.modules.groups as G
import cellprofiler.pipeline as cpp
import cellprofiler.workspace as cpw


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
        self.assertEqual(g0.metadata_choice, "Plate")

    def test_01_02_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20120213205828
ModuleCount:1
HasImagePlaneDetails:False

Groups:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Do you want to group your images?:Yes
    grouping metadata count:1
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
        self.assertEqual(g0.metadata_choice, "Plate")

    def make_image_sets(self, key_metadata, channel_metadata):
        '''Make image sets by permuting key and channel metadata

        key_metadata: collection of 2 tuples. 2 tuple is key + collection of
                      values, for instance, [("Well", ("A01", "A02", "A03")),
                      ("Site", ("1","2","3","4"))]

        channel_metadata: collection of tuple of channel name, channel metadata key,
                          channel metadata value and image type

        returns a Groups module and a workspace with the image sets in its measurements
        '''
        iscds = [cpp.Pipeline.ImageSetChannelDescriptor(channel_name,
                                                        image_type)
                 for channel_name, metadata_key, metadata_value, image_type
                 in channel_metadata]
        image_set_key_names = [key for key, values in key_metadata]

        slices = tuple([slice(0, len(values)) for key, values in key_metadata])
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
        image_sets = []
        for k in image_set_keys:
            image_set = []
            image_sets.append(image_set)
            for _, channel_metadata_key, channel_metadata_value, _ in channel_metadata:
                file_name = "_".join(list(k) + [channel_metadata_value]) + ".tif"
                metadata = dict([
                                    (kn, kv) for kn, kv in
                                    zip(image_set_key_names + [channel_metadata_key],
                                        list(k) + [channel_metadata_value])])
                image_set.append((file_name, metadata))
        #
        # Scramble the image sets
        #
        r = np.random.RandomState()
        r.seed(np.frombuffer("".join(["%s=%s" % kv for kv in key_metadata]), np.uint8))
        image_sets = [image_sets[i] for i in r.permutation(len(image_sets))]

        m = cpmeas.Measurements()
        m.set_metadata_tags(["_".join((cpmeas.C_METADATA, k))
                             for k in image_set_key_names])
        m.set_channel_descriptors(iscds)
        for i, ipds in enumerate(image_sets):
            image_number = i + 1
            for (file_name, metadata), iscd in zip(ipds, iscds):
                for feature, value in (
                        (cpmeas.C_FILE_NAME, file_name),
                        (cpmeas.C_PATH_NAME, os.pathsep + "images"),
                        (cpmeas.C_URL, "file://images/" + file_name)):
                    m[cpmeas.IMAGE, feature + "_" + iscd.name, image_number] = value
                for key, value in metadata.iteritems():
                    feature = "_".join((cpmeas.C_METADATA, key))
                    m[cpmeas.IMAGE, feature, image_number] = value

        pipeline = cpp.Pipeline()
        module = G.Groups()
        module.module_num = 1
        pipeline.add_module(module)
        workspace = cpw.Workspace(pipeline, module, m, None, m, None)
        return module, workspace

    def test_02_00_compute_no_groups(self):
        groups, workspace = self.make_image_sets(
                (("Plate", ("P-12345", "P-23456")),
                 ("Well", ("A01", "A02", "A03")),
                 ("Site", ("1", "2", "3", "4"))),
                (("DNA", "Wavelength", "1", cpp.Pipeline.ImageSetChannelDescriptor.CT_GRAYSCALE),
                 ("GFP", "Wavelength", "1", cpp.Pipeline.ImageSetChannelDescriptor.CT_GRAYSCALE)))
        groups = G.Groups()
        groups.wants_groups.value = False
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        image_numbers = m.get_image_numbers()
        expected_file_names = m[cpmeas.IMAGE, cpmeas.C_FILE_NAME + "_" + "DNA", image_numbers]
        self.assertTrue(groups.prepare_run(workspace))
        self.assertEqual(len(image_numbers), 2 * 3 * 4)
        output_file_names = m[cpmeas.IMAGE, cpmeas.C_FILE_NAME + "_" + "DNA", image_numbers]
        self.assertSequenceEqual(list(expected_file_names),
                                 list(output_file_names))

    def test_02_01_group_on_one(self):
        groups = G.Groups()
        groups, workspace = self.make_image_sets(
                (("Plate", ("P-12345", "P-23456")),
                 ("Well", ("A01", "A02", "A03")),
                 ("Site", ("1", "2", "3", "4"))),
                (("DNA", "Wavelength", "1", cpp.Pipeline.ImageSetChannelDescriptor.CT_GRAYSCALE),
                 ("GFP", "Wavelength", "1", cpp.Pipeline.ImageSetChannelDescriptor.CT_GRAYSCALE)))
        groups.wants_groups.value = True
        groups.grouping_metadata[0].metadata_choice.value = "Plate"
        groups.prepare_run(workspace)
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        image_numbers = m.get_image_numbers()
        self.assertEqual(len(image_numbers), 24)
        np.testing.assert_array_equal(
                np.hstack([np.ones(12, int), np.ones(12, int) * 2]),
                m[cpmeas.IMAGE, cpmeas.GROUP_NUMBER, image_numbers])
        np.testing.assert_array_equal(
                np.hstack([np.arange(1, 13)] * 2),
                m[cpmeas.IMAGE, cpmeas.GROUP_INDEX, image_numbers])

        pipeline = workspace.pipeline
        assert isinstance(pipeline, cpp.Pipeline)
        key_list, groupings = pipeline.get_groupings(workspace)
        self.assertEqual(len(key_list), 1)
        self.assertEqual(key_list[0], "Metadata_Plate")
        self.assertEqual(len(groupings), 2)

        for group_number, plate, (grouping, image_set_list) in zip(
                (1, 2), ("P-12345", "P-23456"), groupings):
            self.assertDictEqual(grouping, dict(Metadata_Plate=plate))
            self.assertEqual(len(image_set_list), 3 * 4)
            self.assertSequenceEqual(list(image_set_list),
                                     range((group_number - 1) * 12 + 1,
                                           group_number * 12 + 1))
            for image_number in range(1 + (group_number - 1) * 12,
                                      1 + group_number * 12):
                for image_name in ("DNA", "GFP"):
                    ftr = "_".join((cpmeas.C_FILE_NAME, image_name))
                    self.assertTrue(
                            m[cpmeas.IMAGE, ftr, image_number].startswith(plate))

    def test_02_01_group_on_two(self):
        groups, workspace = self.make_image_sets(
                (("Plate", ("P-12345", "P-23456")),
                 ("Well", ("A01", "A02", "A03")),
                 ("Site", ("1", "2", "3", "4"))),
                (("DNA", "Wavelength", "1", cpp.Pipeline.ImageSetChannelDescriptor.CT_GRAYSCALE),
                 ("GFP", "Wavelength", "1", cpp.Pipeline.ImageSetChannelDescriptor.CT_GRAYSCALE)))
        groups.wants_groups.value = True
        groups.grouping_metadata[0].metadata_choice.value = "Plate"
        groups.add_grouping_metadata()
        groups.grouping_metadata[1].metadata_choice.value = "Site"
        self.assertTrue(groups.prepare_run(workspace))
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        image_numbers = m.get_image_numbers()

        pipeline = workspace.pipeline
        assert isinstance(pipeline, cpp.Pipeline)
        key_list, groupings = pipeline.get_groupings(workspace)
        self.assertEqual(len(key_list), 2)
        self.assertEqual(key_list[0], "Metadata_Plate")
        self.assertEqual(key_list[1], "Metadata_Site")
        self.assertEqual(len(groupings), 8)

        idx = 0
        for plate in ("P-12345", "P-23456"):
            for site in ("1", "2", "3", "4"):
                grouping, image_set_list = groupings[idx]
                idx += 1
                self.assertEqual(grouping["Metadata_Plate"], plate)
                self.assertEqual(grouping["Metadata_Site"], site)
                self.assertEqual(len(image_set_list), 3)
                ftr = "_".join((cpmeas.C_FILE_NAME, "DNA"))
                for image_number in image_set_list:
                    file_name = m[cpmeas.IMAGE, ftr, image_number]
                    p, w, s, rest = file_name.split("_")
                    self.assertEqual(p, plate)
                    self.assertEqual(s, site)

    def test_03_01_get_measurement_columns_nogroups(self):
        #
        # Don't return the metadata grouping tags measurement if no groups
        #
        groups = G.Groups()
        groups.wants_groups.value = False
        columns = groups.get_measurement_columns(None)
        self.assertEqual(len(columns), 0)

    def test_03_02_get_measurement_columns_groups(self):
        #
        # Return the metadata grouping tags measurement if groups
        #
        groups = G.Groups()
        groups.wants_groups.value = True
        choices = ["Plate", "Well", "Site"]
        for i, choice in enumerate(choices):
            if i > 0:
                groups.add_grouping_metadata()
            del groups.grouping_metadata[i].metadata_choice.choices[:]
            groups.grouping_metadata[i].metadata_choice.choices.extend(choices)
            groups.grouping_metadata[i].metadata_choice.value = choice
        columns = groups.get_measurement_columns(None)
        self.assertEqual(len(columns), 4)
        column = columns[0]
        self.assertEqual(column[0], cpmeas.EXPERIMENT)
        self.assertEqual(column[1], cpmeas.M_GROUPING_TAGS)
        self.assertTrue(column[2].startswith(cpmeas.COLTYPE_VARCHAR))
        column_metadata = []
        for column in columns[1:]:
            self.assertEqual(column[0], cpmeas.IMAGE)
            self.assertEqual(column[2], cpmeas.COLTYPE_VARCHAR)
            column_metadata.append(column[1])
        for choice in choices:
            self.assertTrue(cpmeas.C_METADATA + "_" + choice in column_metadata)
