"""test_namesubscriber.py - test the name subscriber classes from settings"""

import unittest

import cellprofiler_core.setting


class TestListNameSubscriber(unittest.TestCase):
    def test_load_image_list_empty(self):
        s = cellprofiler_core.setting.ListImageNameSubscriber("foo")
        self.assertEqual(s.value_text, "")
        self.assertEqual(s.value, [])

    def test_load_image_list_single(self):
        s = cellprofiler_core.setting.ListImageNameSubscriber("foo", value="SampleName")
        self.assertEqual(s.value_text, "SampleName")
        self.assertEqual(s.value, ["SampleName"])

    def test_load_image_list_multiple(self):
        s = cellprofiler_core.setting.ListImageNameSubscriber(
            "foo", value="SampleName1, SampleName2"
        )
        self.assertEqual(s.value_text, "SampleName1, SampleName2")
        self.assertEqual(s.value, ["SampleName1", "SampleName2"])

    def test_set_image_list(self):
        s = cellprofiler_core.setting.ListImageNameSubscriber("foo")
        s.value = "SampleName3, SampleName4"
        self.assertEqual(s.value_text, "SampleName3, SampleName4")
        self.assertEqual(s.value, ["SampleName3", "SampleName4"])

    def test_load_object_list_empty(self):
        s = cellprofiler_core.setting.ListObjectNameSubscriber("foo")
        self.assertEqual(s.value_text, "")
        self.assertEqual(s.value, [])

    def test_load_object_list_single(self):
        s = cellprofiler_core.setting.ListObjectNameSubscriber(
            "foo", value="SampleName"
        )
        self.assertEqual(s.value_text, "SampleName")
        self.assertEqual(s.value, ["SampleName"])

    def test_load_object_list_multiple(self):
        s = cellprofiler_core.setting.ListObjectNameSubscriber(
            "foo", value="SampleName1, SampleName2"
        )
        self.assertEqual(s.value_text, "SampleName1, SampleName2")
        self.assertEqual(s.value, ["SampleName1", "SampleName2"])

    def test_set_object_list(self):
        s = cellprofiler_core.setting.ListObjectNameSubscriber("foo")
        s.value = "SampleName3, SampleName4"
        self.assertEqual(s.value_text, "SampleName3, SampleName4")
        self.assertEqual(s.value, ["SampleName3", "SampleName4"])
