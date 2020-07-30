"""test_namesubscriber.py - test the name subscriber classes from settings"""

import unittest

from cellprofiler_core.setting.subscriber import (
    ImageListSubscriber,
    LabelListSubscriber,
)


class TestListNameSubscriber(unittest.TestCase):
    def test_load_image_list_empty(self):
        s = ImageListSubscriber("foo")
        self.assertEqual(s.value_text, "")
        self.assertEqual(s.value, [])

    def test_load_image_list_single(self):
        s = ImageListSubscriber("foo", value="SampleName")
        self.assertEqual(s.value_text, "SampleName")
        self.assertEqual(s.value, ["SampleName"])

    def test_load_image_list_multiple(self):
        s = ImageListSubscriber("foo", value="SampleName1, SampleName2")
        self.assertEqual(s.value_text, "SampleName1, SampleName2")
        self.assertEqual(s.value, ["SampleName1", "SampleName2"])

    def test_set_image_list(self):
        s = ImageListSubscriber("foo")
        s.value = "SampleName3, SampleName4"
        self.assertEqual(s.value_text, "SampleName3, SampleName4")
        self.assertEqual(s.value, ["SampleName3", "SampleName4"])

    def test_load_object_list_empty(self):
        s = LabelListSubscriber("foo")
        self.assertEqual(s.value_text, "")
        self.assertEqual(s.value, [])

    def test_load_object_list_single(self):
        s = LabelListSubscriber("foo", value="SampleName")
        self.assertEqual(s.value_text, "SampleName")
        self.assertEqual(s.value, ["SampleName"])

    def test_load_object_list_multiple(self):
        s = LabelListSubscriber("foo", value="SampleName1, SampleName2")
        self.assertEqual(s.value_text, "SampleName1, SampleName2")
        self.assertEqual(s.value, ["SampleName1", "SampleName2"])

    def test_set_object_list(self):
        s = LabelListSubscriber("foo")
        s.value = "SampleName3, SampleName4"
        self.assertEqual(s.value_text, "SampleName3, SampleName4")
        self.assertEqual(s.value, ["SampleName3", "SampleName4"])
