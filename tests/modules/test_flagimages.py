"""test_flagimages.py - Test the FlagImages module
"""

import base64
import contextlib
import os
import tempfile
import unittest
import zlib
from six.moves import StringIO

import PIL.Image as PILImage
import numpy as np
import scipy.ndimage

from cellprofiler.preferences import set_headless
from .test_filterobjects import make_classifier_pickle

set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.setting as cps
import cellprofiler.image as cpi
import cellprofiler.workspace as cpw
import cellprofiler.object as cpo
import cellprofiler.measurement as cpmeas
import cellprofiler.preferences as cpprefs

import cellprofiler.modules.flagimage as F


def image_measurement_name(index):
    return "Metadata_ImageMeasurement_%d" % index


OBJECT_NAME = "object"


def object_measurement_name(index):
    return "Measurement_Measurement_%d" % index


MEASUREMENT_CATEGORY = "MyCategory"
MEASUREMENT_FEATURE = "MyFeature"
MEASUREMENT_NAME = "_".join((MEASUREMENT_CATEGORY, MEASUREMENT_FEATURE))


class TestFlagImages:
    def test_load_v2(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9889

FlagImage:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
    Hidden:2
    Hidden:3
    Name the flag\'s category:Metadata
    Name the flag:QCFlag
    Flag if any, or all, measurement(s) fails to meet the criteria?:Flag if any fail
    Skip image set if flagged?:No
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MaxIntensity_DNA
    Flag images based on low values?:No
    Minimum value:0.0
    Flag images based on high values?:Yes
    Maximum value:0.95
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MinIntensity_Cytoplasm
    Flag images based on low values?:Yes
    Minimum value:0.05
    Flag images based on high values?:No
    Maximum value:1.0
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MeanIntensity_DNA
    Flag images based on low values?:Yes
    Minimum value:0.1
    Flag images based on high values?:Yes
    Maximum value:0.9
    Hidden:1
    Name the flag\'s category:Metadata
    Name the flag:HighCytoplasmIntensity
    Flag if any, or all, measurement(s) fails to meet the criteria?:Flag if any fail
    Skip image set if flagged?:Yes
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MeanIntensity_Cytoplasm
    Flag images based on low values?:No
    Minimum value:0.0
    Flag images based on high values?:Yes
    Maximum value:.8
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[0]
        assert isinstance(module, F.FlagImage)
        expected = (
            (
                "QCFlag",
                F.C_ANY,
                False,
                (
                    ("Intensity_MaxIntensity_DNA", None, 0.95),
                    ("Intensity_MinIntensity_Cytoplasm", 0.05, None),
                    ("Intensity_MeanIntensity_DNA", 0.1, 0.9),
                ),
            ),
            (
                "HighCytoplasmIntensity",
                None,
                True,
                (("Intensity_MeanIntensity_Cytoplasm", None, 0.8),),
            ),
        )
        assert len(expected) == module.flag_count.value
        for flag, (feature_name, combine, skip, measurements) in zip(
            module.flags, expected
        ):
            assert isinstance(flag, cps.SettingsGroup)
            assert flag.category == "Metadata"
            assert flag.feature_name == feature_name
            assert flag.wants_skip == skip
            if combine is not None:
                assert flag.combination_choice == combine
            assert len(measurements) == flag.measurement_count.value
            for measurement, (measurement_name, min_value, max_value) in zip(
                flag.measurement_settings, measurements
            ):
                assert isinstance(measurement, cps.SettingsGroup)
                assert measurement.source_choice == F.S_IMAGE
                assert measurement.measurement == measurement_name
                assert measurement.wants_minimum.value == (min_value is not None)
                if measurement.wants_minimum.value:
                    assert (
                        round(abs(measurement.minimum_value.value - min_value), 7) == 0
                    )
                assert measurement.wants_maximum.value == (max_value is not None)
                if measurement.wants_maximum.value:
                    assert (
                        round(abs(measurement.maximum_value.value - max_value), 7) == 0
                    )

    def test_load_v3(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:2
DateRevision:20120306205005

FlagImage:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Hidden:2
    Hidden:3
    Name the flag\'s category:Metadata
    Name the flag:QCFlag
    Flag if any, or all, measurement(s) fails to meet the criteria?:Flag if any fail
    Skip image set if flagged?:No
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MaxIntensity_DNA
    Flag images based on low values?:No
    Minimum value:0.0
    Flag images based on high values?:Yes
    Maximum value:0.95
    Rules file location:Default Input Folder\x7CNone
    Rules file name:foo.txt
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MinIntensity_Cytoplasm
    Flag images based on low values?:Yes
    Minimum value:0.05
    Flag images based on high values?:No
    Maximum value:1.0
    Rules file location:Default Input Folder\x7CNone
    Rules file name:bar.txt
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MeanIntensity_DNA
    Flag images based on low values?:Yes
    Minimum value:0.1
    Flag images based on high values?:Yes
    Maximum value:0.9
    Rules file location:Default Input Folder\x7CNone
    Rules file name:baz.txt
    Hidden:1
    Name the flag\'s category:Metadata
    Name the flag:HighCytoplasmIntensity
    Flag if any, or all, measurement(s) fails to meet the criteria?:Flag if any fail
    Skip image set if flagged?:Yes
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MeanIntensity_Cytoplasm
    Flag images based on low values?:No
    Minimum value:0.0
    Flag images based on high values?:Yes
    Maximum value:.8
    Rules file location:Default Input Folder\x7CNone
    Rules file name:dunno.txt
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[0]
        assert isinstance(module, F.FlagImage)
        expected = (
            (
                "QCFlag",
                F.C_ANY,
                False,
                (
                    ("Intensity_MaxIntensity_DNA", None, 0.95, "foo.txt"),
                    ("Intensity_MinIntensity_Cytoplasm", 0.05, None, "bar.txt"),
                    ("Intensity_MeanIntensity_DNA", 0.1, 0.9, "baz.txt"),
                ),
            ),
            (
                "HighCytoplasmIntensity",
                None,
                True,
                (("Intensity_MeanIntensity_Cytoplasm", None, 0.8, "dunno.txt"),),
            ),
        )
        assert len(expected) == module.flag_count.value
        for flag, (feature_name, combine, skip, measurements) in zip(
            module.flags, expected
        ):
            assert isinstance(flag, cps.SettingsGroup)
            assert flag.category == "Metadata"
            assert flag.feature_name == feature_name
            assert flag.wants_skip == skip
            if combine is not None:
                assert flag.combination_choice == combine
            assert len(measurements) == flag.measurement_count.value
            for (
                measurement,
                (measurement_name, min_value, max_value, rules_file),
            ) in zip(flag.measurement_settings, measurements):
                assert isinstance(measurement, cps.SettingsGroup)
                assert measurement.source_choice == F.S_IMAGE
                assert measurement.measurement == measurement_name
                assert measurement.wants_minimum.value == (min_value is not None)
                if measurement.wants_minimum.value:
                    assert (
                        round(abs(measurement.minimum_value.value - min_value), 7) == 0
                    )
                assert measurement.wants_maximum.value == (max_value is not None)
                if measurement.wants_maximum.value:
                    assert (
                        round(abs(measurement.maximum_value.value - max_value), 7) == 0
                    )
                assert measurement.rules_file_name == rules_file
                assert measurement.rules_class == "1"

    def test_load_v4(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:2
DateRevision:20120306205005

FlagImage:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:4|show_window:True|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)]
    Hidden:2
    Hidden:3
    Name the flag\'s category:Metadata
    Name the flag:QCFlag
    Flag if any, or all, measurement(s) fails to meet the criteria?:Flag if any fail
    Skip image set if flagged?:No
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MaxIntensity_DNA
    Flag images based on low values?:No
    Minimum value:0.0
    Flag images based on high values?:Yes
    Maximum value:0.95
    Rules file location:Default Input Folder\x7CNone
    Rules file name:foo.txt
    Rules class:4
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MinIntensity_Cytoplasm
    Flag images based on low values?:Yes
    Minimum value:0.05
    Flag images based on high values?:No
    Maximum value:1.0
    Rules file location:Default Input Folder\x7CNone
    Rules file name:bar.txt
    Rules class:2
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MeanIntensity_DNA
    Flag images based on low values?:Yes
    Minimum value:0.1
    Flag images based on high values?:Yes
    Maximum value:0.9
    Rules file location:Default Input Folder\x7CNone
    Rules file name:baz.txt
    Rules class:1
    Hidden:1
    Name the flag\'s category:Metadata
    Name the flag:HighCytoplasmIntensity
    Flag if any, or all, measurement(s) fails to meet the criteria?:Flag if any fail
    Skip image set if flagged?:Yes
    Flag is based on:Whole-image measurement
    Select the object whose measurements will be used to flag:None
    Which measurement?:Intensity_MeanIntensity_Cytoplasm
    Flag images based on low values?:No
    Minimum value:0.0
    Flag images based on high values?:Yes
    Maximum value:.8
    Rules file location:Default Input Folder\x7CNone
    Rules file name:dunno.txt
    Rules class:3
"""
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.LoadExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.load(StringIO(data))
        assert len(pipeline.modules()) == 1
        module = pipeline.modules()[0]
        assert isinstance(module, F.FlagImage)
        expected = (
            (
                "QCFlag",
                F.C_ANY,
                False,
                (
                    ("Intensity_MaxIntensity_DNA", None, 0.95, "foo.txt", "4"),
                    ("Intensity_MinIntensity_Cytoplasm", 0.05, None, "bar.txt", "2"),
                    ("Intensity_MeanIntensity_DNA", 0.1, 0.9, "baz.txt", "1"),
                ),
            ),
            (
                "HighCytoplasmIntensity",
                None,
                True,
                (("Intensity_MeanIntensity_Cytoplasm", None, 0.8, "dunno.txt", "3"),),
            ),
        )
        assert len(expected) == module.flag_count.value
        for flag, (feature_name, combine, skip, measurements) in zip(
            module.flags, expected
        ):
            assert isinstance(flag, cps.SettingsGroup)
            assert flag.category == "Metadata"
            assert flag.feature_name == feature_name
            assert flag.wants_skip == skip
            if combine is not None:
                assert flag.combination_choice == combine
            assert len(measurements) == flag.measurement_count.value
            for (
                measurement,
                (measurement_name, min_value, max_value, rules_file, rules_class),
            ) in zip(flag.measurement_settings, measurements):
                assert isinstance(measurement, cps.SettingsGroup)
                assert measurement.source_choice == F.S_IMAGE
                assert measurement.measurement == measurement_name
                assert measurement.wants_minimum.value == (min_value is not None)
                if measurement.wants_minimum.value:
                    assert (
                        round(abs(measurement.minimum_value.value - min_value), 7) == 0
                    )
                assert measurement.wants_maximum.value == (max_value is not None)
                if measurement.wants_maximum.value:
                    assert (
                        round(abs(measurement.maximum_value.value - max_value), 7) == 0
                    )
                assert measurement.rules_file_name == rules_file
                assert measurement.rules_class == rules_class

    def make_workspace(self, image_measurements, object_measurements):
        """Make a workspace with a FlagImage module and the given measurements

        image_measurements - a sequence of single image measurements. Use
                             image_measurement_name(i) to get the name of
                             the i th measurement
        object_measurements - a seequence of sequences of object measurements.
                              These are stored under object, OBJECT_NAME with
                              measurement name object_measurement_name(i) for
                              the i th measurement.

        returns module, workspace
        """
        module = F.FlagImage()
        measurements = cpmeas.Measurements()
        for i in range(len(image_measurements)):
            measurements.add_image_measurement(
                image_measurement_name(i), image_measurements[i]
            )
        for i in range(len(object_measurements)):
            measurements.add_measurement(
                OBJECT_NAME,
                object_measurement_name(i),
                np.array(object_measurements[i]),
            )
        flag = module.flags[0]
        assert isinstance(flag, cps.SettingsGroup)
        flag.category.value = MEASUREMENT_CATEGORY
        flag.feature_name.value = MEASUREMENT_FEATURE
        module.set_module_num(1)
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            assert not isinstance(event, cpp.RunExceptionEvent)

        pipeline.add_listener(callback)
        pipeline.add_module(module)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        workspace = cpw.Workspace(
            pipeline, module, image_set, cpo.ObjectSet(), measurements, image_set_list
        )
        return module, workspace

    @contextlib.contextmanager
    def make_classifier(
        self,
        module,
        answer,
        classes=None,
        class_names=None,
        rules_classes=None,
        name="Classifier",
        n_features=1,
    ):
        assert isinstance(module, F.FlagImage)
        feature_names = [image_measurement_name(i) for i in range(n_features)]
        if classes is None:
            classes = np.arange(1, max(3, answer + 1))
        if class_names is None:
            class_names = ["Class%d" for _ in classes]
        if rules_classes is None:
            rules_classes = [class_names[0]]
        s = make_classifier_pickle(
            np.array([answer]), classes, class_names, name, feature_names
        )
        fd, filename = tempfile.mkstemp(".model")
        os.write(fd, s)
        os.close(fd)
        measurement = module.flags[0].measurement_settings[0]
        measurement.source_choice.value = F.S_CLASSIFIER
        measurement.rules_directory.set_custom_path(os.path.dirname(filename))
        measurement.rules_file_name.value = os.path.split(filename)[1]
        measurement.rules_class.value = rules_classes
        yield
        try:
            os.remove(filename)
        except:
            pass

    def test_positive_image_measurement(self):
        module, workspace = self.make_workspace([1], [])
        flag = module.flags[0]
        assert isinstance(flag, cps.SettingsGroup)
        measurement = flag.measurement_settings[0]
        assert isinstance(measurement, cps.SettingsGroup)
        measurement.measurement.value = image_measurement_name(0)
        measurement.wants_minimum.value = False
        measurement.wants_maximum.value = True
        measurement.maximum_value.value = 0.95
        module.run(workspace)
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        assert MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE)
        assert m.get_current_image_measurement(MEASUREMENT_NAME) == 1
        assert workspace.disposition == cpw.DISPOSITION_CONTINUE

    def test_negative_image_measurement(self):
        module, workspace = self.make_workspace([1], [])
        flag = module.flags[0]
        assert isinstance(flag, cps.SettingsGroup)
        measurement = flag.measurement_settings[0]
        assert isinstance(measurement, cps.SettingsGroup)
        measurement.measurement.value = image_measurement_name(0)
        measurement.wants_minimum.value = True
        measurement.minimum_value.value = 0.1
        measurement.wants_maximum.value = False
        module.run(workspace)
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        assert MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE)
        assert m.get_current_image_measurement(MEASUREMENT_NAME) == 0

    def test_no_ave_object_measurement(self):
        for case in ("minimum", "maximum"):
            module, workspace = self.make_workspace([], [[]])
            flag = module.flags[0]
            assert isinstance(flag, cps.SettingsGroup)
            measurement = flag.measurement_settings[0]
            assert isinstance(measurement, cps.SettingsGroup)
            measurement.source_choice.value = F.S_AVERAGE_OBJECT
            measurement.object_name.value = OBJECT_NAME
            measurement.measurement.value = object_measurement_name(0)
            if case == "minimum":
                measurement.wants_maximum.value = False
                measurement.wants_minimum.value = True
                measurement.minimum_value.value = 0.3
            else:
                measurement.wants_minimum.value = False
                measurement.wants_maximum.value = True
                measurement.maximum_value.value = 0.2
            module.run(workspace)
            m = workspace.measurements
            assert isinstance(m, cpmeas.Measurements)
            assert MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE)
            assert m.get_current_image_measurement(MEASUREMENT_NAME) == 1

    def test_positive_ave_object_measurement(self):
        for case in ("minimum", "maximum"):
            module, workspace = self.make_workspace([], [[0.1, 0.2, 0.3, 0.4]])
            flag = module.flags[0]
            assert isinstance(flag, cps.SettingsGroup)
            measurement = flag.measurement_settings[0]
            assert isinstance(measurement, cps.SettingsGroup)
            measurement.source_choice.value = F.S_AVERAGE_OBJECT
            measurement.object_name.value = OBJECT_NAME
            measurement.measurement.value = object_measurement_name(0)
            if case == "minimum":
                measurement.wants_maximum.value = False
                measurement.wants_minimum.value = True
                measurement.minimum_value.value = 0.3
            else:
                measurement.wants_minimum.value = False
                measurement.wants_maximum.value = True
                measurement.maximum_value.value = 0.2
            module.run(workspace)
            m = workspace.measurements
            assert isinstance(m, cpmeas.Measurements)
            assert MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE)
            assert m.get_current_image_measurement(MEASUREMENT_NAME) == 1

    def test_negative_ave_object_measurement(self):
        for case in ("minimum", "maximum"):
            module, workspace = self.make_workspace([], [[0.1, 0.2, 0.3, 0.4]])
            flag = module.flags[0]
            assert isinstance(flag, cps.SettingsGroup)
            measurement = flag.measurement_settings[0]
            assert isinstance(measurement, cps.SettingsGroup)
            measurement.source_choice.value = F.S_AVERAGE_OBJECT
            measurement.object_name.value = OBJECT_NAME
            measurement.measurement.value = object_measurement_name(0)
            if case == "minimum":
                measurement.wants_maximum.value = False
                measurement.wants_minimum.value = True
                measurement.minimum_value.value = 0.2
            else:
                measurement.wants_minimum.value = False
                measurement.wants_maximum.value = True
                measurement.maximum_value.value = 0.3
            module.run(workspace)
            m = workspace.measurements
            assert isinstance(m, cpmeas.Measurements)
            assert MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE)
            assert m.get_current_image_measurement(MEASUREMENT_NAME) == 0

    def test_no_object_measurements(self):
        for case in ("minimum", "maximum"):
            module, workspace = self.make_workspace([], [[]])
            flag = module.flags[0]
            assert isinstance(flag, cps.SettingsGroup)
            measurement = flag.measurement_settings[0]
            assert isinstance(measurement, cps.SettingsGroup)
            measurement.source_choice.value = F.S_ALL_OBJECTS
            measurement.object_name.value = OBJECT_NAME
            measurement.measurement.value = object_measurement_name(0)
            if case == "maximum":
                measurement.wants_minimum.value = False
                measurement.wants_maximum.value = True
                measurement.maximum_value.value = 0.35
            else:
                measurement.wants_maximum.value = False
                measurement.wants_minimum.value = True
                measurement.minimum_value.value = 0.15
            module.run(workspace)
            m = workspace.measurements
            assert isinstance(m, cpmeas.Measurements)
            assert MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE)
            assert m.get_current_image_measurement(MEASUREMENT_NAME) == 1

    def test_positive_object_measurement(self):
        for case in ("minimum", "maximum"):
            module, workspace = self.make_workspace([], [[0.1, 0.2, 0.3, 0.4]])
            flag = module.flags[0]
            assert isinstance(flag, cps.SettingsGroup)
            measurement = flag.measurement_settings[0]
            assert isinstance(measurement, cps.SettingsGroup)
            measurement.source_choice.value = F.S_ALL_OBJECTS
            measurement.object_name.value = OBJECT_NAME
            measurement.measurement.value = object_measurement_name(0)
            if case == "maximum":
                measurement.wants_minimum.value = False
                measurement.wants_maximum.value = True
                measurement.maximum_value.value = 0.35
            else:
                measurement.wants_maximum.value = False
                measurement.wants_minimum.value = True
                measurement.minimum_value.value = 0.15
            module.run(workspace)
            m = workspace.measurements
            assert isinstance(m, cpmeas.Measurements)
            assert MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE)
            assert m.get_current_image_measurement(MEASUREMENT_NAME) == 1

    def test_negative_object_measurement(self):
        for case in ("minimum", "maximum"):
            module, workspace = self.make_workspace([], [[0.1, 0.2, 0.3, 0.4]])
            flag = module.flags[0]
            assert isinstance(flag, cps.SettingsGroup)
            measurement = flag.measurement_settings[0]
            assert isinstance(measurement, cps.SettingsGroup)
            measurement.source_choice.value = F.S_ALL_OBJECTS
            measurement.object_name.value = OBJECT_NAME
            measurement.measurement.value = object_measurement_name(0)
            if case == "maximum":
                measurement.wants_minimum.value = False
                measurement.wants_maximum.value = True
                measurement.maximum_value.value = 0.45
            else:
                measurement.wants_maximum.value = False
                measurement.wants_minimum.value = True
                measurement.minimum_value.value = 0.05
            module.run(workspace)
            m = workspace.measurements
            assert isinstance(m, cpmeas.Measurements)
            assert MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE)
            assert m.get_current_image_measurement(MEASUREMENT_NAME) == 0

    def test_two_measurements_any(self):
        for measurements, expected in (
            ((0, 0), 0),
            ((0, 1), 1),
            ((1, 0), 1),
            ((1, 1), 1),
        ):
            module, workspace = self.make_workspace(measurements, [])
            flag = module.flags[0]
            assert isinstance(flag, cps.SettingsGroup)
            flag.combination_choice.value = F.C_ANY
            module.add_measurement(flag)
            for i in range(2):
                measurement = flag.measurement_settings[i]
                assert isinstance(measurement, cps.SettingsGroup)
                measurement.measurement.value = image_measurement_name(i)
                measurement.wants_minimum.value = False
                measurement.wants_maximum.value = True
                measurement.maximum_value.value = 0.5
            module.run(workspace)
            m = workspace.measurements
            assert isinstance(m, cpmeas.Measurements)
            assert MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE)
            assert m.get_current_image_measurement(MEASUREMENT_NAME) == expected

    def test_two_measurements_all(self):
        for measurements, expected in (
            ((0, 0), 0),
            ((0, 1), 0),
            ((1, 0), 0),
            ((1, 1), 1),
        ):
            module, workspace = self.make_workspace(measurements, [])
            flag = module.flags[0]
            assert isinstance(flag, cps.SettingsGroup)
            flag.combination_choice.value = F.C_ALL
            module.add_measurement(flag)
            for i in range(2):
                measurement = flag.measurement_settings[i]
                assert isinstance(measurement, cps.SettingsGroup)
                measurement.measurement.value = image_measurement_name(i)
                measurement.wants_minimum.value = False
                measurement.wants_maximum.value = True
                measurement.maximum_value.value = 0.5
            module.run(workspace)
            m = workspace.measurements
            assert isinstance(m, cpmeas.Measurements)
            assert MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE)
            assert m.get_current_image_measurement(MEASUREMENT_NAME) == expected

    def test_get_measurement_columns(self):
        module = F.FlagImage()
        module.add_flag()
        module.flags[0].category.value = "Foo"
        module.flags[0].feature_name.value = "Bar"
        module.flags[1].category.value = "Hello"
        module.flags[1].feature_name.value = "World"
        columns = module.get_measurement_columns(None)
        assert len(columns) == 2
        assert all(
            [
                column[0] == cpmeas.IMAGE
                and column[1] in ("Foo_Bar", "Hello_World")
                and column[2] == cpmeas.COLTYPE_INTEGER
                for column in columns
            ]
        )
        assert columns[0][1] != columns[1][1]
        categories = module.get_categories(None, "foo")
        assert len(categories) == 0
        categories = module.get_categories(None, cpmeas.IMAGE)
        assert len(categories) == 2
        assert "Foo" in categories
        assert "Hello" in categories
        assert len(module.get_measurements(None, cpmeas.IMAGE, "Whatever")) == 0
        for category, feature in (("Foo", "Bar"), ("Hello", "World")):
            features = module.get_measurements(None, cpmeas.IMAGE, category)
            assert len(features) == 1
            assert features[0] == feature

    def test_skip(self):
        module, workspace = self.make_workspace([1], [])
        flag = module.flags[0]
        assert isinstance(flag, cps.SettingsGroup)
        flag.wants_skip.value = True
        measurement = flag.measurement_settings[0]
        assert isinstance(measurement, cps.SettingsGroup)
        measurement.measurement.value = image_measurement_name(0)
        measurement.wants_minimum.value = False
        measurement.wants_maximum.value = True
        measurement.maximum_value.value = 0.95
        module.run(workspace)
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        assert MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE)
        assert m.get_current_image_measurement(MEASUREMENT_NAME) == 1
        assert workspace.disposition == cpw.DISPOSITION_SKIP

    def test_dont_skip(self):
        module, workspace = self.make_workspace([1], [])
        flag = module.flags[0]
        assert isinstance(flag, cps.SettingsGroup)
        flag.wants_skip.value = True
        measurement = flag.measurement_settings[0]
        assert isinstance(measurement, cps.SettingsGroup)
        measurement.measurement.value = image_measurement_name(0)
        measurement.wants_minimum.value = True
        measurement.minimum_value.value = 0.1
        measurement.wants_maximum.value = False
        module.run(workspace)
        m = workspace.measurements
        assert isinstance(m, cpmeas.Measurements)
        assert MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE)
        assert m.get_current_image_measurement(MEASUREMENT_NAME) == 0
        assert workspace.disposition == cpw.DISPOSITION_CONTINUE

    def test_filter_by_rule(self):
        rules_file_contents = "IF (%s > 2.0, [1.0,-1.0], [-1.0,1.0])\n" % (
            "_".join((cpmeas.IMAGE, image_measurement_name(0)))
        )
        rules_path = tempfile.mktemp()
        rules_dir, rules_file = os.path.split(rules_path)
        fd = open(rules_path, "wt")
        try:
            fd.write(rules_file_contents)
            fd.close()
            for value, choice, expected in (
                (1.0, 1, 0),
                (3.0, 1, 1),
                (1.0, 2, 1),
                (3.0, 2, 0),
            ):
                module, workspace = self.make_workspace([value], [])
                flag = module.flags[0]
                assert isinstance(flag, cps.SettingsGroup)
                flag.wants_skip.value = False
                measurement = flag.measurement_settings[0]
                assert isinstance(measurement, cps.SettingsGroup)
                measurement.source_choice.value = F.S_RULES
                measurement.rules_file_name.value = rules_file
                measurement.rules_directory.dir_choice = cpprefs.ABSOLUTE_FOLDER_NAME
                measurement.rules_directory.custom_path = rules_dir
                measurement.rules_class.set_value([str(choice)])
                module.run(workspace)
                m = workspace.measurements
                assert isinstance(m, cpmeas.Measurements)
                assert MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE)
                assert m.get_current_image_measurement(MEASUREMENT_NAME) == expected
        finally:
            os.remove(rules_path)

    def test_filter_by_3class_rule(self):
        f = "_".join((cpmeas.IMAGE, image_measurement_name(0)))
        rules_file_contents = (
            "IF (%(f)s > 2.0, [1.0,-1.0,-1.0], [-0.5,0.5,0.5])\n"
            "IF (%(f)s > 1.6, [0.5,0.5,-0.5], [-1.0,-1.0,1.0])\n"
        ) % locals()
        measurement_values = [1.5, 2.3, 1.8]
        expected_classes = ["3", "1", "2"]
        rules_path = tempfile.mktemp()
        rules_dir, rules_file = os.path.split(rules_path)
        fd = open(rules_path, "wt")
        fd.write(rules_file_contents)
        fd.close()
        try:
            for rules_classes in (
                ["1"],
                ["2"],
                ["3"],
                ["1", "2"],
                ["1", "3"],
                ["2", "3"],
            ):
                for expected_class, measurement_value in zip(
                    expected_classes, measurement_values
                ):
                    module, workspace = self.make_workspace([measurement_value], [])
                    flag = module.flags[0]
                    assert isinstance(flag, cps.SettingsGroup)
                    flag.wants_skip.value = False
                    measurement = flag.measurement_settings[0]
                    assert isinstance(measurement, cps.SettingsGroup)
                    measurement.source_choice.value = F.S_RULES
                    measurement.rules_file_name.value = rules_file
                    measurement.rules_directory.dir_choice = (
                        cpprefs.ABSOLUTE_FOLDER_NAME
                    )
                    measurement.rules_directory.custom_path = rules_dir
                    measurement.rules_class.set_value(rules_classes)

                    m = workspace.measurements
                    assert isinstance(m, cpmeas.Measurements)
                    module.run(workspace)
                    assert MEASUREMENT_NAME in m.get_feature_names(cpmeas.IMAGE)
                    value = m.get_current_image_measurement(MEASUREMENT_NAME)
                    expected_value = 1 if expected_class in rules_classes else 0
                    assert value == expected_value
        finally:
            os.remove(rules_path)

    def test_classify_true(self):
        module, workspace = self.make_workspace([1], [])
        with self.make_classifier(module, 1):
            module.run(workspace)
            m = workspace.measurements
            assert m[cpmeas.IMAGE, MEASUREMENT_NAME] == 1

    def test_classify_false(self):
        module, workspace = self.make_workspace([1], [])
        with self.make_classifier(module, 2):
            module.run(workspace)
            m = workspace.measurements
            assert m[cpmeas.IMAGE, MEASUREMENT_NAME] == 0

    def test_classify_multiple_select_true(self):
        module, workspace = self.make_workspace([1], [])
        with self.make_classifier(
            module,
            2,
            classes=[1, 2, 3],
            class_names=["Foo", "Bar", "Baz"],
            rules_classes=["Bar", "Baz"],
        ):
            module.run(workspace)
            m = workspace.measurements
            assert m[cpmeas.IMAGE, MEASUREMENT_NAME] == 1

    def test_classify_multiple_select_false(self):
        module, workspace = self.make_workspace([1], [])
        with self.make_classifier(
            module,
            2,
            classes=[1, 2, 3],
            class_names=["Foo", "Bar", "Baz"],
            rules_classes=["Foo", "Baz"],
        ):
            module.run(workspace)
            m = workspace.measurements
            assert m[cpmeas.IMAGE, MEASUREMENT_NAME] == 0

    def test_batch(self):
        orig_path = "/foo/bar"

        def fn_alter_path(path, **varargs):
            assert path == orig_path
            return "/imaging/analysis"

        module = F.FlagImage()
        rd = module.flags[0].measurement_settings[0].rules_directory
        rd.dir_choice = cps.ABSOLUTE_FOLDER_NAME
        rd.custom_path = orig_path
        module.prepare_to_create_batch(None, fn_alter_path)
        assert rd.custom_path == "/imaging/analysis"
