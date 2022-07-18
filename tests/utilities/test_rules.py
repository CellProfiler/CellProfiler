"""test_rules - test the CPA rules parser
"""

import unittest
from io import StringIO

import numpy as np
import pytest

import cellprofiler_core.measurement
import cellprofiler.utilities.rules as R

OBJECT_NAME = "MyObject"
M_FEATURES = ["Measurement%d" % i for i in range(1, 11)]
MISSPELLED_FEATURE = "Measuremen11"


class TestRules(unittest.TestCase):
    def test_01_01_load_rules(self):
        data = """IF (Nuclei_Intensity_UpperQuartileIntensity_CorrDend > 0.12762499999999999, [0.79607587785712131, -0.79607587785712131], [-0.94024303819690347, 0.94024303819690347])
IF (Nuclei_Intensity_MinIntensity_CorrAxon > 0.026831299999999999, [0.68998040630066426, -0.68998040630066426], [-0.80302016375137986, 0.80302016375137986])
IF (Nuclei_Intensity_UpperQuartileIntensity_CorrDend > 0.19306000000000001, [0.71934712791500899, -0.71934712791500899], [-0.47379648809429048, 0.47379648809429048])
IF (Nuclei_Intensity_UpperQuartileIntensity_CorrDend > 0.100841, [0.24553066971563919, -0.24553066971563919], [-1.0, 1.0])
IF (Nuclei_Location_Center_Y > 299.32499999999999, [-0.61833689824912363, 0.61833689824912363], [0.33384091087462237, -0.33384091087462237])
IF (Nuclei_Intensity_MaxIntensity_CorrNuclei > 0.76509499999999997, [-0.95683305069726121, 0.95683305069726121], [0.22816438860290775, -0.22816438860290775])
IF (Nuclei_Neighbors_NumberOfNeighbors_5 > 2.0, [-0.92848043428127192, 0.92848043428127192], [0.20751332434602923, -0.20751332434602923])
IF (Nuclei_Intensity_MedianIntensity_CorrDend > 0.15327499999999999, [0.79784566567342285, -0.79784566567342285], [-0.35665314560825129, 0.35665314560825129])
IF (Nuclei_Neighbors_SecondClosestXVector_5 > -11.5113, [-0.21859862538067179, 0.21859862538067179], [0.71270785008847592, -0.71270785008847592])
IF (Nuclei_Intensity_StdIntensity_CorrDend > 0.035382299999999998, [0.28838229530755011, -0.28838229530755011], [-0.75312050069265968, 0.75312050069265968])
IF (Nuclei_Intensity_MaxIntensityEdge_CorrNuclei > 0.63182899999999997, [-0.93629855522957672, 0.93629855522957672], [0.1710257492070047, -0.1710257492070047])
IF (Nuclei_Intensity_StdIntensityEdge_CorrNuclei > 0.037909400000000003, [0.28514731668218346, -0.28514731668218346], [-0.60783543053602795, 0.60783543053602795])
IF (Nuclei_Intensity_MedianIntensity_CorrAxon > 0.042631500000000003, [0.20227787378316109, -0.20227787378316109], [-0.78282539096589077, 0.78282539096589077])
IF (Nuclei_Intensity_MinIntensity_CorrDend > 0.042065400000000003, [0.52616744135942872, -0.52616744135942872], [-0.32613209033812068, 0.32613209033812068])
IF (Nuclei_Neighbors_FirstClosestYVector_5 > 3.8226100000000001, [0.69128399165300047, -0.69128399165300047], [-0.34874605597401531, 0.34874605597401531])
IF (Nuclei_Intensity_MeanIntensity_CorrNuclei > 0.283188, [-0.79881507037552979, 0.79881507037552979], [0.24825909570051025, -0.24825909570051025])
IF (Nuclei_Location_Center_Y > 280.154, [-0.50545174099468504, 0.50545174099468504], [0.3297202808867149, -0.3297202808867149])
IF (Nuclei_Intensity_UpperQuartileIntensity_CorrDend > 0.132241, [0.35771841831789791, -0.35771841831789791], [-0.63545019489162846, 0.63545019489162846])
IF (Nuclei_AreaShape_MinorAxisLength > 6.4944899999999999, [0.5755128363506562, -0.5755128363506562], [-0.41737581982462335, 0.41737581982462335])
IF (Nuclei_Intensity_LowerQuartileIntensity_CorrDend > 0.075424000000000005, [0.50557978238660795, -0.50557978238660795], [-0.35606081901385256, 0.35606081901385256])
"""
        fd = StringIO(data)
        rules = R.Rules()
        rules.parse(fd)
        self.assertEqual(len(rules.rules), 20)
        for rule in rules.rules:
            self.assertEqual(rule.object_name, "Nuclei")
            self.assertEqual(rule.comparitor, ">")

        rule = rules.rules[0]
        self.assertEqual(rule.feature, "Intensity_UpperQuartileIntensity_CorrDend")
        self.assertAlmostEqual(rule.threshold, 0.127625)
        self.assertAlmostEqual(rule.weights[0, 0], 0.79607587785712131)
        self.assertAlmostEqual(rule.weights[0, 1], -0.79607587785712131)
        self.assertAlmostEqual(rule.weights[1, 0], -0.94024303819690347)
        self.assertAlmostEqual(rule.weights[1, 1], 0.94024303819690347)

    def test_02_00_no_measurements(self):
        m = cellprofiler_core.measurement.Measurements()
        m.add_measurement(OBJECT_NAME, M_FEATURES[0], np.array([], float))
        rules = R.Rules()
        rules.rules += [
            R.Rules.Rule(
                OBJECT_NAME, M_FEATURES[0], ">", 0, np.array([[1.0, -1.0], [-1.0, 1.0]]), False, 1
            )
        ]
        score = rules.score(m)
        self.assertEqual(score.shape[0], 0)
        self.assertEqual(score.shape[1], 2)

    def test_02_01_score_one_positive(self):
        m = cellprofiler_core.measurement.Measurements()
        m.add_measurement(OBJECT_NAME, M_FEATURES[0], np.array([1.5], float))
        rules = R.Rules()
        rules.rules += [
            R.Rules.Rule(
                OBJECT_NAME, M_FEATURES[0], ">", 0, np.array([[1.0, -0.5], [-2.0, 0.6]]), False, 1
            )
        ]
        score = rules.score(m)
        self.assertEqual(score.shape[0], 1)
        self.assertEqual(score.shape[1], 2)
        self.assertAlmostEqual(score[0, 0], 1.0)
        self.assertAlmostEqual(score[0, 1], -0.5)

    def test_02_02_score_one_negative(self):
        m = cellprofiler_core.measurement.Measurements()
        m.add_measurement(OBJECT_NAME, M_FEATURES[0], np.array([1.5], float))
        rules = R.Rules()
        rules.rules += [
            R.Rules.Rule(
                OBJECT_NAME,
                M_FEATURES[0],
                ">",
                2.0,
                np.array([[1.0, -0.5], [-2.0, 0.6]]),
                False, 
                1
            )
        ]
        score = rules.score(m)
        self.assertEqual(score.shape[0], 1)
        self.assertEqual(score.shape[1], 2)
        self.assertAlmostEqual(score[0, 0], -2.0)
        self.assertAlmostEqual(score[0, 1], 0.6)

    def test_02_03_score_one_nan(self):
        m = cellprofiler_core.measurement.Measurements()
        m.add_measurement(OBJECT_NAME, M_FEATURES[0], np.array([np.NaN], float))
        rules = R.Rules()
        rules.rules += [
            R.Rules.Rule(
                OBJECT_NAME,
                M_FEATURES[0],
                ">",
                2.0,
                np.array([[1.0, -0.5], [-2.0, 0.6]]), 
                False, 
                1
            )
        ]
        score = rules.score(m)
        self.assertEqual(score.shape[0], 1)
        self.assertEqual(score.shape[1], 2)
        self.assertTrue(score[0, 0], -2)
        self.assertTrue(score[0, 1], 0.6)

    def test_02_04_score_one_positive_fuzzy(self):
        m = cellprofiler_core.measurement.Measurements()
        m.add_measurement(OBJECT_NAME, M_FEATURES[0], np.array([1.5], float))
        rules = R.Rules()
        rules.rules += [
            R.Rules.Rule(
                OBJECT_NAME, MISSPELLED_FEATURE, ">", 0, np.array([[1.0, -0.5], [-2.0, 0.6]]), True, 0.7
            )
        ]
        score = rules.score(m)
        self.assertEqual(score.shape[0], 1)
        self.assertEqual(score.shape[1], 2)
        self.assertAlmostEqual(score[0, 0], 1.0)
        self.assertAlmostEqual(score[0, 1], -0.5)
        rules = R.Rules()
        rules.rules += [
            R.Rules.Rule(
                OBJECT_NAME, MISSPELLED_FEATURE, ">", 0, np.array([[1.0, -0.5], [-2.0, 0.6]]), False, 1
            )
        ]
        with pytest.raises(AssertionError):
            score = rules.score(m)


    def test_03_01_score_two_rules(self):
        m = cellprofiler_core.measurement.Measurements()
        m.add_measurement(OBJECT_NAME, M_FEATURES[0], np.array([1.5], float))
        m.add_measurement(OBJECT_NAME, M_FEATURES[1], np.array([-1.5], float))
        rules = R.Rules()
        rules.rules += [
            R.Rules.Rule(
                OBJECT_NAME, M_FEATURES[0], ">", 0, np.array([[1.0, -0.5], [-2.0, 0.6]]), False, 1
            ),
            R.Rules.Rule(
                OBJECT_NAME, M_FEATURES[1], ">", 0, np.array([[1.5, -0.7], [-2.3, 0.9]]), False, 1
            ),
        ]
        score = rules.score(m)
        self.assertEqual(score.shape[0], 1)
        self.assertEqual(score.shape[1], 2)
        self.assertAlmostEqual(score[0, 0], 1.0 - 2.3)
        self.assertAlmostEqual(score[0, 1], -0.5 + 0.9)

    def test_03_02_score_two_objects(self):
        m = cellprofiler_core.measurement.Measurements()
        m.add_measurement(OBJECT_NAME, M_FEATURES[0], np.array([1.5, 2.5], float))
        rules = R.Rules()
        rules.rules += [
            R.Rules.Rule(
                OBJECT_NAME,
                M_FEATURES[0],
                "<",
                2.0,
                np.array([[1.0, -0.5], [-2.0, 0.6]]), 
                False, 
                1
            )
        ]
        score = rules.score(m)
        self.assertEqual(score.shape[0], 2)
        self.assertEqual(score.shape[1], 2)
        self.assertAlmostEqual(score[0, 0], 1.0)
        self.assertAlmostEqual(score[0, 1], -0.5)
        self.assertAlmostEqual(score[1, 0], -2.0)
        self.assertAlmostEqual(score[1, 1], 0.6)
