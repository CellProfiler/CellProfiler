import os
import tempfile

import numpy
import six.moves

import cellprofiler_core.image
import cellprofiler_core.measurement


import cellprofiler.modules.calculatestatistics
import cellprofiler_core.object
import cellprofiler_core.pipeline
import cellprofiler_core.preferences
import cellprofiler_core.setting
import cellprofiler_core.workspace

import tests.frontend.modules

INPUT_OBJECTS = "my_object"
TEST_FTR = "my_measurement"
FIGURE_NAME = "figname"


def test_load_v2():
    path = tests.frontend.modules.get_test_resources_directory("calculatestatistics/v2.pipeline")
    with open(path, "r") as fd:
        data = fd.read()

    pipeline = cellprofiler_core.pipeline.Pipeline()

    def callback(caller, event):
        assert not isinstance(event, cellprofiler_core.pipeline.event.LoadException)

    pipeline.add_listener(callback)
    pipeline.load(six.moves.StringIO(data))
    assert len(pipeline.modules()) == 1
    module = pipeline.modules()[0]
    assert isinstance(
        module, cellprofiler.modules.calculatestatistics.CalculateStatistics
    )
    assert module.grouping_values == "Metadata_Controls"
    assert len(module.dose_values) == 1
    dv = module.dose_values[0]
    assert dv.measurement == "Metadata_SBS_Doses"
    assert not dv.log_transform
    assert dv.wants_save_figure
    assert dv.figure_name == "DoseResponsePlot"
    assert (
        dv.pathname.dir_choice
        == cellprofiler_core.preferences.DEFAULT_OUTPUT_FOLDER_NAME
    )
    assert dv.pathname.custom_path == "Test"


# def test_compare_to_matlab():
#     expected = {
#         'EC50_DistCytoplasm_Correlation_Correlation_CorrGreenCorrBlue':3.982812,
#         'EC50_DistCytoplasm_Intensity_LowerQuartileIntensity_CorrGreen':4.139827,
#         'EC50_DistCytoplasm_Intensity_MedianIntensity_CorrGreen':4.178600,
#         'EC50_DistCytoplasm_Intensity_MinIntensityEdge_CorrGreen':4.059770,
#         'EC50_DistCytoplasm_Intensity_MinIntensity_CorrGreen':4.066357,
#         'EC50_DistCytoplasm_Math_Ratio1':4.491367,
#         'EC50_DistCytoplasm_Math_Ratio2':3.848722,
#         'EC50_DistCytoplasm_Texture_AngularSecondMoment_CorrGreen_1':4.948056,
#         'EC50_DistCytoplasm_Texture_Entropy_CorrGreen_1':4.687104,
#         'EC50_DistCytoplasm_Texture_InfoMeas2_CorrGreen_1':5.0285,
#         'EC50_DistCytoplasm_Texture_InverseDifferenceMoment_CorrGreen_1':4.319017,
#         'EC50_DistCytoplasm_Texture_SumAverage_CorrGreen_1':4.548876,
#         'EC50_DistCytoplasm_Texture_SumEntropy_CorrGreen_1':4.779139,
#         'EC50_DistCytoplasm_Texture_Variance_CorrGreen_1':4.218379,
#         'EC50_DistanceCells_Correlation_Correlation_CorrGreenCorrBlue':3.708711,
#         'EC50_DistanceCells_Intensity_IntegratedIntensityEdge_CorrGreen':4.135146,
#         'EC50_DistanceCells_Intensity_LowerQuartileIntensity_CorrGreen':4.5372,
#         'EC50_DistanceCells_Intensity_MeanIntensityEdge_CorrGreen':4.1371,
#         'EC50_DistanceCells_Intensity_MinIntensityEdge_CorrGreen':4.033999,
#         'EC50_DistanceCells_Intensity_MinIntensity_CorrGreen':4.079470,
#         'EC50_DistanceCells_Texture_AngularSecondMoment_CorrGreen_1':5.118689,
#         'EC50_DistanceCells_Texture_Correlation_CorrGreen_1':4.002074,
#         'EC50_DistanceCells_Texture_Entropy_CorrGreen_1':5.008000,
#         'EC50_DistanceCells_Texture_InfoMeas1_CorrGreen_1':3.883586,
#         'EC50_DistanceCells_Texture_InverseDifferenceMoment_CorrGreen_1':3.977216,
#         'EC50_DistanceCells_Texture_SumAverage_CorrGreen_1':4.9741,
#         'EC50_DistanceCells_Texture_SumEntropy_CorrGreen_1':5.1455,
#         'EC50_DistanceCells_Texture_SumVariance_CorrGreen_1':4.593041,
#         'EC50_DistanceCells_Texture_Variance_CorrGreen_1':4.619517,
#         'EC50_Nuclei_Correlation_Correlation_CorrGreenCorrBlue':3.751133,
#         'EC50_Nuclei_Math_Ratio1':4.491367,
#         'EC50_Nuclei_Math_Ratio2':3.848722,
#         'EC50_Nuclei_Texture_SumAverage_CorrGreen_1':3.765297,
#         'EC50_PropCells_AreaShape_Area':4.740853,
#         'EC50_PropCells_AreaShape_MajorAxisLength':5.064460,
#         'EC50_PropCells_AreaShape_MinorAxisLength':4.751471,
#         'EC50_PropCells_AreaShape_Perimeter':4.949292,
#         'EC50_PropCells_Correlation_Correlation_CorrGreenCorrBlue':3.772565,
#         'EC50_PropCells_Texture_GaborX_CorrGreen_1':5.007167,
#         'EC50_PropCells_Texture_InfoMeas2_CorrBlue_1':4.341353,
#         'EC50_PropCells_Texture_SumVariance_CorrBlue_1':4.298359,
#         'EC50_PropCells_Texture_SumVariance_CorrGreen_1':4.610826,
#         'EC50_PropCells_Texture_Variance_CorrBlue_1':4.396352,
#         'EC50_PropCells_Texture_Variance_CorrGreen_1':4.632468,
#         'EC50_PropCytoplasm_AreaShape_Area':4.669679,
#         'EC50_PropCytoplasm_AreaShape_MinorAxisLength':4.754476,
#         'EC50_PropCytoplasm_AreaShape_Perimeter':4.949292,
#         'EC50_PropCytoplasm_Correlation_Correlation_CorrGreenCorrBlue':4.072830,
#         'EC50_PropCytoplasm_Intensity_IntegratedIntensity_CorrGreen':4.0934,
#         'EC50_PropCytoplasm_Intensity_LowerQuartileIntensity_CorrGreen':3.925800,
#         'EC50_PropCytoplasm_Intensity_MedianIntensity_CorrGreen':3.9252,
#         'EC50_PropCytoplasm_Texture_AngularSecondMoment_CorrGreen_1':4.777481,
#         'EC50_PropCytoplasm_Texture_Entropy_CorrGreen_1':4.4432,
#         'EC50_PropCytoplasm_Texture_GaborX_CorrGreen_1':5.163371,
#         'EC50_PropCytoplasm_Texture_InfoMeas2_CorrGreen_1':4.701046,
#         'EC50_PropCytoplasm_Texture_SumEntropy_CorrGreen_1':4.510543,
#         'EC50_ThresholdedCells_Texture_AngularSecondMoment_CorrBlue_1':4.560315,
#         'EC50_ThresholdedCells_Texture_AngularSecondMoment_CorrGreen_1':4.966674,
#         'EC50_ThresholdedCells_Texture_Entropy_CorrBlue_1':4.457866,
#         'EC50_ThresholdedCells_Texture_InfoMeas2_CorrBlue_1':4.624049,
#         'EC50_ThresholdedCells_Texture_SumAverage_CorrBlue_1':4.686706,
#         'EC50_ThresholdedCells_Texture_SumEntropy_CorrBlue_1':4.537378,
#         'EC50_ThresholdedCells_Texture_SumVariance_CorrBlue_1':4.322820,
#         'EC50_ThresholdedCells_Texture_SumVariance_CorrGreen_1':4.742158,
#         'EC50_ThresholdedCells_Texture_Variance_CorrBlue_1':4.265549,
#         'EC50_ThresholdedCells_Texture_Variance_CorrGreen_1':4.860020,
#         'OneTailedZfactor_DistCytoplasm_Intensity_MedianIntensity_CorrGreen':-4.322503,
#         'OneTailedZfactor_DistCytoplasm_Intensity_MinIntensityEdge_CorrGreen':-4.322503,
#         'OneTailedZfactor_DistCytoplasm_Intensity_MinIntensity_CorrGreen':-4.322503,
#         'OneTailedZfactor_DistCytoplasm_Math_Ratio1':0.622059,
#         'OneTailedZfactor_DistCytoplasm_Math_Ratio2':-4.508284,
#         'OneTailedZfactor_DistCytoplasm_Texture_Entropy_CorrGreen_1':-4.645887,
#         'OneTailedZfactor_DistCytoplasm_Texture_InfoMeas2_CorrGreen_1':-4.279118,
#         'OneTailedZfactor_DistCytoplasm_Texture_SumAverage_CorrGreen_1':-4.765570,
#         'OneTailedZfactor_DistCytoplasm_Texture_SumEntropy_CorrGreen_1':-4.682335,
#         'OneTailedZfactor_DistCytoplasm_Texture_Variance_CorrGreen_1':-4.415607,
#         'OneTailedZfactor_DistanceCells_Intensity_MeanIntensityEdge_CorrGreen':-4.200105,
#         'OneTailedZfactor_DistanceCells_Intensity_MinIntensityEdge_CorrGreen':-4.316452,
#         'OneTailedZfactor_DistanceCells_Intensity_MinIntensity_CorrGreen':-4.316452,
#         'OneTailedZfactor_DistanceCells_Texture_Correlation_CorrGreen_1':0.202500,
#         'OneTailedZfactor_DistanceCells_Texture_Entropy_CorrGreen_1':-4.404815,
#         'OneTailedZfactor_DistanceCells_Texture_InfoMeas1_CorrGreen_1':-4.508513,
#         'OneTailedZfactor_DistanceCells_Texture_SumAverage_CorrGreen_1':-4.225356,
#         'OneTailedZfactor_DistanceCells_Texture_SumEntropy_CorrGreen_1':-4.382768,
#         'OneTailedZfactor_DistanceCells_Texture_SumVariance_CorrGreen_1':0.492125,
#         'OneTailedZfactor_DistanceCells_Texture_Variance_CorrGreen_1':0.477360,
#         'OneTailedZfactor_Nuclei_Correlation_Correlation_CorrGreenCorrBlue':0.563780,
#         'OneTailedZfactor_Nuclei_Math_Ratio1':0.622059,
#         'OneTailedZfactor_Nuclei_Math_Ratio2':-4.508284,
#         'OneTailedZfactor_Nuclei_Texture_SumAverage_CorrGreen_1':0.426178,
#         'OneTailedZfactor_PropCells_AreaShape_Area':-4.216674,
#         'OneTailedZfactor_PropCells_AreaShape_MajorAxisLength':-4.119131,
#         'OneTailedZfactor_PropCells_AreaShape_MinorAxisLength':-4.109793,
#         'OneTailedZfactor_PropCells_AreaShape_Perimeter':-4.068050,
#         'OneTailedZfactor_PropCells_Correlation_Correlation_CorrGreenCorrBlue':0.765440,
#         'OneTailedZfactor_PropCells_Texture_GaborX_CorrGreen_1':0.114982,
#         'OneTailedZfactor_PropCells_Texture_InfoMeas2_CorrBlue_1':0.108409,
#         'OneTailedZfactor_PropCells_Texture_SumVariance_CorrBlue_1':0.191251,
#         'OneTailedZfactor_PropCells_Texture_SumVariance_CorrGreen_1':0.559865,
#         'OneTailedZfactor_PropCells_Texture_Variance_CorrBlue_1':0.254078,
#         'OneTailedZfactor_PropCells_Texture_Variance_CorrGreen_1':0.556108,
#         'OneTailedZfactor_PropCytoplasm_AreaShape_Area':-4.223021,
#         'OneTailedZfactor_PropCytoplasm_AreaShape_MinorAxisLength':-4.095632,
#         'OneTailedZfactor_PropCytoplasm_AreaShape_Perimeter':-4.068050,
#         'OneTailedZfactor_PropCytoplasm_Intensity_MedianIntensity_CorrGreen':-4.194663,
#         'OneTailedZfactor_PropCytoplasm_Texture_Entropy_CorrGreen_1':-4.443338,
#         'OneTailedZfactor_PropCytoplasm_Texture_GaborX_CorrGreen_1':0.207265,
#         'OneTailedZfactor_PropCytoplasm_Texture_InfoMeas2_CorrGreen_1':-4.297250,
#         'OneTailedZfactor_PropCytoplasm_Texture_SumEntropy_CorrGreen_1':-4.525324,
#         'OneTailedZfactor_ThresholdedCells_Texture_Entropy_CorrBlue_1':0.167795,
#         'OneTailedZfactor_ThresholdedCells_Texture_InfoMeas2_CorrBlue_1':0.067560,
#         'OneTailedZfactor_ThresholdedCells_Texture_SumAverage_CorrBlue_1':0.478527,
#         'OneTailedZfactor_ThresholdedCells_Texture_SumEntropy_CorrBlue_1':0.155119,
#         'OneTailedZfactor_ThresholdedCells_Texture_SumVariance_CorrBlue_1':0.535907,
#         'OneTailedZfactor_ThresholdedCells_Texture_SumVariance_CorrGreen_1':0.572801,
#         'OneTailedZfactor_ThresholdedCells_Texture_Variance_CorrBlue_1':0.423454,
#         'OneTailedZfactor_ThresholdedCells_Texture_Variance_CorrGreen_1':0.440500,
#         'Vfactor_DistCytoplasm_Correlation_Correlation_CorrGreenCorrBlue':0.500429,
#         'Vfactor_DistCytoplasm_Intensity_LowerQuartileIntensity_CorrGreen':0.325675,
#         'Vfactor_DistCytoplasm_Intensity_MedianIntensity_CorrGreen':0.323524,
#         'Vfactor_DistCytoplasm_Intensity_MinIntensityEdge_CorrGreen':0.138487,
#         'Vfactor_DistCytoplasm_Intensity_MinIntensity_CorrGreen':0.128157,
#         'Vfactor_DistCytoplasm_Math_Ratio1':0.503610,
#         'Vfactor_DistCytoplasm_Math_Ratio2':0.319610,
#         'Vfactor_DistCytoplasm_Texture_AngularSecondMoment_CorrGreen_1':0.522880,
#         'Vfactor_DistCytoplasm_Texture_Entropy_CorrGreen_1':0.504303,
#         'Vfactor_DistCytoplasm_Texture_InfoMeas2_CorrGreen_1':0.289432,
#         'Vfactor_DistCytoplasm_Texture_InverseDifferenceMoment_CorrGreen_1':0.234123,
#         'Vfactor_DistCytoplasm_Texture_SumAverage_CorrGreen_1':0.591687,
#         'Vfactor_DistCytoplasm_Texture_SumEntropy_CorrGreen_1':0.520356,
#         'Vfactor_DistCytoplasm_Texture_Variance_CorrGreen_1':-0.007649,
#         'Vfactor_DistanceCells_Correlation_Correlation_CorrGreenCorrBlue':0.761198,
#         'Vfactor_DistanceCells_Intensity_IntegratedIntensityEdge_CorrGreen':0.234655,
#         'Vfactor_DistanceCells_Intensity_LowerQuartileIntensity_CorrGreen':0.252240,
#         'Vfactor_DistanceCells_Intensity_MeanIntensityEdge_CorrGreen':0.195125,
#         'Vfactor_DistanceCells_Intensity_MinIntensityEdge_CorrGreen':0.138299,
#         'Vfactor_DistanceCells_Intensity_MinIntensity_CorrGreen':0.126784,
#         'Vfactor_DistanceCells_Texture_AngularSecondMoment_CorrGreen_1':0.342691,
#         'Vfactor_DistanceCells_Texture_Correlation_CorrGreen_1':0.314396,
#         'Vfactor_DistanceCells_Texture_Entropy_CorrGreen_1':0.311771,
#         'Vfactor_DistanceCells_Texture_InfoMeas1_CorrGreen_1':0.410631,
#         'Vfactor_DistanceCells_Texture_InverseDifferenceMoment_CorrGreen_1':0.170576,
#         'Vfactor_DistanceCells_Texture_SumAverage_CorrGreen_1':0.223147,
#         'Vfactor_DistanceCells_Texture_SumEntropy_CorrGreen_1':0.269519,
#         'Vfactor_DistanceCells_Texture_SumVariance_CorrGreen_1':0.571528,
#         'Vfactor_DistanceCells_Texture_Variance_CorrGreen_1':0.566272,
#         'Vfactor_Nuclei_Correlation_Correlation_CorrGreenCorrBlue':0.705051,
#         'Vfactor_Nuclei_Math_Ratio1':0.503610,
#         'Vfactor_Nuclei_Math_Ratio2':0.319610,
#         'Vfactor_Nuclei_Texture_SumAverage_CorrGreen_1':0.553708,
#         'Vfactor_PropCells_AreaShape_Area':0.340093,
#         'Vfactor_PropCells_AreaShape_MajorAxisLength':0.243838,
#         'Vfactor_PropCells_AreaShape_MinorAxisLength':0.320691,
#         'Vfactor_PropCells_AreaShape_Perimeter':0.238915,
#         'Vfactor_PropCells_Correlation_Correlation_CorrGreenCorrBlue':0.723520,
#         'Vfactor_PropCells_Texture_GaborX_CorrGreen_1':0.213161,
#         'Vfactor_PropCells_Texture_InfoMeas2_CorrBlue_1':0.199791,
#         'Vfactor_PropCells_Texture_SumVariance_CorrBlue_1':0.078959,
#         'Vfactor_PropCells_Texture_SumVariance_CorrGreen_1':0.642844,
#         'Vfactor_PropCells_Texture_Variance_CorrBlue_1':0.199105,
#         'Vfactor_PropCells_Texture_Variance_CorrGreen_1':0.640818,
#         'Vfactor_PropCytoplasm_AreaShape_Area':0.325845,
#         'Vfactor_PropCytoplasm_AreaShape_MinorAxisLength':0.312258,
#         'Vfactor_PropCytoplasm_AreaShape_Perimeter':0.238915,
#         'Vfactor_PropCytoplasm_Correlation_Correlation_CorrGreenCorrBlue':0.337565,
#         'Vfactor_PropCytoplasm_Intensity_IntegratedIntensity_CorrGreen':0.292900,
#         'Vfactor_PropCytoplasm_Intensity_LowerQuartileIntensity_CorrGreen':0.175528,
#         'Vfactor_PropCytoplasm_Intensity_MedianIntensity_CorrGreen':0.193308,
#         'Vfactor_PropCytoplasm_Texture_AngularSecondMoment_CorrGreen_1':0.276152,
#         'Vfactor_PropCytoplasm_Texture_Entropy_CorrGreen_1':0.239567,
#         'Vfactor_PropCytoplasm_Texture_GaborX_CorrGreen_1':0.332380,
#         'Vfactor_PropCytoplasm_Texture_InfoMeas2_CorrGreen_1':0.379141,
#         'Vfactor_PropCytoplasm_Texture_SumEntropy_CorrGreen_1':0.337740,
#         'Vfactor_ThresholdedCells_Texture_AngularSecondMoment_CorrBlue_1':0.334520,
#         'Vfactor_ThresholdedCells_Texture_AngularSecondMoment_CorrGreen_1':0.192882,
#         'Vfactor_ThresholdedCells_Texture_Entropy_CorrBlue_1':0.276245,
#         'Vfactor_ThresholdedCells_Texture_InfoMeas2_CorrBlue_1':0.139166,
#         'Vfactor_ThresholdedCells_Texture_SumAverage_CorrBlue_1':0.465237,
#         'Vfactor_ThresholdedCells_Texture_SumEntropy_CorrBlue_1':0.355399,
#         'Vfactor_ThresholdedCells_Texture_SumVariance_CorrBlue_1':0.453937,
#         'Vfactor_ThresholdedCells_Texture_SumVariance_CorrGreen_1':0.564371,
#         'Vfactor_ThresholdedCells_Texture_Variance_CorrBlue_1':0.360566,
#         'Vfactor_ThresholdedCells_Texture_Variance_CorrGreen_1':0.548770,
#         'Zfactor_DistCytoplasm_Correlation_Correlation_CorrGreenCorrBlue':0.531914,
#         'Zfactor_DistCytoplasm_Intensity_LowerQuartileIntensity_CorrGreen':0.265558,
#         'Zfactor_DistCytoplasm_Intensity_MedianIntensity_CorrGreen':0.178586,
#         'Zfactor_DistCytoplasm_Intensity_MinIntensityEdge_CorrGreen':0.084566,
#         'Zfactor_DistCytoplasm_Intensity_MinIntensity_CorrGreen':0.086476,
#         'Zfactor_DistCytoplasm_Math_Ratio1':0.623284,
#         'Zfactor_DistCytoplasm_Math_Ratio2':0.358916,
#         'Zfactor_DistCytoplasm_Texture_AngularSecondMoment_CorrGreen_1':0.429510,
#         'Zfactor_DistCytoplasm_Texture_Entropy_CorrGreen_1':0.508275,
#         'Zfactor_DistCytoplasm_Texture_InfoMeas2_CorrGreen_1':0.068695,
#         'Zfactor_DistCytoplasm_Texture_InverseDifferenceMoment_CorrGreen_1':0.347949,
#         'Zfactor_DistCytoplasm_Texture_SumAverage_CorrGreen_1':0.646576,
#         'Zfactor_DistCytoplasm_Texture_SumEntropy_CorrGreen_1':0.494276,
#         'Zfactor_DistCytoplasm_Texture_Variance_CorrGreen_1':0.179011,
#         'Zfactor_DistanceCells_Correlation_Correlation_CorrGreenCorrBlue':0.824686,
#         'Zfactor_DistanceCells_Intensity_IntegratedIntensityEdge_CorrGreen':0.027644,
#         'Zfactor_DistanceCells_Intensity_LowerQuartileIntensity_CorrGreen':0.088491,
#         'Zfactor_DistanceCells_Intensity_MeanIntensityEdge_CorrGreen':0.065056,
#         'Zfactor_DistanceCells_Intensity_MinIntensityEdge_CorrGreen':0.089658,
#         'Zfactor_DistanceCells_Intensity_MinIntensity_CorrGreen':0.078017,
#         'Zfactor_DistanceCells_Texture_AngularSecondMoment_CorrGreen_1':0.238131,
#         'Zfactor_DistanceCells_Texture_Correlation_CorrGreen_1':0.301107,
#         'Zfactor_DistanceCells_Texture_Entropy_CorrGreen_1':0.251143,
#         'Zfactor_DistanceCells_Texture_InfoMeas1_CorrGreen_1':0.564957,
#         'Zfactor_DistanceCells_Texture_InverseDifferenceMoment_CorrGreen_1':0.302767,
#         'Zfactor_DistanceCells_Texture_SumAverage_CorrGreen_1':0.036459,
#         'Zfactor_DistanceCells_Texture_SumEntropy_CorrGreen_1':0.159798,
#         'Zfactor_DistanceCells_Texture_SumVariance_CorrGreen_1':0.516938,
#         'Zfactor_DistanceCells_Texture_Variance_CorrGreen_1':0.501186,
#         'Zfactor_Nuclei_Correlation_Correlation_CorrGreenCorrBlue':0.691408,
#         'Zfactor_Nuclei_Math_Ratio1':0.623284,
#         'Zfactor_Nuclei_Math_Ratio2':0.358916,
#         'Zfactor_Nuclei_Texture_SumAverage_CorrGreen_1':0.587347,
#         'Zfactor_PropCells_AreaShape_Area':0.132425,
#         'Zfactor_PropCells_AreaShape_MajorAxisLength':0.034809,
#         'Zfactor_PropCells_AreaShape_MinorAxisLength':0.113864,
#         'Zfactor_PropCells_AreaShape_Perimeter':0.005984,
#         'Zfactor_PropCells_Correlation_Correlation_CorrGreenCorrBlue':0.717632,
#         'Zfactor_PropCells_Texture_GaborX_CorrGreen_1':0.251023,
#         'Zfactor_PropCells_Texture_InfoMeas2_CorrBlue_1':0.149719,
#         'Zfactor_PropCells_Texture_SumVariance_CorrBlue_1':0.102050,
#         'Zfactor_PropCells_Texture_SumVariance_CorrGreen_1':0.611960,
#         'Zfactor_PropCells_Texture_Variance_CorrBlue_1':0.197090,
#         'Zfactor_PropCells_Texture_Variance_CorrGreen_1':0.614879,
#         'Zfactor_PropCytoplasm_AreaShape_Area':0.205042,
#         'Zfactor_PropCytoplasm_AreaShape_MinorAxisLength':0.072682,
#         'Zfactor_PropCytoplasm_AreaShape_Perimeter':0.005984,
#         'Zfactor_PropCytoplasm_Correlation_Correlation_CorrGreenCorrBlue':0.272017,
#         'Zfactor_PropCytoplasm_Intensity_IntegratedIntensity_CorrGreen':0.115327,
#         'Zfactor_PropCytoplasm_Intensity_LowerQuartileIntensity_CorrGreen':0.141850,
#         'Zfactor_PropCytoplasm_Intensity_MedianIntensity_CorrGreen':0.105803,
#         'Zfactor_PropCytoplasm_Texture_AngularSecondMoment_CorrGreen_1':0.107640,
#         'Zfactor_PropCytoplasm_Texture_Entropy_CorrGreen_1':0.067896,
#         'Zfactor_PropCytoplasm_Texture_GaborX_CorrGreen_1':0.136688,
#         'Zfactor_PropCytoplasm_Texture_InfoMeas2_CorrGreen_1':0.334749,
#         'Zfactor_PropCytoplasm_Texture_SumEntropy_CorrGreen_1':0.208829,
#         'Zfactor_ThresholdedCells_Texture_AngularSecondMoment_CorrBlue_1':0.263467,
#         'Zfactor_ThresholdedCells_Texture_AngularSecondMoment_CorrGreen_1':0.124355,
#         'Zfactor_ThresholdedCells_Texture_Entropy_CorrBlue_1':0.236433,
#         'Zfactor_ThresholdedCells_Texture_InfoMeas2_CorrBlue_1':0.125845,
#         'Zfactor_ThresholdedCells_Texture_SumAverage_CorrBlue_1':0.449333,
#         'Zfactor_ThresholdedCells_Texture_SumEntropy_CorrBlue_1':0.323243,
#         'Zfactor_ThresholdedCells_Texture_SumVariance_CorrBlue_1':0.507477,
#         'Zfactor_ThresholdedCells_Texture_SumVariance_CorrGreen_1':0.599000,
#         'Zfactor_ThresholdedCells_Texture_Variance_CorrBlue_1':0.361424,
#         'Zfactor_ThresholdedCells_Texture_Variance_CorrGreen_1':0.481393
#     }
#     temp_dir = tempfile.mkdtemp()
#     m = None
#     try:
#         cpprefs.set_default_output_directory(temp_dir)
#         print "Writing output to %s"%temp_dir
#         path = os.path.split(__file__)[0]
#         matfile_path = os.path.join(path,'calculatestatistics.mat')
#         if not os.path.isfile(matfile_path):
#             # Download from GIT URL
#             matfile_path = os.path.join(temp_dir, 'calculatestatistics.mat')
#             url = github_url + (
#                 "/cellprofiler/modules/tests/"
#                 "calculatestatistics.mat")
#             urllib.urlretrieve(url, matfile_path)
#         measurements = loadmat(matfile_path,
#                                struct_as_record = True)
#         measurements = measurements['m']
#         image_set_list = cpi.ImageSetList()
#         image_set = image_set_list.get_image_set(0)
#         m = cpmeas.cellprofiler_core.measurement.Measurements()
#         doses = [0 ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10,
#                  0 ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10,
#                  0 ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10,
#                  0 ,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,10,
#                  10,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,0,
#                  10,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,0,
#                  10,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,0,
#                  10,0 ,1 ,2 ,3 ,4 ,5 ,6 ,7 ,8 ,9 ,0]
#         for i,dose in enumerate(doses):
#             m.add_image_measurement("Dose",dose)
#             for object_name in measurements.dtype.fields:
#                 omeasurements = measurements[object_name][0,0]
#                 for feature_name in omeasurements.dtype.fields:
#                     data = omeasurements[feature_name][0,0][0,i].flatten()
#                     m.add_measurement(object_name, feature_name, data)
#             if i < len(doses)-1:
#                 m.next_image_set()
#         pipeline = cpp.Pipeline()
#         module = C.CalculateStatistics()
#         module.grouping_values.value = "Dose"
#         module.dose_values[0].log_transform.value = False
#         module.dose_values[0].measurement.value = "Dose"
#         module.dose_values[0].wants_save_figure.value = True
#         module.dose_values[0].figure_name.value = "EC49_"
#         module.set_module_num(1)
#         pipeline.add_module(module)
#         def callback(caller, event):
#             assertFalse(isinstance(event, cpp.event.RunException))
#         workspace = cpw.Workspace(pipeline, module, image_set,
#                                   cpo.ObjectSet(), m,
#                                   image_set_list)
#         module.post_run(workspace)
#         for feature_name in m.get_feature_names(cpmeas.EXPERIMENT):
#             if not expected.has_key(feature_name):
#                 print "Missing measurement: %s"%feature_name
#                 continue
#             value = m.get_experiment_measurement(feature_name)
#             e_value = expected[feature_name]
#             diff = abs(value-e_value) *2 /abs(value+e_value)
#             assertTrue(diff < .05, "%s: Matlab: %f, Python: %f diff: %f" %
#                             (feature_name, e_value, value, diff))
#             if diff > .01:
#                 print ("Warning: > 1%% difference for %s: Matlab: %f, Python: %f diff: %f" %
#                        (feature_name, e_value, value, diff))
#             if feature_name.startswith("EC50"):
#                 filename = "EC49_"+feature_name[5:]+".pdf"
#                 assertTrue(os.path.isfile(os.path.join(temp_dir, filename)))
#     finally:
#         try:
#             if m is not None:
#                 m.close()
#         except:
#             pass
#         for filename in os.listdir(temp_dir):
#             path = os.path.join(temp_dir, filename)
#             os.remove(path)
#         os.rmdir(temp_dir)


def make_workspace(mdict, controls_measurement, dose_measurements=[]):
    """Make a workspace and module for running CalculateStatistics

    mdict - a two-level dictionary that mimics the measurements structure
            for instance:
            mdict = { cpmeas.Image: { "M1": [ 1,2,3] }}
            for the measurement M1 with values for 3 image sets
    controls_measurement - the name of the controls measurement
    """
    module = cellprofiler.modules.calculatestatistics.CalculateStatistics()
    module.set_module_num(1)
    module.grouping_values.value = controls_measurement

    pipeline = cellprofiler_core.pipeline.Pipeline()
    pipeline.add_module(module)

    m = cellprofiler_core.measurement.Measurements()
    nimages = None
    for object_name in list(mdict.keys()):
        odict = mdict[object_name]
        for feature in list(odict.keys()):
            m.add_all_measurements(object_name, feature, odict[feature])
            if nimages is None:
                nimages = len(odict[feature])
            else:
                assert nimages == len(odict[feature])
            if (
                object_name == "Image"
                and feature in dose_measurements
            ):
                if len(module.dose_values) > 1:
                    module.add_dose_value()
                dv = module.dose_values[-1]
                dv.measurement.value = feature
    m.image_set_number = nimages
    image_set_list = cellprofiler_core.image.ImageSetList()
    for i in range(nimages):
        image_set = image_set_list.get_image_set(i)
    workspace = cellprofiler_core.workspace.Workspace(
        pipeline,
        module,
        image_set,
        cellprofiler_core.object.ObjectSet(),
        m,
        image_set_list,
    )
    return workspace, module


def test_NAN():
    """Regression test of IMG-762

    If objects have NAN values, the means are NAN and the
    z-factors are NAN too.
    """
    mdict = {
        "Image": {
            "Metadata_Controls": [1, 0, -1],
            "Metadata_Doses": [0, 0.5, 1],
        },
        INPUT_OBJECTS: {
            TEST_FTR: [
                numpy.array([1.0, numpy.NaN, 2.3, 3.4, 2.9]),
                numpy.array([5.3, 2.4, numpy.NaN, 3.2]),
                numpy.array([numpy.NaN, 3.1, 4.3, 2.2, 1.1, 0.1]),
            ]
        },
    }
    workspace, module = make_workspace(mdict, "Metadata_Controls", ["Metadata_Doses"])
    module.post_run(workspace)
    m = workspace.measurements
    assert isinstance(m,cellprofiler_core.measurement.Measurements)
    for category in ("Zfactor", "OneTailedZfactor", "Vfactor"):
        feature = "_".join((category, INPUT_OBJECTS, TEST_FTR))
        value = m.get_experiment_measurement(feature)
        assert not numpy.isnan(value)


def test_make_path():
    # regression test of issue #1478
    # If the figure directory doesn't exist, it should be created
    #
    mdict = {
        "Image": {
            "Metadata_Controls": [1, 0, -1],
            "Metadata_Doses": [0, 0.5, 1],
        },
        INPUT_OBJECTS: {
            TEST_FTR: [
                numpy.array([1.0, 2.3, 3.4, 2.9]),
                numpy.array([5.3, 2.4, 3.2]),
                numpy.array([3.1, 4.3, 2.2, 1.1, 0.1]),
            ]
        },
    }
    workspace, module = make_workspace(mdict, "Metadata_Controls", ["Metadata_Doses"])
    assert isinstance(
        module, cellprofiler.modules.calculatestatistics.CalculateStatistics
    )
    my_dir = tempfile.mkdtemp()
    my_subdir = os.path.join(my_dir, "foo")
    fnfilename = FIGURE_NAME + INPUT_OBJECTS + "_" + TEST_FTR + ".pdf"
    fnpath = os.path.join(my_subdir, fnfilename)
    try:
        dose_group = module.dose_values[0]
        dose_group.wants_save_figure.value = True
        dose_group.pathname.dir_choice = (
            cellprofiler_core.preferences.ABSOLUTE_FOLDER_NAME
        )
        dose_group.pathname.custom_path = my_subdir
        dose_group.figure_name.value = FIGURE_NAME
        module.post_run(workspace)
        assert os.path.isfile(fnpath)
    finally:
        if os.path.exists(fnpath):
            os.remove(fnpath)
        if os.path.exists(my_subdir):
            os.rmdir(my_subdir)
        os.rmdir(my_dir)
