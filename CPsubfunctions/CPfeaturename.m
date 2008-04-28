function FeatureName = CPfeaturename(handles,MeasurementCategory,FeatureNumber)


flds = handles.Measurements.(ObjectName).(MeasurementCategory){FeatureNumber};

% MeasureName = [MeasurementCategory 'Features']; %%OLD way, previous to Measurements overhaul
% MeasureName = MeasurementCategory; %%NEW way, after Measurements overhaul
FeatureName = handles.Measurements.(ObjectName).(MeasurementCategory){FeatureNumber};

% switch MeasurementCategory
%     case 'AreaShape'
%         if FeatureNumber == 1,      FeatureName = 'Area';
%         elseif FeatureNumber == 2,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 3,  FeatureName = 'Solidity';
%         elseif FeatureNumber == 4,  FeatureName = 'Extent';
%         elseif FeatureNumber == 5,  FeatureName = 'EulerNumber';
%         elseif FeatureNumber == 6,  FeatureName = 'Perimeter';
%         elseif FeatureNumber == 7,  FeatureName = 'FormFactor';
%         elseif FeatureNumber == 8,  FeatureName = 'MajorAxisLength';
%         elseif FeatureNumber == 9,  FeatureName = 'MinorAxisLength';
%         elseif FeatureNumber == 10,  FeatureName = 'Orientation';
%         elseif FeatureNumber == 11,  FeatureName = 'Zernike_0_0';
%         elseif FeatureNumber == 12,  FeatureName = 'Zernike_1_1';
%         elseif FeatureNumber == 13,  FeatureName = 'Zernike_2_0';
%         elseif FeatureNumber == 14,  FeatureName = 'Zernike_0_0';
%         elseif FeatureNumber == 15,  FeatureName = 'Zernike_0_0';
%         elseif FeatureNumber == 16,  FeatureName = 'Zernike_0_0';
%         elseif FeatureNumber == 17,  FeatureName = 'Zernike_0_0';
%         elseif FeatureNumber == 18,  FeatureName = 'Zernike_0_0';
%         elseif FeatureNumber == 19,  FeatureName = 'Zernike_0_0';
%         elseif FeatureNumber == 20,  FeatureName = 'Zernike_0_0';
%         elseif FeatureNumber == 21,  FeatureName = 'Zernike_0_0';
%         elseif FeatureNumber == 22,  FeatureName = 'Zernike_0_0';
%         elseif FeatureNumber == 23,  FeatureName = 'Zernike_0_0';
%         elseif FeatureNumber == 24,  FeatureName = 'Zernike_0_0';
%         elseif FeatureNumber == 25,  FeatureName = 'Zernike_0_0';
%         elseif FeatureNumber == 26,  FeatureName = 'Zernike_0_0';
%         elseif FeatureNumber == 27,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 28,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 29,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 30,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 31,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 32,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 33,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 34,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 35,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 36,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 37,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 38,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 39,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 40,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 41,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 42,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 43,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 44,  FeatureName = 'Eccentricity';
%         elseif FeatureNumber == 45,  FeatureName = 'Eccentricity';
%             
%             FeatureName = '';
%             FeatureName = '';
%             FeatureName = '';
%             FeatureName = '';
%             FeatureName = '';
%             FeatureName = '';
%             FeatureName = '';
%             FeatureName = '';
%             FeatureName = '';
%     case 'Correlation'
%         
%     case 'Intensity'
%         
%     case 'Neighbors'
% 
%     case 'Texture'
%         
% end

% MeasureObjectAreaShape (For any compartment)
% [PER OBJECT ONLY]
% % Basic shape features:     Feature Number:
% % Area                    |       1
% % Eccentricity            |       2
% % Solidity                |       3
% % Extent                  |       4
% % EulerNumber             |       5
% % Perimeter               |       6
% % FormFactor              |       7
% % MajorAxisLength         |       8
% % MinorAxisLength         |       9
% % Orientation             |      10
% %
% % Zernike shape features:
% %     'Zernike_0_0'       |      11
% %     'Zernike_1_1'       |      12
% %     'Zernike_2_0'       |      13
% %     'Zernike_2_2'       |      14
% %     'Zernike_3_1'       |      15
% %     'Zernike_3_3'       |      16
% %     'Zernike_4_0'       |      17
% %     'Zernike_4_2'       |      18
% %     'Zernike_4_4'       |      19
% %     'Zernike_5_1'       |      20
% %     'Zernike_5_3'       |      21
% %     'Zernike_5_5'       |      22
% %     'Zernike_6_0'       |      23
% %     'Zernike_6_2'       |      24
% %     'Zernike_6_4'       |      25
% %     'Zernike_6_6'       |      26
% %     'Zernike_7_1'       |      27
% %     'Zernike_7_3'       |      28
% %     'Zernike_7_5'       |      29
% %     'Zernike_7_7'       |      30
% %     'Zernike_8_0'       |      31
% %     'Zernike_8_2'       |      32
% %     'Zernike_8_4'       |      33
% %     'Zernike_8_6'       |      34
% %     'Zernike_8_8'       |      35
% %     'Zernike_9_1'       |      36
% %     'Zernike_9_3'       |      37
% %     'Zernike_9_5'       |      38
% %     'Zernike_9_7'       |      39
% %     'Zernike_9_9'       |      40
% %
% 
% MeasureObjectIntensity (for any compartment in combination with any channel)
% [PER OBJECT ONLY]
% % Features measured:      Feature Number:
% % IntegratedIntensity     |       1
% % MeanIntensity           |       2
% % StdIntensity            |       3
% % MinIntensity            |       4
% % MaxIntensity            |       5
% % IntegratedIntensityEdge |       6
% % MeanIntensityEdge       |       7
% % StdIntensityEdge        |       8
% % MinIntensityEdge        |       9
% % MaxIntensityEdge        |      10
% % MassDisplacement        |      11
% 
% MeasureObjectNeighbors
% [PER OBJECT ONLY]
% % Features measured:      Feature Number:
% % NumberOfNeighbors         |    1
% % PercentTouching           |    2
% % FirstClosestObjectNumber  |    3
% % FirstClosestXVector       |    4
% % FirstClosestYVector       |    5
% % SecondClosestObjectNumber |    6
% % SecondClosestXVector      |    7
% % SecondClosestYVector      |    8
% % AngleBetweenNeighbors     |    9
% 
% MeasureTexture (for any compartment - or whole image - in combination with any channel, *and* at any 'Scale of texture' - a numerical value the user enters)
% [CAN BE PER OBJECT OR PER IMAGE OR BOTH]
% 
% % Features measured:      Feature Number:
% % AngularSecondMoment     |       1
% % Contrast                |       2
% % Correlation             |       3
% % Variance                |       4
% % InverseDifferenceMoment |       5
% % SumAverage              |       6
% % SumVariance             |       7
% % SumEntropy              |       8
% % Entropy                 |       9
% % DifferenceVariance      |      10
% % DifferenceEntropy       |      11
% % InfoMeas                |      12
% % InfoMeas2               |      13
% % GaborX                  |      14
% % GaborY                  |      15
% (note that the first 13 are Haralick features)
% 
% MeasureCorrelation (for any compartment - or whole image - in combination with any pairs of channels)
% [CAN BE PER OBJECT OR PER IMAGE OR BOTH]
% % Features measured:      Feature Number:
% % Correlation          |         1
% % Slope                |         2
% 
% MeasureImageIntensity (for any channel)
% [PER IMAGE ONLY]
% % Features measured:      Feature Number:
% % TotalIntensity       |         1
% % MeanIntensity        |         2
% % TotalArea            |         3
% 
% MeasureImageAreaOccupied (for any channel)
% [PER IMAGE ONLY]
% % Features measured:      Feature Number:
% % AreaOccupied        |        1
% % TotalImageArea      |        2
% % ThresholdUsed       |        3
% 
% MeasureImageSaturationBlur (for any channel)
% [PER IMAGE ONLY]
% % Features measured:      Feature Number:
% % FocusScore           |         1
% % PercentSaturated     |         2
% 
% MeasureImageGranularity (for any channel)
% [PER IMAGE ONLY]
% % Features measured:      Feature Number:
% % GS1                 |        1
% % GS2                 |        2
% % GS3                 |        3
% % GS4                 |        4
% % GS5                 |        5
% % GS6                 |        6
% % GS7                 |        7
% % GS8                 |        8
% % GS9                 |        9
% % GS10                |        10
% % GS11                |        11
% % GS12                |        12
% % GS13                |        13
% % GS14                |        14
% % GS15                |        15
% % GS16                |        16
% 
% Align module (for each channel, except the first one specified):
% [PER IMAGE]
% % Features measured:    Feature Number:
% % ImageXAlign         |      1
% % ImageYAlign         |      2
% 
% CalculateRatios and CalculateMath module:
% [PER IMAGE OR PER OBJECT]
% User-defined ratios
% 
% CalculateStatistics:
% [PER EXPERIMENT] This is the only module, I think, that records one set of values for the whole experiment:
% % Features measured:   Feature Number:
% % Zfactor            |      1
% % Vfactor            |      2
% % EC50               |      3
% 
% ClassifyObjects module:
% [PER IMAGE]
% User-defined classification bins (the fraction or absolute number of objects in each classification ‘bin’)
% 
% DefineGrid module:
% [PER IMAGE]
% % Features measured:      Feature Number:
% % XLocationOfLowestXSpot |      1
% % YLocationOfLowestYSpot |      2
% % XSpacing               |      3 
% % YSpacing               |      4 
% % Rows                   |      5
% % Columns                |      6 
% % TotalHeight            |      7
% % TotalWidth             |      8
% % LeftOrRightNum         |      9
% % TopOrBottomNum         |     10
% % RowsOrColumnsNum       |     11
% 
% LoadText module:
% [PER IMAGE ONLY]
% User-defined strings
% 
% Rotate module: 
% [PER IMAGE ONLY]
% % Features measured:   Feature Number:
% Rotation             |      1
% 
% Crop module:
% [PER IMAGE ONLY]
% % Features measured:          Feature Number:
% % AreaRetainedAfterCropping |   1
% % OriginalImageArea         |   2
% 
% SubtractBackground module:
% [PER IMAGE ONLY]
% % Features measured:    Feature Number:
% % IntensityToShift    |   1