function handles = AlgMeasureAreaShapeIntensTxtr(handles)

% Help for the Measure module:
%
% Retrieves a segmented image, in label matrix format, and its
% corresponding original grayscale image and makes lots of measurements of
% the objects that are segmented in the image. The label matrix image
% should be "compacted": I mean that each number should correspond to an
% object, with no numbers skipped.  So, if some objects were discarded from
% the label matrix image, the image should be converted to binary and
% re-made into a label matrix image before feeding into this module.

% The contents of this file are subject to the Mozilla Public License Version
% 1.1 (the "License"); you may not use this file except in compliance with
% the License. You may obtain a copy of the License at
% http://www.mozilla.org/MPL/
%
% Software distributed under the License is distributed on an "AS IS" basis,
% WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
% for the specific language governing rights and limitations under the
% License.
%
%
% The Original Code is the Measure Area, Shape, Intensity, and Texture module.
%
% The Initial Developer of the Original Code is
% Whitehead Institute for Biomedical Research
% Portions created by the Initial Developer are Copyright (C) 2003,2004
% the Initial Developer. All Rights Reserved.
%
% Contributor(s):
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current algorithm number, since this is needed to find
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;
CurrentAlgorithmNum = str2num(handles.currentalgorithm);

%textVAR01 = What did you call the segmented objects that you want to measure?
%defaultVAR01 = Nuclei
ObjectName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What did you call the greyscale images you want to measure?
%defaultVAR02 = OrigBlue
ImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = Measure the percent of cells with a total intensity greater
%textVAR04 = than or equal to this threshold.  Type N to skip this measurement.
%defaultVAR04 = N
Threshold = char(handles.Settings.Vvariable{CurrentAlgorithmNum,4});

%textVAR06 = The measurements made by this module will be named based on
%textVAR07 = your entries, e.g. "OrigRedwithinNuclei".

%%% Retrieves the pixel size that the user entered (micrometers per pixel).
PixelSize = str2num(handles.Vpixelsize{1});

%%% POTENTIAL IMPROVEMENT: Allow the user to select which measurements will
%%% be made, particularly for those which take a long time to calculate?
%%% Probably not a good idea: we want the measurements coming out to be
%%% uniform from experiment to experiment so as to have comparable data for
%%% comparing experiments.  If the user wants to skip some measurements,
%%% they can alter this .m file to comment out the measurements.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImageToBeAnalyzed".
fieldname = ['dOT',ImageName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Measure algorithm, you must have previously run an algorithm that loads a greyscale image.  You specified in the Measure module that the desired image was named ', ImageName, ' which should have produced an image in the handles structure called ', fieldname, '. The Measure module cannot locate this image.']);
end
OrigImageToBeAnalyzed = handles.(fieldname);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImageToBeAnalyzed) ~= 2
    error('Image processing was canceled because the Measure Area Shape Intensity Texture module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%% Retrieves the label matrix image that contains the segmented objects which
%%% will be measured with this algorithm.
fieldname = ['dOTSegmented',ObjectName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the Measure algorithm, you must have previously run an algorithm that generates an image with the objects identified.  You specified in the Measure module that the primary objects were named ',ObjectName,' which should have produced an image in the handles structure called ', fieldname, '. The Measure module cannot locate this image.']);
end
LabelMatrixImage = handles.(fieldname);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Note to programmers on the format in which to acquire measurements:
%%% the measurements must be stored in double format, because the
%%% extraction part of the program is designed to deal with that type of
%%% array only, not cell or structure arrays. I also thought it wise to
%%% disallow more than one "column" of data per object, to allow for
%%% uniform extraction of data later.  So, for example, instead of storing
%%% the X and Y position together, they are stored separately.

%%% Note to programmers on how to extract measurements:
%%% Examples:
%%% handles.dMCCenterXNuclei{1}(2) gives the X position for
%%% the second object in the first image.  handles.dMCAreaNuclei{2}(1) gives
%%% the area of the first object in the second image.

%%% Note to programmers on saving measurements to the handles structure:
%%% Take note that the fields in the handles structure are named
%%% appropriately based on the user's input for "ObjectName".
%%% The prefix 'dMT' is added to the beginning of the field name for the
%%% Object Count, since there is one measurement made per image. (T =
%%% Total).  The prefix 'dMC' is added to the beginning of the field name
%%% for the Object XY positions since there are multiple measurements per
%%% image (one X,Y position for each object in the image) (C = Cell by
%%% cell).  Due to these prefixes, these fields will be deleted from the
%%% handles structure at the end of the analysis batch.

%%%
%%% COUNT
%%%

if sum(sum(LabelMatrixImage)) == 0
    %%% None of the measurements are made if there are no objects in the label
    %%% matrix image.
    %%% Saves the count to the handles structure.
    fieldname = ['dMTCount', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {0};
else

    %%% The regionprops command extracts a lot of measurements.  It
    %%% is most efficient to call the regionprops command once for all the
    %%% properties rather than calling it for each property separately.
    Statistics = regionprops(LabelMatrixImage,'Area', 'ConvexArea', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity', 'Solidity', 'Extent', 'Centroid');

    %%% CATCH NAN's -->>
    if sum(isnan(cat(1,Statistics.Solidity))) ~= 0
        error('Image processing was canceled because there was a problem in the Measure Area Shape Intensity Texture module. Some of the measurements could not be made.  This might be because some objects had zero area or because some measurements were attempted that were divided by zero. If you want to make measurements despite this problem, remove the 3 lines in the .m file for this module following the line "%%% CATCH NAN''s". This will result in some non-numeric values in the output file, which will be represented as "NaN" (Not a Number).')
    end

    %%%
    %%% AREA
    %%%

    %%% Makes the Area array a double object rather than a cell or struct
    %%% object.
    Area = cat(1,Statistics.Area);
    %%% Converts the measurement to micrometers.  Converts the number of pixels
    %%% to micrometers squared.
    Area = Area.*(PixelSize*PixelSize);
    %%% Saves the areas to the handles structure.
    fieldname = ['dMCArea', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {[Area]};
    fieldname = ['dMTMeanArea', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {mean(Area)};
    fieldname = ['dMTStdevArea', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {std(Area)};
    fieldname = ['dMTMedianArea', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {median(Area)};
    fieldname = ['dMTSumArea', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {sum(Area)};

    %%%
    %%% CONVEX AREA
    %%%

    %%% Makes the ConvexAreas array a double object rather than a cell or struct
    %%% object.
    ConvexArea = cat(1,Statistics.ConvexArea);
    %%% Converts the measurement to micrometers. The number of pixels is
    %%% converted to micrometers squared.
    ConvexArea = ConvexArea.*(PixelSize*PixelSize);
    %%% Saves the areas to the handles structure.
    fieldname = ['dMCConvexArea', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {[ConvexArea]};
    fieldname = ['dMTMeanConvexArea', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {mean(ConvexArea)};
    fieldname = ['dMTStdevConvexArea', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {std(ConvexArea)};
    fieldname = ['dMTMedianConvexArea', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {median(ConvexArea)};

    %%%
    %%% MAJOR AXIS
    %%%

    %%% Makes the major axis array a double object rather than a cell or struct
    %%% object.
    MajorAxis = cat(1,Statistics.MajorAxisLength);
    %%% Converts the measurement to micrometers.
    MajorAxis = MajorAxis*PixelSize;
    %%% Saves the major axis lengths to the handles structure.
    fieldname = ['dMCMajorAxis', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {[MajorAxis]};
    fieldname = ['dMTMeanMajorAxis', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {mean(MajorAxis)};
    fieldname = ['dMTStdevMajorAxis', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {std(MajorAxis)};
    fieldname = ['dMTMedianMajorAxis', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {median(MajorAxis)};

    %%%
    %%% MINOR AXIS
    %%%

    %%% Makes the minor axis array a double object rather than a cell or struct
    %%% object.
    MinorAxis = cat(1,Statistics.MinorAxisLength);
    %%% Converts the measurement to micrometers.
    MinorAxis = MinorAxis*PixelSize;
    %%% Saves the minor axis lengths to the handles structure.
    fieldname = ['dMCMinorAxis', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {[MinorAxis]};
    fieldname = ['dMTMeanMinorAxis', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {mean(MinorAxis)};
    fieldname = ['dMTStdevMinorAxis', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {std(MinorAxis)};
    fieldname = ['dMTMedianMinorAxis', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {median(MinorAxis)};

    %%%
    %%% ECCENTRICITY
    %%%

    %%% The eccentricity of the ellipse that has the same second-moments as the
    %%% region. The eccentricity is the ratio of the distance between the foci
    %%% of the ellipse and its major axis length. The value is between 0 and 1.
    %%% (0 and 1 are degenerate cases; an ellipse whose eccentricity is 0 is
    %%% actually a circle, while an ellipse whose eccentricity is 1 is a line
    %%% segment.)  Other sources define Eccentricity as the ratio of the major
    %%% axis to the minor axis, but I have named that "Aspect Ratio" below
    %%% since it is apparently calculated differently than Matlab's
    %%% eccentricity measurement.

    %%% Makes the Eccentricity array a double object rather than a cell or struct
    %%% object.
    Eccentricity = cat(1,Statistics.Eccentricity);
    %%% Note: No need to convert the measurement to micrometers because it is
    %%% dimensionless.
    %%% Saves the Eccentricities to the handles structure.
    fieldname = ['dMCEccentricity', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {[Eccentricity]};
    fieldname = ['dMTMeanEccentricity', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {mean(Eccentricity)};
    fieldname = ['dMTStdevEccentricity', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {std(Eccentricity)};
    fieldname = ['dMTMedianEccentricity', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {median(Eccentricity)};

    %%%
    %%% SOLIDITY
    %%%

    %%% Solidity is the proportion of pixels in the convex hull that are also
    %%% in the region. Defined as Area/Convex area.

    %%% Makes the solidity array a double object rather than a cell or struct
    %%% object.
    Solidity = cat(1,Statistics.Solidity);
    %%% Note: No need to convert the measurement to micrometers because it is
    %%% dimensionless.
    %%% Saves the Solidities to the handles structure.
    fieldname = ['dMCSolidity', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {[Solidity]};
    fieldname = ['dMTMeanSolidity', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {mean(Solidity)};
    fieldname = ['dMTStdevSolidity', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {std(Solidity)};
    fieldname = ['dMTMedianSolidity', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {median(Solidity)};

    %%%
    %%% EXTENT
    %%%

    %%% Extent is the proportion of the pixels in the convex hull that are also
    %%% in the region. (Area divided by the area of the bounding box).

    %%% Makes the Extent array a double object rather than a cell or struct
    %%% object.
    Extent = cat(1,Statistics.Extent);
    %%% Note: No need to convert the measurement to micrometers because it is
    %%% dimensionless.
    %%% Saves the Extents to the handles structure.
    fieldname = ['dMCExtent', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {[Extent]};
    fieldname = ['dMTMeanExtent', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {mean(Extent)};
    fieldname = ['dMTStdevExtent', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {std(Extent)};
    fieldname = ['dMTMedianExtent', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {median(Extent)};

    %%%
    %%% CENTER POSITIONS
    %%%

    %%% Note that the X, Y locations are stored as the pixel locations (not
    %%% converted to micrometers.)

    %%% Makes the Centers array a double object rather than a cell or struct
    %%% object.  There are two columns in this array, the first is X and the
    %%% second is Y, so they are extracted into two separate variables, CentersX
    %%% and CentersY.
    CentersXY = cat(1,Statistics.Centroid);
    CentersX = CentersXY(:,1);
    CentersY = CentersXY(:,2);
    %%% Saves X and Y positions to handles structure.
    fieldname = ['dMCCenterX', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {[CentersX]};
    fieldname = ['dMCCenterY', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {[CentersY]};

    %%%
    %%% INTEGRATED INTENSITY (TOTAL INTENSITY PER OBJECT)
    %%%

    %%% The find function (when used as follows) returns the linear index
    %%% position of all the nonzero elements in the label matrix image.
    ForegroundPixels = find(LabelMatrixImage);
    %%% Returns the actual label matrix value at each foreground pixel in the
    %%% label matrix.
    LabelValue = LabelMatrixImage(ForegroundPixels);
    %%% Creates a sparse matrix: Can think of it this way (not sure if I have
    %%% rows and columns mixed up, but it doesn't matter): each object is a
    %%% column, identified by the LabelValue, which is really equivalent to the
    %%% object number.  Each row of the matrix is a position in the original
    %%% image, identified by linear indexing, so that the number of rows is equal
    %%% to the linear index value of the last nonzero pixel in the label matrix
    %%% image.  The value of each cell in this matrix is the intensity value from
    %%% the original image at that position.
    AllObjectsPixelValues = sparse(ForegroundPixels, LabelValue, OrigImageToBeAnalyzed(ForegroundPixels));
    %%% Sums all pixel intensity values in each column.
    AlmostIntegratedIntensity = sum(AllObjectsPixelValues(:,:));
    %%% Converts from sparse to full to end up with one column of numbers.
    IntegratedIntensity = full(AlmostIntegratedIntensity');
    %%% Integrated Intensity is in arbitrary intensity units, dimensionless
    %%% with respect to area.

    %%% Saves Integrated Intensities to handles structure.
    fieldname = ['dMCIntegratedIntensity', ImageName, 'within', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {[IntegratedIntensity]};
    fieldname = ['dMTMeanIntegratedIntensity', ImageName, 'within', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {mean(IntegratedIntensity)};
    fieldname = ['dMTStdevIntegratedIntensity', ImageName, 'within', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {std(IntegratedIntensity)};
    fieldname = ['dMTMedianIntegratedIntensity', ImageName, 'within', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {median(IntegratedIntensity)};
    fieldname = ['dMTSumIntegratedIntensity', ImageName, 'within', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {sum(IntegratedIntensity)};

    %%% Calculates the percent of cells whose integrated intensity is above the
    %%% user's threshold.
    if strcmp(upper(Threshold),'N') ~= 1
        NumberObjectsAboveThreshold = sum(IntegratedIntensity >= str2num(Threshold));
        TotalNumberObjects = length(IntegratedIntensity);
        PercentObjectsAboveThreshold = NumberObjectsAboveThreshold/TotalNumberObjects;
        fieldname = ['dMTPercentAboveThreshold', ImageName, 'within', ObjectName];
        handles.(fieldname)(handles.setbeinganalyzed) = {PercentObjectsAboveThreshold};
    end

    %%%
    %%% MEAN INTENSITY (PER OBJECT)
    %%%

    %%% Note: this depends on the ForegroundPixels and LabelValue variables
    %%% determined in the integrated intensity code above.
    %%% Computes the mean.
    MeanIntensity =  IntegratedIntensity ./ Area;
    MeanIntensity = MeanIntensity';
    %%% Subtracts the mean from each region.
    Map1 = [0 MeanIntensity];
    try
        OrigImageToBeAnalyzed2 = OrigImageToBeAnalyzed - Map1(LabelMatrixImage + 1);
    catch error('There was a problem in the Measure module.  The image to be analyzed is a different size than the image of identified objects.  If the objects were identified from a cropped image, the cropped image should be used by the Measure module.')
    end
    %%% Avoids divide by zero.
    NonZeroArea = Area;
    NonZeroArea(NonZeroArea < 2) = 2;
    NonZeroArea = NonZeroArea';
    %%% Estimates the standard deviation.
    Temp2 = sparse(ForegroundPixels, LabelValue, OrigImageToBeAnalyzed2(ForegroundPixels));
    StDevIntensity = sqrt(full(sum(Temp2.^2)) ./ (NonZeroArea - 1));
    %%% Converts to a column.
    MeanIntensity = MeanIntensity';
    StDevIntensity = StDevIntensity';
    %%% Saves data to handles structure.
    fieldname = ['dMCMeanIntensity', ImageName, 'within', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {[MeanIntensity]};
    fieldname = ['dMTMeanMeanIntensity', ImageName, 'within', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {mean(MeanIntensity)};
    fieldname = ['dMTStdevMeanIntensity', ImageName, 'within', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {std(MeanIntensity)};
    fieldname = ['dMTMedianMeanIntensity', ImageName, 'within', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {median(MeanIntensity)};

    fieldname = ['dMCStDevIntensity', ImageName, 'within', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {[StDevIntensity]};
    fieldname = ['dMTMeanStDevIntensity', ImageName, 'within', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {mean(StDevIntensity)};
    fieldname = ['dMTStdevStDevIntensity', ImageName, 'within', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {std(StDevIntensity)};
    fieldname = ['dMTMedianStDevIntensity', ImageName, 'within', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {median(StDevIntensity)};

    %%%
    %%% PERIMETER
    %%%

    %%% Shifts labels in each of the 4 cardinal directions (stretching the
    %%% exposed row/column), and compares to the original labels.
    temp_labels = LabelMatrixImage .* ((LabelMatrixImage ~= LabelMatrixImage([1 1:end-1], :)) | ...
        (LabelMatrixImage ~= LabelMatrixImage([2:end end], :)) | ...
        (LabelMatrixImage ~= LabelMatrixImage(:, [1 1:end-1], :)) | ...
        (LabelMatrixImage ~= LabelMatrixImage(:, [2:end end])));
    %%% Finds the locations and labels for perimeter pixels.
    perim_locations = find(temp_labels);
    perim_labels = LabelMatrixImage(perim_locations);
    %%% Creates a sparse matrix with column as label and row as location,
    %%% with a 1 at (A,B) if location A has label B.  Summing the columns
    %%% gives the count of perimeter pixels with a given label.
    Perimeter = full(sum(sparse(perim_locations, perim_labels, 1)));
    Perimeter = Perimeter';
    %%% Converts the measurement to micrometers.
    Perimeter = Perimeter*PixelSize;
    %%% Saves Perimeters to handles structure.
    fieldname = ['dMCPerimeter', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {[Perimeter]};
    fieldname = ['dMTMeanPerimeter', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {mean(Perimeter)};
    fieldname = ['dMTStdevPerimeter', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {std(Perimeter)};
    fieldname = ['dMTMedianPerimeter', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {median(Perimeter)};

    %%%
    %%% CIRCULARITY
    %%%

    %%% Defined as Perimeter squared divided by Area.  Conversion to
    %%% micrometers was already done above; the result of the calculation below
    %%% is dimensionless anyway.
    Circularity = (Perimeter.*Perimeter)./Area;
    fieldname = ['dMCCircularity', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {[Circularity]};
    fieldname = ['dMTMeanCircularity', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {mean(Circularity)};
    fieldname = ['dMTStdevCircularity', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {std(Circularity)};
    fieldname = ['dMTMedianCircularity', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {median(Circularity)};

    %%%
    %%% FORM FACTOR
    %%%

    %%% Defined as 4pi*Area divided by the Perimeter squared. Conversion to
    %%% micrometers was already done above; the result of the calculation below
    %%% is dimensionless anyway.
    FormFactor = 4*pi.*Area./(Perimeter.*Perimeter);
    fieldname = ['dMCFormFactor', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {[FormFactor]};
    fieldname = ['dMTMeanFormFactor', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {mean(FormFactor)};
    fieldname = ['dMTStdevFormFactor', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {std(FormFactor)};
    fieldname = ['dMTMedianFormFactor', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {median(FormFactor)};

    %%%
    %%% AREA TO PERIMETER RATIO
    %%%

    %%% Conversion to micrometers was already done above.
    AreaPerimRatio = Area./Perimeter;

    fieldname = ['dMCAreaPerimRatio', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {[AreaPerimRatio]};
    fieldname = ['dMTMeanAreaPerimRatio', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {mean(AreaPerimRatio)};
    fieldname = ['dMTStdevAreaPerimRatio', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {std(AreaPerimRatio)};
    fieldname = ['dMTMedianAreaPerimRatio', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {median(AreaPerimRatio)};

    %%%
    %%% ASPECT RATIO
    %%%

    %%% This should really be the maximum diameter divided by the minimum
    %%% diameter, but those measurements would add a lot of time to measure.
    %%% Conversion to micrometers was already done above; the result of the
    %%% calculation below is dimensionless anyway.

    AspectRatio = MajorAxis./MinorAxis;

    fieldname = ['dMCAspectRatio', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {[AspectRatio]};
    fieldname = ['dMTMeanAspectRatio', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {mean(AspectRatio)};
    fieldname = ['dMTStdevAspectRatio', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {std(AspectRatio)};
    fieldname = ['dMTMedianAspectRatio', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {median(AspectRatio)};

    %%%
    %%% COUNT
    %%%

    %%% Counts the number of objects, by counting the number of area
    %%% measurements made. NOTE: This depends on the Area calculation above, so
    %%% if you remove the area calculation you will need to include the line
    %%% Areas = regionprops(LabelMatrixImage,'Area') before the following, or
    %%% substitute a different measurement name for "Area".
    CellCount = length(Area);
    %%% Saves the count to the handles structure.
    fieldname = ['dMTCount', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {CellCount};

end % Goes with: if no objects are in the image.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS IN THE FIGURE WINDOW %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% Checks whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this algorithm.
if any(findobj == ThisAlgFigureNumber) == 1;
    figure(ThisAlgFigureNumber);
    originalsize = get(ThisAlgFigureNumber, 'position');
    newsize = originalsize;
    if handles.setbeinganalyzed == 1
        newsize(3) = originalsize(3)*.5;
        set(ThisAlgFigureNumber, 'position', newsize);
    end
    newsize(1) = 0;
    newsize(2) = 0;
    displaytexthandle = uicontrol(ThisAlgFigureNumber,'style','text', 'position', newsize,'fontname','fixedwidth','backgroundcolor',[0.7,0.7,0.7]);
    %%% Note that the number of spaces after each measurement name results in
    %%% the measurement numbers lining up properly when displayed in a fixed
    %%% width font.  Also, it costs less than 0.1 seconds to do all of these
    %%% calculations, so I won't bother to retrieve the already calculated
    %%% means and sums from each measurement's code above.
    %%% Checks whether any objects were found in the image.
    if sum(sum(LabelMatrixImage)) == 0
        displaytext = strvcat(['      Image Set # ',num2str(handles.setbeinganalyzed)],...
            ['Number of ', ObjectName ,':      zero']);
    else
        displaytext = strvcat(['      Image Set # ',num2str(handles.setbeinganalyzed)],...
            ['Number of ', ObjectName ,':      ', num2str(CellCount)],...
            ['MeanArea:                 ', num2str(mean(Area))],...
            ['MeanConvexArea:           ', num2str(mean(ConvexArea))],...
            ['MeanPerimeter:            ', num2str(mean(Perimeter))],...
            ['MeanMajorAxis:            ', num2str(mean(MajorAxis))],...
            ['MeanMinorAxis:            ', num2str(mean(MinorAxis))],...
            ['MeanEccentricity:         ', num2str(mean(Eccentricity))],...
            ['MeanSolidity:             ', num2str(mean(Solidity))],...
            ['MeanExtent:               ', num2str(mean(Extent))],...
            ['MeanCircularity:          ', num2str(mean(Circularity))],...
            ['MeanFormFactor:           ', num2str(mean(FormFactor))],...
            ['MeanAreaPerimRatio:       ', num2str(mean(AreaPerimRatio))],...
            ['MeanAspectRatio:          ', num2str(mean(AspectRatio))],...
            ['MeanIntegratedIntensity:  ', num2str(mean(IntegratedIntensity))],...
            ['MeanMeanIntensity:        ', num2str(mean(MeanIntensity))],...
            ['MeanStDevIntensity:       ', num2str(mean(StDevIntensity))],...
            ['SumIntegratedIntensity:   ', num2str(sum(IntegratedIntensity))],...
            ['SumArea:                  ', num2str(sum(Area))]);
        if strcmp(upper(Threshold),'N') ~= 1
            displaytext = strvcat(displaytext,...
                ['Percent above intensity threshold:', num2str(PercentObjectsAboveThreshold)]);
        end
    end % Goes with: if no objects were in the label matrix image.
    set(displaytexthandle,'string',displaytext)
end