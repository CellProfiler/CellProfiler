function handles = AlgMeasureIntensityTexture(handles)


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
% The Original Code is the Measure Intensity and Texture Module.
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

%%% Retrieves a segmented image, in label matrix format, and a
%%% corresponding original grayscale image and makes 
%%% measurements of the objects that are segmented in the image.
%%% This module differs from the AlgMeasure module because it lacks
%%% measurements of shape and area and includes only intensity and
%%% texture.
%%% The label matrix image should be "compacted": I mean that each number
%%% should correspond to an object, with no numbers skipped.  So, if some
%%% objects were discarded from the label matrix image, the image should be
%%% converted to binary and re-made into a label matrix image before
%%% feeding into this module.

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

%textVAR01 = What did you call the segmented objects that you want to measure?
%defaultVAR01 = Nuclei
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
ObjectName = handles.(fieldname);

%textVAR02 = What did you call the greyscale images you want to measure? 
%defaultVAR02 = OrigRed
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
ImageName = handles.(fieldname);

%textVAR03 = Note: it is advisable to use the original image rather than an adjusted image.
%textVAR04 = Adjustments alter the intensity levels from one image relative to another
%textVAR05 = so that intensity measurements cannot be accurately compared. If you have
%textVAR06 = cropped an image, however, the cropped image should be specified here.

%textVAR07 = The measurements made by this module will be named after the two boxes above, 
%textVAR08 = e.g. "OrigRedwithinNuclei".

%%% Retrieves the pixel size that the user entered (micrometers per pixel).
PixelSize = str2num(handles.Vpixelsize{1});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Retrieves the label matrix image that contains the segmented objects which
%%% will be measured with this algorithm.  
fieldname = ['dOTSegmented',ObjectName];
%%% Checks whether the image exists in the handles structure.
    if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the MeasureIntensityTexture algorithm, you must have previously run an algorithm that generates an image with the objects identified.  You specified in the MeasureIntensityTexture module that the primary objects were named ',ObjectName,' which should have produced an image in the handles structure called ', fieldname, '. The MeasureIntensityTexture module cannot locate this image.']);
    end
LabelMatrixImage = handles.(fieldname);
        % figure, imshow(LabelMatrixImage), title('LabelMatrixImage')
        
%%% Read (open) the image you want to analyze and assign it to a variable,
%%% "OrigImageToBeAnalyzed".
fieldname = ['dOT',ImageName];
%%% Checks whether the image exists in the handles structure.
    if isfield(handles, fieldname) == 0
    error(['Image processing has been canceled. Prior to running the MeasureIntensityTexture algorithm, you must have previously run an algorithm that loads a greyscale image.  You specified in the MeasureIntensityTexture module that the desired image was named ', ImageName, ' which should have produced an image in the handles structure called ', fieldname, '. The MeasureIntensityTexture module cannot locate this image.']);
    end
OrigImageToBeAnalyzed = handles.(fieldname);
%%% Update the handles structure.
%%% Removed for parallel: guidata(gcbo, handles);
        
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Note to programmers on the format in which to acquire measurements: 
%%% the measurements must be stored in double format, 
%%% because the extraction part of the program is designed to deal with
%%% that type of array only, not cell or structure arrays.
%%% I also thought it wise to disallow more than one
%%% "column" of data per object, to allow for uniform extraction of data
%%% later.  So, for example, instead of storing the X and Y position
%%% together, they are stored separately. 

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

%%% None of the measurements are made if there are no objects in the label
%%% matrix image.
if sum(sum(LabelMatrixImage)) == 0
    %%% Save the count to the handles structure.  The field is named 
    %%% appropriately based on the user's input, with the 'dMT' prefix added so
    %%% that this field will be deleted at the end of the analysis batch.
    fieldname = ['dMTCount', ObjectName];
    handles.(fieldname)(handles.setbeinganalyzed) = {0};
else

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImageToBeAnalyzed) ~= 2
    error('Image processing was canceled because the Segment Intensity module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%% The ObjectName is changed to include the Object Name plus the name of
%%% the grayscale image used, so that the measurements do not overwrite any
%%% measurements made on the original objects.  For example, if
%%% measurements were made for the Nuclei using the original
%%% blue image and this module is being
%%% used to measure the intensities, etc. of the OrigRed channel at the
%%% nuclei, the blue measurements will be called: MeanAreaNuclei whereas
%%% the red measurements will be called: MeanAreaOrigRedWithinNuclei.
ObjectName = strcat(ImageName , 'within', ObjectName);

%%%
%%% INTEGRATED INTENSITY (TOTAL INTENSITY PER OBJECT)
%%%

% The find function (when used as follows) returns the linear index
% position of all the nonzero elements in the label matrix image.
ForegroundPixels = find(LabelMatrixImage);
% The find function (when used as follows) returns the x and y position of
% the nonzero elements of the label matrix image (which we don't care
% about), as well as the actual label matrix value at that point (i.e. 1,
% 2, 3).
[x,y,LabelValue] = find(LabelMatrixImage);
% Creates a sparse matrix: Can think of it this way (not sure if I have
% rows and columns mixed up, but it doesn't matter): each object is a
% column, identified by the LabelValue, which is really equivalent to the
% object number.  Each row of the matrix is a position in the original
% image, identified by linear indexing, so that the number of rows is equal
% to the linear index value of the last nonzero pixel in the label matrix
% image.  The value of each cell in this matrix is the intensity value from
% the original image at that position.
AllObjectsPixelValues = sparse(ForegroundPixels, LabelValue, OrigImageToBeAnalyzed(ForegroundPixels));
% The sum command sums all pixel intensity values in each column.
AlmostIntegratedIntensity = sum(AllObjectsPixelValues(:,:));
% Need to convert from sparse to full to end up with one column of numbers.
IntegratedIntensity = full(AlmostIntegratedIntensity');

%%% Save Integrated Intensities to handles structure.  The fields are named 
%%% appropriately based on the user's input, with the 'dMC' prefix added so
%%% that these fields will be deleted at the end of the analysis batch.
fieldname = ['dMCIntegratedIntensity', ObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = {[IntegratedIntensity]};
fieldname = ['dMTMeanIntegratedIntensity', ObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = {mean(IntegratedIntensity)};
fieldname = ['dMTStdevIntegratedIntensity', ObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = {std(IntegratedIntensity)};
fieldname = ['dMTMedianIntegratedIntensity', ObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = {median(IntegratedIntensity)};
fieldname = ['dMTSumIntegratedIntensity', ObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = {sum(IntegratedIntensity)};

%%%
%%% MEAN INTENSITY (PER OBJECT)
%%%

%%% Finds the locations and labels for different objects.
ObjectLocations = find(LabelMatrixImage);
ObjectLabels = LabelMatrixImage(ObjectLocations);
%%% Creates a sparse matrix with column as label and row as location,
%%% with a 1 at (A,B) if location A has label B.  Summing the columns
%%% gives the count of area pixels with a given label.
Areas1 = full(sum(sparse(ObjectLocations, ObjectLabels, 1)));
%%% Computes the mean.
Temp1 = sparse(ObjectLocations, ObjectLabels, OrigImageToBeAnalyzed(ObjectLocations));
MeanIntensity =  full(sum(Temp1)) ./ Areas1;
%%% Subtracts the mean from each region.
Map1 = [0 MeanIntensity];
try
OrigImageToBeAnalyzed2 = OrigImageToBeAnalyzed - Map1(LabelMatrixImage + 1);
catch error('There was a problem in the MeasureIntensityTexture module.  The image to be analyzed is a different size than the image of identified objects.  If the objects were identified from a cropped image, the cropped image should be used by the Measure module.')  
end
%%% Avoids divide by zero.
Areas1(Areas1 < 2) = 2;
%%% Estimates the standard deviation.
Temp2 = sparse(ObjectLocations, ObjectLabels, OrigImageToBeAnalyzed2(ObjectLocations));
StDevIntensity = sqrt(full(sum(Temp2.^2)) ./ (Areas1 - 1));
%%% Converts to a column.
MeanIntensity = MeanIntensity';
StDevIntensity = StDevIntensity';
%%% Saves data to handles structure.
fieldname = ['dMCMeanIntensity', ObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = {[MeanIntensity]};
fieldname = ['dMTMeanMeanIntensity', ObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = {mean(MeanIntensity)};
fieldname = ['dMTStdevMeanIntensity', ObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = {std(MeanIntensity)};
fieldname = ['dMTMedianMeanIntensity', ObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = {median(MeanIntensity)};

fieldname = ['dMCStDevIntensity', ObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = {[StDevIntensity]};
fieldname = ['dMTMeanStDevIntensity', ObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = {mean(StDevIntensity)};
fieldname = ['dMTStdevStDevIntensity', ObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = {std(StDevIntensity)};
fieldname = ['dMTMedianStDevIntensity', ObjectName];
handles.(fieldname)(handles.setbeinganalyzed) = {median(StDevIntensity)};

end % Goes with: if no objects are in the image.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS IN THE FIGURE WINDOW %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% Check whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this algorithm.
if any(findobj == ThisAlgFigureNumber) == 1;
    figure(ThisAlgFigureNumber);
     originalsize = get(ThisAlgFigureNumber, 'position');
%     newsize = originalsize;
%     newsize(3) = 305;
%     set(ThisAlgFigureNumber, 'position', newsize);
%     displaytext = ' ';
newsize = originalsize;
newsize(1) = 0;
newsize(2) = 0;
if handles.setbeinganalyzed == 1
    newsize(3) = originalsize(3)*.5;
    originalsize(3) = originalsize(3)*.5;
    set(ThisAlgFigureNumber, 'position', originalsize);
end
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
        ['MeanIntegratedIntensity:  ', num2str(mean(IntegratedIntensity))],...
        ['MeanMeanIntensity:        ', num2str(mean(MeanIntensity))],...
        ['MeanStDevIntensity:       ', num2str(mean(StDevIntensity))],...
        ['SumIntegratedIntensity:   ', num2str(sum(IntegratedIntensity))]);
end % Goes with: if no objects were in the label matrix image.
set(displaytexthandle,'string',displaytext)
end

%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for the MeasureIntensityTexture module: 
%%%%% .
