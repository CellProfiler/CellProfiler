function handles = FilterByObjectMeasurement(handles)

% Help for the Filter by Object Measurement module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Eliminates objects based on their measurements (e.g. area, shape,
% texture, intensity).
% *************************************************************************
%
% This module removes objects based on their measurements produced by
% another module (e.g. Measure Object Area Shape, Measure Object Intensity,
% Measure Texture). All objects outside of the specified parameters will be
% discarded.
%
% Feature Number:
% The feature number specifies which feature from the Measure module will
% be used for filtering. See each Measure module's help for the numbered
% list of the features measured by that module.
%
% Special note on saving images: Using the settings in this module, object
% outlines can be passed along to the module Overlay Outlines and then
% saved with the Save Images module. Objects themselves can be passed along
% to the object processing module Convert To Image and then saved with the
% Save Images module. This module produces several additional types of
% objects with names that are automatically passed along with the following
% naming structure: (1) The unedited segmented image, which includes
% objects on the edge of the image and objects that are outside the size
% range, can be saved using the name: UneditedSegmented + whatever you
% called the objects (e.g. UneditedSegmentedNuclei). (2) The segmented
% image which excludes objects smaller than your selected size range can be
% saved using the name: SmallRemovedSegmented + whatever you called the
% objects (e.g. SmallRemovedSegmented Nuclei).
%
% See also MeasureObjectAreaShape, MeasureObjectIntensity, MeasureTexture,
% MeasureCorrelation, CalculateRatios, MeasureObjectNeighbors.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision: 2221 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the objects you want to filter?
%infotypeVAR01 = objectgroup
%inputtypeVAR01 = popupmenu
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call the filtered objects?
%defaultVAR02 = FilteredNuclei
%infotypeVAR02 = objectgroup indep
TargetName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Which category of measurements do you want to filter by?  This module must be run after a Measure module.
%choiceVAR03 = AreaShape
%choiceVAR03 = Correlation
%choiceVAR03 = Intensity
%choiceVAR03 = Neighbors
%choiceVAR03 = Ratio
%choiceVAR03 = Texture
%inputtypeVAR03 = popupmenu
MeasureChoice = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = For INTENSITY or TEXTURE features, which image's measurements do you want to use (for other measurements, this will only affect the display)?
%infotypeVAR04 = imagegroup
%inputtypeVAR04 = popupmenu
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = For RATIO, which object was used to calculate the numerator?
%infotypeVAR05 = objectgroup
%inputtypeVAR05 = popupmenu
RatioNum = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Which feature number do you want to use as a filter? See help for details.
%defaultVAR06 = 1
FeatureNum = char(handles.Settings.VariableValues{CurrentModuleNum,6});
FeatureNum = str2num(FeatureNum);

%textVAR07 = Minimum value required:
%choiceVAR07 = No minimum
%inputtypeVAR07 = popupmenu custom
MinValue1 = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Maximum value allowed:
%choiceVAR08 = No maximum
%inputtypeVAR08 = popupmenu custom
MaxValue1 = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = What do you want to call the outlines of the identified objects (optional)?
%defaultVAR09 = Do not save
%infotypeVAR09 = outlinegroup indep
SaveOutlines = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','CheckScale');
LabelMatrixImage = CPretrieveimage(handles,['Segmented' ObjectName],ModuleName,'MustBeGray','DontCheckScale');

if strcmp(MeasureChoice,'Intensity')
    fieldname = ['Intensity_',ImageName];
    MeasureInfo = handles.Measurements.(ObjectName).(fieldname){handles.Current.SetBeingAnalyzed}(:,FeatureNum);
elseif strcmp(MeasureChoice,'Texture')
    fieldname = ['Texture_',ImageName];
    MeasureInfo = handles.Measurements.(ObjectName).(fieldname){handles.Current.SetBeingAnalyzed}(:,FeatureNum);
elseif strcmp(MeasureChoice,'Neighbors')
    fieldname = 'NumberNeighbors';
    MeasureInfo = handles.Measurements.(ObjectName).(fieldname){handles.Current.SetBeingAnalyzed}(:,1);
elseif strcmp(MeasureChoice,'Ratio')
    MeasureInfo = handles.Measurements.(RatioNum).(MeasureChoice){handles.Current.SetBeingAnalyzed}(:,FeatureNum);
else
    MeasureInfo = handles.Measurements.(ObjectName).(MeasureChoice){handles.Current.SetBeingAnalyzed}(:,FeatureNum);
end

if strcmpi(MinValue1, 'No minimum')
    MinValue1 = -Inf;
else
    MinValue1 = str2double(MinValue1);
end

if strcmpi(MaxValue1, 'No maximum')
    MaxValue1 = Inf;
else
    MaxValue1 = str2double(MaxValue1);
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

Filter = find((MeasureInfo < MinValue1) | (MeasureInfo > MaxValue1));
FinalLabelMatrixImage = LabelMatrixImage;
for i=1:numel(Filter)
    FinalLabelMatrixImage(FinalLabelMatrixImage == Filter(i)) = 0;
end

x = sortrows(unique([LabelMatrixImage(:) FinalLabelMatrixImage(:)],'rows'),1);
x(x(:,2)>0,2)=1:sum(x(:,2)>0);
LookUpColumn = x(:,2);

FinalLabelMatrixImage = LookUpColumn(FinalLabelMatrixImage+1);

%%% Note: these outlines are not perfectly accurate; for some reason it
%%% produces more objects than in the original image.  But it is OK for
%%% display purposes.
%%% Maximum filters the image with a 3x3 neighborhood.
MaxFilteredImage = ordfilt2(FinalLabelMatrixImage,9,ones(3,3),'symmetric');
%%% Determines the outlines.
IntensityOutlines = FinalLabelMatrixImage - MaxFilteredImage;
%%% Converts to logical.
warning off MATLAB:conversionToLogical
LogicalOutlines = logical(IntensityOutlines);
warning on MATLAB:conversionToLogical
%%% Determines the grayscale intensity to use for the cell outlines.
LineIntensity = max(OrigImage(:));
%%% Overlays the outlines on the original image.
ObjectOutlinesOnOrigImage = OrigImage;
ObjectOutlinesOnOrigImage(LogicalOutlines) = LineIntensity;

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'TwoByTwo',ThisModuleFigureNumber);
    end
    %%% A subplot of the figure window is set to display the original
    %%% image.
    subplot(2,2,1); 
    CPimagesc(OrigImage,handles);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the label
    %%% matrix image.
    subplot(2,2,3); 
    CPimagesc(LabelMatrixImage,handles);
    title(['Original ',ObjectName]);
    %%% A subplot of the figure window is set to display the Overlaid image,
    %%% where the maxima are imposed on the inverted original image
    ColoredLabelMatrixImage = CPlabel2rgb(handles,FinalLabelMatrixImage);
    
    subplot(2,2,2); 
    CPimagesc(ColoredLabelMatrixImage,handles);
    title(['Filtered ' ObjectName]);
    subplot(2,2,4); 
    CPimagesc(ObjectOutlinesOnOrigImage,handles);
    title([TargetName, ' Outlines on Input Image']);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

handles.Pipeline.(['Segmented' TargetName]) = FinalLabelMatrixImage;

fieldname = ['SmallRemovedSegmented', ObjectName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)
    handles.Pipeline.(['SmallRemovedSegmented' TargetName]) = handles.Pipeline.(['SmallRemovedSegmented',ObjectName]);
end

fieldname = ['UneditedSegmented',ObjectName];
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, fieldname)
    handles.Pipeline.(['UneditedSegmented' TargetName]) = handles.Pipeline.(['UneditedSegmented',ObjectName]);
end

%%% Saves the ObjectCount, i.e., the number of segmented objects.
%%% See comments for the Threshold saving above
if ~isfield(handles.Measurements.Image,'ObjectCountFeatures')
    handles.Measurements.Image.ObjectCountFeatures = {};
    handles.Measurements.Image.ObjectCount = {};
end
column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ObjectCountFeatures,TargetName)));
if isempty(column)
    handles.Measurements.Image.ObjectCountFeatures(end+1) = {['ObjectCount ' TargetName]};
    column = length(handles.Measurements.Image.ObjectCountFeatures);
end
handles.Measurements.Image.ObjectCount{handles.Current.SetBeingAnalyzed}(1,column) = max(FinalLabelMatrixImage(:));

%%% Saves the location of each segmented object
handles.Measurements.(TargetName).LocationFeatures = {'CenterX','CenterY'};
tmp = regionprops(FinalLabelMatrixImage,'Centroid');
Centroid = cat(1,tmp.Centroid);
handles.Measurements.(TargetName).Location(handles.Current.SetBeingAnalyzed) = {Centroid};

if ~strcmp(SaveOutlines,'Do not save')
    try handles.Pipeline.(SaveOutlines) = LogicalOutlines;
    catch
        error(['The object outlines were not calculated by the ', ModuleName, ' module so these images were not saved to the handles structure. Image processing is still in progress, but the Save Images module will fail if you attempted to save these images.'])
    end
end