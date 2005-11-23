function handles = FilterByObjectMeasurement(handles)

% Help for the Filter Objects by Measurement module: 
% Category: Object Processing
%
% SHORT DESCRIPTION:
% Eliminates objects based on their measurements (e.g. Area, Shape,
% Texture, Intensity).
% *************************************************************************
%
% This module applies a filter using measurements produced by either
% MeasureObjectAreaShape, MeasureObjectIntensity, or MeasureObjectTexture
% modules. All objects outside of the specified parameters will be
% discarded.
%
% Feature Number:
% The feature number is the parameter from the chosen module (AreaShape,
% Intensity, Texture) which will be used for the ratio. Please see
% individual measurement module help for list of measurements and their
% feature numbers.
%
% Special note on saving images: Using the settings in this module, object
% outlines can be passed along to the module OverlayOutlines and then saved
% with SaveImages. Objects themselves can be passed along to the object
% processing module ConvertToImage and then saved with SaveImages. This
% module produces several additional types of objects with names that are
% automatically passed along with the following naming structure: (1) The
% unedited segmented image, which includes objects on the edge of the image
% and objects that are outside the size range, can be saved using the name:
% UneditedSegmented + whatever you called the objects (e.g.
% UneditedSegmentedNuclei). (2) The segmented image which excludes objects
% smaller than your selected size range can be saved using the name:
% SmallRemovedSegmented + whatever you called the objects (e.g.
% SmallRemovedSegmented Nuclei).
%
% See also MEASUREOBJECTAREASHAPE, MEASUREOBJECTINTENSITY,
% MEASUREOBJECTTEXTURE.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter
%   Thouis Jones
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

%textVAR01 = What did you call the objects you want to process?
%infotypeVAR01 = objectgroup
%inputtypeVAR01 = popupmenu
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call the filtered objects?
%defaultVAR02 = FilteredNuclei
%infotypeVAR02 = objectgroup indep
TargetName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What measurement do you want to filter by?  This module must be run after a MeasureObject module.
%choiceVAR03 = AreaShape
%choiceVAR03 = Intensity
%choiceVAR03 = Texture
%inputtypeVAR03 = popupmenu
MeasureChoice = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = If using Intensity or Texture, what image was used to make the measurements? (This will also be used for the final display)
%infotypeVAR04 = imagegroup
%inputtypeVAR04 = popupmenu
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = What feature number do you want to  use as a filter? See the help for this module.
%defaultVAR05 = 1
FeatureNum = char(handles.Settings.VariableValues{CurrentModuleNum,5});
FeatureNum = str2num(FeatureNum);

%textVAR06 = Minimum value required:
%choiceVAR06 = 0.5
%choiceVAR06 = Do not use
%inputtypeVAR06 = popupmenu custom
MinValue1 = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Maximum value allowed:
%choiceVAR07 = Do not use
%inputtypeVAR07 = popupmenu custom
MaxValue1 = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = What do you want to call the image of the outlines of the objects?
%choiceVAR08 = Do not save
%infotypeVAR08 = outlinegroup indep
%inputtypeVAR08 = popupmenu custom
SaveOutlined = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

OrigImage = handles.Pipeline.(ImageName);
LabelMatrixImage = handles.Pipeline.(['Segmented' ObjectName]);

if strcmp(MeasureChoice,'Intensity')
    fieldname = ['Intensity_',ImageName];
    MeasureInfo = handles.Measurements.(ObjectName).(fieldname){handles.Current.SetBeingAnalyzed}(:,FeatureNum);
elseif strcmp(MeasureChoice,'AreaShape')
    MeasureInfo = handles.Measurements.(ObjectName).AreaShape{handles.Current.SetBeingAnalyzed}(:,FeatureNum);
elseif strcmp(MeasureChoice,'Texture')
    fieldname = ['Texture_',ImageName];
    MeasureInfo = handles.Measurements.(ObjectName).(fieldname){handles.Current.SetBeingAnalyzed}(:,FeatureNum);
end

if strcmp(MinValue1, 'Do not use')
    MinValue1 = -Inf;
else
    MinValue1 = str2double(MinValue1);
end

if strcmp(MaxValue1, 'Do not use')
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

FinalLabelMatrixImage = bwlabel(FinalLabelMatrixImage);

%%% Finds the perimeter of the objects
PerimObjects = bwperim(FinalLabelMatrixImage > 0);
%%% Pre-allocates space
PrimaryObjectOutlines = logical(zeros(size(FinalLabelMatrixImage,1),size(FinalLabelMatrixImage,2)));
%%% Places outlines on image
PrimaryObjectOutlines(PerimObjects) = 1;

%%% Overlays the object outlines on the original image.
ObjectOutlinesOnOrigImage = OrigImage;
ObjectOutlinesOnOrigImage(PrimaryObjectOutlines == 1) = 1;

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow 

ThisModuleFigureNumber = CPwhichmodulefigurenumber(CurrentModule);
if any(findobj == ThisModuleFigureNumber) == 1
    drawnow
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1);
    CPimagesc(OrigImage);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,3);
    CPimagesc(LabelMatrixImage);
    title(['Segmented ',ObjectName]);
    %%% A subplot of the figure window is set to display the Overlaid image,
    %%% where the maxima are imposed on the inverted original image
    try
        ColoredLabelMatrixImage = CPlabel2rgb(handles,FinalLabelMatrixImage);
    catch
        ColoredLabelMatrixImage = FinalLabelMatrixImage;
    end
    subplot(2,2,2);
    CPimagesc(ColoredLabelMatrixImage);
    title(['Filtered ' ObjectName]);
    %%% A subplot of the figure window is set to display the inverted original
    %%% image with watershed lines drawn to divide up clusters of objects.
    subplot(2,2,4);
    CPimagesc(ObjectOutlinesOnOrigImage);
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

if ~strcmp(SaveOutlined,'Do not save')
    try handles.Pipeline.(SaveOutlined) = PrimaryObjectOutlines;
    catch
        error(['The object outlines were not calculated by the ', ModuleName, ' module so these images were not saved to the handles structure. Image processing is still in progress, but the Save Images module will fail if you attempted to save these images.'])
    end
end