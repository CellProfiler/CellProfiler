function handles = FilterByAreaShape(handles)

% Help for the Filter Objects by AreaShape module: 
% Category: Object Processing
%
% This module applies a filter using statistics measured by the 
% MeasureObjectAreaShape module to select objects with desired area or shape. For
% example, it can be used to eliminate objects with a Solidity value below
% a certain threshold.
%
% See also MEASUREOBJECTAREASHAPE

CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the original image?
%infotypeVAR01 = imagegroup
%inputtypeVAR01 = popupmenu
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What did you call the objects you want to process?
%infotypeVAR02 = objectgroup
%inputtypeVAR02 = popupmenu
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What do you want to call the filtered objects?
%defaultVAR03 = FilteredNuclei
%infotypeVAR03 = objectgroup indep
TargetName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What feature do you want to  use as a filter? Please run this module after MeasureObjectAreaShape module.
%choiceVAR04 = Area
%choiceVAR04 = Eccentricity
%choiceVAR04 = Solidity
%choiceVAR04 = Extent
%choiceVAR04 = Euler Number
%choiceVAR04 = Perimeter
%choiceVAR04 = Form factor
%choiceVAR04 = MajorAxisLength
%choiceVAR04 = MinorAxisLength
%inputtypeVAR04 = popupmenu
FeatureName1 = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Minimum value required:
%choiceVAR05 = 0.5
%choiceVAR05 = Do not use
%inputtypeVAR05 = popupmenu custom
MinValue1 = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Maximum value allowed:
%choiceVAR06 = Do not use
%inputtypeVAR06 = popupmenu custom
MaxValue1 = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = What do you want to call the image of the colored objects?
%choiceVAR07 = Do not save
%infotypeVAR07 = imagegroup indep
%inputtypeVAR07 = popupmenu custom
SaveColored = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = What do you want to call the image of the outlines of the objects?
%choiceVAR08 = Do not save
%infortypeVAR08 = imagegroup indep
%inputtypeVAR08 = popupmenu custom
SaveOutlined = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%%%VariableRevisionNumber = 2



OrigImage = handles.Pipeline.(ImageName);
LabelMatrixImage = handles.Pipeline.(['Segmented' ObjectName]);
FilterType = strmatch(FeatureName1, handles.Measurements.(ObjectName).AreaShapeFeatures);
AreaShapeInfo = handles.Measurements.(ObjectName).AreaShape{handles.Current.SetBeingAnalyzed}(:,FilterType);

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

Filter = find((AreaShapeInfo < MinValue1) | (AreaShapeInfo > MaxValue1));
FinalLabelMatrixImage = LabelMatrixImage;
for i=1:numel(Filter)
    FinalLabelMatrixImage(FinalLabelMatrixImage == Filter(i)) = 0;
end

FinalLabelMatrixImage = bwlabel(FinalLabelMatrixImage);

%%% Calculates the object outlines, which are overlaid on the original
%%% image and displayed in figure subplot (2,2,4).
%%% Creates the structuring element that will be used for dilation.
StructuringElement = strel('square',3);
%%% Converts the FinalLabelMatrixImage to binary.
FinalBinaryImage = im2bw(FinalLabelMatrixImage,.1);
%%% Dilates the FinalBinaryImage by one pixel (8 neighborhood).
DilatedBinaryImage = imdilate(FinalBinaryImage, StructuringElement);
%%% Subtracts the FinalBinaryImage from the DilatedBinaryImage,
%%% which leaves the PrimaryObjectOutlines.
PrimaryObjectOutlines = DilatedBinaryImage - FinalBinaryImage;
%%% Overlays the object outlines on the original image.
ObjectOutlinesOnOrigImage = OrigImage;
%%% Determines the grayscale intensity to use for the cell outlines.
LineIntensity = max(OrigImage(:));
ObjectOutlinesOnOrigImage(PrimaryObjectOutlines == 1) = LineIntensity;

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow 

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1
    drawnow
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(OrigImage);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,3); imagesc(LabelMatrixImage); title(['Segmented ',ObjectName]);
    %%% A subplot of the figure window is set to display the Overlaid image,
    %%% where the maxima are imposed on the inverted original image
    try
        ColoredLabelMatrixImage = CPlabel2rgb(handles,FinalLabelMatrixImage);
    catch
        ColoredLabelMatrixImage = FinalLabelMatrixImage;
    end
    subplot(2,2,2); imagesc(ColoredLabelMatrixImage); title(['Filtered ' ObjectName]);
    %%% A subplot of the figure window is set to display the inverted original
    %%% image with watershed lines drawn to divide up clusters of objects.
    subplot(2,2,4); imagesc(ObjectOutlinesOnOrigImage); title([TargetName, ' Outlines on Input Image']);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow



handles.Pipeline.(['Segmented' TargetName]) = FinalLabelMatrixImage;

if ~strcmp(SaveColored,'Do not save')
    try handles.Pipeline.(SaveColored) = ColoredLabelMatrixImage;
    catch
        errordlg('The object outlines were not calculated by the IdentifyPrimAutomatic module (possibly because the window is closed) so these images were not saved to the handles structure. Image processing is still in progress, but the Save Images module will fail if you attempted to save these images.')
    end
end
if ~strcmp(SaveOutlined,'Do not save')
    try handles.Pipeline.(SaveOutlined) = ObjectOutlinesOnOrigImage;
    catch
        errordlg('The object outlines were not calculated by the IdentifyPrimAutomatic module (possibly because the window is closed) so these images were not saved to the handles structure. Image processing is still in progress, but the Save Images module will fail if you attempted to save these images.')
    end
end