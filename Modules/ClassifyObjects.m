function handles = ClassifyObjects(handles)

% Help for the Classify Objects module:
% Category: Other
%
% This module classifies objects into a number of different
% classes according to the size of a measurement specified
% by the user.
%
%
% SAVING IMAGES: To save images using this module, use the SaveImages
% module with the images to be saved called 'ColorClassified' plus
% the object name you enter in the module, e.g.
% 'ColorClassifiedCells'.
%
% See also <nothing relevant>.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$

drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the segmented objects?
%infotypeVAR01 = objectgroup
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = Enter the feature type, e.g. AreaShape, Texture, Intensity,...
%choiceVAR02 = AreaShape
%choiceVAR02 = Correlation
%choiceVAR02 = Texture
%choiceVAR02 = Intensity
%choiceVAR02 = Neighbors
%choiceVAR02 = Ratios
FeatureType = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu custom

%textVAR03 = Enter feature number
%defaultVAR03 = 1
FeatureNbr = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%textVAR04 = Enter the lower limit of the lower bin
%defaultVAR04 = 0
LowerBinMin = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = Enter the upper limit for the upper bin
%defaultVAR05 = 100
UpperBinMax = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = Enter number of bins (Note: to measure the percent of objects that are above a threshold, type P:XXX in this box, where XXX is the threshold).
%defaultVAR06 = 3
NbrOfBins = char(handles.Settings.VariableValues{CurrentModuleNum,6});

try
    if strncmpi(NbrOfBins,'P',1)
        MidPointToUse = str2double(NbrOfBins(3:end));
        NbrOfBins = 0;
    else
        NbrOfBins = str2double(NbrOfBins);
        if isempty(NbrOfBins) | NbrOfBins < 1
            errordlg('Image processing has been canceled because an error was found in the number of bins specification in the ClassifyObjects module.');
        end
    end
catch
    error('Image processing was canceled because you must enter a number, or the letter P for the number of bins in the Classify Objects module.')
end

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Retrieves the label matrix image that contains the segmented objects
fieldname = ['Segmented', ObjectName];

%%% Checks whether the image exists in the handles structure.
if ~isfield(handles.Pipeline, fieldname)
    errordlg(['Image processing has been canceled. Prior to running the ClassifyObject module, you must have previously run a module that generates an image with the objects identified.  You specified in the ClassifyObject module that the primary objects were named ',ObjectName,' which should have produced an image in the handles structure called ', fieldname, '. The ClassifyObject module cannot locate this image.']);
end
LabelMatrixImage = handles.Pipeline.(fieldname);

%%% Checks whether the feature type exists in the handles structure.
if ~isfield(handles.Measurements.(ObjectName),FeatureType)
    errordlg('The feature type entered in the ClassifyObjects module does not exist.');
end

if isempty(LowerBinMin) | isempty(UpperBinMax) | LowerBinMin > UpperBinMax
    errordlg('Image processing has been canceled because an error in the specification of the lower and upper limits was found in the ClassifyObjects module.');
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

% Get Measurements
Measurements = handles.Measurements.(ObjectName).(FeatureType){handles.Current.SetBeingAnalyzed}(:,FeatureNbr);

if NbrOfBins == 0
    edges = [LowerBinMin,MidPointToUse,UpperBinMax];
    NbrOfBins = 2;
else
    % Quantize measurements
    edges = linspace(LowerBinMin,UpperBinMax,NbrOfBins+1);
end
edges(1) = edges(1) - sqrt(eps);                               % Just a fix so that objects with a measurement that equals the lower bin edge of the lowest bin are counted
QuantizedMeasurements = zeros(size(Measurements));
bins = zeros(1,NbrOfBins);
for k = 1:NbrOfBins
    index = find(Measurements > edges(k) & Measurements <= edges(k+1));
    QuantizedMeasurements(index) = k;
    bins(k) = length(index);
end

% Produce image where the the objects are colored according to the original
% measurements and the quantized measurements
NonQuantizedImage = zeros(size(LabelMatrixImage));
NbrOfObjects = max(LabelMatrixImage(:));
props = regionprops(LabelMatrixImage,'PixelIdxList');              % Pixel indexes for objects fast
for k = 1:NbrOfObjects
    NonQuantizedImage(props(k).PixelIdxList) = Measurements(k);
end
QuantizedMeasurements = [0;QuantizedMeasurements];                 % Add a background class
QuantizedImage = QuantizedMeasurements(LabelMatrixImage+1);
cmap = [0 0 0;jet(length(bins))];
QuantizedRGBimage = ind2rgb(QuantizedImage+1,cmap);
    FeatureName = handles.Measurements.(ObjectName).([FeatureType,'Features']){FeatureNbr};

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;

    drawnow
    %%% Sets the width of the figure window to be appropriate (half width).
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = originalsize(3);
        set(ThisModuleFigureNumber, 'position', newsize);
    end
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);

    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1)
    ImageHandle = imagesc(NonQuantizedImage,[min(Measurements) max(Measurements)]);
    set(ImageHandle,'ButtonDownFcn','ImageTool(gco)','Tag',sprintf('%s colored accoring to %s',ObjectName,FeatureName))
    axis image
    set(gca,'Fontsize',handles.Current.FontSize)
    title(sprintf('%s colored accoring to %s',ObjectName,FeatureName))

    %%% Produce and plot histogram of original data
    subplot(2,2,2)
    Nbins = min(round(NbrOfObjects/5),40);
    hist(Measurements,Nbins)
    set(get(gca,'Children'),'FaceVertexCData',hot(Nbins));
    set(gca,'Fontsize',handles.Current.FontSize);
    xlabel(FeatureName),ylabel(['#',ObjectName]);
    title(sprintf('Histogram of %s',FeatureName));
    ylimits = ylim;
    axis tight
    xlimits = xlim;
    axis([xlimits ylimits])

    %%% A subplot of the figure window is set to display the quantized image.
    subplot(2,2,3)
    ImageHandle = image(QuantizedRGBimage);axis image
    set(ImageHandle,'ButtonDownFcn','ImageTool(gco)','Tag',['Classified ', ObjectName])
    set(gca,'Fontsize',handles.Current.FontSize)
    title(['Classified ', ObjectName],'fontsize',handles.Current.FontSize);

    %%% Produce and plot histogram
    subplot(2,2,4)
    x = edges(1:end-1) + (edges(2)-edges(1))/2;
    h = bar(x,bins,1);
    set(gca,'Fontsize',handles.Current.FontSize);
    xlabel(FeatureName),ylabel(['#',ObjectName])
    title(sprintf('Histogram of %s',FeatureName))
    axis tight
    xlimits(1) = min(xlimits(1),LowerBinMin);                          % Extend limits if necessary and save them
    xlimits(2) = max(xlimits(2),UpperBinMax);                          % so they can be used for the second histogram
    axis([xlimits ylim])
    set(get(h,'Children'),'FaceVertexCData',jet(NbrOfBins));

    CPFixAspectRatio(NonQuantizedImage);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% Saves images to the handles structure so they can be saved to the hard
%%% drive, if the user requests.
fieldname = ['ColorClassified',ObjectName];
handles.Pipeline.(fieldname) = QuantizedRGBimage;

ClassifyFeatureNames = cell(1,NbrOfBins);
for k = 1:NbrOfBins
    ClassifyFeatureNames{k} = ['Bin ',num2str(k)];
end
FeatureName = FeatureName(~isspace(FeatureName));                    % Remove spaces in the feature name
handles.Measurements.Image.(['ClassifyObjects_',ObjectName,'_',FeatureName,'Features']) = ClassifyFeatureNames;
handles.Measurements.Image.(['ClassifyObjects_',ObjectName,'_',FeatureName])(handles.Current.SetBeingAnalyzed) = {bins};


