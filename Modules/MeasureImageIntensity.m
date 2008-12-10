function handles = MeasureImageIntensity(handles,varargin)

% Help for the Measure Image Intensity module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Measures the total image intensity by summing every pixel's intensity,
% but can discard some pixel values if desired.
% *************************************************************************
%
% This module will sum all pixel values to measure the total image
% intensity. The user can also choose to ignore pixels below or above a
% particular intensity level.
%
% Features measured:      Feature Number:
% TotalIntensity       |         1
% MeanIntensity        |         2
% TotalArea            |         3
%
%
% Settings:
%
% You may tell the module to ignore pixels above or below a pixel intensity
% value that you specify, in the range 0 to 1 (use the CellProfiler image
% tool 'ShowOrHidePixelData' to see the pixel intensities for your images
% in the appropriate range of 0 to 1). Leaving these values at 0 and 1
% means that every pixel intensity will be included in the measurement.
% This setting is useful to adjust when you are attempting to exclude
% bright artifactual objects: you can first set the threshold to exclude
% these bright objects, but it may also be desirable to expand the
% thresholded region around those bright objects by a certain distance so
% as to avoid a 'halo' effect.
%
% For publication purposes, it is important to note that the units of
% intensity from microscopy images are usually described as "Intensity
% units" or "Arbitrary intensity units" since microscopes are not 
% callibrated to an absolute scale. Also, it is important to note whether 
% you are reporting either the mean or the total intensity, so specify
% "Mean intensity units" or "Total intensity units" accordingly.
%
% See also MeasureObjectIntensity.
%
% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the images you want to process?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = Ignore pixels below this intensity level (Range = 0-1)
%defaultVAR02 = 0
LowThreshold = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,2}));

%textVAR03 = Ignore pixels above this intensity level (Range = 0-1)
%defaultVAR03 = 1
HighThreshold = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%textVAR04 = Exclude pixels within this many pixels of an excluded bright object
%defaultVAR04 = 0
ExpansionDistance = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%%% Retrieves the pixel size that the user entered (micrometers per pixel).
PixelSize = str2double(handles.Settings.PixelSize);

%%%%%%%%%%%%%%%%
%%% FEATURES %%%
%%%%%%%%%%%%%%%%

if nargin > 1 
    switch varargin{1}
%feature:categories
        case 'categories'
            if nargin == 1 || strcmp(varargin{2},'Image')
                result = { 'Intensity' };
            else
                result = {};
            end
%feature:measurements
        case 'measurements'
            result = {};
            if nargin >= 3 &&...
                strcmp(varargin{3},'Intensity') &&...
                strcmp(varargin{2},'Image')
                result = {'TotalIntensity','MeanIntensity','TotalArea' };
            end
        otherwise
            error(['Unhandled category: ',varargin{1}]);
    end
    handles=result;
    return;
end

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the incoming image and assigns it to a variable.
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','CheckScale');

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Subtracts the threshold from the original image.
ThresholdedOrigImage = OrigImage - LowThreshold;
ThresholdedOrigImage(ThresholdedOrigImage < 0) = 0;

%%% Expands the mask around bright regions if requested.
if ExpansionDistance ~= 0
    BinaryBrightRegions = im2bw(ThresholdedOrigImage,HighThreshold-LowThreshold);
    ExpandedBinaryBrightRegions = bwmorph(BinaryBrightRegions, 'dilate', ExpansionDistance);
    ThresholdedOrigImage(ExpandedBinaryBrightRegions == 1) = 0;
else
    %%% The low threshold is subtracted because it was subtracted from the
    %%% whole image above.
    ThresholdedOrigImage(ThresholdedOrigImage > (HighThreshold-LowThreshold)) = 0;
end

TotalIntensity = sum(sum(ThresholdedOrigImage));
TotalArea = sum(sum(ThresholdedOrigImage>0));
%%% Converts to micrometers.
TotalArea = TotalArea*PixelSize*PixelSize;
MeanIntensity = TotalIntensity/TotalArea;

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'TwoByOne',ThisModuleFigureNumber)
    end
    %%% A subplot of the figure window is set to display the original
    %%% image.
    hAx=subplot(2,1,1,'Parent',ThisModuleFigureNumber); 
    CPimagesc(OrigImage,handles,hAx);
    title(hAx,['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the processed
    %%% image.
    hAx=subplot(2,1,2,'Parent',ThisModuleFigureNumber); 
    CPimagesc(ThresholdedOrigImage,handles,hAx); 
    title(hAx,'Thresholded Image');
    if isempty(findobj('Parent',ThisModuleFigureNumber,'tag','TextUIControl'))
        displaytexthandle = uicontrol(ThisModuleFigureNumber,'tag','TextUIControl','style','text', 'position', [0 0 200 60],'fontname','helvetica','backgroundcolor',[.7 .7 .9],'FontSize',handles.Preferences.FontSize);
    else
        displaytexthandle = findobj('Parent',ThisModuleFigureNumber,'tag','TextUIControl');
    end
    displaytext = {['Total intensity:      ', num2str(TotalIntensity, '%2.1E')],...
        ['Mean intensity:      ', num2str(MeanIntensity)],...
        ['Total area after thresholding:', num2str(TotalArea, '%2.1E')]};
    set(displaytexthandle,'string',displaytext, 'HorizontalAlignment', 'left')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves measurements to the handles structure.
handles = CPaddmeasurements(handles, 'Image', ...
                ['Intensity_TotalIntensity_',ImageName], ...
			    TotalIntensity);
handles = CPaddmeasurements(handles, 'Image', ...
                ['Intensity_MeanIntensity_',ImageName], ...
			    MeanIntensity);
handles = CPaddmeasurements(handles, 'Image', ...
                ['Intensity_TotalArea_',ImageName], ...
			    TotalArea);