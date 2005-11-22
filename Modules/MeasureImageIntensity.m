function handles = MeasureImageIntensity(handles)

% Help for the Measure Image Intensity module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Measures the total image intensity by summing every pixels intensity. The
% user can choose to ignore pixels below or above a particular intensity
% level.
% *************************************************************************
%
% Settings:
%
% You may tell the module to ignore pixels above or below a pixel
% intensity value that you specify, in the range 0 to 1 (use 'Show
% pixel data' to see the pixel intensities for your images in the
% appropriate range of 0 to 1). Leaving these values at 0 and 1 means
% that every pixel intensity will be included in the measurement.
%
% See also MEASUREAREAOCCUPIED, MEASUREOBJECTINTENSITY, MEASURECORRELATION.

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
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find  the
%%% variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

%textVAR01 = What did you call the images you want to process?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the staining measured by this module?
%defaultVAR02 = Fluorescence
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Ignore pixels below this intensity level (Range = 0-1)
%defaultVAR03 = 0
LowThreshold = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%textVAR04 = Ignore pixels above this intensity level (Range = 0-1)
%defaultVAR04 = 1
HighThreshold = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = Exclude pixels within this many pixels of an excluded bright object
%defaultVAR05 = 0
ExpansionDistance = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%%% Retrieves the pixel size that the user entered (micrometers per pixel).
PixelSize = str2double(handles.Settings.PixelSize);

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
%%% Checks whether image has been loaded.
if isfield(handles.Pipeline, ImageName)==0,
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
OrigImage = handles.Pipeline.(ImageName);

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error(['Image processing was canceled in the ', ModuleName, ' module because it requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.'])
end

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

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        %%% Sets the width of the figure window to be appropriate (half width).
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 280;
        set(ThisModuleFigureNumber, 'position', newsize);
    end
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original
    %%% image.
    subplot(2,1,1); CPimagesc(OrigImage);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the processed
    %%% image.
    subplot(2,1,2); CPimagesc(ThresholdedOrigImage); title('Thresholded Image');

    displaytexthandle = uicontrol(ThisModuleFigureNumber,'style','text', 'position', [65 -10 265 55],'fontname','helvetica','backgroundcolor',[.7 .7 .9],'FontSize',handles.Preferences.FontSize);
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
featurefieldname = ['Intensity_',ImageName,'Features'];
fieldname = ['Intensity_',ImageName];
handles.Measurements.Image.(featurefieldname) = {'Total intensity','Mean intensity','Total area'};
handles.Measurements.Image.(fieldname)(handles.Current.SetBeingAnalyzed) = {[TotalIntensity MeanIntensity TotalArea]};