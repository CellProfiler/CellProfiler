function handles = MeasureImageAreaOccupied(handles)

% Help for the Measure Image Area Occupied module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Measures total area covered by stain in an image.
% *************************************************************************
%
% This module simply measures the total area covered by stain in an
% image.
%
% Settings:
%
% Threshold: The threshold affects the stringency of the lines between
% the objects and the background. You may enter an absolute number
% between 0 and 1 for the threshold (use 'Show pixel data' to see the
% pixel intensities for your images in the appropriate range of 0 to
% 1), or you may have it calculated for each image individually by
% typing 0.  There are advantages either way.  An absolute number
% treats every image identically, but an automatically calculated
% threshold is more realistic/accurate, though occasionally subject to
% artifacts.  The threshold which is used for each image is recorded
% as a measurement in the output file, so if you find unusual
% measurements from one of your images, you might check whether the
% automatically calculated threshold was unusually high or low
% compared to the remaining images.  When an automatic threshold is
% selected, it may consistently be too stringent or too lenient, so an
% adjustment factor can be entered as well. The number 1 means no
% adjustment, 0 to 1 makes the threshold more lenient and greater than
% 1 (e.g. 1.3) makes the threshold more stringent.
%
% How it works:
% This module applies a threshold to the incoming image so that any
% pixels brighter than the specified value are assigned the value 1
% (white) and the remaining pixels are assigned the value zero
% (black), producing a binary image.  The number of white pixels are
% then counted.  This provides a measurement of the area occupied by
% fluorescence.  The threshold is calculated automatically and then
% adjusted by a user-specified factor. It might be desirable to write
% a new module where the threshold can be set to a constant value.
%
% See also MEASUREAREASHAPECOUNTLOCATION,
% MEASURECORRELATION,
% MEASUREINTENSITYTEXTURE,
% MEASURETOTALINTENSITY.

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

%textVAR02 = What do you want to call the staining measured by this module?
%defaultVAR02 = CellStain
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Select thresholding method or enter a threshold in the range [0,1].
%choiceVAR03 = MoG Global
%choiceVAR03 = MoG Adaptive
%choiceVAR03 = Otsu Global
%choiceVAR03 = Otsu Adaptive
%choiceVAR03 = All
%choiceVAR03 = Test Mode
Threshold = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu custom

%textVAR04 = Threshold correction factor
%defaultVAR04 = 1
ThresholdCorrection = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = Lower and upper bounds on threshold (in the range [0,1])
%defaultVAR05 = 0,1
ThresholdRange = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Approximate percentage of image covered by objects (for MoG thresholding only):
%choiceVAR06 = 10%
%choiceVAR06 = 20%
%choiceVAR06 = 30%
%choiceVAR06 = 40%
%choiceVAR06 = 50%
%choiceVAR06 = 60%
%choiceVAR06 = 70%
%choiceVAR06 = 80%
%choiceVAR06 = 90%
pObject = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

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
if ~isfield(handles.Pipeline, ImageName),
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
OrigImage = handles.Pipeline.(ImageName);

%%% Checks that the Min and Max threshold bounds have valid values
index = strfind(ThresholdRange,',');
if isempty(index)
    error(['Image processing was canceled in the ', ModuleName, ' module because the Min and Max threshold bounds are invalid.'])
end
MinimumThreshold = ThresholdRange(1:index-1);
MaximumThreshold = ThresholdRange(index+1:end);

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% STEP 1. Find threshold and apply to image

[handles,Threshold] = CPthreshold(handles,Threshold,pObject,MinimumThreshold,MaximumThreshold,ThresholdCorrection,OrigImage,ImageName,ModuleName);

%%% Thresholds the original image.
ThresholdedOrigImage = im2bw(OrigImage,Threshold);
AreaOccupiedPixels = sum(ThresholdedOrigImage(:));
AreaOccupied = AreaOccupiedPixels*PixelSize*PixelSize;

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber) == 1;
    drawnow
    %%% Sets the width of the figure window to be appropriate (half width).
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 0.5*originalsize(3);
        newsize(4) = 1.2*originalsize(4);
        newsize(2) = originalsize(2)-0.2*originalsize(4);
        set(ThisModuleFigureNumber, 'position', newsize);
    end
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1); CPimagesc(OrigImage);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,1,2); CPimagesc(ThresholdedOrigImage); title('Thresholded Image');
    if handles.Current.SetBeingAnalyzed == 1
        displaytexthandle = uicontrol(ThisModuleFigureNumber,'tag','DisplayText','style','text', 'position', [20 0 250 40],'fontname','fixedwidth','backgroundcolor',[0.7 0.7 0.9],'FontSize',handles.Preferences.FontSize);
    else
        displaytexthandle = findobj('Parent',ThisModuleFigureNumber,'tag','DisplayText');
    end
    displaytext = {['  Cycle # ',num2str(handles.Current.SetBeingAnalyzed)];...
        ['  Area occupied by ',ObjectName,':      ',num2str(AreaOccupied,'%2.1E')]};
    set(displaytexthandle,'string',displaytext)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

FeatureNames = {'ImageAreaOccupied','ImageAreaOccupiedThreshold'};
fieldname = ['AreaOccupied_',ObjectName,'Features'];
handles.Measurements.Image.(fieldname) = FeatureNames;

fieldname = ['AreaOccupied',ObjectName];
handles.Measurements.Image.(fieldname){handles.Current.SetBeingAnalyzed}(:,1) = AreaOccupied;
handles.Measurements.Image.(fieldname){handles.Current.SetBeingAnalyzed}(:,2) = Threshold;