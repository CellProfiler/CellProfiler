function handles = MeasureImageAreaOccupied(handles)

% Help for the Measure Image Area Occupied module:
% Category: Measurement
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
% SAVING IMAGES: If you want to save images produced by this module,
% alter the code for this module to save those images to the handles
% structure (see the SaveImages module help) and then use the Save
% Images module.
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
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$




drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%



%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the images you want to process?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the region measured by this module?
%defaultVAR02 = StainedRegion
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Enter the threshold (Positive number, Max = 1):
%choiceVAR03 = Automatic
Threshold = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu custom

%textVAR04 = If auto threshold, enter an adjustment factor (Positive number, 1 = no adjustment):
%defaultVAR04 = 1
ThresholdAdjustmentFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%%% Retrieves the pixel size that the user entered (micrometers per pixel).
PixelSize = str2double(handles.Settings.PixelSize);

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
fieldname = ['', ImageName];
%%% Checks whether image has been loaded.
if isfield(handles.Pipeline, fieldname)==0,
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Area Occupied module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
OrigImage = handles.Pipeline.(fieldname);


%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% Determines the threshold to be used, if the user has left the Threshold
%%% variable set to 0.
if strcmp(Threshold,'Automatic')
    Threshold = CPgraythresh(OrigImage,handles,ImageName);
    %%% Replaced the following line to accomodate calculating the
    %%% threshold for images that have been masked.
%    Threshold = CPgraythresh(OrigImage);
    Threshold = Threshold*ThresholdAdjustmentFactor;
else
    Threshold=str2double(Threshold);
end
%%% Thresholds the original image.
ThresholdedOrigImage = im2bw(OrigImage, Threshold);
AreaOccupiedPixels = sum(ThresholdedOrigImage(:));
AreaOccupied = AreaOccupiedPixels*PixelSize*PixelSize;

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
        newsize(3) = 0.5*originalsize(3);
        set(ThisModuleFigureNumber, 'position', newsize);
    end
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1); imagesc(OrigImage);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,1,2); imagesc(ThresholdedOrigImage); title('Thresholded Image');
    if handles.Current.SetBeingAnalyzed == 1
        displaytexthandle = uicontrol(ThisModuleFigureNumber,'style','text', 'position', [0 0 235 30],'fontname','fixedwidth','backgroundcolor',[0.7,0.7,0.7],'FontSize',handles.Current.FontSize);
    else
        displaytexthandle = findobj('Parent',ThisModuleFigureNumber,'Position',[0 0 235 30]);
    end
    displaytext = {['      Image Set # ',num2str(handles.Current.SetBeingAnalyzed)];...
        ['Area occupied by ', ObjectName ,':      ', num2str(AreaOccupied, '%2.1E')]};
    set(displaytexthandle,'string',displaytext)
    set(ThisModuleFigureNumber,'toolbar','figure')
    numel(findobj('Parent',ThisModuleFigureNumber))
   end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% Saves the Area Occupied measurement to the handles structure.
fieldname = ['ImageAreaOccupied', ObjectName];
handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {AreaOccupied};

%%% Saves the Threshold value to the handles structure.
fieldname = ['ImageAreaOccupiedThreshold', ObjectName];
handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {Threshold};
