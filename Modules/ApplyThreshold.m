function handles = ApplyThreshold(handles)

% Help for the Apply Threshold module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Pixels below (or above) a certain threshold are set to zero. The
% remaining pixels retain their original value or (optional) are shifted to
% match the threshold used.
% *************************************************************************
%
% Settings: 
% If your answer was not 1, you can expand the thresholding around those
% excluded bright pixels by entering the number of pixels to expand here:
% This setting is useful to adjust when you are attempting to
% exclude bright artifactual objects: you can first set the threshold to
% exclude these bright objects, but it may also be desirable to expand the
% thresholded region around those bright objects by a certain distance so
% as to avoid a 'halo' effect.

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

%textVAR01 = What did you call the image to be thresholded?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the thresholded image?
%defaultVAR02 = ThreshBlue
%infotypeVAR02 = imagegroup indep
ThresholdedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Pixels below this value (Range = 0-1) will be set to zero (0 will not threshold any pixels)
%defaultVAR03 = 0
LowThreshold = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%textVAR04 = If your answer was not 0, do you want to shift the remaining pixels' intensities down by that intensity or retain their original values?
%choiceVAR04 = Retain
%choiceVAR04 = Shift
%inputtypeVAR04 = popupmenu
Shift = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Pixels above this value (Range = 0-1) will be set to zero (1 will not threshold any pixels)
%defaultVAR05 = 1
HighThreshold = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = If your answer was not 1, you can expand the thresholding around those excluded bright pixels by entering the number of pixels to expand here:
%defaultVAR06 = 0
DilationValue = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = Binary option: Enter the threshold to use to make the incoming image binary (black and white) where pixels equal to or below this value will be zero and above this value will be 1. If instead you want to use the settings above to preserve grayscale information, enter 0 here.
%defaultVAR07 = 0
BinaryChoice = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%%%VariableRevisionNumber = 4

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image to be analyzed and assigns it to a variable,
%%% "OrigImage".
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','CheckScale');

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

if BinaryChoice == 0
    %%% Identifies bright object pixels.
    BinaryBrightObjectsImage = zeros(size(OrigImage));
    BinaryBrightObjectsImage(OrigImage >= HighThreshold) = 1;
    StructuringElement = strel('disk',DilationValue,8);
    DilatedBinaryBrightObjectsImage = imdilate(BinaryBrightObjectsImage,StructuringElement);
    ThresholdedImage = OrigImage;
    ThresholdedImage(DilatedBinaryBrightObjectsImage == 1) = 0;
    if strcmp(Shift,'Retain')
        ThresholdedImage(ThresholdedImage <= LowThreshold) = 0;
    elseif strcmp(Shift,'Shift')
        ThresholdedImage = ThresholdedImage - LowThreshold;
        ThresholdedImage(ThresholdedImage < 0) = 0;
    end
else
    ThresholdedImage = im2bw(OrigImage,BinaryChoice);
end

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
    subplot(2,1,1);
    CPimagesc(OrigImage,handles);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the Thresholded
    %%% image.
    subplot(2,1,2);
    CPimagesc(ThresholdedImage,handles);
    title('Thresholded Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The Thresholded image is saved to the handles structure so it can be
%%% used by subsequent modules.
handles.Pipeline.(ThresholdedImageName) = ThresholdedImage;