function handles = RescaleIntensity(handles)

% Help for the Rescale Intensity module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Changes intensity range of an image to desired specifications.
% *************************************************************************
%
% The intensity of the incoming images are rescaled by one of several
% methods. This is especially helpful for converting 12-bit images saved in
% 16-bit format to the correct range (see method E).
%
% Settings:
%
% Rescaling method:
% (S) Stretch the image so that the minimum is zero and the maximum is
% one.
% (E) Enter the minimum and maximum values of the original image and the
% desired resulting image. Pixels are scaled from their user-specified original
% range to a new user-specified range.  If the user enters "AE", then the
% highest and lowest pixel values will be Automatically computed for Each
% image by taking the maximum and minimum pixel values in Each image.  If
% the user enters "AA", then the highest and lowest pixel values will be
% Automatically computed by taking the maximum and minimum pixel values in
% All the images in the set. Pixels in the original image that are above or
% below the original range are pinned to the high/low values of that range
% before being scaled. To convert 12-bit images saved in 16-bit format to
% the correct range, use the settings 0, 0.0625, 0, 1.  The value 0.0625 is
% equivalent to 2^12 divided by 2^16, so it will convert a 16 bit image
% containing only 12 bits of data to the proper range.
% (G) rescale the image so that all pixels are equal to or Greater
% than one.
% (M) Match the maximum of one image to the maximum of another.
% (C) Convert to 8 bit: Images in CellProfiler are normally stored as
% numerical class double in the range of 0 to 1. This option converts these
% images to class uint8, meaning an 8 bit integer in the range of 0 to 255.
% This is useful to reduce the amount of memory required to store the
% image. Warning: Most CellProfiler modules require the incoming image to
% be in the standard 0 to 1 range, so this conversion may cause downstream
% modules to behave unexpectedly.

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

%textVAR01 = What did you call the image to be rescaled?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the rescaled image?
%defaultVAR02 = RescaledBlue
%infotypeVAR02 = imagegroup indep
RescaledImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Rescaling method. (S) Stretch the image (0 to 1). (E) Enter the minimum and maximum values in the boxes below. (G) rescale so all pixels are equal to or Greater than one. (M) Match the maximum of one image to the maximum of another. (C) Convert to 8 bit. See the help for details.
%choiceVAR03 = Stretch 0 to 1
%choiceVAR03 = Enter min/max below
%choiceVAR03 = Greater than one
%choiceVAR03 = Match Maximum
%choiceVAR03 = Convert to 8 bit
RescaleOption = char(handles.Settings.VariableValues{CurrentModuleNum,3});
RescaleOption = RescaleOption(1);
%inputtypeVAR03 = popupmenu

%textVAR04 = (Method E only): Enter the intensity from the original image that should be set to the lowest value in the rescaled image, or type AA to calculate the lowest intensity automatically from all of the images to be analyzed and AE to calculate the lowest intensity from each image independently.
%defaultVAR04 = AA
LowestPixelOrig = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = (Method E only): Enter the intensity from the original image that should be set to the highest value in the rescaled image, or type AA to calculate the highest intensity automatically from all of the images to be analyzed and AE to calculate the highest intensity from each image independently.
%defaultVAR05 = AA
HighestPixelOrig = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = (Method E only): What should the lowest intensity of the rescaled image be (range [0,1])?
%defaultVAR06 = 0
LowestPixelRescale = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = (Method E only): What should the highest intensity of the rescaled image be (range [0,1])?
%defaultVAR07 = 1
HighestPixelRescale = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%textVAR08 = (Method M only): What did you call the image whose maximum you want the rescaled image to match?
%infotypeVAR08 = imagegroup
OtherImageName = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image to be analyzed and assigns it to a variable,
%%% "OrigImage".
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,0,1);

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strncmpi(RescaleOption,'S',1)
    MethodSpecificArguments = [];
elseif strncmpi(RescaleOption,'M',1)
    %%% Reads (opens) the image to be analyzed and assigns it to a variable,
    %%% "MethodSpecificArguments".
    MethodSpecificArguments = CPretrieveimage(handles,OtherImageName,ModuleName,0,1);
elseif strncmpi(RescaleOption,'G',1)
    MethodSpecificArguments = [];
elseif strncmpi(RescaleOption,'E',1)
    MethodSpecificArguments{1} = LowestPixelOrig;
    MethodSpecificArguments{2} = HighestPixelOrig;
    MethodSpecificArguments{3} = LowestPixelRescale;
    MethodSpecificArguments{4} = HighestPixelRescale;
    MethodSpecificArguments{5} = ImageName;
elseif strncmpi(RescaleOption,'C',1)
    MethodSpecificArguments = [];
end

%%% Uses a CellProfiler subfunction.
[handles,RescaledImage] = CPrescale(handles,OrigImage,RescaleOption,MethodSpecificArguments);

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
%%% Check whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this module.
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'TwoByOne')
    end
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1); CPimagesc(OrigImage);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the Rescaled
    %%% Image.
    subplot(2,1,2); CPimagesc(RescaledImage); 
    title('Rescaled Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The Rescaled image is saved to the handles structure so it can be
%%% used by subsequent modules.
handles.Pipeline.(RescaledImageName) = RescaledImage;