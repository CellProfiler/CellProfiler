function handles = RescaleIntensity(handles)

% Help for the Rescale Intensity module:
% Category: Image Processing
%
% The intensity of the incoming images are rescaled by one of several
% methods.
%
% Settings:
%
% Rescaling method:
% (S) Stretch the image so that the minimum is zero and the maximum is
% one.
% (E) Enter the minimum and maximum values of the original image
% and the resulting image. Pixels are scaled from their user-specified
% original range to a new user-specified range.  If the user enters
% "AE", then the highest and lowest pixel values will be Automatically
% computed for Each image by taking the maximum and minimum pixel
% values in Each image.  If the user enters "AA", then the highest and
% lowest pixel values will be Automatically computed by taking the
% maximum and minimum pixel values in All the images in the set.
% Pixels in the original image that are above or below the original
% range are pinned to the high/low values of that range before being
% scaled.
% (G) rescale the image so that all pixels are equal to or Greater
% than one.
% (M) Match the maximum of one image to the maximum of another.
% (C) Convert to 8 bit: Images in CellProfiler are
% normally stored as numerical class double in the range of 0 to 1.
% This option converts these images to class uint8, meaning an 8 bit
% integer in the range of 0 to 255. This is useful to reduce the amount
% of memory required to store the image. Warning: Most CellProfiler
% modules require the incoming image to be in the standard 0 to 1
% range, so this conversion may cause downstream modules to behave
% unexpectedly.
%
% SAVING IMAGES: The rescaled images produced by this module can be
% easily saved using the Save Images module, using the name you
% assign.

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
%   Ola Friman     <friman@bwh.harvard.edu>
%   Steve Lowe     <stevelowe@alum.mit.edu>
%   Joo Han Chang  <joohan.chang@gmail.com>
%   Colin Clarke   <colinc@mit.edu>
%   Mike Lamprecht <mrl@wi.mit.edu>
%   Susan Ma       <xuefang_ma@wi.mit.edu>
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

%textVAR06 = (Method E only): What should the lowest intensity of the rescaled image be?
%defaultVAR06 = 0
LowestPixelRescale = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = (Method E only): What should the highest intensity of the rescaled image be?
%defaultVAR07 = 1
HighestPixelRescale = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%textVAR08 = (Method M only): What did you call the image whose maximum you want the rescaled image to match?
%infotypeVAR08 = imagegroup
OtherImageName = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image to be analyzed and assigns it to a variable,
%%% "OrigImage".
fieldname = ['', ImageName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Rescale Intensity module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
OrigImage = handles.Pipeline.(fieldname);

if max(OrigImage(:)) > 1 || min(OrigImage(:)) < 0
    CPwarndlg('The images you have loaded are outside the 0-1 range, and you may be losing data.','Outside 0-1 Range','replace');
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS%%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

if strncmpi(RescaleOption,'S',1) == 1
    MethodSpecificArguments = [];
elseif strncmpi(RescaleOption,'M',1) == 1
    %%% Reads (opens) the image to be analyzed and assigns it to a variable,
    %%% "OrigImage".
    fieldname = ['', OtherImageName];
    %%% Checks whether the image to be analyzed exists in the handles structure.
    if isfield(handles.Pipeline, fieldname)==0,
        %%% If the image is not there, an error message is produced.  The error
        %%% is not displayed: The error function halts the current function and
        %%% returns control to the calling function (the analyze all images
        %%% button callback.)  That callback recognizes that an error was
        %%% produced because of its try/catch loop and breaks out of the image
        %%% analysis loop without attempting further modules.
        error(['Image processing was canceled because the Rescale Intensity module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    %%% Reads the image.
    MethodSpecificArguments = handles.Pipeline.(fieldname);
elseif strncmpi(RescaleOption,'G',1) == 1
    MethodSpecificArguments = [];
elseif strncmpi(RescaleOption,'E',1) == 1
    MethodSpecificArguments{1} = LowestPixelOrig;
    MethodSpecificArguments{2} = HighestPixelOrig;
    MethodSpecificArguments{3} = LowestPixelRescale;
    MethodSpecificArguments{4} = HighestPixelRescale;
    MethodSpecificArguments{5} = ImageName;
elseif strncmpi(RescaleOption,'C',1) == 1
    MethodSpecificArguments = [];
end

%%% Uses a CellProfiler subfunction.
[handles,RescaledImage] = CPrescale(handles,OrigImage,RescaleOption,MethodSpecificArguments);

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
%%% Check whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this module.
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
    %%% A subplot of the figure window is set to display the Rescaled
    %%% Image.
    subplot(2,1,2); imagesc(RescaledImage); title('Rescaled Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The Rescaled image is saved to the handles structure so it can be
%%% used by subsequent modules.
handles.Pipeline.(RescaledImageName) = RescaledImage;