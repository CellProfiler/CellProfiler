function handles = ColorToGray(handles)

% Help for the RGB To Gray module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Converts RGB (Red, Green Blue) color images to grayscale. All channels
% can be merged into one grayscale image or each channel can extracted into
% a seperate grayscale image. All identify modules require grayscale
% images.
% *************************************************************************
%
% Settings:
%
% Split:
% Takes an RGB image and splits into three separate grayscale images.
%
% Gray:
% Takes an RGB image and converts it to grayscale.  Each color's
% contribution to the final image can be adjusted independently.
%
% Adjustment factors: Leaving the adjustment factors set to 1 will
% balance all three colors equally in the final image, which will
% use the same range of intensities as the incoming image.  To weight
% colors relative to each other, the adjustment factor can be
% increased (to increase the weighting) or decreased (to decrease the
% weighting).
%
% SAVING IMAGES: The grayscale image produced by this module can be
% easily saved using the Save Images module, using the names you
% assign. If you want to save other intermediate images, alter the
% code for this module to save those images to the handles structure
% (see the SaveImages module help) and then use the Save Images
% module.
%
% See also RGBSPLIT, RGBMERGE.

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
% $Revision: 1718 $

drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

%textVAR01 = What did you call the image to be converted to Gray?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = How do you want to convert the RGB image?
%choiceVAR02 = Gray
%choiceVAR02 = Split
GrayOrSplit = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = GRAY OPTIONS:

%textVAR04 = What do you want to call the grayscale image?
%defaultVAR04 = OrigGray
%infotypeVAR04 = imagegroup indep
GrayscaleImageName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Enter the relative contribution of the red channel
%defaultVAR05 = 1
RedIntensity = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = Enter the relative contribution of the green channel
%defaultVAR06 = 1
GreenIntensity = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%textVAR07 = Enter the relative contribution of the blue channel
%defaultVAR07 = 1
BlueIntensity = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,7}));

%textVAR08 = SPLIT OPTIONS:

%textVAR09 = What do you want to call the image that was red? Type N to ignore red.
%defaultVAR09 = OrigRed
%infotypeVAR09 = imagegroup indep
RedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = What do you want to call the image that was green? Type N to ignore green.
%defaultVAR10 = OrigGreen
%infotypeVAR10 = imagegroup indep
GreenImageName = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = What do you want to call the image that was blue? Type N to ignore blue.
%defaultVAR11 = OrigBlue
%infotypeVAR11 = imagegroup indep
BlueImageName = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%%%VariableRevisionNumber = 1

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
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
OrigImage = handles.Pipeline.(fieldname);

if max(OrigImage(:)) > 1 || min(OrigImage(:)) < 0
    CPwarndlg('The images you have loaded are outside the 0-1 range, and you may be losing data.','Outside 0-1 Range','replace');
end

%%% Checks that the original image is three-dimensional (i.e. a color
%%% image)
if ndims(OrigImage) ~= 3
    error(['Image processing was canceled because the ', ModuleName, ' module requires a color image (an input image that is three-dimensional), but the image loaded does not fit this requirement.  This may be because the image is a grayscale image already.'])
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS%%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(GrayOrSplit,'Gray')
    %%% Converts Image to Gray
    InitialGrayscaleImage = OrigImage(:,:,1)*RedIntensity+OrigImage(:,:,2)*GreenIntensity+OrigImage(:,:,3)*BlueIntensity;
    %%% Divides by the sum of the weights to make sure the image is in the proper 0 to 1 range.
    GrayscaleImage = InitialGrayscaleImage/sum(RedIntensity+GreenIntensity+BlueIntensity);
elseif strcmp(GrayOrSplit,'Split')
    %%% Determines whether the user has specified an image to be loaded in
    %%% red.
    if ~strcmp(upper(RedImageName), 'N')
        RedImage = OrigImage(:,:,1);
    else
        RedImage = zeros(size(OrigImage(:,:,1)));
    end
    if ~strcmp(upper(GreenImageName), 'N')
        GreenImage = OrigImage(:,:,2);
    else
        GreenImage = zeros(size(OrigImage(:,:,1)));
    end
    if ~strcmp(upper(BlueImageName), 'N')
        BlueImage = OrigImage(:,:,3);
    else
        BlueImage = zeros(size(OrigImage(:,:,1)));
    end
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);

if strcmp(GrayOrSplit,'Gray')
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
        subplot(2,1,1);
        ImageHandle = imagesc(OrigImage);
        set(ImageHandle,'ButtonDownFcn','ImageTool(gco)');
        colormap(handles.Preferences.IntensityColorMap);
        title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
        %%% A subplot of the figure window is set to display the Grayscale
        %%% Image.
        subplot(2,1,2);
        ImageHandle = imagesc(GrayscaleImage);
        set(ImageHandle,'ButtonDownFcn','ImageTool(gco)');
        title('Grayscale Image');
    end
elseif strcmp(GrayOrSplit,'Split')
    if any(findobj == ThisModuleFigureNumber) == 1;
        drawnow
        %%% Activates the appropriate figure window.
        CPfigure(handles,ThisModuleFigureNumber);
        %%% A subplot of the figure window is set to display the Splitd RGB
        %%% image.  Using imagesc or image instead of imshow doesn't work when
        %%% some of the pixels are saturated.
        subplot(2,2,1);
        ImageHandle = imagesc(OrigImage);
        set(ImageHandle,'ButtonDownFcn','ImageTool(gco)');
        title(['Input RGB Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
        %%% A subplot of the figure window is set to display the blue image.
        subplot(2,2,2);
        ImageHandle = imagesc(BlueImage);
        set(ImageHandle,'ButtonDownFcn','ImageTool(gco)');
        title('Blue Image');
        %%% A subplot of the figure window is set to display the green image.
        subplot(2,2,3);
        ImageHandle = imagesc(GreenImage);
        set(ImageHandle,'ButtonDownFcn','ImageTool(gco)');
        title('Green Image');
        %%% A subplot of the figure window is set to display the red image.
        subplot(2,2,4);
        ImageHandle = imagesc(RedImage);
        set(ImageHandle,'ButtonDownFcn','ImageTool(gco)');
        title('Red Image');
        CPFixAspectRatio(OrigImage);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(GrayOrSplit,'Gray')
    %%% Saves the Grayscaled image to the handles structure so it can be
    %%% used by subsequent modules.
    fieldname = GrayscaleImageName;
    handles.Pipeline.(fieldname) = GrayscaleImage;
elseif strcmp(GrayOrSplit,'Split')
    %%% Saves the adjusted image to the handles structure so it can be used by
    %%% subsequent modules.
    handles.Pipeline.(RedImageName) = RedImage;
    handles.Pipeline.(GreenImageName) = GreenImage;
    handles.Pipeline.(BlueImageName) = BlueImage;
end