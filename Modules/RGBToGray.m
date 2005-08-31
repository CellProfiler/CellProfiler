function handles = RGBToGray(handles)

% Help for the RGB To Gray module:
% Category: Image Processing
%
% Takes an RGB image and converts it to grayscale.  Each color's
% contribution to the final image can be adjusted independently.
%
% Settings:
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

%textVAR01 = What did you call the image to be converted to Gray?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the grayscale image?
%defaultVAR02 = OrigGray
%infotypeVAR02 = imagegroup indep
GrayscaleImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Enter the relative contribution of the red channel
%defaultVAR03 = 1
RedIntensity = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%textVAR04 = Enter the relative contribution of the green channel
%defaultVAR04 = 1
GreenIntensity = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = Enter the relative contribution of the blue channel
%defaultVAR05 = 1
BlueIntensity = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

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
    error(['Image processing was canceled because the RGB to Gray module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
OrigImage = handles.Pipeline.(fieldname);

%%% Checks that the original image is three-dimensional (i.e. a color
%%% image)
if ndims(OrigImage) ~= 3
    error('Image processing was canceled because the RGB to Gray module requires a color image (an input image that is three-dimensional), but the image loaded does not fit this requirement.  This may be because the image is a grayscale image already.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS%%%
%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% Converts Image to Gray
InitialGrayscaleImage = OrigImage(:,:,1)*RedIntensity+OrigImage(:,:,2)*GreenIntensity+OrigImage(:,:,3)*BlueIntensity;
%%% Divides by the sum of the weights to make sure the image is in the proper 0 to 1 range.
GrayscaleImage = InitialGrayscaleImage/sum(RedIntensity+GreenIntensity+BlueIntensity);

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
    subplot(2,1,1); imagesc(OrigImage);CPcolormap(handles);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the Grayscale
    %%% Image.
    subplot(2,1,2); imagesc(GrayscaleImage); title('Grayscale Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% Saves the Grayscaled image to the handles structure so it can be
%%% used by subsequent modules.
fieldname = [GrayscaleImageName];
handles.Pipeline.(fieldname) = GrayscaleImage;
