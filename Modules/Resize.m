function handles = Resize(handles)

% Help for the Resize module:
% Category: Image Processing
%
% Images are resized (smaller or larger) based on the user's inputs.
% This module uses the Matlab built-in function imresize.
%
% SAVING IMAGES: The thresholded images produced by this module can be
% easily saved using the Save Images module, using the name you
% assign. If you want to save other intermediate images, alter the
% code for this module to save those images to the handles structure
% (see the SaveImages module help) and then use the Save Images
% module.

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

%textVAR01 = What did you call the image to be resized?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the resized image?
%defaultVAR02 = ResizedBlue
%infotypeVAR02 = imagegroup indep
ResizedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = To shrink the image, enter the resizing factor (0 to 1). To enlarge the image, enter the resizing factor (greater than 1)
%defaultVAR03 = .25
ResizingFactor = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%textVAR04 = Alternately, leave the shrinking factor set to 1 and enter the desired resulting size in pixels: height,width  (This may change the aspect ratio of the image.)
%defaultVAR04 = 100,100
SpecificSize = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = Enter the interpolation method
%choiceVAR05 = Nearest Neighbor
%choiceVAR05 = Bilinear
%choiceVAR05 = Bicubic
InterpolationMethod = char(handles.Settings.VariableValues{CurrentModuleNum,5});
if strcmp(InterpolationMethod,'Bilinear')
    InterpolationMethod = 'L';
elseif strcmp(InterpolationMethod,'Bicubic')
    InterpolationMethod = 'C';
else
    InterpolationMethod = 'N';
end
%inputtypeVAR05 = popupmenu

%%%VariableRevisionNumber = 01

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
    error(['Image processing was canceled because the Resize module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
OrigImage = handles.Pipeline.(fieldname);

%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS%%%
%%%%%%%%%%%%%%%%%%%%
drawnow



if ResizingFactor == 1
    ResizeData = SpecificSize;
else ResizeData = ResizingFactor;
end

if strncmpi(InterpolationMethod,'N',1) == 1
    InterpolationMethod = 'nearest';
elseif strncmpi(InterpolationMethod,'L',1) == 1
    InterpolationMethod = 'bilinear';
elseif strncmpi(InterpolationMethod,'C',1) == 1
    InterpolationMethod = 'bicubic';
else error('Image processing was canceled because you must enter "N", "L", or "C" for the interpolation method in the Resize Images module.')
end

ResizedImage = imresize(OrigImage,ResizeData,InterpolationMethod);
%%% If the interpolation method is bicubic (maybe bilinear too; not sure), there is a chance
%%% the image will be slightly out of the range 0 to 1, so the image is
%%% rescaled here.
if strncmpi(InterpolationMethod,'bicubic',1) == 1
    if min(OrigImage(:)) < 0 | max(OrigImage(:)) > 1
        error('Image processing was canceled because the intensity of the image coming into the Resize module is outside the range 0 to 1')
    else
        %%% As long as the incoming image was within 0 to 1, it's ok to
        %%% truncate the resized image at 0 and 1 without losing much image
        %%% data.
        ResizedImage(ResizedImage<0) = 0;
        ResizedImage(ResizedImage>1) = 1;
    end
end

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
    %%% A subplot of the figure window is set to display the Resized
    %%% Image.
    subplot(2,1,2); imagesc(ResizedImage); title('Resized Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% The Resized image is saved to the handles structure so it can be
%%% used by subsequent modules.
handles.Pipeline.(ResizedImageName) = ResizedImage;
