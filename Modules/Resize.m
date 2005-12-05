function handles = Resize(handles)

% Help for the Resize module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Resizes images.
% *************************************************************************
%
% Images are resized (smaller or larger) based on the user's inputs.
% This module uses the Matlab built-in function imresize.

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

%textVAR01 = What did you call the image to be resized?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the resized image?
%defaultVAR02 = ResizedBlue
%infotypeVAR02 = imagegroup indep
ResizedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = To shrink the image, enter the resizing factor (0 to 1). To enlarge the image, enter the resizing factor (greater than 1).
%defaultVAR03 = .25
ResizingFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%textVAR04 = Alternately, leave the shrinking factor set to 1 and enter the desired resulting size in pixels: height,width. This may change the aspect ratio of the image.
%defaultVAR04 = 100,100
SpecificSize = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,4})); %#ok Ignore MLint

%textVAR05 = Enter the interpolation method
%choiceVAR05 = Nearest Neighbor
%choiceVAR05 = Bilinear
%choiceVAR05 = Bicubic
InterpolationMethod = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%%%VariableRevisionNumber = 1

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

if ResizingFactor == 1
    ResizeData = SpecificSize;
else ResizeData = ResizingFactor;
end

if strncmpi(InterpolationMethod,'N',1)
    InterpolationMethod = 'nearest';
elseif ~strcmp(InterpolationMethod,'Bilinear') && ~strcmp(InterpolationMethod,'Bicubic')
    error(['Image processing was canceled in the ', ModuleName, ' module because you must enter "Nearest Neighbor", "Bilinear", or "Bicubic" for the interpolation method.'])
end

ResizedImage = imresize(OrigImage,ResizeData,InterpolationMethod);
%%% If the interpolation method is bicubic (maybe bilinear too; not sure), there is a chance
%%% the image will be slightly out of the range 0 to 1, so the image is
%%% rescaled here.
if strncmpi(InterpolationMethod,'bicubic',1)
    if min(OrigImage(:)) < 0 || max(OrigImage(:)) > 1
        error(['Image processing was canceled in the ', ModuleName, ' module because the intensity of the input image is outside the range 0 to 1, which is incompatible with the bicubic interpolation method.'])
    else
        %%% As long as the incoming image was within 0 to 1, it's ok to
        %%% truncate the resized image at 0 and 1 without losing much image
        %%% data.
        ResizedImage(ResizedImage<0) = 0;
        ResizedImage(ResizedImage>1) = 1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
%%% Check whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this module.
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'TwoByOne',ThisModuleFigureNumber)
    end
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1); 
    CPimagesc(OrigImage,handles);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the Resized
    %%% Image.
    subplot(2,1,2); 
    CPimagesc(ResizedImage,handles); 
    title('Resized Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The processed image is saved to the handles structure so it can be
%%% used by subsequent modules.
handles.Pipeline.(ResizedImageName) = ResizedImage;