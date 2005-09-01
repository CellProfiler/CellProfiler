function handles = Subtract(handles)

% Help for the Subtract Images module:
% Category: Image Processing
%
% Sorry, this module has not yet been documented. It was written for a
% very specific purpose and it allows blurring and subtracting images.
%
% SPEED OPTIMIZATION: Note that increasing the blur radius increases
% the processing time exponentially.
%
% SAVING IMAGES: The resulting image produced by this module can be easily
% saved using the Save Images module, using the name you assign. If
% you want to save other intermediate images, alter the code for this
% module to save those images to the handles structure (see the
% SaveImages module help) and then use the Save Images module.
%
% See also <nothing relevant>

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

%textVAR01 = Subtract this image (enter the name here)
%infotypeVAR01 = imagegroup
SubtractImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = From this image (enter the name here)
%defaultVAR02 = NHSw1
%infotypeVAR02 = imagegroup
BasicImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What do you want to call the resulting image?
%defaultVAR03 = SubtractedCellStain
%infotypeVAR03 = imagegroup custom
ResultingImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Enter the factor to multiply the subtracted image by:
%defaultVAR04 = 1
MultiplyFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = Contrast stretch the resulting image?
%choiceVAR05 = No
%choiceVAR05 = Yes
Stretch = char(handles.Settings.VariableValues{CurrentModuleNum,5});
Stretch = Stretch(1);
%inputtypeVAR05 = popupmenu

%textVAR06 = Blur radius for the basic image
%defaultVAR06 = 3
BlurRadius = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,6}));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the images you want to analyze and assigns them to
%%% variables.
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, BasicImageName) == 0
    error(['Image processing has been canceled. Prior to running the Subtract Images module, you must have previously run a module to load an image. You specified in the Subtract Images module that this image was called ', BasicImageName, ' which should have produced a field in the handles structure called ', BasicImageName, '. The Subtract Images module cannot find this image.']);
end
BasicImage = handles.Pipeline.(BasicImageName);
%%% Checks whether the image exists in the handles structure.
if isfield(handles.Pipeline, SubtractImageName) == 0
    error(['Image processing has been canceled. Prior to running the Subtract Images module, you must have previously run a module to load an image. You specified in the Subtract Images module that this image was called ', SubtractImageName, ' which should have produced a field in the handles structure called ', SubtractImageName, '. The Subtract Images module cannot find this image.']);
end
SubtractImage = handles.Pipeline.(SubtractImageName);

%%% Checks that the original images are two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(BasicImage) ~= 2
    error('Image processing was canceled because the Subtract Images module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end
if ndims(SubtractImage) ~= 2
    error('Image processing was canceled because the Subtract Images module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow



AdjustedSubtractImage = MultiplyFactor*SubtractImage;
if BlurRadius ~= 0
    %%% Blurs the image.
    %%% Note: using filter2 is much faster than imfilter (e.g. 14.5 sec vs. 99.1 sec).
    FiltSize = max(3,ceil(4*BlurRadius));
    BasicImage = filter2(fspecial('gaussian',FiltSize, BlurRadius), BasicImage);
end
ResultingImage = imsubtract(BasicImage,AdjustedSubtractImage);
if strcmp(upper(Stretch),'Y') == 1
    ResultingImage = imadjust(ResultingImage,stretchlim(ResultingImage,[.01 .99]));
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow



fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;

    drawnow
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(BasicImage);
    title([BasicImageName, ' input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,2); imagesc(SubtractImage); title([SubtractImageName, ' input image']);
    subplot(2,2,3); imagesc(ResultingImage); title([BasicImageName,' minus ',SubtractImageName,' = ',ResultingImageName]);
    CPFixAspectRatio(BasicImage);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% Saves the processed image to the handles structure.
handles.Pipeline.(ResultingImageName) = ResultingImage;
