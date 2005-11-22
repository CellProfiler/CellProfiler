function handles = Subtract(handles)

% Help for the Subtract Images module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Subtracts the intensities of one image from another.
% *************************************************************************
%
% Sorry, this module has not yet been documented. It was written for a
% very specific purpose and it allows blurring and then subtracting
% images.
%
% SPEED OPTIMIZATION: Note that increasing the blur radius increases
% the processing time exponentially.
%
% See also <nothing relevant>

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter
%   Thouis Jones
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

%textVAR01 = Subtract this image:
%infotypeVAR01 = imagegroup
SubtractImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = From this image (called 'basic image'):
%infotypeVAR02 = imagegroup
BasicImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What do you want to call the resulting image?
%defaultVAR03 = SubtractedCellStain
%infotypeVAR03 = imagegroup indep
ResultingImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Enter the factor to multiply the subtracted image by before subtracting:
%defaultVAR04 = 1
MultiplyFactor = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    error(['Image processing was canceled in the ', ModuleName, ' module. Prior to running the Subtract Images module, you must have previously run a module to load an image. You specified in the Subtract Images module that this image was called ', SubtractImageName, ' which should have produced a field in the handles structure called ', SubtractImageName, '. The Subtract Images module cannot find this image.']);
end
SubtractImage = handles.Pipeline.(SubtractImageName);

%%% Checks that the original images are two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(BasicImage) ~= 2
    error(['Image processing was canceled in the ', ModuleName, ' module because it requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.'])
end
if ndims(SubtractImage) ~= 2
    error(['Image processing was canceled in the ', ModuleName, ' module because it requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.'])
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

AdjustedSubtractImage = MultiplyFactor*SubtractImage;
ResultingImage = imsubtract(BasicImage,AdjustedSubtractImage);
ResultingImage(ResultingImage < 0) = 0;

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = CPwhichmodulefigurenumber(CurrentModule);
if any(findobj == ThisModuleFigureNumber)
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); CPimagesc(BasicImage); title([BasicImageName, ' image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,2,2); CPimagesc(SubtractImage); title([SubtractImageName, ' image']);
    subplot(2,2,3); CPimagesc(ResultingImage); title([BasicImageName,' minus ',SubtractImageName,' = ',ResultingImageName]);
    CPFixAspectRatio(BasicImage);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the processed image to the handles structure.
handles.Pipeline.(ResultingImageName) = ResultingImage;