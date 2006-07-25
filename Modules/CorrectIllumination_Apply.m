function handles = CorrectIllumination_Apply(handles)

% Help for the Correct Illumination Apply module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Applies an illumination function, created by
% CorrectIllumination_Calculate, to an image in order to correct for uneven
% illumination (uneven shading).
% *************************************************************************
%
% This module corrects for uneven illumination of each image. An
% illumination function image that represents the variation in
% illumination across the field of view is either made by a previous
% module or loaded by a previous module in the pipeline.  This module
% then applies the illumination function to each image coming through
% the pipeline to produce the corrected image.
%
% Settings:
%
% Divide or Subtract:
% This module either divides each image by the illumination function,
% or the illumination function is subtracted from each image. The
% choice depends on how the illumination function was calculated and
% on your physical model of how illumination variation affects the
% background of images relative to the objects in images. If the
% background is significant relative to the real signal coming from
% cells (a somewhat empirical decision), then the Subtract option may be
% preferable. If, in contrast, the signal to background ratio is quite
% high (the cells are stained strongly), then the Divide option is
% probably preferable. Typically, Subtract is used if the illumination
% function was calculated using the background option in the
% CORRECTILLUMINATION_CALCULATE module and divide is used if the
% illumination function was calculated using the regular option.
%
% Rescaling:
% If subtracting the illumination function, any pixels that end up
% negative are set to zero, so no rescaling of the corrected image is
% necessary. If dividing, the resulting corrected image may be in a
% very different range of intensity values relative to the original,
% depending on the values of the illumination function. If you are not
% rescaling, you should confirm that the illumination function is in a
% reasonable range (e.g. 1 to some number), so that the resulting
% image is in a reasonable range (0 to 1). Otherwise, you have two
% options to rescale the resulting image: either stretch the image
% so that the minimum is zero and the maximum is one, or match the
% maximum of the corrected image to the the maximum of the original.
% Either of these options has the potential to disturb the brightness
% of images relative to other images in the set, so caution should be
% used in interpreting intensity measurements from images that have
% been rescaled. See the help for the Rescale Intensity module for details.
%
% See also CorrectIllumination_Calculate, RescaleIntensity.

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
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image to be corrected?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the corrected image?
%defaultVAR02 = CorrBlue
%infotypeVAR02 = imagegroup indep
CorrectedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What did you call the illumination correction function image to be used to carry out the correction (produced by another module or loaded as a .mat format image using Load Single Image)?
%infotypeVAR03 = imagegroup
IllumCorrectFunctionImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = How do you want to apply the illumination correction function?
%choiceVAR04 = Divide
%choiceVAR04 = Subtract
DivideOrSubtract = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = If you chose division, Choose rescaling method.
%choiceVAR05 = No rescaling
%choiceVAR05 = Stretch 0 to 1
%choiceVAR05 = Match maximums
RescaleOption = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','CheckScale');

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
IllumCorrectFunctionImage = CPretrieveimage(handles,IllumCorrectFunctionImageName,ModuleName,'MustBeGray','DontCheckScale',size(OrigImage));

if strcmp(RescaleOption,'No rescaling') == 1
    MethodSpecificArguments = [];
    RescaleOption = 'N';
elseif strcmp(RescaleOption,'Stretch 0 to 1') == 1
    MethodSpecificArguments = [];
    RescaleOption = 'S';
elseif strcmp(RescaleOption,'Match maximums') == 1
    MethodSpecificArguments = OrigImage;
    RescaleOption = 'M';
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(DivideOrSubtract,'Divide') == 1
    %%% Corrects the original image based on the IlluminationImage,
    %%% by dividing each pixel by the value in the IlluminationImage.
    CorrectedImage1 = OrigImage ./ IllumCorrectFunctionImage;
    %%% Rescales using a CP subfunction, if requested.
    [handles,CorrectedImage] = CPrescale(handles,CorrectedImage1,RescaleOption,MethodSpecificArguments);
elseif strcmp(DivideOrSubtract,'Subtract') == 1
    %%% Corrects the original image based on the IlluminationImage,
    %%% by subtracting each pixel by the value in the IlluminationImage.
    CorrectedImage = imsubtract(OrigImage, single(IllumCorrectFunctionImage));
    %%% Converts negative values to zero.  I have essentially truncated the
    %%% data at zero rather than trying to rescale the data, because negative
    %%% values should be fairly rare (and minor), since the minimum is used to
    %%% calculate the IlluminationImage.
    CorrectedImage(CorrectedImage < 0) = 0;
else error(['Image processing was canceled in the ', ModuleName, ' module because you must choose Divide or Subtract for the method by which to apply the illumination correction.'])
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
        CPresizefigure(OrigImage,'TwoByTwo',ThisModuleFigureNumber);
    end
    %%% A subplot of the figure window is set to display the original
    %%% image, some intermediate images, and the final corrected image.
    subplot(2,2,1);
    CPimagesc(OrigImage,handles);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% The mean image does not absolutely have to be present in order to
    %%% carry out the calculations if the illumination image is provided,
    %%% so the following subplot is only shown if MeanImage exists in the
    %%% workspace.
    subplot(2,2,2);
    CPimagesc(CorrectedImage,handles);
    title('Illumination Corrected Image');
    subplot(2,2,3);
    CPimagesc(IllumCorrectFunctionImage,handles);
    title('Illumination Correction Function Image');
    text(1,50,['Min Value: ' num2str(min(min(IllumCorrectFunctionImage)))],'Color','red','fontsize',handles.Preferences.FontSize);
    text(1,150,['Max Value: ' num2str(max(max(IllumCorrectFunctionImage)))],'Color','red','fontsize',handles.Preferences.FontSize);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the corrected image to the
%%% handles structure so it can be used by subsequent modules.
handles.Pipeline.(CorrectedImageName) = CorrectedImage;