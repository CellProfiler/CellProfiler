function handles = MeasureImageSaturationBlur(handles)

% Help for the Measure Image Saturation and Blur module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Measures the percentage of pixels in the image that are saturated and
% measures blur using a focus score.
% *************************************************************************
%
% The percentage of pixels that are saturated is calculated and stored as a
% measurement in the output file. 'Saturated' means that the pixel's
% intensity value is equal to the maximum possible intensity value for that
% image type.
%
% The module can also measure blur by calculating a focus score (higher =
% better focus). This calculation takes much longer than the saturation
% checking, so it is optional. We are calculating the focus using the
% normalized variance. We used this algorithm because it was ranked best in
% this paper:
% Sun, Y., Duthaler, S., Nelson, B. "Autofocusing in Computer Microscopy:
%    Selecting the optimals focus algorithm." Microscopy Research and
%    Technique 65:139-149 (2004)
%
% The calculation of the focus score is as follows:
% [m,n] = size(Image);
% MeanImageValue = mean(Image(:));
% SquaredNormalizedImage = (Image-MeanImageValue).^2;
% BlurScore{ImageNumber} = ...
%    sum(SquaredNormalizedImage(:))/(m*n*MeanImageValue);
%
% Example Output:
%
% Percent of pixels that are Saturated:
% OrigBlue:   0.086173
% OrigGreen:  0
% OrigRed:    0
%
% Focus Score:
% OrigBlue:   0.47135
% OrigGreen:  0.03440
% OrigRed:    0.04652

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

%textVAR01 = What did you call the image you want to check for saturation?
%choiceVAR01 = Do not use
%infotypeVAR01 = imagegroup
NameImageToCheck{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the image you want to check for saturation?
%choiceVAR02 = Do not use
%infotypeVAR02 = imagegroup
NameImageToCheck{2} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What did you call the image you want to check for saturation?
%choiceVAR03 = Do not use
%infotypeVAR03 = imagegroup
NameImageToCheck{3} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = What did you call the image you want to check for saturation?
%choiceVAR04 = Do not use
%infotypeVAR04 = imagegroup
NameImageToCheck{4} = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = What did you call the image you want to check for saturation?
%choiceVAR05 = Do not use
%infotypeVAR05 = imagegroup
NameImageToCheck{5} = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = What did you call the image you want to check for saturation?
%choiceVAR06 = Do not use
%infotypeVAR06 = imagegroup
NameImageToCheck{6} = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 =  Do you want to also check the above images for blur?
%choiceVAR07 = Yes
%choiceVAR07 = No
BlurCheck = char(handles.Settings.VariableValues{CurrentModuleNum,7});
BlurCheck = BlurCheck(1);
%inputtypeVAR07 = popupmenu

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS, FILE HANDLING, IMAGE ANALYSIS, STORE DATA IN HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

tmp1 = {};
for n = 1:6
    if ~strcmp(NameImageToCheck{n}, 'Do not use')
        tmp1{end+1} = NameImageToCheck{n};
    end
end
NameImageToCheck = tmp1;

for ImageNumber = 1:length(NameImageToCheck);
    %%% Reads (opens) the images you want to analyze and assigns them to
    %%% variables.
    fieldname = ['', NameImageToCheck{ImageNumber}];
    %%% Checks whether the image to be analyzed exists in the handles structure.
    if ~isfield(handles.Pipeline, fieldname),
        %%% If the image is not there, an error message is produced.  The error
        %%% is not displayed: The error function halts the current function and
        %%% returns control to the calling function (the analyze all images
        %%% button callback.)  That callback recognizes that an error was
        %%% produced because of its try/catch loop and breaks out of the image
        %%% analysis loop without attempting further modules.
        error(['Image processing was canceled in the ', ModuleName, ' module because it could not find the input image.  It was supposed to be named ', NameImageToCheck{ImageNumber}, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
    end
    %%% Reads the image.
    ImageToCheck{ImageNumber} = handles.Pipeline.(fieldname); %#ok Ignore MLint

    %%% Checks that the original image is two-dimensional (i.e. not a color
    %%% image), which would disrupt several of the image functions.
    if ndims(ImageToCheck{ImageNumber}) ~= 2
        error(['Image processing was canceled in the ', ModuleName, ' module because it requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image. You can run a Color To Gray module to convert your image to grayscale. Also, you can modify the code to handle each channel of a color image; we just have not done it yet.  This requires making the proper headings in the measurements file and displaying the results properly.'])
    end
    NumberPixelsSaturated = sum(sum(ImageToCheck{ImageNumber} == 1));
    [m,n] = size(ImageToCheck{ImageNumber});
    TotalPixels = m*n;
    PercentPixelsSaturated = 100*NumberPixelsSaturated/TotalPixels;
    PercentSaturation{ImageNumber} = PercentPixelsSaturated;  %#ok Ignore MLint

    Measurefieldname = ['SaturationBlur_',NameImageToCheck{ImageNumber}];
    Featurefieldname = ['SaturationBlur_',NameImageToCheck{ImageNumber},'Features'];
    %%% Checks the focus of the images, if desired.
    if ~strcmpi(BlurCheck,'N')
        %         Old method of scoring focus, not justified
        %         RightImage = ImageToCheck{ImageNumber}(:,2:end);
        %         LeftImage = ImageToCheck{ImageNumber}(:,1:end-1);
        %         MeanImageValue = mean(ImageToCheck{ImageNumber}(:));
        %         if MeanImageValue == 0
        %             BlurScore{ImageNumber} = 0;
        %         else
        %             BlurScore{ImageNumber} = std(RightImage(:) - LeftImage(:)) / MeanImageValue;
        %         end
        Image = ImageToCheck{ImageNumber};
        if ~strcmp(class(Image),'double')
            Image = im2double(Image);
        end
        [m,n] = size(Image);
        MeanImageValue = mean(Image(:));
        SquaredNormalizedImage = (Image-MeanImageValue).^2;
        if MeanImageValue == 0
            BlurScore{ImageNumber} = 0;  %#ok Ignore MLint
        else
            BlurScore{ImageNumber} = sum(SquaredNormalizedImage(:))/(m*n*MeanImageValue);
        end
        Featurenames = {'BlurScore','PercentSaturated'};
        handles.Measurements.Image.(Featurefieldname) = Featurenames;
        handles.Measurements.Image.(Measurefieldname){handles.Current.SetBeingAnalyzed}(:,1) = BlurScore{ImageNumber};
        handles.Measurements.Image.(Measurefieldname){handles.Current.SetBeingAnalyzed}(:,2) = PercentSaturation{ImageNumber};
    else
        Featurenames = {'PercentSaturated'};
        handles.Measurements.Image.(Featurefieldname) = Featurenames;
        handles.Measurements.Image.(Measurefieldname){handles.Current.SetBeingAnalyzed}(:,1) = PercentSaturation{ImageNumber};
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber) == 1;
    CPfigure(handles,ThisModuleFigureNumber);
    originalsize = get(ThisModuleFigureNumber, 'position');
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        originalsize(3) = originalsize(3)*.5;
        set(ThisModuleFigureNumber, 'position', originalsize,'color',[.7 .7 .9]);
    end
    displaytexthandle = uicontrol(ThisModuleFigureNumber,'style','text', 'units','normalized','position',[0.1 0.1 0.8 0.8],...
        'fontname','Helvetica','fontsize',handles.Preferences.FontSize,'backgroundcolor',[.7 .7 .9],'horizontalalignment','left');
    DisplayText = strvcat(['    Cycle # ',num2str(handles.Current.SetBeingAnalyzed)],... %#ok We want to ignore MLint error checking for this line.
        '      ',...
        'Percent of pixels that are Saturated:');
    for ImageNumber = 1:length(PercentSaturation)
        if ~isempty(PercentSaturation{ImageNumber})
            try DisplayText = strvcat(DisplayText, ... %#ok We want to ignore MLint error checking for this line.
                    [NameImageToCheck{ImageNumber}, ':    ', num2str(PercentSaturation{ImageNumber})]);%#ok We want to ignore MLint error checking for this line.
            end
        end
    end
    DisplayText = strvcat(DisplayText, '      ','      ','Focus Score:'); %#ok We want to ignore MLint error checking for this line.
    if strcmp(upper(BlurCheck), 'N') ~= 1
        for ImageNumber = 1:length(BlurScore)
            if ~isempty(BlurScore{ImageNumber})
                try DisplayText = strvcat(DisplayText, ... %#ok We want to ignore MLint error checking for this line.
                        [NameImageToCheck{ImageNumber}, ':    ', num2str(BlurScore{ImageNumber})]);%#ok We want to ignore MLint error checking for this line.
                end
            end
        end
    end
    set(displaytexthandle,'string',DisplayText)
end