function handles = MeasureImageSaturationBlur(handles)

% Help for the Measure Image Saturation & Blur module:
% Category: Measurement
%
% The percentage of pixels that are saturated (their intensity value
% is equal to the maximum possible intensity value for that image
% type) is calculated and stored as a measurement in the output file.
%
% The module can also compute and record a focus score (lower =
% better focus). This calculation takes much longer than the
% saturation checking, so it is optional. We are calculating the focus
% using the normalized variance.
%
% How it works:
% The calculation of the focus score is as follows:
% [m,n] = size(Image);
% MeanImageValue = mean(Image(:));
% SquaredNormalizedImage = (Image-MeanImageValue).^2;
% BlurScore{ImageNumber} = ...
%    sum(SquaredNormalizedImage(:))/(m*n*MeanImageValue);
%
% SAVING IMAGES: If you want to save images produced by this module,
% alter the code for this module to save those images to the handles
% structure (see the SaveImages module help) and then use the Save
% Images module.
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
%   Ola Friman     <friman@bwh.harvard.edu>
%   Steve Lowe     <stevelowe@alum.mit.edu>
%   Joo Han Chang  <joohan.chang@gmail.com>
%   Colin Clarke   <colinc@mit.edu>
%   Mike Lamprecht <mrl@wi.mit.edu>
%   Susan Ma       <xuefang_ma@wi.mit.edu>
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

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
    ImageToCheck{ImageNumber} = handles.Pipeline.(fieldname);

    %%% Checks that the original image is two-dimensional (i.e. not a color
    %%% image), which would disrupt several of the image functions.
    if ndims(ImageToCheck{ImageNumber}) ~= 2
        error(['Image processing was canceled in the ', ModuleName, ' module because it requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image. You can run an RGB Split module or RGB to Grayscale module to convert your image to grayscale. Also, you can modify the code to handle each channel of a color image; we just have not done it yet.  This requires making the proper headings in the measurements file and displaying the results properly.'])
    end
    NumberPixelsSaturated = sum(sum(ImageToCheck{ImageNumber} == 1));
    [m,n] = size(ImageToCheck{ImageNumber});
    TotalPixels = m*n;
    PercentPixelsSaturated = 100*NumberPixelsSaturated/TotalPixels;
    PercentSaturation{ImageNumber} = PercentPixelsSaturated;

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
            BlurScore{ImageNumber} = 0;
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

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;
    CPfigure(handles,ThisModuleFigureNumber);
    originalsize = get(ThisModuleFigureNumber, 'position');
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        originalsize(3) = originalsize(3)*.5;
        set(ThisModuleFigureNumber, 'position', originalsize,'color',[.7 .7 .9]);
    end

    %delete(findobj('Parent',ThisModuleFigureNumber));

    displaytexthandle = uicontrol(ThisModuleFigureNumber,'style','text', 'units','normalized','position',[0.1 0.1 0.8 0.8],...
        'fontname','times','fontsize',handles.Current.FontSize,'backgroundcolor',[.7 .7 .9],'horizontalalignment','left');
    DisplayText = strvcat(['    Image Set # ',num2str(handles.Current.SetBeingAnalyzed)],... %#ok We want to ignore MLint error checking for this line.
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