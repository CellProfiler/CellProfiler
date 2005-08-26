function handles = MeasureImageSaturationBlur(handles)

% Help for the Measure Image Saturation & Blur module:
% Category: Measurement
%
% The percentage of pixels that are saturated (their intensity value
% is equal to the maximum possible intensity value for that image
% type) is calculated and stored as a measurement in the output file.
%
% The module can also compute and record a focus score (higher =
% better focus). This calculation takes much longer than the
% saturation checking, so it is optional.
%
% How it works:
% The calculation of the focus score is as follows:
% RightImage = Image(:,2:end)
% LeftImage = Image(:,1:end-1)
% MeanImageValue = mean(Image(:))
% FocusScore = std(RightImage(:) - LeftImage(:)) / MeanImageValue
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

%textVAR01 = What did you call the image you want to check for saturation?
%infotypeVAR01 = imagegroup
%choiceVAR01 = Do not use
NameImageToCheck{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the image you want to check for saturation?
%infotypeVAR02 = imagegroup
%choiceVAR02 = Do not use
NameImageToCheck{2} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = What did you call the image you want to check for saturation?
%infotypeVAR03 = imagegroup
%choiceVAR03 = Do not use
NameImageToCheck{3} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = What did you call the image you want to check for saturation?
%infotypeVAR04 = imagegroup
%choiceVAR04 = Do not use
NameImageToCheck{4} = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = What did you call the image you want to check for saturation?
%infotypeVAR05 = imagegroup
%choiceVAR05 = Do not use
NameImageToCheck{5} = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = What did you call the image you want to check for saturation?
%infotypeVAR06 = imagegroup
%choiceVAR06 = Do not use
NameImageToCheck{6} = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 =  Do you want to also check the above images for blur?
%choiceVAR07 = Yes
%choiceVAR07 = No
BlurCheck = char(handles.Settings.VariableValues{CurrentModuleNum,8});
BlurCheck = BlurCheck(1);
%inputtypeVAR07 = popupmenu

%%%VariableRevisionNumber = 03

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS, FILE HANDLING, IMAGE ANALYSIS, STORE DATA IN HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

for ImageNumber = 1:6;
    %%% Reads (opens) the images you want to analyze and assigns them to
    %%% variables.
    if strcmp(NameImageToCheck{ImageNumber}), 'Do not use') ~= 1
        fieldname = ['', NameImageToCheck{ImageNumber}];
        %%% Checks whether the image to be analyzed exists in the handles structure.
        if isfield(handles.Pipeline, fieldname)==0,
            %%% If the image is not there, an error message is produced.  The error
            %%% is not displayed: The error function halts the current function and
            %%% returns control to the calling function (the analyze all images
            %%% button callback.)  That callback recognizes that an error was
            %%% produced because of its try/catch loop and breaks out of the image
            %%% analysis loop without attempting further modules.
            error(['Image processing was canceled because the Saturation & Blur Check module could not find the input image.  It was supposed to be named ', NameImageToCheck{ImageNumber}, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
        end
        %%% Reads the image.
        ImageToCheck{ImageNumber} = handles.Pipeline.(fieldname);

        %%% Checks that the original image is two-dimensional (i.e. not a color
        %%% image), which would disrupt several of the image functions.
        if ndims(ImageToCheck{ImageNumber}) ~= 2
            error('Image processing was canceled because the Saturation Blur Check module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image. You can run an RGB Split module or RGB to Grayscale module to convert your image to grayscale. Also, you can modify the code to handle each channel of a color image; we just have not done it yet.  This requires making the proper headings in the measurements file and displaying the results properly.')
        end
        NumberPixelsSaturated = sum(sum(ImageToCheck{ImageNumber} == 1));
        [m,n] = size(ImageToCheck{ImageNumber});
        TotalPixels = m*n;
        PercentPixelsSaturated = 100*NumberPixelsSaturated/TotalPixels;
        PercentSaturation{ImageNumber} = PercentPixelsSaturated;

        %%% Checks the focus of the images, if desired.
        if strcmp(upper(BlurCheck), 'N') ~= 1
            RightImage = ImageToCheck{ImageNumber}(:,2:end);
            LeftImage = ImageToCheck{ImageNumber}(:,1:end-1);
            MeanImageValue = mean(ImageToCheck{ImageNumber}(:));
            if MeanImageValue == 0
                FocusScore{ImageNumber} = 0;
            else
                FocusScore{ImageNumber} = std(RightImage(:) - LeftImage(:)) / MeanImageValue;
            end
            FocusScoreFeaturenames(ImageNumber) = {['FocusScore ', NameImageToCheck{ImageNumber}]};
        end
        PercentSaturationFeaturenames(ImageNumber) = {['PercentSaturation ', NameImageToCheck{ImageNumber}]};
    end
end

% Remove empty cells, in case there is an 'N' in between two actual images
if strcmp(upper(BlurCheck), 'N') ~= 1
    FocusScoreFeaturenames = cellstr(strvcat(FocusScoreFeaturenames{:}))';
end
PercentSaturationFeaturenames = cellstr(strvcat(PercentSaturationFeaturenames{:}))';

% Store measurements
if strcmp(upper(BlurCheck), 'N') ~= 1
    handles.Measurements.Image.FocusScoreFeatures = FocusScoreFeaturenames;
    handles.Measurements.Image.FocusScore{handles.Current.SetBeingAnalyzed} = cat(2,FocusScore{:});;
end
handles.Measurements.Image.PercentSaturationFeatures = PercentSaturationFeaturenames;
handles.Measurements.Image.PercentSaturation{handles.Current.SetBeingAnalyzed} = cat(2,PercentSaturation{:});

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;
    CPfigure(handles,ThisModuleFigureNumber);
    originalsize = get(ThisModuleFigureNumber, 'position');
    newsize = originalsize;
    newsize(1) = 0;
    newsize(2) = 0;
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        newsize(3) = originalsize(3)*.5;
        originalsize(3) = originalsize(3)*.5;
        set(ThisModuleFigureNumber, 'position', originalsize,'color',[1 1 1]);
    end

    delete(findobj('Parent',ThisModuleFigureNumber));

    displaytexthandle = uicontrol(ThisModuleFigureNumber,'style','text', 'units','normalized','position', [0.1 0.1 0.8 0.8],...
        'fontname','times','fontsize',handles.Current.FontSize,'backgroundcolor',[1 1 1],'horizontalalignment','left');
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
        for ImageNumber = 1:length(FocusScore)
            if ~isempty(FocusScore{ImageNumber})
                try DisplayText = strvcat(DisplayText, ... %#ok We want to ignore MLint error checking for this line.
                        [NameImageToCheck{ImageNumber}, ':    ', num2str(FocusScore{ImageNumber})]);%#ok We want to ignore MLint error checking for this line.
                end
            end
        end
    end
    set(displaytexthandle,'string',DisplayText)
end

%%% ABANDONED WAYS TO MEASURE FOCUS:
%             eval(['ImageToCheck',ImageNumber,' = histeq(eval([''ImageToCheck'',ImageNumber]));'])
%             if str2num(BlurRadius) == 0
%                 BlurredImage = eval(['ImageToCheck',ImageNumber]);
%             else
%                 %%% Blurs the image.
%                 %%% Note: using filter2 is much faster than imfilter (e.g. 14.5 sec vs. 99.1 sec).
%                 FiltSize = max(3,ceil(4*BlurRadius));
%                 BlurredImage = filter2(fspecial('gaussian',FiltSize, str2num(BlurRadius)), eval(['ImageToCheck',ImageNumber]));
%                 % figure, imshow(BlurredImage, []), title('BlurredImage')
%                 % imwrite(BlurredImage, [BareFileName,'BI','.',FileFormat], FileFormat);
%             end
%             %%% Subtracts the BlurredImage from the original.
%             SubtractedImage = imsubtract(eval(['ImageToCheck',ImageNumber]), BlurredImage);
%             handles.FocusScore(handles.Current.SetBeingAnalyzed) = std(SubtractedImage(:));
%             handles.FocusScore2(handles.Current.SetBeingAnalyzed) = sum(sum(SubtractedImage.*SubtractedImage));
%             FocusScore = handles.FocusScore
%             FocusScore2 = handles.FocusScore2
%
