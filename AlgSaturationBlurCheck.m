function handles = AlgSaturationBlurCheck(handles)

% Help for the Saturation & Blur Check module: 
% Sorry, this module has not yet been documented.

% The contents of this file are subject to the Mozilla Public License Version 
% 1.1 (the "License"); you may not use this file except in compliance with 
% the License. You may obtain a copy of the License at 
% http://www.mozilla.org/MPL/
% 
% Software distributed under the License is distributed on an "AS IS" basis,
% WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
% for the specific language governing rights and limitations under the
% License.
% 
% 
% The Original Code is the Saturation and Blur Check Module.
% 
% The Initial Developer of the Original Code is
% Whitehead Institute for Biomedical Research
% Portions created by the Initial Developer are Copyright (C) 2003,2004
% the Initial Developer. All Rights Reserved.
% 
% Contributor(s):
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current algorithm number, since this is needed to find
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;
CurrentAlgorithmNum = str2double(handles.currentalgorithm);

%textVAR01 = What did you call the image you want to check for saturation and blur?
%defaultVAR01 = OrigBlue
NameImageToCheck{1} = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What did you call the image you want to check for saturation and blur?
%defaultVAR02 = OrigGreen
NameImageToCheck{2} = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = What did you call the image you want to check for saturation and blur?
%defaultVAR03 = OrigRed
NameImageToCheck{3} = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});

%textVAR04 = What did you call the image you want to check for saturation and blur?
%defaultVAR04 = N
NameImageToCheck{4} = char(handles.Settings.Vvariable{CurrentAlgorithmNum,4});

%textVAR05 = What did you call the image you want to check for saturation and blur?
%defaultVAR05 = N
NameImageToCheck{5} = char(handles.Settings.Vvariable{CurrentAlgorithmNum,5});

%textVAR06 = What did you call the image you want to check for saturation and blur?
%defaultVAR06 = N
NameImageToCheck{6} = char(handles.Settings.Vvariable{CurrentAlgorithmNum,6});

%textVAR07 =  For unused colors, leave "N" in the boxes above.
%textVAR09 = Do you want to check for blur?
%defaultVAR09 = Y
BlurCheck = char(handles.Settings.Vvariable{CurrentAlgorithmNum,9});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS, FILE HANDLING, IMAGE ANALYSIS, STORE DATA IN HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

for ImageNumber = 1:6;
    %%% Reads (opens) the images you want to analyze and assigns them to
    %%% variables.
    if strcmp(upper(NameImageToCheck{ImageNumber}), 'N') ~= 1
        fieldname = ['dOT', NameImageToCheck{ImageNumber}];
        %%% Checks whether the image to be analyzed exists in the handles structure.
        if isfield(handles, fieldname) == 0
            %%% If the image is not there, an error message is produced.  The error
            %%% is not displayed: The error function halts the current function and
            %%% returns control to the calling function (the analyze all images
            %%% button callback.)  That callback recognizes that an error was
            %%% produced because of its try/catch loop and breaks out of the image
            %%% analysis loop without attempting further modules.
            error(['Image processing was canceled because the Saturation & Blur Check module could not find the input image.  It was supposed to be named ', NameImageToCheck{ImageNumber}, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
        end
        %%% Reads the image.
        ImageToCheck{ImageNumber} = handles.(fieldname);
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
%             handles.FocusScore(handles.setbeinganalyzed) = std(SubtractedImage(:));
%             handles.FocusScore2(handles.setbeinganalyzed) = sum(sum(SubtractedImage.*SubtractedImage));
%             FocusScore = handles.FocusScore
%             FocusScore2 = handles.FocusScore2
% 
            %%% Saves the Focus Score to the handles structure.  The field is named
            %%% appropriately based on the user's input, with the 'dMT' prefix added so
            %%% that this field will be deleted at the end of the analysis batch.
            fieldname = ['dMTFocusScore', NameImageToCheck{ImageNumber}];
            handles.(fieldname)(handles.setbeinganalyzed) = {FocusScore{ImageNumber}};
        end
        %%% Saves the Percent Saturation to the handles structure.  The field is named
        %%% appropriately based on the user's input, with the 'dMT' prefix added so
        %%% that this field will be deleted at the end of the analysis batch.
        fieldname = ['dMTPercentSaturation', NameImageToCheck{ImageNumber}];
        handles.(fieldname)(handles.setbeinganalyzed) = {PercentSaturation{ImageNumber}};
    end
    drawnow
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%

fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% Checks whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this algorithm.
if any(findobj == ThisAlgFigureNumber) == 1;
    figure(ThisAlgFigureNumber);
    originalsize = get(ThisAlgFigureNumber, 'position');
    newsize = originalsize;
    newsize(1) = 0;
    newsize(2) = 0;
    if handles.setbeinganalyzed == 1
        newsize(3) = originalsize(3)*.5;
        originalsize(3) = originalsize(3)*.5;
        set(ThisAlgFigureNumber, 'position', originalsize);
    end
    displaytexthandle = uicontrol(ThisAlgFigureNumber,'style','text', 'position', newsize,'fontname','fixedwidth');
    DisplayText = strvcat(['    Image Set # ',num2str(handles.setbeinganalyzed)],...
        '      ',...
        'Percent of pixels that are Saturated:');
    for ImageNumber = 1:6
        try DisplayText = strvcat(DisplayText, ... %#ok We want to ignore MLint error checking for this line.
                [NameImageToCheck{ImageNumber}, ':    ', num2str(PercentSaturation{ImageNumber})]);
        end
    end
    DisplayText = strvcat(DisplayText, '      ','      ','Focus Score:');
    for ImageNumber = 1:6
        try DisplayText = strvcat(DisplayText, ... %#ok We want to ignore MLint error checking for this line.
                [NameImageToCheck{ImageNumber}, ':    ', num2str(FocusScore{ImageNumber})]);
        end
    end
    set(displaytexthandle,'string',DisplayText)
end