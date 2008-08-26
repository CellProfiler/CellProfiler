function handles = MeasureImageQuality(handles)

% Help for the Measure Image Quality module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Measures the percentage of pixels in the image that are saturated and
% measures blur (poor focus). 
% *************************************************************************
%
% This module replaces the module previously known as Image Saturation and
% Blur.
%
% Features measured:      Feature Number:
% FocusScore           |         1
% LocalFocusScore      |         2
% WindowSize           |         3
% PercentSaturation    |         4
% PercentMaximal       |         5
%
% In addition, an OrigThreshold value is added to the Image measurements
% under the MeasureImageQuality category. 
%
% Lastly, the following measurements are placed in the Experiment category: 
% MeanThreshold 
% MedianThreshold
% StdevThreshold
%
% Please note that these Experiment measurements are calculated once the
% pipeline has run through all of the cycles consecutively. It will not
% produce a result for a batch run, since the cycles are processed
% independently from each other.
%
% The percentage of pixels that are saturated is calculated and stored as a
% measurement in the output file. 'Saturated' means that the pixel's
% intensity value is equal to the maximum possible intensity value for that
% image type.
%
% Update [Oct-11-2007]
% Because the saturated pixels may not reach to the maximum possible
% intensity value of the image type for some reasons such as CCDs saturate
% before 255 in graylevel, we also calculate the percentage of the maximal
% intensity value.  Even though we may capture the maximal intensity
% percentage of 'dark' images, the maximal percentage is mostly very minimal or
% ignorable. So, PercentMaximal is another good indicator for saturation
% detection.
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
% FocusScore{ImageNumber} = ...
%    sum(SquaredNormalizedImage(:))/(m*n*MeanImageValue);
%
% Update (2007-10-11):
%
% The above score is to measure a relative score given a focus setting of 
% a certain microscope. Using this, one can calibrrate the microscope's
% focus setting. However it doesn't necessarily tell you how well an image
% was focused when taken. That means these scores obtained from many different
% images probably taken in different situations and with different cell
% contents can not be used for focus comparison.
% 
% The newly added LocalFocusScore is a local version of the original 
% FocusScore. LocalFocusScore was just named after the original one to be
% consistent with naming. Note that these focus scores do not necessarily 
% represent the qualities of focusing between different images. 
% LocalFocusScore was added to differentiate good segmentation and bad 
% segmentation images in the cases when bad segmentation images usually 
% contain no cell objects with high background noise.
%
% Example Output:
% 
% Percent of pixels that are Saturated:
% RescaledOrig:     0.002763
% 
% Percent of pixels that are in the Maximal
% Intensity:
% RescaledOrig:     0.0002763
%
% 
% Focus Score:
% RescaledOrig: 0.016144
% 
% Suggested Threshold:
% Orig: 0.0022854
%
% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003--2008.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision: 5237 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 =  Do you want to check the following selected images for image quality (called blur earlier)?
%choiceVAR01 = No
%choiceVAR01 = Yes
BlurCheck = char(handles.Settings.VariableValues{CurrentModuleNum,1});
BlurCheck = BlurCheck(1);
%inputtypeVAR01 = popupmenu

%textVAR02 = If you chose to check images for image quality above, enter the window size of LocalFocusScore measurement (A suggested value is 2 times ObjectSize)?
%defaultVAR02 = 20
WindowSize = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,2}));

%textVAR03 = Which grayscale image would you like to use to check for saturation?
%choiceVAR03 = Do not use
%infotypeVAR03 = imagegroup
NameImageToCheck{1} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = Which grayscale image would you like to use to calculate a suggested threshold?
%choiceVAR04 = Do not use
%infotypeVAR04 = imagegroup
NameImageToThresh{1} = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Which automatic thresholding method would you like to use to calculate a suggested threshold?
%choiceVAR05 = Otsu Global
%choiceVAR05 = Otsu Adaptive
%choiceVAR05 = Otsu PerObject
%choiceVAR05 = MoG Global
%choiceVAR05 = MoG Adaptive
%choiceVAR05 = MoG PerObject
%choiceVAR05 = Background Global
%choiceVAR05 = Background Adaptive
%choiceVAR05 = Background PerObject
%choiceVAR05 = RobustBackground Global
%choiceVAR05 = RobustBackground Adaptive
%choiceVAR05 = RobustBackground PerObject
%choiceVAR05 = RidlerCalvard Global
%choiceVAR05 = RidlerCalvard Adaptive
%choiceVAR05 = RidlerCalvard PerObject
%choiceVAR05 = Kapur Global
%choiceVAR05 = Kapur Adaptive
%choiceVAR05 = Kapur PerObject
ThresholdMethod{1} = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu 

%textVAR06 = Which grayscale image would you like to use to check for saturation?
%choiceVAR06 = Do not use
%infotypeVAR06 = imagegroup
NameImageToCheck{2} = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 = Which grayscale image would you like to use to calculate a suggested threshold?
%choiceVAR07 = Do not use
%infotypeVAR07 = imagegroup
NameImageToThresh{2} = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = Which automatic thresholding method would you like to use to calculate a suggested threshold?
%choiceVAR08 = Otsu Global
%choiceVAR08 = Otsu Adaptive
%choiceVAR08 = Otsu PerObject
%choiceVAR08 = MoG Global
%choiceVAR08 = MoG Adaptive
%choiceVAR08 = MoG PerObject
%choiceVAR08 = Background Global
%choiceVAR08 = Background Adaptive
%choiceVAR08 = Background PerObject
%choiceVAR08 = RobustBackground Global
%choiceVAR08 = RobustBackground Adaptive
%choiceVAR08 = RobustBackground PerObject
%choiceVAR08 = RidlerCalvard Global
%choiceVAR08 = RidlerCalvard Adaptive
%choiceVAR08 = RidlerCalvard PerObject
%choiceVAR08 = Kapur Global
%choiceVAR08 = Kapur Adaptive
%choiceVAR08 = Kapur PerObject
ThresholdMethod{2} = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%textVAR09 = Which grayscale image would you like to use to check for saturation?
%choiceVAR09 = Do not use
%infotypeVAR09 = imagegroup
NameImageToCheck{3} = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%textVAR10 = Which grayscale image would you like to use to calculate a suggested threshold?
%choiceVAR10 = Do not use
%infotypeVAR10 = imagegroup
NameImageToThresh{3} = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu

%textVAR11 = Which automatic thresholding method would you like to use to calculate a suggested threshold?
%choiceVAR11 = Otsu Global
%choiceVAR11 = Otsu Adaptive
%choiceVAR11 = Otsu PerObject
%choiceVAR11 = MoG Global
%choiceVAR11 = MoG Adaptive
%choiceVAR11 = MoG PerObject
%choiceVAR11 = Background Global
%choiceVAR11 = Background Adaptive
%choiceVAR11 = Background PerObject
%choiceVAR11 = RobustBackground Global
%choiceVAR11 = RobustBackground Adaptive
%choiceVAR11 = RobustBackground PerObject
%choiceVAR11 = RidlerCalvard Global
%choiceVAR11 = RidlerCalvard Adaptive
%choiceVAR11 = RidlerCalvard PerObject
%choiceVAR11 = Kapur Global
%choiceVAR11 = Kapur Adaptive
%choiceVAR11 = Kapur PerObject
ThresholdMethod{3} = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu

%textVAR12 = Which grayscale image would you like to use to check for saturation?
%choiceVAR12 = Do not use
%infotypeVAR12 = imagegroup
NameImageToCheck{4} = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu

%textVAR13 = Which grayscale image would you like to use to calculate a suggested threshold?
%choiceVAR13 = Do not use
%infotypeVAR13 = imagegroup
NameImageToThresh{4} = char(handles.Settings.VariableValues{CurrentModuleNum,13});
%inputtypeVAR13 = popupmenu

%textVAR14 = Which automatic thresholding method would you like to use to calculate a suggested threshold?
%choiceVAR14 = Otsu Global
%choiceVAR14 = Otsu Adaptive
%choiceVAR14 = Otsu PerObject
%choiceVAR14 = MoG Global
%choiceVAR14 = MoG Adaptive
%choiceVAR14 = MoG PerObject
%choiceVAR14 = Background Global
%choiceVAR14 = Background Adaptive
%choiceVAR14 = Background PerObject
%choiceVAR14 = RobustBackground Global
%choiceVAR14 = RobustBackground Adaptive
%choiceVAR14 = RobustBackground PerObject
%choiceVAR14 = RidlerCalvard Global
%choiceVAR14 = RidlerCalvard Adaptive
%choiceVAR14 = RidlerCalvard PerObject
%choiceVAR14 = Kapur Global
%choiceVAR14 = Kapur Adaptive
%choiceVAR14 = Kapur PerObject
ThresholdMethod{4} = char(handles.Settings.VariableValues{CurrentModuleNum,14});
%inputtypeVAR14 = popupmenu


%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS, FILE HANDLING, IMAGE ANALYSIS, STORE DATA IN HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

tmp1 = {};
for n = 1:4
    if ~strcmp(NameImageToCheck{n}, 'Do not use')
        tmp1{end+1} = NameImageToCheck{n};
    end
end
if isempty(tmp1)
    error('You have not selected any images to check for saturation and blur.')
end

tmp2 = {};
for n = 1:4
    if ~strcmp(NameImageToThresh{n}, 'Do not use')
        tmp2{end+1} = NameImageToThresh{n};
    end
end
if isempty(tmp2)
    error('You have not selected any images to calculate the suggested threshold.')
end

NameImageToCheck = tmp1;


%%% Calculate the Saturation and Blur
for ImageNumber = 1:length(NameImageToCheck);
    %%% Reads (opens) the images you want to analyze and assigns them to
    %%% variables.
    ImageToCheck{ImageNumber} = CPretrieveimage(handles,NameImageToCheck{ImageNumber},ModuleName,'MustBeGray','CheckScale'); %#ok Ignore MLint
   
    NumberPixelsSaturated = sum(sum(ImageToCheck{ImageNumber} == 1));
    NumberPixelsMaximal = sum(sum(ImageToCheck{ImageNumber} == max(ImageToCheck{ImageNumber}(:))));
    [m,n] = size(ImageToCheck{ImageNumber});
    TotalPixels = m*n;
    PercentPixelsSaturated = 100*NumberPixelsSaturated/TotalPixels;
    PercentSaturation{ImageNumber} = PercentPixelsSaturated;  %#ok Ignore MLint
    PercentMaximal{ImageNumber} = 100*NumberPixelsMaximal/TotalPixels;
    
    %%% Checks the focus of the images, if desired.
    if ~strcmpi(BlurCheck,'N')
        %         Old method of scoring focus, not justified
        %         RightImage = ImageToCheck{ImageNumber}(:,2:end);
        %         LeftImage = ImageToCheck{ImageNumber}(:,1:end-1);
        %         MeanImageValue = mean(ImageToCheck{ImageNumber}(:));
        %         if MeanImageValue == 0
        %             FocusScore{ImageNumber} = 0;
        %         else
        %             FocusScore{ImageNumber} = std(RightImage(:) - LeftImage(:)) / MeanImageValue;
        %         end
        Image = ImageToCheck{ImageNumber};
        if ~strcmp(class(Image),'double')
            Image = im2double(Image);
        end
        [m,n] = size(Image);
        MeanImageValue = mean(Image(:));
        SquaredNormalizedImage = (Image-MeanImageValue).^2;
        if MeanImageValue == 0
            FocusScore{ImageNumber} = 0;  %#ok Ignore MLint
        else
            FocusScore{ImageNumber} = sum(SquaredNormalizedImage(:))/(m*n*MeanImageValue);
        end

        GlobalFocusScore = FocusScore{ImageNumber};       
        %%% Local normalized variance 
        % WindowSize = 15; %%I'm commenting out this line because it looks
        % like an error.  Why would we want to ask the user to specify a
        % window size if we are going to over-ride it? -Martha 2008-05-22
        m_numblocks = floor(m/WindowSize);
        n_numblocks = floor(n/WindowSize);        
        for i = 1 : m_numblocks
            for j = 1 : n_numblocks
                SubImage = Image((i-1)*WindowSize+1:i*WindowSize,(j-1)*WindowSize+1:j*WindowSize);
                    SubMeanImageValue = mean(SubImage(:));
                    SubSquaredNormalizedImage = (SubImage-SubMeanImageValue).^2;
                if SubMeanImageValue == 0
                    LocalNormVar(i,j) = 0;  %#ok Ignore MLint
                else
                    LocalNormVar(i,j) = sum(SubSquaredNormalizedImage(:))/(WindowSize*WindowSize*SubMeanImageValue);
                end
            end
        end      
        %%% Different statistics testing and chose normvarLocalNormVar 
        %meanLocalNormVar{ImageNumber} = mean(LocalNormVar(:));
        %medianLocalNormVar{ImageNumber} = median(LocalNormVar(:));
        %minLocalNormVar{ImageNumber} = min(LocalNormVar(:));
        %maxLocalNormVar{ImageNumber} = max(LocalNormVar(:));
        %modeLocalNormVar{ImageNumber} = mode(LocalNormVar(:));
        %varLocalNormVar{ImageNumber} = var(LocalNormVar(:));
        %normvarLocalNormVar{ImageNumber} = var(LocalNormVar(:))/mean(LocalNormVar(:));
        normvarLocalNormVar2{ImageNumber} = var(LocalNormVar(:))/median(LocalNormVar(:));
        LocalFocusScore{ImageNumber} = normvarLocalNormVar2{ImageNumber};

        handles = CPaddmeasurements(handles, 'Image', ...
            CPjoinstrings('ImageQuality','FocusScore',NameImageToCheck{ImageNumber}), ...
            FocusScore{ImageNumber});
        handles = CPaddmeasurements(handles, 'Image', ...
            CPjoinstrings('ImageQuality','LocalFocusScore',NameImageToCheck{ImageNumber}), ...
            LocalFocusScore{ImageNumber});
        handles = CPaddmeasurements(handles, 'Image', ...
            CPjoinstrings('ImageQuality','WindowSize',NameImageToCheck{ImageNumber}), ...
            WindowSize);
    else
        handles = CPaddmeasurements(handles, 'Image', ...
            CPjoinstrings('ImageQuality','FocusScore',NameImageToCheck{ImageNumber}),[]);
        handles = CPaddmeasurements(handles, 'Image', ...
            CPjoinstrings('ImageQuality','LocalFocusScore',NameImageToCheck{ImageNumber}),[]);
        handles = CPaddmeasurements(handles, 'Image', ...
            CPjoinstrings('ImageQuality','WindowSize',NameImageToCheck{ImageNumber}),[]);
    end
    handles = CPaddmeasurements(handles, 'Image', ...
        CPjoinstrings('ImageQuality','PercentSaturated',NameImageToCheck{ImageNumber}), ...
        PercentSaturation{ImageNumber});
    handles = CPaddmeasurements(handles, 'Image', ...
        CPjoinstrings('ImageQuality','PercentMaximal',NameImageToCheck{ImageNumber}), ...
        PercentMaximal{ImageNumber});
end

%%% Calculate the Suggested Threshold
NameImageToThresh = tmp2;
pObject='10%';
MinimumThreshold= char('0');
MaximumThreshold=char('1');
ThresholdCorrection=str2num('1');


%%% Now, loop through tmp1 to grab the 'OrigThreshold' from CPthreshold
for ImageNumber = 1:length(tmp2);
    OrigImageThresh = double(CPretrieveimage(handles,NameImageToThresh{ImageNumber},ModuleName,'MustBeGray','CheckScale'));
    [handles,OrigThreshold,WeightedVariance, SumOfEntropies] = CPthreshold(handles,ThresholdMethod{ImageNumber},pObject,MinimumThreshold,MaximumThreshold,ThresholdCorrection,OrigImageThresh,NameImageToThresh{ImageNumber},ModuleName, '');
    handles = CPaddmeasurements(handles,'Image',CPjoinstrings(ModuleName,'Threshold',NameImageToThresh{ImageNumber}), OrigThreshold);
end

SetBeingAnalyzed=handles.Current.SetBeingAnalyzed;
TotalNumberOfImageSets = handles.Current.NumberOfImageSets;

%%% At the end of the image set, calculate the mean, median, and stdev
%%% for the entire image set.
if SetBeingAnalyzed == TotalNumberOfImageSets
        for ImageNumber = 1:length(tmp2),
            Threshold = handles.Measurements.Image.(CPjoinstrings(ModuleName,'Threshold',NameImageToThresh{ImageNumber}));
        end
        MeanThreshold = mean(cellfun(@mean,Threshold));
        MedianThreshold = median(cellfun(@median,Threshold));
        StdevThreshold = std(cellfun(@mean,Threshold));
        handles = CPaddmeasurements(handles,'Experiment', CPjoinstrings(ModuleName,'MeanThresh_AllImages'), MeanThreshold);
        handles = CPaddmeasurements(handles,'Experiment', CPjoinstrings(ModuleName,'MedianThresh_AllImages'), MedianThreshold);
        handles = CPaddmeasurements(handles,'Experiment', CPjoinstrings(ModuleName,'StdThresh_AllImages'), StdevThreshold);
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Text',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure('','NarrowText',ThisModuleFigureNumber)
    end
    
    if isempty(findobj('Parent',ThisModuleFigureNumber,'tag','DisplayText'))
        displaytexthandle = uicontrol(ThisModuleFigureNumber,'tag','DisplayText','style','text','units','normalized','position', [0.1 0.1 0.8 0.8],'fontname','helvetica','backgroundcolor',[.7 .7 .9],'horizontalalignment','left','FontSize',handles.Preferences.FontSize);
    else
        displaytexthandle = findobj('Parent',ThisModuleFigureNumber,'tag','DisplayText');
    end

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
    DisplayText = strvcat(DisplayText,'      ',...
        'Percent of pixels that are in the Maximal Intensity:');
    for ImageNumber = 1:length(PercentMaximal)
        if ~isempty(PercentMaximal{ImageNumber})
            try DisplayText = strvcat(DisplayText, ... %#ok We want to ignore MLint error checking for this line.
                    [NameImageToCheck{ImageNumber}, ':    ', num2str(PercentMaximal{ImageNumber})]);%#ok We want to ignore MLint error checking for this line.
            end
        end
    end

    if strcmpi(BlurCheck, 'N') ~= 1
        DisplayText = strvcat(DisplayText, '      ','      ','Focus Score:'); %#ok We want to ignore MLint error checking for this line.
        for ImageNumber = 1:length(FocusScore)
            if ~isempty(FocusScore{ImageNumber})
                try DisplayText = strvcat(DisplayText, ... %#ok We want to ignore MLint error checking for this line.
                        [NameImageToCheck{ImageNumber}, ':    ', num2str(FocusScore{ImageNumber})]);%#ok We want to ignore MLint error checking for this line.
                end
            end
        end
    end
    if strcmpi(BlurCheck, 'N') ~= 1
        DisplayText = strvcat(DisplayText, '      ','Local Focus Score:'); %#ok We want to ignore MLint error checking for this line.
        for ImageNumber = 1:length(LocalFocusScore)
            if ~isempty(LocalFocusScore{ImageNumber})
                try DisplayText = strvcat(DisplayText, ... %#ok We want to ignore MLint error checking for this line.
                        [NameImageToCheck{ImageNumber}, ':    ', num2str(LocalFocusScore{ImageNumber})]);%#ok We want to ignore MLint error checking for this line.
                end
            end
        end
    end  

    DisplayText = strvcat(DisplayText,'      ',...
        'Suggested Threshold:');
    for ImageNumber = 1:length(NameImageToThresh)
        if ~isempty(NameImageToThresh{ImageNumber})
            try DisplayText = strvcat(DisplayText, ... %#ok We want to ignore MLint error checking for this line.
                    [NameImageToThresh{ImageNumber}, ':    ', num2str(handles.Measurements.Image.(CPjoinstrings(ModuleName,'Threshold',NameImageToThresh{ImageNumber})){1})]);%#ok We want to ignore MLint error checking for this line.
            end
        end
    end

    set(displaytexthandle,'string',DisplayText)
end