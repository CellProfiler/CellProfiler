function handles = MeasureImageAreaOccupied(handles)

% Help for the Measure Image Area Occupied module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Measures total area covered by stain in an image.
% *************************************************************************
%
% This module simply measures the total area covered by stain in an
% image, using a threshold to determine stain vs background.
%
% How it works:
% This module applies a threshold to the incoming image so that any pixels
% brighter than the specified value are assigned the value 1 (white) and
% the remaining pixels are assigned the value zero (black), producing a
% binary image. The number of white pixels are then counted. This provides
% a measurement of the area occupied by the staining.
%
% Features measured:      Feature Number:
% AreaOccupied        |        1
% TotalImageArea      |        2
% ThresholdUsed       |        3
%
% (Note: to use with Calculate modules, the "category of measures you would
% like to use" should be entered as: AreaOccupied_Name, where Name is the 
% name you entered in the MeasureAreaOccupied module.)
%
% Settings:
%
% * Select automatic thresholding method:
%    The threshold affects the stringency of the lines between the
% objects and the background. You can have the threshold automatically
% calculated using several methods, or you can enter an absolute number
% between 0 and 1 for the threshold (to see the pixel intensities for your
% images in the appropriate range of 0 to 1, use the CellProfiler Image
% Tool, 'Show Or Hide Pixel Data', in a window showing your image).
% There are advantages either way.  An absolute number treats every
% image identically, but is not robust to slight changes in
% lighting/staining conditions between images. An automatically
% calculated threshold adapts to changes in lighting/staining
% conditions between images and is usually more robust/accurate, but
% it can occasionally produce a poor threshold for unusual/artifactual
% images. It also takes a short time to calculate.
%    The threshold which is used for each image is recorded as a
% measurement in the output file, so if you find unusual measurements
% from one of your images, you might check whether the automatically
% calculated threshold was unusually high or low compared to the
% other images.
%    There are four methods for finding thresholds automatically, Otsu's
% method, the Mixture of Gaussian (MoG) method, the Background method, and
% the Ridler-Calvard method. The Otsu method uses our version of the Matlab
% function graythresh (the code is in the CellProfiler subfunction
% CPthreshold). Our modifications include taking into account the max and
% min values in the image and log-transforming the image prior to
% calculating the threshold. Otsu's method is probably better if you don't
% know anything about the image, or if the percent of the image covered by
% objects varies substantially from image to image. But if you know the
% object coverage percentage and it does not vary much from image to image,
% the MoG can be better, especially if the coverage percentage is not near
% 50%. Note, however, that the MoG function is experimental and has not
% been thoroughly validated. The background function is very simple and is
% appropriate for images in which most of the image is background. It finds
% the mode of the histogram of the image, which is assumed to be the
% background of the image, and chooses a threshold at twice that value
% (which you can adjust with a Threshold Correction Factor, see below).
% This can be very helpful, for example, if your images vary in overall
% brightness but the objects of interest are always twice (or actually, any
% constant) as bright as the background of the image. The Ridler-Calvard
% method is simple and its results are often very similar to Otsu's. It
% chooses and initial threshold, and then iteratively calculates the next
% one by taking the mean of the average intensities of the background and
% foreground pixels determined by the first threshold, repeating this until
% the threshold converges.
%
% * Threshold correction factor:
% When the threshold is calculated automatically, it may consistently be
% too stringent or too lenient. You may need to enter an adjustment factor
% which you empirically determine is suitable for your images. The number 1
% means no adjustment, 0 to 1 makes the threshold more lenient and greater
% than 1 (e.g. 1.3) makes the threshold more stringent. For example, the
% Otsu automatic thresholding inherently assumes that 50% of the image is
% covered by objects. If a larger percentage of the image is covered, the
% Otsu method will give a slightly biased threshold that may have to be
% corrected using a threshold correction factor.
%
% * Lower and upper bounds on threshold:
% Can be used as a safety precaution when the threshold is calculated
% automatically. For example, if there are no objects in the field of view,
% the automatic threshold will be unreasonably low. In such cases, the
% lower bound you enter here will override the automatic threshold.
%
% * Approximate percentage of image covered by objects:
% An estimate of how much of the image is covered with objects. This
% information is currently only used in the MoG (Mixture of Gaussian)
% thresholding but may be used for other thresholding methods in the
% future.
%
% See also IdentifyPrimAutomatic, IdentifyPrimManual, and
% MeasureObjectAreaShape modules.


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
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the images you want to process?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu custom

%textVAR02 = What do you want to call the staining measured by this module?
%defaultVAR02 = CellStain
%infotypeVAR02 = imagegroup indep
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Select an automatic thresholding method or enter an absolute threshold in the range [0,1]. Choosing 'All' will use the Otsu Global method to calculate a single threshold for the entire image group. The other methods calculate a threshold for each image individually. Set interactively will allow you to manually adjust the threshold to determine what will work well.
%choiceVAR03 = Otsu Global
%choiceVAR03 = Otsu Adaptive
%choiceVAR03 = MoG Global
%choiceVAR03 = MoG Adaptive
%choiceVAR03 = Background Global
%choiceVAR03 = Background Adaptive
%choiceVAR03 = RobustBackground Global
%choiceVAR03 = RobustBackground Adaptive
%choiceVAR03 = RidlerCalvard Global
%choiceVAR03 = RidlerCalvard Adaptive
%choiceVAR03 = Kapur Global
%choiceVAR03 = Kapur Adaptive
%choiceVAR03 = All
%choiceVAR03 = Set interactively
Threshold = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu custom

%textVAR04 = Threshold correction factor
%defaultVAR04 = 1
ThresholdCorrection = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = Lower and upper bounds on threshold, in the range [0,1]
%defaultVAR05 = 0,1
ThresholdRange = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = For MoG thresholding, what is the approximate fraction of image covered by objects? 
%choiceVAR06 = 0.01
%choiceVAR06 = 0.1
%choiceVAR06 = 0.2
%choiceVAR06 = 0.3
%choiceVAR06 = 0.4
%choiceVAR06 = 0.5
%choiceVAR06 = 0.6
%choiceVAR06 = 0.7
%choiceVAR06 = 0.8
%choiceVAR06 = 0.9
%choiceVAR06 = 0.99
pObject = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu custom

%%% Retrieves the pixel size that the user entered (micrometers per pixel).
PixelSize = str2double(handles.Settings.PixelSize);

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','CheckScale');

%%% Checks that the Min and Max threshold bounds have valid values
index = strfind(ThresholdRange,',');
if isempty(index)
    error(['Image processing was canceled in the ', ModuleName, ' module because the Min and Max threshold bounds are invalid.'])
end
MinimumThreshold = ThresholdRange(1:index-1);
MaximumThreshold = ThresholdRange(index+1:end);

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% STEP 1. Find threshold and apply to image
[handles,Threshold] = CPthreshold(handles,Threshold,pObject,MinimumThreshold,MaximumThreshold,ThresholdCorrection,OrigImage,ImageName,ModuleName);

%%% Thresholds the original image.
ThresholdedOrigImage = OrigImage > Threshold;   
% ThresholdedOrigImage = im2bw(OrigImage,Threshold);
AreaOccupiedPixels = sum(ThresholdedOrigImage(:));
AreaOccupied = AreaOccupiedPixels*PixelSize*PixelSize;

[rows,columns] = size(OrigImage);
TotalImageArea = rows*columns*PixelSize*PixelSize;

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber) == 1;
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'TwoByOne',ThisModuleFigureNumber)
    end
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1); 
    CPimagesc(OrigImage,handles);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,1,2); 
    CPimagesc(ThresholdedOrigImage,handles); 
    title('Thresholded Image');
    if isempty(findobj('Parent',ThisModuleFigureNumber,'tag','DisplayText'))
        displaytexthandle = uicontrol(ThisModuleFigureNumber,'tag','DisplayText','style','text', 'position', [0 0 200 40],'fontname','helvetica','backgroundcolor',[0.7 0.7 0.9],'FontSize',handles.Preferences.FontSize);
    else
        displaytexthandle = findobj('Parent',ThisModuleFigureNumber,'tag','DisplayText');
    end
    displaytext = {['Area occupied by ',ObjectName,':      ',num2str(AreaOccupied,'%2.1E')]};
    set(displaytexthandle,'string',displaytext)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

FeatureNames = {'AreaOccupied','TotalImageArea','ThresholdUsed'};
fieldname = ['AreaOccupied_',ObjectName,'Features'];
handles.Measurements.Image.(fieldname) = FeatureNames;

fieldname = ['AreaOccupied_',ObjectName];
handles.Measurements.Image.(fieldname){handles.Current.SetBeingAnalyzed}(:,1) = AreaOccupied;
handles.Measurements.Image.(fieldname){handles.Current.SetBeingAnalyzed}(:,2) = TotalImageArea;
handles.Measurements.Image.(fieldname){handles.Current.SetBeingAnalyzed}(:,3) = Threshold;

%% Save the thresholded image in handles.Pipeline for later use.
handles.Pipeline.(ObjectName) = ThresholdedOrigImage;
