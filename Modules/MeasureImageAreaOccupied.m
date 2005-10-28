function handles = MeasureImageAreaOccupied(handles)

% Help for the Measure Image Area Occupied module:
% Category: Measurement
%
% This module simply measures the total area covered by stain in an
% image.
%
% Settings:
%
% Threshold: The threshold affects the stringency of the lines between
% the objects and the background. You may enter an absolute number
% between 0 and 1 for the threshold (use 'Show pixel data' to see the
% pixel intensities for your images in the appropriate range of 0 to
% 1), or you may have it calculated for each image individually by
% typing 0.  There are advantages either way.  An absolute number
% treats every image identically, but an automatically calculated
% threshold is more realistic/accurate, though occasionally subject to
% artifacts.  The threshold which is used for each image is recorded
% as a measurement in the output file, so if you find unusual
% measurements from one of your images, you might check whether the
% automatically calculated threshold was unusually high or low
% compared to the remaining images.  When an automatic threshold is
% selected, it may consistently be too stringent or too lenient, so an
% adjustment factor can be entered as well. The number 1 means no
% adjustment, 0 to 1 makes the threshold more lenient and greater than
% 1 (e.g. 1.3) makes the threshold more stringent.
%
% How it works:
% This module applies a threshold to the incoming image so that any
% pixels brighter than the specified value are assigned the value 1
% (white) and the remaining pixels are assigned the value zero
% (black), producing a binary image.  The number of white pixels are
% then counted.  This provides a measurement of the area occupied by
% fluorescence.  The threshold is calculated automatically and then
% adjusted by a user-specified factor. It might be desirable to write
% a new module where the threshold can be set to a constant value.
%
% SAVING IMAGES: If you want to save images produced by this module,
% alter the code for this module to save those images to the handles
% structure (see the SaveImages module help) and then use the Save
% Images module.
%
% See also MEASUREAREASHAPECOUNTLOCATION,
% MEASURECORRELATION,
% MEASUREINTENSITYTEXTURE,
% MEASURETOTALINTENSITY.

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

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

%textVAR01 = What did you call the images you want to process?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the region measured by this module?
%defaultVAR02 = StainedRegion
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Select thresholding method or enter a threshold in the range [0,1].
%choiceVAR03 = MoG Global
%choiceVAR03 = MoG Adaptive
%choiceVAR03 = Otsu Global
%choiceVAR03 = Otsu Adaptive
Threshold = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu custom

%textVAR04 = Threshold correction factor
%defaultVAR04 = 1
ThresholdCorrection = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,4}));

%textVAR05 = Lower bound on threshold in the range [0,1].
%defaultVAR05 = 0
MinimumThreshold = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = Approximate percentage of image covered by objects (for MoG thresholding only):
%choiceVAR06 = 10%
%choiceVAR06 = 20%
%choiceVAR06 = 30%
%choiceVAR06 = 40%
%choiceVAR06 = 50%
%choiceVAR06 = 60%
%choiceVAR06 = 70%
%choiceVAR06 = 80%
%choiceVAR06 = 90%
pObject = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%%% Retrieves the pixel size that the user entered (micrometers per pixel).
PixelSize = str2double(handles.Settings.PixelSize);

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
fieldname = ['', ImageName];
%%% Checks whether image has been loaded.
if isfield(handles.Pipeline, fieldname)==0,
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
OrigImage = handles.Pipeline.(fieldname);

pObject = str2num(pObject(1:2))/100;

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% STEP 1. Find threshold and apply to image
if strfind(Threshold,'Global')
    if strfind(Threshold,'Otsu')
        Threshold = CPgraythresh(OrigImage,handles,ImageName);
    elseif strfind(Threshold,'MoG')
        Threshold = MixtureOfGaussians(OrigImage,pObject);
    end
elseif strfind(Threshold,'Adaptive')
    %%% Choose the block size that best covers the original image in the sense
    %%% that the number of extra rows and columns is minimal.
    %%% Get size of image
    [m,n] = size(OrigImage);

    %%% Deduce a suitable block size based on the image size and the percentage of image
    %%% covered by objects. We want blocks to be big enough to contain both background and
    %%% objects. The more uneven the ratio between background pixels and object pixels the
    %%% larger the block size need to be. The minimum block size is about 50x50 pixels.
    %%% The line below divides the image in 10x10 blocks, and makes sure that the block size is
    %%% at least 50x50 pixels.
    BlockSize = max(50,min(round(m/10),round(n/10)));

    %%% Calculates a range of acceptable block sizes as plus-minus 10% of the suggested block size.
    BlockSizeRange = floor(1.1*BlockSize):-1:ceil(0.9*BlockSize);
    [ignore,index] = min(ceil(m./BlockSizeRange).*BlockSizeRange-m + ceil(n./BlockSizeRange).*BlockSizeRange-n);
    BestBlockSize = BlockSizeRange(index);

    %%% Pads the image so that the blocks fit properly.
    RowsToAdd = BestBlockSize*ceil(m/BestBlockSize) - m;
    ColumnsToAdd = BestBlockSize*ceil(n/BestBlockSize) - n;
    RowsToAddPre = round(RowsToAdd/2);
    RowsToAddPost = RowsToAdd - RowsToAddPre;
    ColumnsToAddPre = round(ColumnsToAdd/2);
    ColumnsToAddPost = ColumnsToAdd - ColumnsToAddPre;
    PaddedImage = padarray(OrigImage,[RowsToAddPre ColumnsToAddPre],'replicate','pre');
    PaddedImage = padarray(PaddedImage,[RowsToAddPost ColumnsToAddPost],'replicate','post');

    %%% Calculates the threshold for each block in the image, and a global threshold used
    %%% to constrain the adaptive threshholds.
    if strfind(Threshold,'Otsu')
        GlobalThreshold = graythresh(OrigImage);
        Threshold = blkproc(PaddedImage,[BestBlockSize BestBlockSize],'graythresh(x)');
    elseif strfind(Threshold,'MoG')
        GlobalThreshold = MixtureOfGaussians(OrigImage,pObject);
        Threshold = blkproc(PaddedImage,[BestBlockSize BestBlockSize],@MixtureOfGaussians,pObject);
    end
    %%% Resizes the block-produced image to be the size of the padded image.
    %%% Bilinear prevents dipping below zero. The crop the image
    %%% get rid of the padding, to make the result the same size as the original image.
    Threshold = imresize(Threshold, size(PaddedImage), 'bilinear');
    Threshold = Threshold(RowsToAddPre+1:end-RowsToAddPost,ColumnsToAddPre+1:end-ColumnsToAddPost);

    %%% For any of the threshold values that is lower than the user-specified
    %%% minimum threshold, set to equal the minimum threshold.  Thus, if there
    %%% are no objects within a block (e.g. if cells are very sparse), an
    %%% unreasonable threshold will be overridden by the minimum threshold.
    Threshold(Threshold <= 0.7*GlobalThreshold) = 0.7*GlobalThreshold;
    Threshold(Threshold >= 1.5*GlobalThreshold) = 1.5*GlobalThreshold;
else
    %%% The threshold is manually set by the user
    %%% Checks that the Threshold parameter has a valid value
    Threshold = str2double(Threshold);
    if isnan(Threshold) | Threshold > 1 | Threshold < 0
        error('The threshold entered in the IdentifyPrimAutomatic module is not a number, or is outside the acceptable range of 0 to 1.')
    end
end
%%% Correct the threshold using the correction factor given by the user
%%% and make sure that the threshold is not larger than the minimum threshold
Threshold = ThresholdCorrection*Threshold;
Threshold = max(Threshold,MinimumThreshold);
drawnow

%%% Thresholds the original image.
ThresholdedOrigImage = im2bw(OrigImage,Threshold);
AreaOccupiedPixels = sum(ThresholdedOrigImage(:));
AreaOccupied = AreaOccupiedPixels*PixelSize*PixelSize;

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;
    drawnow
    %%% Sets the width of the figure window to be appropriate (half width).
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 0.5*originalsize(3);
        newsize(4) = 1.2*originalsize(4);
        newsize(2) = originalsize(2)-0.2*originalsize(4);
        set(ThisModuleFigureNumber, 'position', newsize);
    end
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1); imagesc(OrigImage);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,1,2); imagesc(ThresholdedOrigImage); title('Thresholded Image');
    if handles.Current.SetBeingAnalyzed == 1
        displaytexthandle = uicontrol(ThisModuleFigureNumber,'tag','DisplayText','style','text', 'position', [20 0 250 40],'fontname','fixedwidth','backgroundcolor',[0.7 0.7 0.9],'FontSize',handles.Current.FontSize);
    else
        displaytexthandle = findobj('Parent',ThisModuleFigureNumber,'tag','DisplayText');
    end
    displaytext = {['  Image Set # ',num2str(handles.Current.SetBeingAnalyzed)];...
        ['  Area occupied by ',ObjectName,':      ',num2str(AreaOccupied,'%2.1E')]};
    set(displaytexthandle,'string',displaytext)
    set(ThisModuleFigureNumber,'toolbar','figure')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

FeatureNames = {'ImageAreaOccupied','ImageAreaOccupiedThreshold'};
fieldname = ['ImageAreaOccupied',ObjectName,'Features'];
handles.Measurements.Image.(fieldname) = FeatureNames;

fieldname = ['ImageAreaOccupied',ObjectName];
handles.Measurements.Image.(fieldname){handles.Current.SetBeingAnalyzed}(:,1) = AreaOccupied;
handles.Measurements.Image.(fieldname){handles.Current.SetBeingAnalyzed}(:,2) = Threshold;

%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%

function Threshold = MixtureOfGaussians(OrigImage,pObject)
%%% This function finds a suitable threshold for the input image
%%% OrigImage. It assumes that the pixels in the image belong to either
%%% a background class or an object class. 'pObject' is an initial guess
%%% of the prior probability of an object pixel, or equivalently, the fraction
%%% of the image that is covered by objects. Essentially, there are two steps.
%%% First, a number of Gaussian distributions are estimated to match the
%%% distribution of pixel intensities in OrigImage. Currently 3 Gaussian
%%% distributions are fitted, one corresponding to a background class, one
%%% corresponding to an object class, and one distribution for an intermediate
%%% class. The distributions are fitted using the Expectation-Maximization (EM)
%%% algorithm, a procedure referred to as Mixture of Gaussians modeling. When
%%% the 3 Gaussian distributions have been fitted, it's decided whether the
%%% intermediate class models background pixels or object pixels based on the
%%% probability of an object pixel 'pObject' given by the user.

%%% The number of classes is set to 3
NumberOfClasses = 3;

%%% Transform the image into a vector. Also, if the image is (larger than 512x512),
%%% select a subset of 512^2 pixels for speed. This should be enough to capture the
%%% statistics in the image.
Intensities = OrigImage(:);
if length(Intensities) > 512^2
    indexes = randperm(length(Intensities));
    Intensities = Intensities(indexes(1:512^2));
end

%%% Get the probability for a background pixel
pBackground = 1 - pObject;

%%% Initialize mean and standard deviations of the three Gaussian distributions
%%% by looking at the pixel intensities in the original image and by considering
%%% the percentage of the image that is covered by object pixels. Class 1 is the
%%% background class and Class 3 is the object class. Class 2 is an intermediate
%%% class and we will decide later if it encodes background or object pixels.
%%% Also, for robustness the we remove 1% of the smallest and highest intensities
%%% in case there are any quantization effects that have resulted in unaturally many
%%% 0:s or 1:s in the image.
Intensities = sort(Intensities);
Intensities = Intensities(round(length(Intensities)*0.01):round(length(Intensities)*0.99));
ClassMean(1) = Intensities(round(length(Intensities)*pBackground/2));                      %%% Initialize background class
ClassMean(3) = Intensities(round(length(Intensities)*(1 - pObject/2)));                    %%% Initialize object class
ClassMean(2) = (ClassMean(1) + ClassMean(3))/2;                                            %%% Initialize intermediate class
%%% Initialize standard deviations of the Gaussians. They should be the same to avoid problems.
ClassStd(1:3) = 0.15;
%%% Initialize prior probabilities of a pixel belonging to each class. The intermediate
%%% class is gets some probability from the background and object classes.
pClass(1) = 3/4*pBackground;
pClass(2) = 1/4*pBackground + 1/4*pObject;
pClass(3) = 3/4*pObject;

%%% Apply transformation.  a < x < b, transform to log((x-a)/(b-x)).
%a = - 0.000001;
%b = 1.000001;
%Intensities = log((Intensities-a)./(b-Intensities));
%ClassMean = log((ClassMean-a)./(b - ClassMean))
%ClassStd(1:3) = [1 1 1];

%%% Expectation-Maximization algorithm for fitting the three Gaussian distributions/classes
%%% to the data. Note, the code below is general and works for any number of classes.
%%% Iterate until parameters don't change anymore.
delta = 1;
while delta > 0.001
    %%% Store old parameter values to monitor change
    oldClassMean = ClassMean;

    %%% Update probabilities of a pixel belonging to the background or object1 or object2
    for k = 1:NumberOfClasses
        pPixelClass(:,k) = pClass(k)* 1/sqrt(2*pi*ClassStd(k)^2) * exp(-(Intensities - ClassMean(k)).^2/(2*ClassStd(k)^2));
    end
    pPixelClass = pPixelClass ./ repmat(sum(pPixelClass,2) + eps,[1 NumberOfClasses]);

    %%% Update parameters in Gaussian distributions
    for k = 1:NumberOfClasses
        pClass(k) = mean(pPixelClass(:,k));
        ClassMean(k) = sum(pPixelClass(:,k).*Intensities)/(length(Intensities)*pClass(k));
        ClassStd(k)  = sqrt(sum(pPixelClass(:,k).*(Intensities - ClassMean(k)).^2)/(length(Intensities)*pClass(k))) + sqrt(eps);    % Add sqrt(eps) to avoid division by zero
    end

    %%% Calculate change
    delta = sum(abs(ClassMean - oldClassMean));
end

%%% Now the Gaussian distributions are fitted and we can describe the histogram of the pixel
%%% intensities as the sum of these Gaussian distributions. To find a threshold we first have
%%% to decide if the intermediate class 2 encodes background or object pixels. This is done by
%%% choosing the combination of class probabilities 'pClass' that best matches the user input 'pObject'.
Threshold = linspace(ClassMean(1),ClassMean(3),10000);
Class1Gaussian = pClass(1) * 1/sqrt(2*pi*ClassStd(1)^2) * exp(-(Threshold - ClassMean(1)).^2/(2*ClassStd(1)^2));
Class2Gaussian = pClass(2) * 1/sqrt(2*pi*ClassStd(2)^2) * exp(-(Threshold - ClassMean(2)).^2/(2*ClassStd(2)^2));
Class3Gaussian = pClass(3) * 1/sqrt(2*pi*ClassStd(3)^2) * exp(-(Threshold - ClassMean(3)).^2/(2*ClassStd(3)^2));
if abs(pClass(2) + pClass(3) - pObject) < abs(pClass(3) - pObject)
    %%% Intermediate class 2 encodes object pixels
    BackgroundDistribution = Class1Gaussian;
    ObjectDistribution = Class2Gaussian + Class3Gaussian;
else
    %%% Intermediate class 2 encodes background pixels
    BackgroundDistribution = Class1Gaussian + Class2Gaussian;
    ObjectDistribution = Class3Gaussian;
end

%%% Now, find the threshold at the intersection of the background distribution
%%% and the object distribution.
[ignore,index] = min(abs(BackgroundDistribution - ObjectDistribution));
Threshold = Threshold(index);