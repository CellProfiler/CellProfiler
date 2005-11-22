function [handles,Threshold] = CPthreshold(handles,Threshold,pObject,MinimumThreshold,MaximumThreshold,ThresholdCorrection,OrigImage,ImageName,ModuleName)

%%% Convert user-specified percentage of image covered by objects to a prior probability
%%% of a pixel being part of an object.
pObject = str2double(pObject(1:2))/100;

%%% Check the MinimumThreshold entry. If no minimum threshold has been set, set it to zero.
%%% Otherwise make sure that the user gave a valid input.
if strcmp(MinimumThreshold,'Do not use')
    MinimumThreshold = 0;
else
    MinimumThreshold = str2double(MinimumThreshold);
    if isnan(MinimumThreshold) |  MinimumThreshold < 0 | MinimumThreshold > 1 %#ok Ignore MLint
        error(['The Minimum threshold entry in the ', ModuleName, ' module is invalid.'])
    end
end

if strcmp(MaximumThreshold,'Do not use')
    MaximumThreshold = 1;
else
    MaximumThreshold = str2double(MaximumThreshold);
    if isnan(MaximumThreshold) | MaximumThreshold < 0 | MaximumThreshold > 1 %#ok Ignore MLint
        error(['The Maximum bound on the threshold in the ', ModuleName, ' module is invalid.'])
    end

    if MinimumThreshold > MaximumThreshold,
        error(['Min bound on the threshold larger the Max bound on the threshold in the ', ModuleName, ' module.'])
    end
end

%%% STEP 1. Find threshold and apply to image
if strfind(Threshold,'Global')
    if strfind(Threshold,'Otsu')
        Threshold = CPgraythresh(OrigImage,handles,ImageName);
    elseif strfind(Threshold,'MoG')
        Threshold = MixtureOfGaussians(OrigImage,handles,pObject,ImageName);
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
    [ignore,index] = min(ceil(m./BlockSizeRange).*BlockSizeRange-m + ceil(n./BlockSizeRange).*BlockSizeRange-n); %#ok Ignore MLint
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
        GlobalThreshold = CPgraythresh(OrigImage);
        Threshold = blkproc(PaddedImage,[BestBlockSize BestBlockSize],@CPgraythresh);
    elseif strfind(Threshold,'MoG')
        GlobalThreshold = MixtureOfGaussians(OrigImage,handles,pObject,ImageName);
        Threshold = blkproc(PaddedImage,[BestBlockSize BestBlockSize],@MixtureOfGaussians,handles,pObject,ImageName);
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
elseif strcmp(Threshold,'All')
    if handles.Current.SetBeingAnalyzed == 1
        try
            %%% Notifies the user that the first image set will take much longer than
            %%% subsequent sets.
            %%% Obtains the screen size.
            ScreenSize = get(0,'ScreenSize');
            ScreenHeight = ScreenSize(4);
            PotentialBottom = [0, (ScreenHeight-720)];
            BottomOfMsgBox = max(PotentialBottom);
            PositionMsgBox = [500 BottomOfMsgBox 350 100];
            h = CPmsgbox('Preliminary calculations are under way for the Identify Primary Threshold module.  Subsequent image sets will be processed much more quickly than the first image set.');
            set(h, 'Position', PositionMsgBox)
            drawnow
            %%% Retrieves the path where the images are stored from the handles
            %%% structure.
            fieldname = ['Pathname', ImageName];
            try Pathname = handles.Pipeline.(fieldname);
            catch error(['Image processing was canceled in the ', ModuleName, ' module because it must be run using images straight from a load images module (i.e. the images cannot have been altered by other image processing modules). This is because you have asked the Identify Primary Threshold module to calculate a threshold based on all of the images before identifying objects within each individual image as CellProfiler cycles through them. One solution is to process the entire batch of images using the image analysis modules preceding this module and save the resulting images to the hard drive, then start a new stage of processing from this Identify Primary Threshold module onward.'])
            end
            %%% Retrieves the list of filenames where the images are stored from the
            %%% handles structure.
            fieldname = ['FileList', ImageName];
            FileList = handles.Pipeline.(fieldname);
            %%% Calculates the threshold based on all of the images.
            Counts = zeros(256,1);
            NumberOfBins = 256;
            for i=1:length(FileList)
                [Image, handles] = CPimread(fullfile(Pathname,char(FileList(i))),handles);
                Counts = Counts + imhist(im2uint8(Image(:)), NumberOfBins);
                drawnow
            end
            % Variables names are chosen to be similar to the formulas in
            % the Otsu paper.
            P = Counts / sum(Counts);
            Omega = cumsum(P);
            Mu = cumsum(P .* (1:NumberOfBins)');
            Mu_t = Mu(end);
            % Saves the warning state and disable warnings to prevent divide-by-zero
            % warnings.
            State = warning;
            warning off Matlab:DivideByZero
            SigmaBSquared = (Mu_t * Omega - Mu).^2 ./ (Omega .* (1 - Omega));
            % Restores the warning state.
            warning(State);
            % Finds the location of the maximum value of sigma_b_squared.
            % The maximum may extend over several bins, so average together the
            % locations.  If maxval is NaN, meaning that sigma_b_squared is all NaN,
            % then return 0.
            Maxval = max(SigmaBSquared);
            if isfinite(Maxval)
                Idx = mean(find(SigmaBSquared == Maxval));
                % Normalizes the threshold to the range [0, 1].
                Threshold = (Idx - 1) / (NumberOfBins - 1);
            else
                Threshold = 0.0;
            end
        catch [ErrorMessage, ErrorMessage2] = lasterr;
            error(['An error occurred in the ', ModuleName, ' module. Matlab says the problem is: ', ErrorMessage, ErrorMessage2])
        end
        fieldname = ['Threshold', ImageName];
        handles.Pipeline.(fieldname) = Threshold;
    else fieldname = ['Threshold', ImageName];
        Threshold = handles.Pipeline.(fieldname);
    end
elseif strcmp(Threshold,'Test Mode')
    fieldname = ['Threshold',ImageName];
    if handles.Current.SetBeingAnalyzed == 1
        Threshold = CPthresh_tool(OrigImage);
        handles.Pipeline.(fieldname) = Threshold;
    else
        Threshold = handles.Pipeline.(fieldname);
    end
else
    %%% The threshold is manually set by the user
    %%% Checks that the Threshold parameter has a valid value
    Threshold = str2double(Threshold);
    if isnan(Threshold) | Threshold > 1 | Threshold < 0 %#ok Ignore MLint
        error(['The threshold entered in the ', ModuleName, ' module is not a number, or is outside the acceptable range of 0 to 1.'])
    end
end
%%% Correct the threshold using the correction factor given by the user
%%% and make sure that the threshold is not larger than the minimum threshold
Threshold = ThresholdCorrection*Threshold;
Threshold = max(Threshold,MinimumThreshold);
Threshold = min(Threshold,MaximumThreshold);

function Threshold = MixtureOfGaussians(OrigImage,handles,pObject,ImageName)
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

%%% If the image was produced using a cropping mask, we do not
%%% want to include the Masked part in the calculation of the
%%% proper threshold, because there will be many zeros in the
%%% image.  So, we check to see whether there is a field in the
%%% handles structure that goes along with the image of interest.
fieldname = ['CropMask', ImageName];
if isfield(handles.Pipeline,fieldname)
    %%% Retrieves previously selected cropping mask from handles
    %%% structure.
    BinaryCropImage = handles.Pipeline.(fieldname);
    if numel(OrigImage) == numel(BinaryCropImage)
        %%% Masks the image and I think turns it into a linear
        %%% matrix.
        OrigImage = OrigImage(logical(BinaryCropImage));
    end
end

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
[ignore,index] = min(abs(BackgroundDistribution - ObjectDistribution)); %#ok Ignore MLint
Threshold = Threshold(index);

function level = CPgraythresh(varargin)
%%% This is the Otsu method of thresholding.

if nargin == 1
    im = varargin{1};
else
    im = varargin{1};
    handles = varargin{2};
    ImageName = varargin{3};
    %%% If the image was produced using a cropping mask, we do not
    %%% want to include the Masked part in the calculation of the
    %%% proper threshold, because there will be many zeros in the
    %%% image.  So, we check to see whether there is a field in the
    %%% handles structure that goes along with the image of interest.
    fieldname = ['CropMask', ImageName];
    if isfield(handles.Pipeline,fieldname)
        %%% Retrieves previously selected cropping mask from handles
        %%% structure.
        BinaryCropImage = handles.Pipeline.(fieldname);
        if numel(im) == numel(BinaryCropImage)
            %%% Masks the image and I think turns it into a linear
            %%% matrix.
            im = im(logical(BinaryCropImage));
        end
    end
end

%%% The threshold is calculated using the matlab function graythresh
%%% but with our modifications that work in log space, and take into
%%% account the max and min values in the image.
im = double(im(:));

if max(im) == min(im),
    level = im(1);
else
    %%% We want to limit the dynamic range of the image to 256.
    %%% Otherwise, an image with almost all values near zero can give a
    %%% bad result.
    minval = max(im)/256;
    im(im < minval) = minval;
    im = log(im);
    minval = min (im);
    maxval = max (im);
    im = (im - minval) / (maxval - minval);
    level = exp(minval + (maxval - minval) * graythresh(im));
end