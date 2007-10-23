function [handles,Threshold,varargout] = CPthreshold(handles,Threshold,pObject,MinimumThreshold,MaximumThreshold,ThresholdCorrection,OrigImage,ImageName,ModuleName,ObjectVar)
%
% Returns an automatically computed threshold, and if requested in
% varargout, the Otsu and Kapur measures of thresholding quality
% (weighted variance and sum of entropies, resp.).
%
% CellProfiler is distributed under the GNU General Public License. See the
% accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research. Copyright
% 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter Thouis Ray Jones In Han Kang Ola Friman Steve Lowe
%   Joo Han Chang Colin Clarke Mike Lamprecht Peter Swire Rodrigo Ipince
%   Vicky Lay Jun Liu Chris Gang
%
% Website: http://www.cellprofiler.org
%
% $Revision$

if nargin == 9
    ObjectVar = [];
else
    ObjectVar = [ObjectVar,'_'];
end

%%% If we are running the Histogram data tool we do not want to limit the
%%% threshold with a maximum of 1 or minimum of 0; otherwise we check for
%%% values outside the range here.
if ~strcmpi('Histogram Data tool',ModuleName)
    %%% Check the MinimumThreshold entry. If no minimum threshold has been
    %%% set, set it to zero. Otherwise make sure that the user gave a valid
    %%% input.
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
            error(['Min bound on the threshold larger than the Max bound on the threshold in the ', ModuleName, ' module.'])
        end
    end
end

%%% If the image was produced using a cropping mask, we do not want to
%%% include the Masked part in the calculation of the proper threshold,
%%% because there will be many zeros in the image.  So, we check to see
%%% whether there is a crop mask image in the handles structure that goes
%%% along with the image of interest.

%%% However, we do not need to retrieve the binary crop mask in a few
%%% cases: if the user is going to set the threshold interactively, if they
%%% are using All images together to calculate the threshold, or if they
%%% have manually entered a numerical value for the threshold.
if strcmp(Threshold,'Set interactively') || strcmp(Threshold,'All') || ~isempty(str2num(Threshold)) 
    %%% In these cases, don't do anything.
else
    fieldname = ['CropMask', ImageName];
    if isfield(handles.Pipeline,fieldname)
        %%% Retrieves crop mask from handles structure. In some cases, it
        %%% might be a label matrix, if we are cropping based on objects
        %%% being present, so we make sure the resulting image is of the
        %%% logical class. This yields a warning at times, because the
        %%% image has values above one, so we temporarily turn the warning
        %%% off.
        OrigWarnState = warning('off','MATLAB:conversionToLogical');
        RetrievedCropMask = handles.Pipeline.(fieldname);
        RetrievedBinaryCropMask = logical(RetrievedCropMask);
        warning(OrigWarnState);
        %%% Handle the case where there are no pixels on in the mask, in
        %%% which case the threshold should be set to a numnber higher than
        %%% any pixels in the original image. In this case, the automatic
        %%% calculations are aborted below, because now Threshold = a
        %%% numerical value rather than a method name like 'Otsu', etc. So
        %%% from this point forward, in this case, the threshold is as if
        %%% entered manually.
        if (~any(RetrievedBinaryCropMask)),
            Threshold = 1;
        end
        %%% Checks whether the size of the RetrievedBinaryCropMask matches
        %%% the size of the OrigImage.
        if numel(OrigImage) == numel(RetrievedBinaryCropMask)
            BinaryCropMask = RetrievedBinaryCropMask;
            %%% Masks the image based on its BinaryCropMask and
            %%% simultaneously makes it a linear set of numbers.
            LinearMaskedImage = OrigImage(BinaryCropMask~=0);
        else
            Warning = CPwarndlg(['In CPthreshold, within the ',ModuleName,' module, the retrieved binary crop mask image (handles.Pipeline.',fieldname,') is not being used because it does not match the size of the original image(',ImageName,').']);
            %%% If the sizes do not match, then it is as if the crop mask
            %%% does not exist. I don't think this should ever actually
            %%% happen, but it might be needed for debugging.
        end
    end
    %%% If we have not masked the image for some reason, we need to create
    %%% the LinearMaskedImage variable, and simultaneously make it a linear
    %%% set of numbers.
    if ~exist('LinearMaskedImage','var')
        LinearMaskedImage = OrigImage(:);
    end
end

%%% STEP 1. Find threshold and apply to image
if ~isempty(strfind(Threshold,'Global')) || ~isempty(strfind(Threshold,'Adaptive')) || ~isempty(strfind(Threshold,'PerObject'))
    if ~isempty(strfind(Threshold,'Global'))
        MethodFlag = 0;
    elseif ~isempty(strfind(Threshold,'Adaptive'))
        MethodFlag = 1;
    elseif ~isempty(strfind(Threshold,'PerObject'))
        MethodFlag = 2;
    end
    %%% Chooses the first word of the method name (removing 'Global' or 'Adaptive' or 'PerObject').
    ThresholdMethod = strtok(Threshold);
    %%% Makes sure we are using an existing thresholding method.
    if isempty(strmatch(ThresholdMethod,{'Otsu','MoG','Background','RobustBackground','RidlerCalvard','Kapur'},'exact'))
        error(['The method chosen for thresholding, ',Threshold,', in the ',ModuleName,' module was not recognized by the CPthreshold subfunction. Adjustment to the code of CellProfiler is needed; sorry for the inconvenience.'])
    end
    
    %%% For all methods, Global or Adaptive or PerObject, we want to
    %%% calculate the global threshold. Sends the linear masked image to
    %%% the appropriate thresholding subfunction.
    eval(['Threshold = ',ThresholdMethod,'(LinearMaskedImage,handles,ImageName,pObject);']);

    %%% This evaluates to something like: Threshold =
    %%% Otsu(LinearMaskedImage,handles,ImageName,pObject);

    %%% The global threshold is used to constrain the Adaptive or PerObject
    %%% thresholds.
    GlobalThreshold = Threshold;

    %%% For Global, we are done. There are more steps involved for Adaptive
    %%% and PerObject methods.

    if MethodFlag == 1 %%% The Adaptive method.
        %%% Choose the block size that best covers the original image in
        %%% the sense that the number of extra rows and columns is minimal.
        %%% Get size of image
        [m,n] = size(OrigImage);
        %%% Deduce a suitable block size based on the image size and the
        %%% percentage of image covered by objects. We want blocks to be
        %%% big enough to contain both background and objects. The more
        %%% uneven the ratio between background pixels and object pixels
        %%% the larger the block size need to be. The minimum block size is
        %%% about 50x50 pixels. The line below divides the image in 10x10
        %%% blocks, and makes sure that the block size is at least 50x50
        %%% pixels.
        BlockSize = max(50,min(round(m/10),round(n/10)));
        %%% Calculates a range of acceptable block sizes as plus-minus 10%
        %%% of the suggested block size.
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
        PaddedImageandCropMask = PaddedImage;
        if exist('BinaryCropMask','var')
            %%% Pad the crop mask too.
            PaddedCropMask = padarray(BinaryCropMask,[RowsToAddPre ColumnsToAddPre],'replicate','pre');
            PaddedCropMask = padarray(PaddedCropMask,[RowsToAddPost ColumnsToAddPost],'replicate','post');
            %%% For the CPblkproc function, the original image and the crop
            %%% mask image (if it exists) must be combined into one.
            PaddedImageandCropMask(:,:,2) = PaddedCropMask;
            %%% And the Block must have two layers, too.
            Block = [BestBlockSize BestBlockSize 2];
            %%% Sends the linear masked image to the appropriate
            %%% thresholding subfunction, in blocks.
            eval(['Threshold = CPblkproc(PaddedImageandCropMask,Block,@',ThresholdMethod,',handles,ImageName,pObject);']);
            %%% This evaluates to something like: Threshold =
            %%% CPblkproc(PaddedImageandCropMask,Block,@Otsu,handles,ImageN
            %%% ame);
        else
            %%% If there is no crop mask, then we can simply use the
            %%% blkproc function rather than CPblkproc.
            Block = [BestBlockSize BestBlockSize];
            eval(['Threshold = blkproc(PaddedImageandCropMask,Block,@',ThresholdMethod,',handles,ImageName,pObject);']);
        end

        %%% Resizes the block-produced image to be the size of the padded
        %%% image. Bilinear prevents dipping below zero. The crop the image
        %%% get rid of the padding, to make the result the same size as the
        %%% original image.
        Threshold = imresize(Threshold, size(PaddedImage), 'bilinear');
        Threshold = Threshold(RowsToAddPre+1:end-RowsToAddPost,ColumnsToAddPre+1:end-ColumnsToAddPost);
        
    elseif MethodFlag == 2 %%% The PerObject method.
        %%% This method require the Retrieved CropMask, which should be a
        %%% label matrix of objects, where each object consists of an
        %%% integer that is its label.
        if ~exist('RetrievedCropMask','var')
            error(['Image processing was canceled in the ',ModuleName,' module because you have chosen to calculate the threshold on a per-object basis, but CellProfiler could not find the image of the objects you want to use.'])
        end
        %%% Initializes the Threshold variable (which will end up being the
        %%% same size as the original image).
        Threshold = ones(size(OrigImage));
        NumberOfLabelsInLabelMatrix = max(RetrievedCropMask(:));
        for i = 1:NumberOfLabelsInLabelMatrix
            %%% Chooses out the pixels in the orig image that correspond
            %%% with i in the label matrix. This simultaneously produces a
            %%% linear set of numbers (and masking of pixels outside the
            %%% object is done automatically, in a sense).
            Intensities = OrigImage(RetrievedCropMask == i);
            
            
            
            
            
%             %%% Diagnostic:
%             PerObjectImage = zeros(size(OrigImage));
%             PerObjectImage(RetrievedCropMask == i) = OrigImage(RetrievedCropMask == i);
%             %figure(31)
%             %imagesc(PerObjectImage)
% 
%             %%% Removes Rows and Columns that are completely blank.
%             ColumnTotals = sum(PerObjectImage,1);
%             warning off all
%             ColumnsToDelete = ~logical(ColumnTotals);
%             warning on all
%             drawnow
%             CroppedImage = PerObjectImage;
%             CroppedImage(:,ColumnsToDelete,:) = [];
%             CroppedImagePlusRange = CroppedImage;
%             [rows,columns] = size(CroppedImage);
%             CroppedImagePlusRange(:,end+1) = 1;
% 
%             [ManualThreshold,bw] = CPthresh_tool(CroppedImagePlusRange,'gray',1);
%             ManualThreshold = log(ManualThreshold);
%             %%% Initializes the variables.
%             if ~exist('ManualThresholds','var')
%                 ManualThresholds = [];
%             end
%             ManualThresholds(end+1) = ManualThreshold;
%             save('Batch_32Manualdata','ManualThresholds');

            

            
            
            
            
            
            %%% Sends those pixels to the appropriate threshold
            %%% subfunctions.
            eval(['CalculatedThreshold = ',ThresholdMethod,'(Intensities,handles,ImageName,pObject);']);
            %%% This evaluates to something like: Threshold =
            %%% Otsu(Intensities,handles,ImageName,pObject);

            

            %%% Sets the pixels corresponding to object i to equal the
            %%% calculated threshold.
            Threshold(RetrievedCropMask == i) = CalculatedThreshold;
%            figure(32), imagesc(Threshold), colormap('gray')
        end
    end

    if MethodFlag == 1 || MethodFlag == 2 %%% For the Adaptive and the PerObject methods.
        %%% Adjusts any of the threshold values that are significantly
        %%% lower or higher than the global threshold.  Thus, if there are
        %%% no objects within a block (e.g. if cells are very sparse), an
        %%% unreasonable threshold will be overridden.
        Threshold(Threshold <= 0.7*GlobalThreshold) = 0.7*GlobalThreshold;
        Threshold(Threshold >= 1.5*GlobalThreshold) = 1.5*GlobalThreshold;
    end

elseif strcmp(Threshold,'All')
    if handles.Current.SetBeingAnalyzed == 1
        try
            %%% Notifies the user that the first image set will take much
            %%% longer than subsequent sets. Obtains the screen size.
            ScreenSize = get(0,'ScreenSize');
            ScreenHeight = ScreenSize(4);
            PotentialBottom = [0, (ScreenHeight-720)];
            BottomOfMsgBox = max(PotentialBottom);
            h = CPmsgbox('Preliminary calculations are under way for the Identify Primary Threshold module.  Subsequent image sets will be processed much more quickly than the first image set.');
            OrigSize = get(h, 'Position');
            PositionMsgBox = [500 BottomOfMsgBox OrigSize(3) OrigSize(4)];
            set(h, 'Position', PositionMsgBox)
            drawnow
            
            %%% Retrieves the path where the images are stored from the
            %%% handles structure.
            fieldname = ['Pathname', ImageName];
            try Pathname = handles.Pipeline.(fieldname);
            catch error(['Image processing was canceled in the ', ModuleName, ' module because it must be run using images straight from a load images module (i.e. the images cannot have been altered by other image processing modules). This is because you have asked the Identify Primary Threshold module to calculate a threshold based on all of the images before identifying objects within each individual image as CellProfiler cycles through them. One solution is to process the entire batch of images using the image analysis modules preceding this module and save the resulting images to the hard drive, then start a new stage of processing from this Identify Primary Threshold module onward.'])
            end
            %%% Retrieves the list of filenames where the images are stored
            %%% from the handles structure.
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
            % Saves the warning state and disable warnings to prevent
            % divide-by-zero warnings.
            State = warning;
            warning off Matlab:DivideByZero
            SigmaBSquared = (Mu_t * Omega - Mu).^2 ./ (Omega .* (1 - Omega));
            % Restores the warning state.
            warning(State);
            % Finds the location of the maximum value of sigma_b_squared.
            % The maximum may extend over several bins, so average together
            % the locations.  If maxval is NaN, meaning that
            % sigma_b_squared is all NaN, then return 0.
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
elseif strcmp(Threshold,'Set interactively')
    fieldname = ['Threshold',ImageName];
    if handles.Current.SetBeingAnalyzed == 1
        Threshold = CPthresh_tool(OrigImage(:,:,1));
        handles.Pipeline.(fieldname) = Threshold;
    else
        Threshold = handles.Pipeline.(fieldname);
    end
else
    %%% If the threshold is a number, it means that it was manually entered
    %%% by the user, or that we calculated it in the binary crop image
    %%% section above. Checks that the Threshold parameter has a valid
    %%% value
    if strcmp(class(Threshold),'char')
        Threshold = str2double(Threshold);
    end
    if isnan(Threshold) || Threshold > 1 || Threshold < 0 %#ok Ignore MLint
        error(['The threshold entered in the ', ModuleName, ' module is not a number, or is outside the acceptable range of 0 to 1.'])
    end
end
%%% Correct the threshold using the correction factor given by the user and
%%% make sure that the threshold is not larger than the minimum threshold
Threshold = ThresholdCorrection*Threshold;
Threshold = max(Threshold,MinimumThreshold);
Threshold = min(Threshold,MaximumThreshold);
handles = CPaddmeasurements(handles,'Image','OrigThreshold',[ObjectVar,ImageName],mean(mean(Threshold)));

if (nargout >= 3),
    if ~ exist('BinaryCropMask', 'var')
        varargout(1) = {WeightedVariance(OrigImage, true(size(OrigImage)), Threshold)};
    else
        varargout(1) = {WeightedVariance(OrigImage, BinaryCropMask~=0, Threshold)};
    end
end
if (nargout >= 4),
    if ~ exist('BinaryCropMask', 'var')
        varargout(2) = {SumOfEntropies(OrigImage, true(size(OrigImage)), Threshold)};
    else
        varargout(2) = {SumOfEntropies(OrigImage, BinaryCropMask~=0, Threshold)};
    end
end




%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%

function level = Otsu(im,handles,ImageName,pObject)
%%% This is the Otsu method of thresholding, adapted from MATLAB's
%%% graythresh function. Our modifications work in log space, and take into
%%% account the max and min values in the image.

%%% The following is needed for the adaptive cases where there the image
%%% has been cropped. This must be done within this subfunction, rather
%%% than in the main code prior to sending to this function via blkproc,
%%% because the blkproc function takes a single image as input, so we have
%%% to store the image and its cropmask in a single image variable.
if ndims(im) == 3
    Image = im(:,:,1);
    CropMask = im(:,:,2);
    clear im
    im = Image(CropMask==1);
else im = im(:);
end

if max(im) == min(im)
    level = im(1);
elseif isempty(im)
    %%% im will be empty if the entire image is cropped away by the
    %%% CropMask. I am not sure whether it is better to then set the level
    %%% to 0 or 1. Setting the level to empty causes problems downstream.
    %%% Presumably setting the level to 1 will not cause major problems
    %%% because the other blocks will average it out as we get closer to
    %%% real objects?
    level = 1;
else
    %%% We want to limit the dynamic range of the image to 256. Otherwise,
    %%% an image with almost all values near zero can give a bad result.
    minval = max(im)/256;
    im(im < minval) = minval;
    im = log(im);
    minval = min (im);
    maxval = max (im);
    im = (im - minval) / (maxval - minval);
    level = exp(minval + (maxval - minval) * graythresh(im));
end

% %%% For debugging:
% data = TrimmedImage;
% figure(30)
% subplot(1,2,1)
% hist(data(:),100);
% title(['trimmed data; Mean = ',num2str(Mean),'; StDev = ',num2str(StDev)])
% data = im;
% subplot(1,2,2)
% [Contents,BinLocations] = hist(data(:),100);
% hist(data(:),100);
% title(['Thresh = ',num2str(level),'; log data'])
% hold on
% plot([level;level],[0,max(Contents)])
% hold off
% figure(30)

function level = MoG(im,handles,ImageName,pObject)
%%% Stands for Mixture of Gaussians. This function finds a suitable
%%% threshold for the input image Block. It assumes that the pixels in the
%%% image belong to either a background class or an object class. 'pObject'
%%% is an initial guess of the prior probability of an object pixel, or
%%% equivalently, the fraction of the image that is covered by objects.
%%% Essentially, there are two steps. First, a number of Gaussian
%%% distributions are estimated to match the distribution of pixel
%%% intensities in OrigImage. Currently 3 Gaussian distributions are
%%% fitted, one corresponding to a background class, one corresponding to
%%% an object class, and one distribution for an intermediate class. The
%%% distributions are fitted using the Expectation-Maximization (EM)
%%% algorithm, a procedure referred to as Mixture of Gaussians modeling.
%%% When the 3 Gaussian distributions have been fitted, it's decided
%%% whether the intermediate class models background pixels or object
%%% pixels based on the probability of an object pixel 'pObject' given by
%%% the user.

%%% The following is needed for the adaptive cases where there the image
%%% has been cropped. This must be done within this subfunction, rather
%%% than in the main code prior to sending to this function via blkproc,
%%% because the blkproc function takes a single image as input, so we have
%%% to store the image and its cropmask in a single image variable.
if ndims(im) == 3
    Image = im(:,:,1);
    CropMask = im(:,:,2);
    clear im
    im = Image(CropMask==1);
else im = im(:);
end

if max(im) == min(im)
    level = im(1);
elseif isempty(im)
    %%% im will be empty if the entire image is cropped away by the
    %%% CropMask. I am not sure whether it is better to then set the level
    %%% to 0 or 1. Setting the level to empty causes problems downstream.
    %%% Presumably setting the level to 1 will not cause major problems
    %%% because the other blocks will average it out as we get closer to
    %%% real objects?
    level = 1;
else

    %%% The number of classes is set to 3
    NumberOfClasses = 3;

    %%% If the image is larger than 512x512, select a subset of 512^2
    %%% pixels for speed. This should be enough to capture the statistics
    %%% in the image.
    % im = im(:);
    if length(im) > 512^2
        rand('seed',0);
        indexes = randperm(length(im));
        im = im(indexes(1:512^2));
    end

    %%% Convert user-specified percentage of image covered by objects to a
    %%% prior probability of a pixel being part of an object.
    
    %%% Since the default list for MoG thresholding contains a % sign, we
    %%% need to remove the percent sign and use only the number to
    %%% calculate the threshold. If the pObject does not contain a % sign,
    %%% it will continue.  
    %%% pObject is important, but pObjectNew is only used in  2 lines of code.blah
    if regexp(pObject, '%')
        pObjectNew = regexprep(pObject, '%', '');
        pObject = (str2double(pObjectNew)/100);
    
    else
        pObject = str2double(pObject);
    end
    %pObject = str2double(pObject(1:2))/100; old code--need to remove
    %%% Get the probability for a background pixel
    pBackground = 1 - pObject;

    %%% Initialize mean and standard deviations of the three Gaussian
    %%% distributions by looking at the pixel intensities in the original
    %%% image and by considering the percentage of the image that is
    %%% covered by object pixels. Class 1 is the background class and Class
    %%% 3 is the object class. Class 2 is an intermediate class and we will
    %%% decide later if it encodes background or object pixels. Also, for
    %%% robustness the we remove 1% of the smallest and highest intensities
    %%% in case there are any quantization effects that have resulted in
    %%% unnaturally many 0:s or 1:s in the image.
    im = sort(im);
    im = im(ceil(length(im)*0.01):round(length(im)*0.99));
    ClassMean(1) = im(round(length(im)*pBackground/2));                      %%% Initialize background class
    ClassMean(3) = im(round(length(im)*(1 - pObject/2)));                    %%% Initialize object class
    ClassMean(2) = (ClassMean(1) + ClassMean(3))/2;                                            %%% Initialize intermediate class
    %%% Initialize standard deviations of the Gaussians. They should be the
    %%% same to avoid problems.
    ClassStd(1:3) = 0.15;
    %%% Initialize prior probabilities of a pixel belonging to each class.
    %%% The intermediate class is gets some probability from the background
    %%% and object classes.
    pClass(1) = 3/4*pBackground;
    pClass(2) = 1/4*pBackground + 1/4*pObject;
    pClass(3) = 3/4*pObject;

    %%% Apply transformation.  a < x < b, transform to log((x-a)/(b-x)).
    %a = - 0.000001; b = 1.000001; im = log((im-a)./(b-im)); ClassMean =
    %log((ClassMean-a)./(b - ClassMean)) ClassStd(1:3) = [1 1 1];

    %%% Expectation-Maximization algorithm for fitting the three Gaussian
    %%% distributions/classes to the data. Note, the code below is general
    %%% and works for any number of classes. Iterate until parameters don't
    %%% change anymore.
    delta = 1;
    while delta > 0.001
        %%% Store old parameter values to monitor change
        oldClassMean = ClassMean;

        %%% Update probabilities of a pixel belonging to the background or
        %%% object1 or object2
        for k = 1:NumberOfClasses
            pPixelClass(:,k) = pClass(k)* 1/sqrt(2*pi*ClassStd(k)^2) * exp(-(im - ClassMean(k)).^2/(2*ClassStd(k)^2));
        end
        pPixelClass = pPixelClass ./ repmat(sum(pPixelClass,2) + eps,[1 NumberOfClasses]);

        %%% Update parameters in Gaussian distributions
        for k = 1:NumberOfClasses
            pClass(k) = mean(pPixelClass(:,k));
            ClassMean(k) = sum(pPixelClass(:,k).*im)/(length(im)*pClass(k));
            ClassStd(k)  = sqrt(sum(pPixelClass(:,k).*(im - ClassMean(k)).^2)/(length(im)*pClass(k))) + sqrt(eps);    % Add sqrt(eps) to avoid division by zero
        end

        %%% Calculate change
        delta = sum(abs(ClassMean - oldClassMean));
    end

    %%% Now the Gaussian distributions are fitted and we can describe the
    %%% histogram of the pixel intensities as the sum of these Gaussian
    %%% distributions. To find a threshold we first have to decide if the
    %%% intermediate class 2 encodes background or object pixels. This is
    %%% done by choosing the combination of class probabilities 'pClass'
    %%% that best matches the user input 'pObject'.
    level = linspace(ClassMean(1),ClassMean(3),10000);
    Class1Gaussian = pClass(1) * 1/sqrt(2*pi*ClassStd(1)^2) * exp(-(level - ClassMean(1)).^2/(2*ClassStd(1)^2));
    Class2Gaussian = pClass(2) * 1/sqrt(2*pi*ClassStd(2)^2) * exp(-(level - ClassMean(2)).^2/(2*ClassStd(2)^2));
    Class3Gaussian = pClass(3) * 1/sqrt(2*pi*ClassStd(3)^2) * exp(-(level - ClassMean(3)).^2/(2*ClassStd(3)^2));
    if abs(pClass(2) + pClass(3) - pObject) < abs(pClass(3) - pObject)
        %%% Intermediate class 2 encodes object pixels
        BackgroundDistribution = Class1Gaussian;
        ObjectDistribution = Class2Gaussian + Class3Gaussian;
    else
        %%% Intermediate class 2 encodes background pixels
        BackgroundDistribution = Class1Gaussian + Class2Gaussian;
        ObjectDistribution = Class3Gaussian;
    end

    %%% Now, find the threshold at the intersection of the background
    %%% distribution and the object distribution.
    [ignore,index] = min(abs(BackgroundDistribution - ObjectDistribution)); %#ok Ignore MLint
    level = level(index);
end

function level = Background(im,handles,ImageName,pObject)
%%% The threshold is calculated by calculating the mode and multiplying by
%%% 2 (an arbitrary empirical factor). The user will presumably adjust the
%%% multiplication factor as needed.

%%% The following is needed for the adaptive cases where there the image
%%% has been cropped. This must be done within this subfunction, rather
%%% than in the main code prior to sending to this function via blkproc,
%%% because the blkproc function takes a single image as input, so we have
%%% to store the image and its cropmask in a single image variable.
if ndims(im) == 3
    Image = im(:,:,1);
    CropMask = im(:,:,2);
    clear im
    im = Image(CropMask==1);
else im = im(:);
end

if max(im) == min(im)
    level = im(1);
elseif isempty(im)
    %%% im will be empty if the entire image is cropped away by the
    %%% CropMask. I am not sure whether it is better to then set the level
    %%% to 0 or 1. Setting the level to empty causes problems downstream.
    %%% Presumably setting the level to 1 will not cause major problems
    %%% because the other blocks will average it out as we get closer to
    %%% real objects?
    level = 1;
else
    level = 2*mode(im(:));
end


function level = RobustBackground(im,handles,ImageName,pObject)
%%% The threshold is calculated by trimming the top and bottom 25% of
%%% pixels off the image, then calculating the mean and standard deviation
%%% of the remaining image. The threshold is then set at 2 (empirical
%%% value) standard deviations above the mean. 

%%% The following is needed for the adaptive cases where there the image
%%% has been cropped. This must be done within this subfunction, rather
%%% than in the main code prior to sending to this function via blkproc,
%%% because the blkproc function takes a single image as input, so we have
%%% to store the image and its cropmask in a single image variable.
if ndims(im) == 3
    Image = im(:,:,1);
    CropMask = im(:,:,2);
    clear im
    im = Image(CropMask==1);
else im = im(:);
end

if max(im) == min(im)
    level = im(1);
elseif isempty(im)
    %%% im will be empty if the entire image is cropped away by the
    %%% CropMask. I am not sure whether it is better to then set the level
    %%% to 0 or 1. Setting the level to empty causes problems downstream.
    %%% Presumably setting the level to 1 will not cause major problems
    %%% because the other blocks will average it out as we get closer to
    %%% real objects?
    level = 1;
else
    %%% First, the image's pixels are sorted from low to high.
    im = sort(im);
    %%% The index of the 5th percentile is calculated, with a minimum of 1.
    LowIndex = max(1,round(.05*length(im)));
    %%% The index of the 95th percentile is calculated, with a maximum of the
    %%% number of pixels in the whole image.
    HighIndex = min(length(im),round(.95*length(im)));
    TrimmedImage = im(LowIndex: HighIndex);
    Mean = mean(TrimmedImage);
    StDev = std(TrimmedImage);
    level = Mean + 2*StDev;
end

% %%% DEBUGGING
% Logim = log(sort(im(im~=0)));
% 
% %%% For debugging:
% figure(30)
% subplot(1,2,1)
% hist(Logim,100);
% title(['Log data; Mean = ',num2str(Mean),'; StDev = ',num2str(StDev)])
% pause(0.1)

% %%% For debugging:
% data = TrimmedImage;
% figure(30)
% subplot(1,2,1)
% hist(data(:),100);
% title(['trimmed data; Mean = ',num2str(Mean),'; StDev = ',num2str(StDev)])
% data = im;
% subplot(1,2,2)
% [Contents,BinLocations] = hist(data(:),100);
% hist(data(:),100);
% title(['Thresh = ',num2str(level),'; raw data'])
% hold on
% plot([level;level],[0,max(Contents)])
% hold off
% 
% figure(30)
% 
% %%% More debugging:
% try
%     load('Batch_80Autodata');
% end
% %%% Initializes the variables.
% if ~exist('Means','var')
%    Means = []; 
%    StDevs = [];
%    Levels = [];
%    TrimmedImages = [];
%    Images = [];
% end
% Means(end+1) = Mean;
% StDevs(end+1) = StDev;
% Levels(end+1) = level;
% TrimmedImages{end+1} = {TrimmedImage};
% Images{end+1} = {im};
% save('Batch_80Autodata','Means','StDevs','Levels','TrimmedImages','Images');


function level = RidlerCalvard(im,handles,ImageName,pObject)

%%% The following is needed for the adaptive cases where there the image
%%% has been cropped. This must be done within this subfunction, rather
%%% than in the main code prior to sending to this function via blkproc,
%%% because the blkproc function takes a single image as input, so we have
%%% to store the image and its cropmask in a single image variable.
if ndims(im) == 3
    Image = im(:,:,1);
    CropMask = im(:,:,2);
    clear im
    im = Image(CropMask==1);
else im = im(:);
end

if max(im) == min(im)
    level = im(1);
elseif isempty(im)
    %%% im will be empty if the entire image is cropped away by the
    %%% CropMask. I am not sure whether it is better to then set the level
    %%% to 0 or 1. Setting the level to empty causes problems downstream.
    %%% Presumably setting the level to 1 will not cause major problems
    %%% because the other blocks will average it out as we get closer to
    %%% real objects?
    level = 1;
else
    %%% We want to limit the dynamic range of the image to 256. Otherwise,
    %%% an image with almost all values near zero can give a bad result.
    MinVal = max(im)/256;
    im(im<MinVal) = MinVal;
    im = log(im);
    MinVal = min(im);
    MaxVal = max(im);
    im = (im - MinVal)/(MaxVal - MinVal);
    PreThresh = 0;
    %%% This method needs an initial value to start iterating. Using
    %%% graythresh (Otsu's method) is probably not the best, because the
    %%% Ridler Calvard threshold ends up being too close to this one and in
    %%% most cases has the same exact value.
    NewThresh = graythresh(im);
    delta = 0.00001;
    while abs(PreThresh - NewThresh)>delta
        PreThresh = NewThresh;
        Mean1 = mean(im(im<PreThresh));
        Mean2 = mean(im(im>=PreThresh));
        NewThresh = mean([Mean1,Mean2]);
    end
    level = exp(MinVal + (MaxVal-MinVal)*NewThresh);
end


function level = Kapur(im,handles,ImageName,pObject)
%%% This is the Kapur, Sahoo, & Wong method of thresholding, adapted to log-space.

%%% The following is needed for the adaptive cases where there the image
%%% has been cropped. This must be done within this subfunction, rather
%%% than in the main code prior to sending to this function via blkproc,
%%% because the blkproc function takes a single image as input, so we have
%%% to store the image and its cropmask in a single image variable.
if ndims(im) == 3
    Image = im(:,:,1);
    CropMask = im(:,:,2);
    clear im
    im = Image(CropMask==1);
else im = im(:);
end

if max(im) == min(im)
    level = im(1);
elseif isempty(im)
    %%% im will be empty if the entire image is cropped away by the
    %%% CropMask. I am not sure whether it is better to then set the level
    %%% to 0 or 1. Setting the level to empty causes problems downstream.
    %%% Presumably setting the level to 1 will not cause major problems
    %%% because the other blocks will average it out as we get closer to
    %%% real objects?
    level = 1;
else
    level = Threshold_Kapur(im, 8);
end


%%% This function computes the threshold of an image by
%%% log-transforming its values, then searching for the threshold that
%%% maximizes the sum of entropies of the foreground and background
%%% pixel values, when treated as separate distributions.
function thresh = Threshold_Kapur(Image, bits)
% Find the smoothed log histogram.
[N, X] = hist(log2(smooth_log_histogram(Image(:), bits)), 256);

% drop any zero bins
drop = (N == 0);
N(drop) = [];
X(drop) = [];

% check for corner cases
if length(X) == 1,
    thresh = X(1);
    return;
end

% Normalize to probabilities
P = N / sum(N);

% Find the probabilities totals up to and above each possible threshold.
loSum = cumsum(P);
hiSum = loSum(end) - loSum;
loE = cumsum(P .* log2(P));
hiE = loE(end) - loE;

% compute the entropies
s = warning('off', 'MATLAB:divideByZero');
loEntropy = loE ./ loSum - log2(loSum);
hiEntropy = hiE ./ hiSum - log2(hiSum);
warning(s);

sumEntropy = loEntropy(1:end-1) + hiEntropy(1:end-1);
sumEntropy(~ isfinite(sumEntropy)) = Inf;
entry = min(find(sumEntropy == min(sumEntropy)));
thresh = 2^((X(entry) + X(entry+1)) / 2);


%%% This function smooths a log-transformed histogram, using noise
%%% proportional to the histogram value.
function Q = smooth_log_histogram(R, bits)
%%% seed random state
state = randn('state');
randn('state', 0);
R(R == 0) = 1 / (2^bits);
Q = exp(log(R) + 0.5*randn(size(R)).*(-log(R)/log(2))/bits);
Q(Q > 1) = 1.0;
Q(Q < 0) = 0.0;
randn('state', state);

%%% Weighted variances of the foreground and background.
function  wv = WeightedVariance(Image, CropMask, Threshold)
if isempty(Image(CropMask)),
    wv = 0;
    return;
end

%%% clamp dynamic range
minval = max(Image(CropMask))/256;
if minval == 0.0,
    wv = 0;
    return;
end
Image(Image < minval) = minval;

%%% Compute the weighted variance
FG = log2(Image((Image >= Threshold) & CropMask));
BG = log2(Image((Image < Threshold) & CropMask));
if isempty(FG),
    wv = var(BG);
elseif isempty(BG);
    wv = var(FG);
else
    wv = (length(FG) * var(FG) + length(BG) * var(BG)) / (length(FG) + length(BG));
end



%%% Sum of entropies of foreground and background as separate distributions.
function  soe = SumOfEntropies(Image, CropMask, Threshold)
if isempty(Image(CropMask)),
    wv = 0;
    return;
end

%%% clamp dynamic range
minval = max(Image(CropMask))/256;
if minval == 0.0,
    wv = 0;
    return;
end
Image(Image < minval) = minval;

%%% Smooth the histogram
Image = smooth_log_histogram(Image, 8);

%%% Find bin locations
[N, X] = hist(log2(Image(CropMask)), 256);

%%% Find counts for FG and BG
FG = Image((Image >= Threshold) & CropMask);
BG = Image((Image < Threshold) & CropMask);
NFG = hist(log2(FG), X);
NBG = hist(log2(BG), X);

%%% drop empty bins
NFG = NFG(NFG > 0);
NBG = NBG(NBG > 0);

if isempty(NFG)
    NFG = [1];
end

if isempty(NBG)
    NBG = [1];
end

% normalize
NFG = NFG / sum(NFG);
NBG = NBG / sum(NBG);

%%% compute sum of entropies
soe = sum(NFG .* log2(NFG)) + sum(NBG .* log2(NBG));
