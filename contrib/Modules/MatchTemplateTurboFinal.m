function handles = MatchTemplateTurbo(handles)

% Help for the Match Template module:
% Category: Object Processing
% Lord of the Rings

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the greyscale images you want to measure?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = How do you want to call the identified objects (matched templates)?
%defaultVAR02 = Ring
%infotypeVAR02 = objectgroup indep
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What shape do you want to use for the outer figure?
%choiceVAR03 = disk
%choiceVAR03 = diamond
%choiceVAR03 = square
%choiceVAR03 = octagon
OuterFigureShape = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu custom

%textVAR04 = Outer diameter of the ring
%defaultVAR04 = 20
OuterDiameter = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,4})); %#ok Ignore MLint

%textVAR05 = What shape do you want to use for the inner figure?
%choiceVAR05 = disk
%choiceVAR05 = diamond
%choiceVAR05 = square
%choiceVAR05 = octagon
%choiceVAR05 = none
InnerFigureShape = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu custom

%textVAR06 = Inner diameter of the ring
%defaultVAR06 = 10
InnerDiameter = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,6})); %#ok Ignore MLint

%textVAR07 = Absolute difference for accepting match, 0 ... 1
%defaultVAR07 = 0.03
ThresholdDifference = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,7})); %#ok Ignore MLint

%textVAR08 = Z-score (Std of cell) for accepting match, 0 ... inf
%defaultVAR08 = 1
ThresholdZScore = str2num(char(handles.Settings.VariableValues{CurrentModuleNum, 08})); %#ok Ignore MLint

%textVAR09 = Correlation value for accepting match, 0 ... inf
%defaultVAR09 = 0.1
ThresholdCorrelation = str2num(char(handles.Settings.VariableValues{CurrentModuleNum, 09})); %#ok Ignore MLint

%textVAR10 = Maxima Suppression Size for Identification of Templates
%defaultVAR10 = 40
MaximaSuppressionSize = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,10})); %#ok Ignore MLint

%textVAR11 = What area do you want to use for pixel extraction?
%defaultVAR11 = 80
AreaExtractionSize = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,11})); %#ok Ignore MLint

%textVAR12 = Do you want to use previously identified objects for overlap testing?
%choiceVAR12 = No
%choiceVAR12 = Yes
UseObjectForOverlap = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu

%textVAR13 = What objects do you want to use for overlap testing?
%choiceVAR13 = Cells
%infotypeVAR13 = objectgroup
OverlapObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,13});
%inputtypeVAR13 = popupmenu

%textVAR14 = Below which percent match do you want to exclude?
%defaultVAR14 = 80
OverlapPercentile = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,14})); %#ok Ignore MLint

%textVAR15 = What do you want to call the outlines of the matched templates (optional)?
%defaultVAR15 = Do not save
%infotypeVAR15 = outlinegroup indep
SaveOutlines = char(handles.Settings.VariableValues{CurrentModuleNum,15});

%%%VariableRevisionNumber = 1

%%% Set up the window for displaying the results
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber);
    CPfigure(handles,'Text',ThisModuleFigureNumber);
    columns = 1;
end

if ~(InnerDiameter < OuterDiameter)
    error(['Image processing was canceled in the ', ModuleName, ' module. Inner diameter must be smaller than outer diameter.'])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     CREATING THE MASKS                   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% constructing Ring-like mask, using InnerDiameter and OuterDiameter
% we start with the outer shape, rescale to the desired size.
OuterShape = getnhood(strel(OuterFigureShape, 48));
mask = imresize (OuterShape, [OuterDiameter OuterDiameter]);
% now we make a hole in the outer mask
if ~strcmp(InnerFigureShape, 'none')
    InnerShape = getnhood(strel(InnerFigureShape, 50));
    temp=zeros(OuterDiameter, OuterDiameter);
    temp(round(OuterDiameter/2)-round(InnerDiameter/2)+1: round(OuterDiameter/2)-round(InnerDiameter/2)+InnerDiameter, ...
        round(OuterDiameter/2)-round(InnerDiameter/2)+1: round(OuterDiameter/2)-round(InnerDiameter/2)+InnerDiameter)...
        =imresize(InnerShape, [InnerDiameter InnerDiameter]);
    mask = and(mask, not(temp));
end

% this mask marks the area outside the Ring, the "Cell" for future calculations of the Z-Score and the difference score  
CellMask = getnhood(strel('disk', 48));
CellMask = imresize (CellMask, [AreaExtractionSize, AreaExtractionSize]);
CellMask ( round(length(CellMask)/2)- round(length(mask)/2)+1 : round(length(CellMask)/2)- round(length(mask)/2) + length(mask),...
    round(length(CellMask)/2)- round(length(mask)/2)+1 : round(length(CellMask)/2)- round(length(mask)/2) + length (mask)) = not(mask);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%     PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable,
OrigImage = CPretrieveimage(handles,ImageName,ModuleName);
[YLength, XLength] = size(OrigImage);

% this is just copied from the CP -code, I don't know, whether we need this
%%% If the image is three dimensional (i.e. color), the three channels
%%% are added together in order to measure object intensity.
if ndims(OrigImage) ~= 2
     s = size(OrigImage);
     if (length(s) == 3 && s(3) == 3)
         OrigImage = OrigImage(:,:,1)+OrigImage(:,:,2)+OrigImage(:,:,3);
     else
         error(['Image processing was canceled in the ', ModuleName, ' module. There was a problem with the dimensions. The image must be grayscale or RGB color.'])
     end
end

%%% If user wants, retrieve the label matrix image that contains the segmented objects of previous module for overlap testing.
if strcmp (UseObjectForOverlap, 'Yes')
    LabelMatrixImage = CPretrieveimage(handles,['Segmented', OverlapObjectName],ModuleName,'MustBeGray','DontCheckScale');
    if any(size(OrigImage) ~= size(LabelMatrixImage))
        error(['Image processing was canceled in the ', ModuleName, ' module. The size of the image you want to measure is not the same as the size of the image from which the ',ObjectName,' objects were identified.'])
    end
end

% this is just copied from the CP -code, I don't know, whether we need this
%%% For the cases where the label matrix was produced from a cropped
%%% image, the sizes of the images will not be equal. So, we crop the
%%% LabelMatrix and try again to see if the matrices are then the
%%% proper size. Removes Rows and Columns that are completely blank.
if strcmp (UseObjectForOverlap, 'Yes')
    if any(size(OrigImage) < size(LabelMatrixImage))
        ColumnTotals = sum(LabelMatrixImage,1);
        RowTotals = sum(LabelMatrixImage,2)';
        warning off all
        ColumnsToDelete = ~logical(ColumnTotals);
        RowsToDelete = ~logical(RowTotals);
        warning on all
        drawnow
        CroppedLabelMatrix = LabelMatrixImage;
        CroppedLabelMatrix(:,ColumnsToDelete,:) = [];
        CroppedLabelMatrix(RowsToDelete,:,:) = [];
        clear LabelMatrixImage
        LabelMatrixImage = CroppedLabelMatrix;
        %%% In case the entire image has been cropped away, we store a single
        %%% zero pixel for the variable.
        if isempty(LabelMatrixImage)
            LabelMatrixImage = 0;
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CALCULATIONS                                  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% this image calls a subfunction which calculates the correlation of mask
% with the OrigImage in Fourier space
CorrelationImage      = StandardCorrelation(OrigImage, mask, 1);

% we extract local maxima from the crrelation image
[MaximaRow MaximaCol] = localMaximum(CorrelationImage, MaximaSuppressionSize, true);

% we need to work with an enlarged image. Some objects might be at the 
% border of the image and the CellMask would be out of the image
EnlFactor = max(AreaExtractionSize, OuterDiameter);
EnlObjectImage   = zeros (2*EnlFactor + YLength, 2*EnlFactor + XLength);

% problem region: % in this region CP spends most of its time. 
% we preallocate the variables for speed
PrimaryObjectCount = length(MaximaRow);
Difference      = zeros(1, PrimaryObjectCount);
ZScore          = zeros(1, PrimaryObjectCount);
Correlation     = zeros(1, PrimaryObjectCount);
Overlap         = zeros(1, PrimaryObjectCount);

% we convert to sparse binary objects for speed
ZeroImage = sparse(zeros(size(EnlObjectImage))>0);
mask = mask>0;
CellMask = CellMask>0;
if strcmp(UseObjectForOverlap, 'Yes')
    BinLabelMatrixImage = LabelMatrixImage > 0;
end

% we preassign for clarity and speed, needed in the loop
% low and high (L and H) coordinates for placement of the Mask
ML = EnlFactor - round(0.5*OuterDiameter) + 1;
MH = EnlFactor - round(0.5*OuterDiameter) + OuterDiameter;
% low and high (L and H) coordinates for placement of the CellMask
CL = EnlFactor - round(0.5*AreaExtractionSize) + 1;
CH = EnlFactor - round(0.5*AreaExtractionSize) + AreaExtractionSize;

% we loop over the objects and calculate the difference, the z-score and the percent overlap 
for f=1 : PrimaryObjectCount
    % we first test, whether the correlation value is above the threshold,
    % if not we can skip the other calculations
    Correlation(f)= CorrelationImage(MaximaRow(f), MaximaCol(f));
    if Correlation(f) > ThresholdCorrelation
        
        % we first generate an enlarged mask image,
        temp = ZeroImage;
        temp(ML + MaximaRow(f) : MH + MaximaRow(f), ML + MaximaCol(f) : MH + MaximaCol(f)) = mask;

        % these image later determines the final outlines of the objects
        EnlObjectImage(temp) = f;

        % now we reduce the size of the mask image to the size of the OrigImage
        MaskImage = temp(EnlFactor+1 : EnlFactor + YLength, EnlFactor+1 : EnlFactor + XLength);

        % and we do the same for the CellMask image (outside)
        temp = ZeroImage;
        temp(CL + MaximaRow(f) : CH + MaximaRow(f), CL + MaximaCol(f) : CH + MaximaCol(f)) = CellMask;
        CellMaskImage = temp(EnlFactor+1 : EnlFactor + YLength, EnlFactor+1 : EnlFactor + XLength);

        % we extract the pixels below the respective masks
        Inside  = OrigImage(MaskImage);
        Outside = OrigImage(CellMaskImage);

        % we calculate or extract the respective values
        Difference(f) = mean(Inside) - mean(Outside);
        StDev         = std(Outside);
        ZScore(f)     = Difference(f)/StDev;

        % if user wants we decide, whether center of the mask overlaps with area of the cell
        if strcmp(UseObjectForOverlap, 'Yes')
            if sum(BinLabelMatrixImage(MaskImage))/sum(sum(mask)) > OverlapPercentile/100;
                Overlap(f) = 1;
            else
                Overlap(f) = 0;
            end
        else
            Overlap(f) = 1;
        end
    end
end
% end of problem region. from now on fast and easy

% we have to resize the padded images
PrimaryObjectImage = EnlObjectImage(EnlFactor+1 : EnlFactor + YLength, EnlFactor+1 : EnlFactor + XLength);

% now we apply the thresholds defined above and save the excluded objects
DiffExcl   = Difference <= ThresholdDifference;
ZscoreExcl = ZScore <= ThresholdZScore;
CorrelExcl = Correlation <= ThresholdCorrelation;
OverlapExcl= ~Overlap;

% this index determines finally the usage
Usage = ~or(or(DiffExcl, ZscoreExcl), or(CorrelExcl, OverlapExcl));
% these are the final values, to be saved
Difference  = Difference(Usage);
ZScore      = ZScore(Usage);
Correlation = Correlation(Usage);
CenterX     = MaximaCol(Usage);
CenterY     = MaximaRow(Usage);

% we calculate the final segmented image
FinalSegmentedImage = zeros(size(OrigImage));
ObjectCount = 0;
for f=1 : PrimaryObjectCount
    if Usage(f)
        ObjectCount = ObjectCount + 1;
        FinalSegmentedImage(PrimaryObjectImage == f) = ObjectCount;
    end
end
FinalOutline = bwperim (FinalSegmentedImage); 

% for some later modules of CP we need the following images, even though
% they do not make sense in the context of the MatchTemplate module
FinalLabelMatrixImage = FinalSegmentedImage;
UneditedLabelMatrixImage = FinalSegmentedImage;
SmallRemovedLabelMatrixImage = FinalSegmentedImage;

%%% Saves the segmented image, not edited for objects along the edges or
%%% for size, to the handles structure.
fieldname = ['UneditedSegmented',ObjectName];
handles.Pipeline.(fieldname) = UneditedLabelMatrixImage;

%%% Saves the segmented image, only edited for small objects, to the
%%% handles structure.
fieldname = ['SmallRemovedSegmented',ObjectName];
handles.Pipeline.(fieldname) = SmallRemovedLabelMatrixImage;

%%% Saves the final segmented label matrix image to the handles structure.
fieldname = ['Segmented',ObjectName];
handles.Pipeline.(fieldname) = FinalLabelMatrixImage;

%%% Saves images to the handles structure so they can be saved to the hard
%%% drive, if the user requested.
if ~strcmpi(SaveOutlines,'Do not save')
    try handles.Pipeline.(SaveOutlines) = FinalOutline;
    catch error(['The object outlines could not be calculated by the ', ModuleName, ' module, so these images were not saved to the handles structure. The Save Images module will therefore not function on these images. This is just for your information - image processing is still in progress, but the Save Images module will fail if you attempted to save these images.'])
    end
end

%%% Saves the ObjectCount, i.e., the number of matched objects.
if ~isfield(handles.Measurements.Image,'ObjectCountFeatures')
    handles.Measurements.Image.ObjectCountFeatures = {};
    handles.Measurements.Image.ObjectCount = {};
end
column = find(~cellfun('isempty',strfind(handles.Measurements.Image.ObjectCountFeatures,ObjectName)));
if isempty(column)
    handles.Measurements.Image.ObjectCountFeatures(end+1) = {ObjectName};
    column = length(handles.Measurements.Image.ObjectCountFeatures);
end
handles.Measurements.Image.ObjectCount{handles.Current.SetBeingAnalyzed}(1,column) = ObjectCount;
%%% Saves the location of each segmented object
handles.Measurements.(ObjectName).LocationFeatures = {'CenterX','CenterY'};
handles.Measurements.(ObjectName).Location(handles.Current.SetBeingAnalyzed) = {[CenterX CenterY]};

%%% Save measurements
% read out features and save to handles structure
BasicFeatures    = {...
   'Z-Score',...
   'Difference'...
   'Correlation',...
   'InnerDiameter',...
   'OuterDiameter'};

if isequal(ObjectCount, 0)
    Features = zeros(1, length(BasicFeatures));
else
    Features (:, 1) = ZScore;
    Features (:, 2) = Difference;
    Features (:, 3) = Correlation;
    Features (:, 4) = InnerDiameter;
    Features (:, 5) = OuterDiameter;
end
handles.Measurements.(ObjectName).(['MatchTempl_',ImageName,'Features']) = BasicFeatures;
handles.Measurements.(ObjectName).(['MatchTempl_',ImageName])(handles.Current.SetBeingAnalyzed) = {Features};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY                   %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber);

    CPfigure(handles,'Image',ThisModuleFigureNumber);
    % ORIGINAL IMAGE
    subplot (2,3,1);
    CPimagesc(OrigImage, handles); title ('OrigImage')
    
    % CORRELATTION IMAGE
    subplot (2,3,2);
    imagesc(CorrelationImage); colormap ('jet'), colorbar ('location', 'southoutside'), title ('Correlation Image')
    
    % MAXIMA IMAGE
    subplot (2,3,3);
    MaxImage = max(OrigImage(:));
    OutlineOrigImage = OrigImage;
    OutlineImage = FinalOutline > 0;
    OutlineOrigImage(OutlineImage)=MaxImage;
    CPimagesc(OutlineOrigImage, handles); title('Outline Overlay')

    % MAXIMA OVERLAY
    Point = getnhood (strel('disk', 4));
    LengthPoint = length(Point);
    MaskPointImage   = zeros (2*EnlFactor + YLength, 2*EnlFactor + XLength);
    % we loop over the primary objects and make the overlay with the points
    for f=1:PrimaryObjectCount
        MaskPointImage((MaximaRow(f) + EnlFactor - round(0.5*LengthPoint) + 1) : (MaximaRow(f) + EnlFactor - round(0.5*LengthPoint) + LengthPoint), ...
        (MaximaCol(f) + EnlFactor - round(0.5*LengthPoint) + 1) : (MaximaCol(f) + EnlFactor - round(0.5*LengthPoint) + LengthPoint)) = Point;
    end
    MaskPointImage     = MaskPointImage(EnlFactor+1 : EnlFactor + YLength, EnlFactor+1 : EnlFactor + XLength);
    MaskPointImage = MaskPointImage > 0;
    subplot (2,3,4);
    MaxImage = max(OrigImage(:));
    PointOrigImage = OrigImage;
    PointOrigImage(MaskPointImage)=MaxImage;
    CPimagesc(PointOrigImage, handles); title('Maxima Overlay')
    
    % EXCLUDED OBJECTS
    tmp = OrigImage/max(OrigImage(:));
    CorrelExclImage  = zeros(size(OrigImage));
    DiffExclImage    = zeros(size(OrigImage));
    ZscoreExclImage  = zeros(size(OrigImage));
    OverlapExclImage = zeros(size(OrigImage));
    
    % we extract the respective objects out of the PrimaryObjectImage
    for f=1:PrimaryObjectCount
        if  CorrelExcl(f)
            CorrelExclImage(PrimaryObjectImage==f) = f;
        elseif DiffExcl(f)
            DiffExclImage(PrimaryObjectImage==f) = f;
        elseif ZscoreExcl(f)
            ZscoreExclImage(PrimaryObjectImage==f) = f;
        elseif OverlapExcl(f)
            OverlapExclImage(PrimaryObjectImage==f) = f;
        end
    end
    
    % we calculate the outlines
    PerimCorrExcl = bwperim (CorrelExclImage);
    PerimDiffExcl = bwperim (DiffExclImage);
    PerimZScExcl  = bwperim (ZscoreExclImage);
    PerimOverlExcl = bwperim (OverlapExclImage);
    
    OutlinedObjectsR = tmp;
    OutlinedObjectsG = tmp;
    OutlinedObjectsB = tmp;
    
    % correlation excluded in red
    OutlinedObjectsR(PerimCorrExcl)  = 1; OutlinedObjectsG(PerimCorrExcl) = 0; OutlinedObjectsB(PerimCorrExcl) = 0;
    % difference excluded in blue
    OutlinedObjectsR(PerimDiffExcl)  = 0; OutlinedObjectsG(PerimDiffExcl) = 0; OutlinedObjectsB(PerimDiffExcl) = 1;
    % zscore excluded in green
    OutlinedObjectsR(PerimZScExcl)   = 0; OutlinedObjectsG(PerimZScExcl)  = 1; OutlinedObjectsB(PerimZScExcl)  = 0;
    % perimeter excluded in ...
    OutlinedObjectsR(PerimOverlExcl) = 1; OutlinedObjectsG(PerimOverlExcl)= 1; OutlinedObjectsB(PerimOverlExcl)= 0;
    % the good objects in white!!
    OutlinedObjectsR(FinalOutline)   = 1; OutlinedObjectsG(FinalOutline)  = 1; OutlinedObjectsB(FinalOutline)  = 1;
    
    subplot (2,3,5);
    OutlinedObjects = cat(3,OutlinedObjectsR,OutlinedObjectsG,OutlinedObjectsB);
    CPimagesc(OutlinedObjects, handles); title('Excluded objects')    
    
    % FINAL OBJECTS
    subplot (2,3,6);
    imagesc (FinalSegmentedImage);
    title ('Final Objects');
    colormap ('jet')
end
return

function ima = StandardCorrelation(ima1,ima2, avgI) %#ok<INUSD>
% this function calculates the correlation between two images.
% courtesy of Peter Horvarth, LMC, ETHZ

% Reads in both images

ima1= double(ima1); avg1 = mean(mean(ima1)); ima1= ima1-avg1;
ima2= double(ima2); avg2 = mean(mean(ima2)); ima2= ima2-avg1;

s1 = size(ima1);
s2 = size(ima2);

% Find the max size of the images
s = max(s1,s2);

% Allocates it to the output image
sx = s(1);
sy = s(2);

% Copy original images into bigger images
image1 = zeros(sx,sy);
image1(sx/2-s1(1)/2+1:sx/2+s1(1)/2,sy/2+1-s1(2)/2:sy/2+s1(2)/2) = ima1;
image2 = zeros(sx,sy);
image2(sx/2-s2(1)/2+1:sx/2+s2(1)/2,sy/2+1-s2(2)/2:sy/2+s2(2)/2) = ima2;

% Calculates FFT of bith images
f1 = fft2(fftshift(image1));

f2 = fft2(fftshift(image2));

% Calculates correlation
corr = fftshift(ifft2(f1.*conj(f2)));

% Normalises it with autocorrelation
corr = corr ./ max(max(ifft2(f2.*conj(f2))));
ima = abs(corr);
%imagesc(ima);

function varargout = localMaximum(x,minDist, exculdeEqualPoints)
% function varargout = localMaximum(x,minDist, exculdeEqualPoints)
%
% This function returns the indexes\subscripts of local maximum in the data x.
% x can be a vector or a matrix of any dimension
%
% minDist is the minimum distance between two peaks (local maxima)
% minDist should be a vector in which each argument corresponds to it's
% relevant dimension OR a number which is the minimum distance for all
% dimensions
%
% exculdeEqualPoints - is a boolean definning either to recognize points with the same value as peaks or not
% x = [1     2     3     4     4     4     4     4     4     3     3     3     2     1];
%  will the program return all the '4' as peaks or not -  defined by the 'exculdeEqualPoints'
% localMaximum(x,3)
% ans =
%      4     5     6     7     8     9    11    12
%
%  localMaximum(x,3,true)
% ans =
%      4     7    12
%
%
% Example:
% a = randn(100,30,10);
% minDist = [10 3 5];
% peaks = localMaximum(a,minDist);
%
% To recieve the subscript instead of the index use:
% [xIn yIn zIn] = localMaximum(a,minDist);
%
% To find local minimum call the function with minus the variable:
% valleys = localMaximum(-a,minDist);

if nargin < 3
    exculdeEqualPoints = false;
    if nargin < 2
        minDist = size(x)/10;
    end
end

if isempty(minDist)
    minDist = size(x)/10;
end

dimX = length ( size(x) );
if length(minDist) ~= dimX
    % In case minimum distance isn't defined for all of x dimensions
    % I use the first value as the default for all of the dimensions
    minDist = minDist( ones(dimX,1) );
end

% validity checks
minDist = ceil(minDist);
minDist = max( [minDist(:)' ; ones(1,length(minDist))] );
minDist = min( [minDist ; size(x)] );

% ---------------------------------------------------------------------
if exculdeEqualPoints
    % this section comes to solve the problem of a plato
    % without this code, points with the same hight will be recognized as peaks
    y = sort(x(:));
    dY = diff(y);
    % finding the minimum step in the data
    minimumDiff = min( dY(dY ~= 0) );
    %adding noise which won't affect the peaks
    x = x + rand(size(x))*minimumDiff;
end
% ---------------------------------------------------------------------


se = ones(minDist);
X = imdilate(x,se);
f = find(x == X);


if nargout
    [varargout{1:nargout}] = ind2sub( size(x), f );
else
    varargout{1} = f;
end