function handles = MeasureTexture(handles,varargin)

% Help for the Measure Texture module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Measures several texture features for identified objects or for entire
% images.
% *************************************************************************
%
% Given an image with objects identified (e.g. nuclei or cells), this
% module extracts texture features for each object based on a corresponding
% grayscale image. Measurements are recorded for each object. If "Image" is
% chosen, the texture of the image overall is measured.
%
% How it works:
% Retrieves objects in label matrix format and a corresponding original
% grayscale image and makes measurements of the objects. The label matrix
% image should be "compacted": that is, each number should correspond to an
% object, with no numbers skipped. So, if some objects were discarded from
% the label matrix image, the image should be converted to binary and
% re-made into a label matrix image before feeding into this module.
%
% The scale of texture measured is chosen by the user, in pixel units. A
% higher number for the scale of texture measures larger patterns of
% texture whereas smaller numbers measure more localized patterns of
% texture. It is best to measure texture on a scale smaller than your
% objects' sizes, so be sure that the value entered for scale of texture is
% smaller than most of your objects. For very small objects (smaller than
% the scale of texture you are measuring), the texture cannot be measured
% and will result in a value of NaN (Not a Number) in the output file.
%
% A range of texture scales may be specified, as a comma-separated list.
% Measurements will be generated for all scales specified.
%
% Note that texture measurements are affected by the overall intensity of 
% the object (or image). For example, if Image1 = Image2 + 0.2, then the 
% texture measurements should be the same for Image1 and Image2. However, 
% if the images are scaled differently, for example Image1 = 0.9*Image2, 
% then this will be reflected in the texture measurements, and they will be
% different. For example, in the extreme case of Image1 = 0*Image2 it is
% obvious that the texture measurements must be different. To make the
% measurements useful (both intensity, texture, etc.), it must be ensured
% that the images are scaled similarly. In other words, if differences in
% intensity are seen between two images or objects, the differences in
% texture cannot be trusted as being completely independent of the
% intensity difference.
%
% Features measured:      Feature Number:
% AngularSecondMoment     |       1
% Contrast                |       2
% Correlation             |       3
% Variance                |       4
% InverseDifferenceMoment |       5
% SumAverage              |       6
% SumVariance             |       7
% SumEntropy              |       8
% Entropy                 |       9
% DifferenceVariance      |      10
% DifferenceEntropy       |      11
% InfoMeas                |      12
% InfoMeas2               |      13
% GaborX                  |      14
% GaborY                  |      15
%
% Texture Measurement descriptions:
%
% Haralick Features:
% Haralick texture features are derived from the co-occurrence matrix, 
% which contains information about how image intensities in pixels with a 
% certain position in relation to each other occur together. For example, 
% how often does a pixel with intensity 0.12 have a neighbor 2 pixels to 
% the right with intensity 0.15? The current implementation in CellProfiler
% uses a shift of 1 pixel to the right for calculating the co-occurence 
% matrix. A different set of measurements is obtained for larger shifts, 
% measuring texture on a larger scale. The original reference for the 
% Haralick features is Haralick et al. (1973) Textural Features for Image
% Classification. IEEE Transaction on Systems Man, Cybernetics,
% SMC-3(6):610-621, where 14 features are described:
% H1. Angular Second Moment
% H2. Contrast
% H3. Correlation
% H4. Sum of Squares: Variation
% H5. Inverse Difference Moment
% H6. Sum Average
% H7. Sum Variance
% H8. Sum Entropy
% H9. Entropy
% H10. Difference Variance
% H11. Difference Entropy
% H12. Information Measure of Correlation 1
% H13. Information Measure of Correlation 2
% H14. Max correlation coefficient
%
% *H14 is disabled because it is computationally demanding.
%
% Gabor "wavelet" features:
% These features are similar to wavelet features, and they are obtained by
% applying so-called Gabor filters to the image. The Gabor filters measure
% the frequency content in different orientations. They are very similar to
% wavelets, and in the current context they work exactly as wavelets, but
% they are not wavelets by a strict mathematical definition. As currently
% implemented, the frequency content of the object is measured along the x-
% and y-axis (i.e. in two different orientations). The original reference
% is Gabor, D. (1946). "Theory of communication" Journal of the Institute
% of Electrical Engineers, 93:429-441.


% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

% MBray 2009_03_20: Comments on variables for pyCP upgrade
%
% Recommended variable order (setting, followed by current variable in MATLAB CP)
% (1) Input grayscale image (ImageName)
% We should reword this to be "What did you call the greyscale images whose texture you
% want to measure?
%
% (2) Input objects (ObjectNameList)
% We should reword this to be "What did you call the objects within which you want
% to measure texture?"
%
% (3) Feature scale (ScaleOfTexture)
% We should reword this to be "What scale of texture do you want to measure?"
%
% (i) A button should be added that lets the user add/subtract images for (1) and objects 
% for (2)
% (ii) The feature scale should let the user specify a range of texture 
% scales, so a module doesn't have to be added for each one. (not sure
% whether we want a "range of textures" (not sure how that would be
% entered) or instead add/subtract buttons to type in individual scales of texture.)

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the greyscale images you want to measure?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the objects that you want to measure?
%choiceVAR02 = Image
%choiceVAR02 = Do not use
%infotypeVAR02 = objectgroup
ObjectNameList{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 =
%choiceVAR03 = Do not use
%infotypeVAR03 = objectgroup
ObjectNameList{2} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 =
%choiceVAR04 = Do not use
%infotypeVAR04 = objectgroup
ObjectNameList{3} = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 =
%choiceVAR05 = Do not use
%infotypeVAR05 = objectgroup
ObjectNameList{4} = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 =
%choiceVAR06 = Do not use
%infotypeVAR06 = objectgroup
ObjectNameList{5} = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 =
%choiceVAR07 = Do not use
%infotypeVAR07 = objectgroup
ObjectNameList{6} = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = What is the scale of texture? A list of texture scales can be specified, separated by commas.
%defaultVAR08 = 3
ScaleOfTexture = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%%%%%%%%%%%%%%%%
%%% FEATURES %%%
%%%%%%%%%%%%%%%%

if nargin > 1 
    switch varargin{1}
%feature:categories
        case 'categories'
            if nargin == 1 || ismember(varargin{2},ObjectNameList)
                result = { 'Texture' };
            else
                result = {};
            end
%feature:measurements
        case 'measurements'
            result = {};
            if nargin >= 3 &&...
                strcmp(varargin{3},'Texture') &&...
                ismember(varargin{2},ObjectNameList)
                result = { ...
                    'AngularSecondMoment','Contrast','Correlation',...
                    'Variance','InverseDifferenceMoment',...
                    'SumAverage','SumVariance','SumEntropy',...
                    'Entropy','DifferenceVariance','DifferenceEntropy',...
                    'InfoMeas','InfoMeas2','GaborX','GaborY' };
            end
        otherwise
            error(['Unhandled category: ',varargin{1}]);
    end
    handles=result;
    return;
end

%%%VariableRevisionNumber = 2

%%% Set up the window for displaying the results
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    CPfigure(handles,'Text',ThisModuleFigureNumber);
    columns = 1;
end

ScaleOfTexture = cellfun(@str2double,strread(ScaleOfTexture,'%s','delimiter',','));

% Loop through all the texture scales specified
for TextureScaleNum = ScaleOfTexture(:)',
    %%% START LOOP THROUGH ALL THE OBJECTS
    for ObjectNameListNum = 1:6
        ObjectName = ObjectNameList{ObjectNameListNum};
        if strcmp(ObjectName,'Do not use')
            continue
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Reads (opens) the image you want to analyze and assigns it to a variable,
        %%% "OrigImage".
        OrigImage = CPretrieveimage(handles,ImageName,ModuleName,'MustBeGray','CheckScale');

        if ~strcmp(ObjectName,'Image')
            %%% Retrieves the label matrix image that contains the segmented objects which
            %%% will be measured with this module.
            LabelMatrixImage = CPretrieveimage(handles,['Segmented', ObjectName],ModuleName,'MustBeGray','DontCheckScale');
            %%% For the cases where the label matrix was produced from a cropped
            %%% image, the sizes of the images will not be equal. So, we crop the
            %%% LabelMatrix and try again to see if the matrices are then the
            %%% proper size. Removes Rows and Columns that are completely blank.
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
            if any(size(OrigImage) ~= size(LabelMatrixImage))
                error(['Image processing was canceled in the ', ModuleName, ' module. The size of the image you want to measure is not the same as the size of the image from which the ',ObjectName,' objects were identified.'])
            end
        end

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        %%% Initialize measurement structure
        Haralick = [];
        HaralickFeatures = {'AngularSecondMoment',...
            'Contrast',...
            'Correlation',...
            'Variance',...
            'InverseDifferenceMoment',...
            'SumAverage',...
            'SumVariance',...
            'SumEntropy',...
            'Entropy',...
            'DifferenceVariance',...
            'DifferenceEntropy',...
            'InfoMeas1',...
            'InfoMeas2'};

        Gabor = [];
        GaborFeatures    = {'GaborX',...
            'GaborY'};

        if strcmp(ObjectName,'Image')
            ObjectCount = 1;
        else
            %%% Count objects
            ObjectCount = max(LabelMatrixImage(:));
        end

        if ObjectCount > 0 || strcmp(ObjectName,'Image')

            %%% Get Gabor features.
            %%% The Gabor features are calculated by convolving the entire
            %%% image with Gabor filters and then extracting the filter output
            %%% value in the centroids of the objects in LabelMatrixImage

            if ~strcmp(ObjectName,'Image')
                % Adjust size of filter to size of objects in the image
                % The centroids indicate where we should measure the Gabor
                % filter output
                tmp = regionprops(LabelMatrixImage,'Area','Centroid');
                Areas = cat(1,tmp.Area);
                MedianArea = median(Areas);

                % Round centroids and find linear index for them.
                % The centroids are stored in [column,row] order.
                Centroids = round(cat(1,tmp.Centroid));
            else
                MedianArea = size(OrigImage,1)*size(OrigImage,2);
            end

            sigma = sqrt(MedianArea/pi)/3;  % Set width of filter to a third of the median radius

            % Use Gabor filters with three different frequencies
            f = 1/(2*TextureScaleNum);

            % Angle direction, filter along the x-axis and y-axis
            theta = [0 pi/2];

            if ~strcmp(ObjectName,'Image')
                % Create kernel coordinates
                KernelSize = round(2.5*sigma); % The filter size is set somewhat arbitrary
            else
                KernelSize = max(size(OrigImage,1)/2,size(OrigImage,2)/2);
            end
            [x,y] = meshgrid(-KernelSize:KernelSize,-KernelSize:KernelSize);

            % Apply Gabor filters and store filter outputs in the Centroid pixels
            GaborFeatureNo = 1;
            Gabor = zeros(ObjectCount,length(f)*length(theta));                              % Initialize measurement matrix
            for m = 1:length(f)
                for n = 1:length(theta)

                    % Calculate Gabor filter kernel
                    % Scale by 1000 to get measurements in a convenient range
                    g = 1000*1/(2*pi*sigma^2)*exp(-(x.^2 + y.^2)/(2*sigma^2)).*exp(2*pi*sqrt(-1)*f(m)*(x*cos(theta(n))+y*sin(theta(n))));
                    g = g - mean(g(:));           % Important that the filters has DC zero, otherwise they will be sensitive to the intensity of the image


                    % Center the Gabor kernel over the centroid and calculate the filter response.
                    if strcmp(ObjectName,'Image')
                        % Cut patch
                        p = OrigImage;

                        if size(OrigImage,1) ~= size(g,1)
                            p = [p;zeros(size(g,1)-size(p,1),size(p,2))];
                        end

                        if size(OrigImage,2) ~= size(g,2)
                            p = [p zeros(size(p,1),size(g,2)-size(p,2))];
                        end
                        % Calculate the filter output
                        Gabor(1,GaborFeatureNo) = abs(sum(sum(g.*p)));
                    else
                        for k = 1:ObjectCount
                            %%% It's possible for objects not to have any pixels,
                            %%% particularly tertiary objects (such as cytoplasm from
                            %%% cells the exact same size as their nucleus).
                            if Areas(k) == 0,
                                Gabor(k, GaborFeatureNo) = 0;
                                continue;
                            end

                            xmin1 = Centroids(k,1)-KernelSize;
                            xmax1 = Centroids(k,1)+KernelSize;
                            ymin1 = Centroids(k,2)-KernelSize;
                            ymax1 = Centroids(k,2)+KernelSize;
                            xmin2 = max(1,xmin1);
                            xmax2 = min(size(OrigImage,2),xmax1);
                            ymin2 = max(1,ymin1);
                            ymax2 = min(size(OrigImage,1),ymax1);

                            % Cut patch
                            p = OrigImage(ymin2:ymax2,xmin2:xmax2);

                            % Pad with zeros if necessary to match the filter kernel size
                            if xmin1 < xmin2
                                p = [zeros(size(p,1),xmin2 - xmin1) p];
                            end
                            if xmax1 > xmax2
                                p = [p zeros(size(p,1),xmax1 - xmax2)];
                            end

                            if ymin1 < ymin2
                                p = [zeros(ymin2 - ymin1,size(p,2));p];
                            end
                            if ymax1 > ymax2
                                p = [p;zeros(ymax1 - ymax2,size(p,2))];
                            end

                            % Calculate the filter output
                            Gabor(k,GaborFeatureNo) = abs(sum(sum(g.*p)));
                        end
                    end
                    GaborFeatureNo = GaborFeatureNo + 1;
                end
            end

            if strcmp(ObjectName,'Image')
                [m,n] = size(OrigImage);
                BWim = ones(m,n);
                %%% Get Haralick features
                Haralick(1,:) = CalculateHaralick(OrigImage,BWim,TextureScaleNum);
            else
                %%% Get Haralick features.
                %%% Have to loop over the objects
                Haralick = zeros(ObjectCount,13);
                [sr sc] = size(LabelMatrixImage);
                props = regionprops(LabelMatrixImage,'PixelIdxList');   % Get pixel indexes in a fast way
                for Object = 1:ObjectCount
                    %%% Cut patch so that we don't have to deal with entire image
                    [r,c] = ind2sub([sr sc],props(Object).PixelIdxList);
                    rmax = min(sr,max(r));
                    rmin = max(1,min(r));
                    cmax = min(sc,max(c));
                    cmin = max(1,min(c));
                    BWim   = LabelMatrixImage(rmin:rmax,cmin:cmax) == Object;
                    Greyim = OrigImage(rmin:rmax,cmin:cmax);
                    %%% Get Haralick features
                    Haralick(Object,:) = CalculateHaralick(Greyim,BWim,TextureScaleNum);
                end
            end
        else
            Haralick = zeros(0,13);
            Gabor = zeros(0,2);
        end
        %%% Save measurements
        AllFeatures = cat(2,HaralickFeatures,GaborFeatures);
        Data = [Haralick Gabor];
        for FeatureNum = 1:length(AllFeatures)
            feature_name = CPjoinstrings('Texture',char(AllFeatures{FeatureNum}),ImageName,num2str(TextureScaleNum));
            handles = CPaddmeasurements(handles, ObjectName, feature_name, Data(:,FeatureNum));
        end

        %%% Report measurements
        FontSize = handles.Preferences.FontSize;

        if any(findobj == ThisModuleFigureNumber);
            % Remove uicontrols from last cycle
            delete(findobj(ThisModuleFigureNumber,'tag','TextUIControl'));

            % This first block writes the same text several times
            % Header

            if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
                delete(findobj('parent',ThisModuleFigureNumber,'string','R'));
                delete(findobj('parent',ThisModuleFigureNumber,'string','G'));
                delete(findobj('parent',ThisModuleFigureNumber,'string','B'));
            end

            uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0 0.95 1 0.04],...
                'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextUIControl',...
                'fontsize',FontSize,'fontweight','bold','string',sprintf(['Average texture features for ',ImageName,', cycle #%d'],handles.Current.SetBeingAnalyzed));

            % Number of objects
            uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.85 0.3 0.03],...
                'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextUIControl',...
                'fontsize',FontSize,'fontweight','bold','string','Number of objects:');

            % Text for Gabor features
            uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8 0.3 0.03],...
                'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextUIControl',...
                'fontsize',FontSize,'fontweight','bold','string','Gabor features:');
            for k = 1:2
                uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8-0.04*k 0.3 0.03],...
                    'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextUIControl',...
                    'fontsize',FontSize,'string',GaborFeatures{k});
            end

            % Text for Haralick features
            uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.65 0.3 0.03],...
                'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextUIControl',...
                'fontsize',FontSize,'fontweight','bold','string','Haralick features:');
            for k = 1:10
                uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.65-0.04*k 0.3 0.03],...
                    'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextUIControl',...
                    'fontsize',FontSize,'string',HaralickFeatures{k});
            end

            % The name of the object image
            uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.35+0.2*(columns-1) 0.9 0.2 0.03],...
                'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextEachObjUIControl',...
                'fontsize',FontSize,'fontweight','bold','string',ObjectName);

            % Number of objects
            uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.35+0.2*(columns-1) 0.85 0.2 0.03],...
                'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextEachObjUIControl',...
                'fontsize',FontSize,'string',num2str(ObjectCount));

            if ObjectCount > 0
                % Gabor features
                for k = 1:2
                    uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.35+0.2*(columns-1) 0.8-0.04*k 0.2 0.03],...
                        'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextEachObjUIControl',...
                        'fontsize',FontSize,'string',sprintf('%0.2f',mean(Gabor(:,k))));
                end

                % Haralick features
                for k = 1:10
                    uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.35+0.2*(columns-1) 0.65-0.04*k 0.2 0.03],...
                        'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextEachObjUIControl',...
                        'fontsize',FontSize,'string',sprintf('%0.2f',mean(Haralick(:,k))));
                end
            end
            % This variable is used to write results in the correct column
            % and to determine the correct window size
            columns = columns + 1;
        end
    end
end
drawnow

function H = CalculateHaralick(im,mask,ScaleOfTexture)

warning off MATLAB:DivideByZero
% This function calculates so-called Haralick features, which are
% based on the co-occurence matrix. The function takes two inputs:
%
% im    - A grey level image
% mask  - A binary mask
%
% Currently, the implementation uses 8 different grey levels
% and calculates the co-occurence matrix for a horizontal shift
% of 1 pixel.
%
% The original reference is:
% Haralick et al. (1973)
% Textural Features for Image Classification.
% IEEE Transaction on Systems
% Man, Cybernetics, SMC-3(6):610-621.
%
% BEWARE: There are lots of erroneous formulas for the Haralick features in
% the literature. There is also an error in the original paper.
%

% Number of greylevels to use
Levels = 8;

% Quantize the image into a lower number of grey levels (specified by Levels)
BinEdges = linspace(0,1,Levels+1);

% Find the max and min values within the mask and normalize so that the
% intenisties within the mask are between 0 and 1.
intensities = im(mask);
Imax = max(intensities(:));
Imin = min(intensities(:));
if Imax ~= Imin                     % Avoid divide by zero
    im = (im - Imin)/(Imax-Imin);
end

% Do the quantization
qim = zeros(size(im));
for k = 1:Levels
    qim(find(im >= BinEdges(k))) = k; %#ok Ignore MLint
end

% Shift ScaleOfTexture step to the right
im1 = qim(:,1:end-ScaleOfTexture); im1 = im1(:);
im2 = qim(:,ScaleOfTexture+1:end); im2 = im2(:);

% Remove cases where at least one position is
% outside the mask.
m1 = mask(:,1:end-ScaleOfTexture); m1 = m1(:);
m2 = mask(:,ScaleOfTexture+1:end); m2 = m2(:);

index = (sum([m1 m2],2) == 2);
if isempty(index)
    H = [0 0 0 0 0 0 0 0 0 0 0 0 0];
    return
end
im1 = im1(index);
im2 = im2(index);

%%% Calculate co-occurence matrix
% P = zeros(Levels);
% for k = 1:Levels
%     index = find(im1==k);
%     if ~isempty(index)
%         P(k,:) = hist(im2(index),(1:Levels));
%     else
%         P(k,:) = zeros(1,Levels);
%     end
% end
% 
% The line below is a fast 2D-histogram in matlab, and is equivalent to the
% loop above.  Ray & Kyungnam, 2007-07-18.
P = full(sparse(im1,im2,1,Levels,Levels)); 
P = P/length(im1);

%%% Calculate features from the co-occurence matrix
% First, pre-calculate a few quantities that are used in
% several features.
px = sum(P,2);
py = sum(P,1);
mux = sum((1:Levels)'.*px);
muy = sum((1:Levels).*py);
sigmax = sqrt(sum(((1:Levels)' - mux).^2.*px));
sigmay = sqrt(sum(((1:Levels) - muy).^2.*py));
HX = -sum(px.*log(px+eps));
HY = -sum(py.*log(py+eps));
HXY = -sum(P(:).*log(P(:)+eps));
HXY1 = -sum(sum(P.*log(px*py+eps)));
HXY2 = -sum(sum(px*py .* log(px*py+eps)));

p_xplusy = zeros(2*Levels-1,1);      % Range 2:2*Levels
p_xminusy = zeros(Levels,1);         % Range 0:Levels-1
for x=1:Levels
    for y = 1:Levels
        p_xplusy(x+y-1) = p_xplusy(x+y-1) + P(x,y);
        p_xminusy(abs(x-y)+1) = p_xminusy(abs(x-y)+1) + P(x,y);
    end
end

% H1. Angular Second Moment
H1 = sum(P(:).^2);

% H2. Contrast
H2 = sum((0:Levels-1)'.^2.*p_xminusy);

% H3. Correlation
H3 = (sum(sum((1:Levels)'*(1:Levels).*P)) - mux*muy)/(sigmax*sigmay);
if isinf(H3),
    H3 = 0;
end

% H4. Sum of Squares: Variation
H4 = sigmax^2;

% H5. Inverse Difference Moment
H5 = sum(sum(1./(1+toeplitz(0:Levels-1).^2).*P));

% H6. Sum Average
H6 = sum((2:2*Levels)'.*p_xplusy);

% H7. Sum Variance (error in Haralick's original paper here)
H7 = sum(((2:2*Levels)' - H6).^2 .* p_xplusy);

% H8. Sum Entropy
H8 = -sum(p_xplusy .* log(p_xplusy+eps));

% H9. Entropy
H9 = HXY;

% H10. Difference Variance
H10 = sum(p_xminusy.*((0:Levels-1)' - sum((0:Levels-1)'.*p_xminusy)).^2);

% H11. Difference Entropy
H11 = - sum(p_xminusy.*log(p_xminusy+eps));

% H12. Information Measure of Correlation 1
H12 = (HXY-HXY1)/max(HX,HY);

% H13. Information Measure of Correlation 2
H13 = real(sqrt(1-exp(-2*(HXY2-HXY))));             % An imaginary result has been encountered once, reason unclear

warning on MATLAB:DivideByZero

% H14. Max correlation coefficient (not currently used)
% Q = zeros(Levels);
% for i = 1:Levels
%     for j = 1:Levels
%         Q(i,j) = sum(P(i,:).*P(j,:)/(px(i)*py(j)));
%     end
% end
% [V,lambda] = eig(Q);
% lambda = sort(diag(lambda));
% H14 = sqrt(max(0,lambda(end-1)));

H = [H1 H2 H3 H4 H5 H6 H7 H8 H9 H10 H11 H12 H13];
H(isnan(H))=0;

% % This function calculates Gabor features in a different way
% % It may be better but it's also considerably slower.
% % It's called by Gabor(Object,:) = CalculateGabor(Greyim,BWim,sigma);
% function G = CalculateGabor(im,mask,sigma,flag)
% %
% % This function calculates Gabor features, which measure
% % the energy in different frequency sub-bands. The Gabor
% % transform is essentially equivalent to a wavelet transform.
% %
% % im    - A grey level image
% % mask  - A binary mask
% % sigma - Scale parameter for the Gaussian weight function
%
% % Use Gabor filters with three different frequencies
% f = [0.06 0.12 0.24];
%
% % Filter along the x-axis and y-axis
% theta = [0 pi/2];
%
% % Match the filter kernel size to the input patch size
% [sr,sc] = size(mask);
% if rem(sr,2) == 0,ty = [-sr/2:sr/2-1];else ty = [-(sr-1)/2:(sr-1)/2];end
% if rem(sc,2) == 0,tx = [-sc/2:sc/2-1];else tx = [-(sc-1)/2:(sc-1)/2];end
% [x,y]=meshgrid(tx,ty);
%
% % Calculate the Gabor features
% G = zeros(length(theta),length(f));
% for m = 1:length(f)
%     for n = 1:length(theta)
%
%         % Calculate Gabor filter kernel
%         g = 1/(2*pi*sigma^2)*exp(-(x.^2 + y.^2)/(2*sigma^2)).*exp(2*pi*sqrt(-1)*f(m)*(x*cos(theta(n))+y*sin(theta(n))));
%
%         % Use Normalized Convolution to calculate filter responses. This
%         % method only include object pixels for calculating the filter
%         % response and excludes surrounding background pixels.
%         % See Farneback, 2002. "Polynomial Expansion for Orientation and
%         % Motion Estimation". PhD Thesis
%         gr = real(g);
%         gi = imag(g);
%         B = [gr(:) gi(:)];
%         Wc = diag(mask(:));
%         r = inv(B'*Wc*B)*B'*Wc*im(:);
%         G(n,m) = sqrt(sum(r.^2));
%
%         % Direct way of calculating filter responses
%         %tmpr = sum(sum(real(g).*im));
%         %tmpi = sum(sum(imag(g).*im));
%         %G(n,m) = sqrt(tmpr.^2+tmpi.^2);
%     end
% end
% G = G(:)';