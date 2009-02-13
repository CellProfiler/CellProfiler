function handles = MeasureObjectIntensity(handles,varargin)

% Help for the Measure Object Intensity module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Measures several intensity features for identified objects.
% *************************************************************************
%
% Given an image with objects identified (e.g. nuclei or cells), this
% module extracts intensity features for each object based on a
% corresponding grayscale image. Measurements are recorded for each object.
%
% Features measured:       Feature Number:
% IntegratedIntensity     |       1
% MeanIntensity           |       2
% StdIntensity            |       3
% MinIntensity            |       4
% MaxIntensity            |       5
% IntegratedIntensityEdge |       6
% MeanIntensityEdge       |       7
% StdIntensityEdge        |       8
% MinIntensityEdge        |       9
% MaxIntensityEdge        |      10
% MassDisplacement        |      11
% LowerQuartileIntensity  |      12
% MedianIntensity         |      13
% UpperQuartileIntensity  |      14
%
% How it works:
% Retrieves objects in label matrix format and a corresponding original
% grayscale image and makes measurements of the objects. The label matrix
% image should be "compacted": that is, each number should correspond to an
% object, with no numbers skipped. So, if some objects were discarded from
% the label matrix image, the image should be converted to binary and
% re-made into a label matrix image before feeding it to this module.
%
% Intensity Measurement descriptions:
%
% * IntegratedIntensity - The sum of the pixel intensities within an
% object.
% * MeanIntensity - The average pixel intensity within an object.
% * StdIntensity - The standard deviation of the pixel intensities within
% an object.
% * MaxIntensity - The maximal pixel intensity within an object.
% * MinIntensity - The minimal pixel intensity within an object.
% * IntegratedIntensityEdge - The sum of the edge pixel intensities of an
% object.
% * MeanIntensityEdge - The average edge pixel intensity of an object.
% * StdIntensityEdge - The standard deviation of the edge pixel intensities
% of an object.
% * MaxIntensityEdge - The maximal edge pixel intensity of an object.
% * MinIntensityEdge - The minimal edge pixel intensity of an object.
% * MassDisplacement - The distance between the centers of gravity in the
% gray-level representation of the object and the binary representation of
% the object.
% * LowerQuartileIntensity - the intensity value of the pixel for which 25%
% of the pixels in the object have lower values.
% * MedianIntensity - the median intensity value within the object
% * UpperQuartileIntensity - the intensity value of the pixel for which 75%
% of the pixels in the object have lower values.
%
% For publication purposes, it is important to note that the units of
% intensity from microscopy images are usually described as "Intensity
% units" or "Arbitrary intensity units" since microscopes are not 
% callibrated to an absolute scale. Also, it is important to note whether 
% you are reporting either the mean or the integrated intensity, so specify
% "Mean intensity units" or "Integrated intensity units" accordingly.
%
% See also MeasureImageIntensity.

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

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the greyscale images you want to measure?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the objects that you want to measure?
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

%%% Initialize measurement structure
BasicFeatures    = {'IntegratedIntensity',...
    'MeanIntensity',...
    'StdIntensity',...
    'MinIntensity',...
    'MaxIntensity',...
    'IntegratedIntensityEdge',...
    'MeanIntensityEdge',...
    'StdIntensityEdge',...
    'MinIntensityEdge',...
    'MaxIntensityEdge',...
    'MassDisplacement',...
    'LowerQuartileIntensity',...
    'MedianIntensity',...
    'UpperQuartileIntensity'};

%%%%%%%%%%%%%%%%
%%% FEATURES %%%
%%%%%%%%%%%%%%%%

if nargin > 1 
    switch varargin{1}
%feature:categories
        case 'categories'
            if nargin == 1 || ismember(varargin{2},ObjectNameList)
                result = { 'Intensity' };
            else
                result = {};
            end
%feature:measurements
        case 'measurements'
            result = {};
            if nargin >= 3 &&...
                strcmp(varargin{3},'Intensity') &&...
                ismember(varargin{2},ObjectNameList)
                result = BasicFeatures;
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
if any(findobj == ThisModuleFigureNumber);
    CPfigure(handles,'Text',ThisModuleFigureNumber);
    columns = 1;
end

%%% START LOOP THROUGH ALL THE OBJECTS
for i = 1:length(ObjectNameList)
    ObjectName = ObjectNameList{i};
    if strcmpi(ObjectName,'Do not use')
        continue
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    drawnow

    %%% Reads (opens) the image you want to analyze and assigns it to a variable,
    %%% "OrigImage".
    OrigImage = CPretrieveimage(handles,ImageName,ModuleName);

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
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    drawnow

    %%% Get pixel indexes (fastest way), and count objects
    [sr sc] = size(LabelMatrixImage);        
    props = regionprops(LabelMatrixImage,'PixelIdxList','Area');
    ObjectCount = length(props);    

    %%% Label-aware boundary finding (even when two objects are adjacent)
    LabelBoundaryImage = CPlabelperim(LabelMatrixImage);
    
    if ObjectCount > 0
        Basic = cell(ObjectCount,length(BasicFeatures));
        
        for Object = 1:ObjectCount
            %%% It's possible for objects not to have any pixels,
            %%% particularly tertiary objects (such as cytoplasm from
            %%% cells the exact same size as their nucleus).
            if isempty(props(Object).PixelIdxList),
                [Basic{Object,:}] = deal(0);
                continue;
            end
         
            %%% Measure basic set of Intensity features
            Basic{Object,1} = sum(OrigImage(props(Object).PixelIdxList));
            Basic{Object,2} = mean(OrigImage(props(Object).PixelIdxList));
            Basic{Object,3} = std(OrigImage(props(Object).PixelIdxList));
            Basic{Object,4} = min(OrigImage(props(Object).PixelIdxList));
            Basic{Object,5} = max(OrigImage(props(Object).PixelIdxList));     
     
            %%% Kyungnam, 2007-Aug-06: optimized code
            %%% Cut patch so that we don't have to deal with entire image
            [r,c] = ind2sub([sr sc],props(Object).PixelIdxList);
            rmax = min(sr,max(r));
            rmin = max(1,min(r));
            cmax = min(sc,max(c));
            cmin = max(1,min(c)); 
            BWim = LabelMatrixImage(rmin:rmax,cmin:cmax) == Object;
            Greyim = OrigImage(rmin:rmax,cmin:cmax);
            Boundaryim = LabelBoundaryImage(rmin:rmax,cmin:cmax) == Object;
            perim = Greyim(Boundaryim(:));
            Basic{Object,6}  = sum(perim);
            Basic{Object,7}  = mean(perim);
            Basic{Object,8}  = std(perim);
            Basic{Object,9}  = min(perim);
            Basic{Object,10} = max(perim);
          
            %%% Kyungnam, 2007-Aug-06: the original old code left commented below
            %%%                        'bwperim' is slow!                         
            %             %%% Get perimeter in order to calculate edge features
            %             perim = bwperim(BWim);
            %             perim = Greyim(find(perim)); %#ok Ignore MLint
            %             Basic(Object,6)  = sum(perim);
            %             Basic(Object,7)  = mean(perim);
            %             Basic(Object,8)  = std(perim);
            %             Basic(Object,9)  = min(perim);
            %             Basic(Object,10) = max(perim);   
           
        end
        %%% Calculate the Mass displacment (taking the pixelsize into account), which is the distance between
        %%% the center of gravity in the gray level image and the binary
        %%% image.
        mask = (LabelMatrixImage > 0);
        masked_labels = LabelMatrixImage(mask);
        masked_intensity = OrigImage(mask);
        [x,y] = meshgrid(1:size(LabelMatrixImage,1),1:size(LabelMatrixImage,2));
        masked_x = x(mask);
        masked_y = y(mask);
        CM_x = full(sparse(masked_labels, 1, masked_x) ./ sparse(masked_labels, 1, 1));
        CM_y = full(sparse(masked_labels, 1, masked_y) ./ sparse(masked_labels, 1, 1));
        
        denom = sparse(masked_labels, 1, masked_intensity);
        if denom ~= 0
            intensity_CM_x = full(sparse(masked_labels, 1, masked_x .*masked_intensity) ./ denom);
            intensity_CM_y = full(sparse(masked_labels, 1, masked_y .*masked_intensity) ./ denom);
        else
            intensity_CM_x = zeros(size(CM_x));
            intensity_CM_y = zeros(size(CM_y));
        end

        PixelSize = str2double(handles.Settings.PixelSize);
        diff_x = CM_x - intensity_CM_x;
        diff_y = CM_y - intensity_CM_y;
        Basic(:,11) = arrayfun(@(x) {x}, sqrt(diff_x.^2+diff_y.^2).*PixelSize);
        %
        % A trick for median, lower & upper quartile:
        %   Add the object # to an intensity scaled between .1 and .9
        %   Sort the resulting array.
        %   Restore the pixels to scaled intensities w/o object #
        %   Do the cumulative sum of the areas of each object
        %   Subtract 1/4, 1/2 and 3/4 of the area and use that to
        %   index into the sorted array to get the values.
        %
        SortedObjectPixels=OrigImage(LabelMatrixImage>0);
        Min = min(SortedObjectPixels);
        Max = max(SortedObjectPixels);
        Scale = (Max-Min) / .8;
        SortedObjectPixels = ((SortedObjectPixels - Min) / (Scale+eps))+.1;
        SortedObjectPixels = SortedObjectPixels + LabelMatrixImage(LabelMatrixImage>0);
        SortedObjectPixels = sort(SortedObjectPixels);
        SortedObjectPixels = SortedObjectPixels - floor(SortedObjectPixels);
        SortedObjectPixels = (SortedObjectPixels - .1) * Scale + Min;
        Address = cumsum([props.Area]);
        idx = max(1,floor(Address-[props.Area]*3/4));
        Basic(:,12) = arrayfun(@(x) {x}, SortedObjectPixels(idx));
        idx = max(1,floor(Address-[props.Area]/2));
        Basic(:,13) = arrayfun(@(x) {x}, SortedObjectPixels(idx));
        idx = max(1,floor(Address-[props.Area]/4));
        Basic(:,14) = arrayfun(@(x) {x}, SortedObjectPixels(idx));
    else
        % Fill in with empty sets
        Basic = cell(1,length(BasicFeatures));
    end
    %%% Save measurements
    for j = 1:size(Basic,2)
        feature_name = CPjoinstrings('Intensity',BasicFeatures{j},ImageName);
        handles = CPaddmeasurements(handles, ObjectName, ...
            feature_name, cat(1,Basic{:,j}));
    end
    
    %%% Report measurements
    if any(findobj == ThisModuleFigureNumber);
        % Remove uicontrols from last cycle
        delete(findobj(ThisModuleFigureNumber,'tag','TextUIControl'));
        
        FontSize = handles.Preferences.FontSize;
        if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
            delete(findobj('parent',ThisModuleFigureNumber,'string','R'));
            delete(findobj('parent',ThisModuleFigureNumber,'string','G'));
            delete(findobj('parent',ThisModuleFigureNumber,'string','B'));
        end
        %%%% This first block writes the same text several times
        %%% Header

        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0 0.95 1 0.04],...
            'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextUIControl',...
            'fontsize',FontSize,'fontweight','bold','string',sprintf(['Average intensity features for ', ImageName,', cycle #%d'],handles.Current.SetBeingAnalyzed));

        %%% Number of objects
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.85 0.3 0.03],...
            'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextUIControl',...
            'fontsize',FontSize,'fontweight','bold','string','Number of objects:');

        %%% Text for Basic features
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8 0.3 0.03],...
            'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextUIControl',...
            'fontsize',FontSize,'fontweight','bold','string','Intensity feature:');
        for k = 1:length(BasicFeatures)
            uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8-0.04*k 0.3 0.03],...
                'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextUIControl',...
                'fontsize',FontSize,'string',BasicFeatures{k});
        end

        %%% The name of the object image
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.35+0.1*(columns-1) 0.9 0.1 0.03],...
            'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextEachObjUIEachOControl',...
            'fontsize',FontSize,'fontweight','bold','string',ObjectName);

        %%% Number of objects
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.35+0.1*(columns-1) 0.85 0.1 0.03],...
            'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextEachObjUIControl',...
            'fontsize',FontSize,'string',num2str(ObjectCount));

        if ObjectCount > 0
            %%% Basic features
            for k = 1:length(BasicFeatures)
                uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.35+0.1*(columns-1) 0.8-0.04*k 0.1 0.03],...
                    'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextEachObjUIControl',...
                    'fontsize',FontSize,'string',sprintf('%0.2f',mean(cat(1,Basic{:,k}))));
            end
        end
        %%% This variable is used to write results in the correct column
        %%% and to determine the correct window size
        columns = columns + 1;
    end
end