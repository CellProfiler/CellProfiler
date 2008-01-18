function handles = MeasureObjectIntensity(handles)

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
%
% See also MeasureImageIntensity.

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
%   Kyungnam Kim
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

    %%% Initialize measurement structure
    Basic = [];
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
        'MassDisplacement'};

    %%% Get pixel indexes (fastest way), and count objects
    [sr sc] = size(LabelMatrixImage);        
    props = regionprops(LabelMatrixImage,'PixelIdxList');
    ObjectCount = length(props);    

    %%% Label-aware boundary finding (even when two objects are adjacent)
    LabelBoundaryImage = CPlabelperim(LabelMatrixImage);
    
    if ObjectCount > 0
        Basic = zeros(ObjectCount,4);

        for Object = 1:ObjectCount
            %%% It's possible for objects not to have any pixels,
            %%% particularly tertiary objects (such as cytoplasm from
            %%% cells the exact same size as their nucleus).
            if isempty(props(Object).PixelIdxList),
                Basic(Object,1:11) = 0;
                continue;
            end
         
            %%% Measure basic set of Intensity features
            Basic(Object,1) = sum(OrigImage(props(Object).PixelIdxList));
            Basic(Object,2) = mean(OrigImage(props(Object).PixelIdxList));
            Basic(Object,3) = std(OrigImage(props(Object).PixelIdxList));
            Basic(Object,4) = min(OrigImage(props(Object).PixelIdxList));
            Basic(Object,5) = max(OrigImage(props(Object).PixelIdxList));     
     
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
            perim = Greyim(find(Boundaryim));
            Basic(Object,6)  = sum(perim);
            Basic(Object,7)  = mean(perim);
            Basic(Object,8)  = std(perim);
            Basic(Object,9)  = min(perim);
            Basic(Object,10) = max(perim);
          
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
           
            %%% Calculate the Mass displacment (taking the pixelsize into account), which is the distance between
            %%% the center of gravity in the gray level image and the binary
            %%% image.
            PixelSize = str2double(handles.Settings.PixelSize);
            BWx = sum((1:size(BWim,2)).*sum(BWim,1))/sum(1:size(BWim,2));
            BWy = sum((1:size(BWim,1))'.*sum(BWim,2))/sum(1:size(BWim,1));
            Greyx = sum((1:size(Greyim,2)).*sum(Greyim,1))/sum(1:size(Greyim,2));
            Greyy = sum((1:size(Greyim,1))'.*sum(Greyim,2))/sum(1:size(Greyim,1));
            Basic(Object,11) = sqrt((BWx-Greyx)^2+(BWy-Greyy)^2)*PixelSize;
        end
    else
        Basic(1,1:11) = 0;
    end
    %%% Save measurements
    handles.Measurements.(ObjectName).(['Intensity_',ImageName,'Features']) = BasicFeatures;
    handles.Measurements.(ObjectName).(['Intensity_',ImageName])(handles.Current.SetBeingAnalyzed) = {Basic};
    
    %%% Report measurements
    if any(findobj == ThisModuleFigureNumber);
        FontSize = handles.Preferences.FontSize;
        if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
            delete(findobj('parent',ThisModuleFigureNumber,'string','R'));
            delete(findobj('parent',ThisModuleFigureNumber,'string','G'));
            delete(findobj('parent',ThisModuleFigureNumber,'string','B'));
        end
        %%%% This first block writes the same text several times
        %%% Header

        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0 0.95 1 0.04],...
            'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica',...
            'fontsize',FontSize,'fontweight','bold','string',sprintf(['Average intensity features for ', ImageName,', cycle #%d'],handles.Current.SetBeingAnalyzed));

        %%% Number of objects
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.85 0.3 0.03],...
            'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica',...
            'fontsize',FontSize,'fontweight','bold','string','Number of objects:');

        %%% Text for Basic features
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8 0.3 0.03],...
            'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica',...
            'fontsize',FontSize,'fontweight','bold','string','Intensity feature:');
        for k = 1:11
            uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8-0.04*k 0.3 0.03],...
                'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica',...
                'fontsize',FontSize,'string',BasicFeatures{k});
        end

        %%% The name of the object image
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.35+0.1*(columns-1) 0.9 0.1 0.03],...
            'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica',...
            'fontsize',FontSize,'fontweight','bold','string',ObjectName);

        %%% Number of objects
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.35+0.1*(columns-1) 0.85 0.1 0.03],...
            'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica',...
            'fontsize',FontSize,'string',num2str(ObjectCount));

        if ObjectCount > 0
            %%% Basic features
            for k = 1:11
                uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.35+0.1*(columns-1) 0.8-0.04*k 0.1 0.03],...
                    'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica',...
                    'fontsize',FontSize,'string',sprintf('%0.2f',mean(Basic(:,k))));
            end
        end
        %%% This variable is used to write results in the correct column
        %%% and to determine the correct window size
        columns = columns + 1;
    end
end