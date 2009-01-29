function handles = MeasureObjectAreaShape(handles, varargin)

% Help for the Measure Object Area Shape module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Measures several area and shape features of identified objects.
% *************************************************************************
%
% Given an image with objects identified (e.g. nuclei or cells), this
% module extracts area and shape features of each object. Note that these
% features are only reliable for objects that are completely inside the
% image borders, so you may wish to exclude objects touching the edge of
% the image in Identify modules.
%
% Basic shape features:     Feature Number:
%
% Zernike shape features measure shape by describing a binary object (or
% more precisely, a patch with background and an object in the center) in a
% basis of Zernike polynomials, using the coefficients as features (Boland
% et al., 1998). Currently, Zernike polynomials from order 0 to order 9 are
% calculated, giving in total 30 measurements. While there is no limit to
% the order which can be calculated (and indeed users could add more by
% adjusting the code), the higher order polynomials carry less information.
%
% Details about how measurements are calculated:
% This module retrieves objects in label matrix format and measures them.
% The label matrix image should be "compacted": that is, each number should
% correspond to an object, with no numbers skipped. So, if some objects
% were discarded from the label matrix image, the image should be converted
% to binary and re-made into a label matrix image before feeding into this
% module.
%
% The following measurements are extracted using the Matlab regionprops.m
% function:
% *Area - Computed from the the actual number of pixels in the region.
% *Eccentricity - Also known as elongation or elongatedness. For an ellipse
% that has the same second-moments as the object, the eccentricity is the
% ratio of the between-foci distance and the major axis length. The value
% is between 0 (a circle) and 1 (a line segment).
% *Solidity - Also known as convexity. The proportion of the pixels in the
% convex hull that are also in the object. Computed as Area/ConvexArea.
% *Extent - The proportion of the pixels in the bounding box that are also
% in the region. Computed as the Area divided by the area of the bounding box.
% *EulerNumber - Equal to the number of objects in the image minus the
% number of holes in those objects. For modules built to date, the number
% of objects in the image is always 1.
% *MajorAxisLength - The length (in pixels) of the major axis of the
% ellipse that has the same normalized second central moments as the
% region.
% *MinorAxisLength - The length (in pixels) of the minor axis of the
% ellipse that has the same normalized second central moments as the
% region.
% *Perimeter - the total number of pixels around the boundary of each
% region in the image.
%
% In addition, the following feature is calculated:
%
% FormFactor = 4*pi*Area/Perimeter^2, equals 1 for a perfectly circular
% object%
%
% HERE IS MORE DETAILED INFORMATION ABOUT THE MEASUREMENTS FOR YOUR
% REFERENCE
%
% 'Area' ? Scalar; the actual number of pixels in the region. (This value
% might differ slightly from the value returned by bwarea, which weights
% different patterns of pixels differently.)
%
% 'Eccentricity' ? Scalar; the eccentricity of the ellipse that has the
% same second-moments as the region. The eccentricity is the ratio of the
% distance between the foci of the ellipse and its major axis length. The
% value is between 0 and 1. (0 and 1 are degenerate cases; an ellipse whose
% eccentricity is 0 is actually a circle, while an ellipse whose eccentricity
% is 1 is a line segment.) This property is supported only for 2-D input
% label matrices.
%
% 'Solidity' -? Scalar; the proportion of the pixels in the convex hull that
% are also in the region. Computed as Area/ConvexArea. This property is
% supported only for 2-D input label matrices.
%
% 'Extent' ? Scalar; the proportion of the pixels in the bounding box that
% are also in the region. Computed as the Area divided by the area of the
% bounding box. This property is supported only for 2-D input label matrices.
%
% 'EulerNumber' ? Scalar; equal to the number of objects in the region
% minus the number of holes in those objects. This property is supported
% only for 2-D input label matrices. regionprops uses 8-connectivity to
% compute the EulerNumber measurement. To learn more about connectivity,
% see Pixel Connectivity.
%
% 'perimeter' ? p-element vector containing the distance around the boundary
% of each contiguous region in the image, where p is the number of regions.
% regionprops computes the perimeter by calculating the distance between
% each adjoining pair of pixels around the border of the region. If the
% image contains discontiguous regions, regionprops returns unexpected
% results. The following figure shows the pixels included in the perimeter
% calculation for this object
%
% 'MajorAxisLength' ? Scalar; the length (in pixels) of the major axis of
% the ellipse that has the same normalized second central moments as the
% region. This property is supported only for 2-D input label matrices.
%
% 'MinorAxisLength' ? Scalar; the length (in pixels) of the minor axis of
% the ellipse that has the same normalized second central moments as the
% region. This property is supported only for 2-D input label matrices.
%
% 'Orientation' ? Scalar; the angle (in degrees ranging from -90 to 90
% degrees) between the x-axis and the major axis of the ellipse that has the
% same second-moments as the region. This property is supported only for
% 2-D input label matrices.
%
% See also MeasureImageAreaOccupied.


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
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the objects that you want to measure?
%choiceVAR01 = Do not use
%infotypeVAR01 = objectgroup
ObjectNameList{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 =
%choiceVAR02 = Do not use
%infotypeVAR02 = objectgroup
ObjectNameList{2} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 =
%choiceVAR03 = Do not use
%infotypeVAR03 = objectgroup
ObjectNameList{3} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 =
%choiceVAR04 = Do not use
%infotypeVAR04 = objectgroup
ObjectNameList{4} = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 =
%choiceVAR05 = Do not use
%infotypeVAR05 = objectgroup
ObjectNameList{5} = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 =
%choiceVAR06 = Do not use
%infotypeVAR06 = objectgroup
ObjectNameList{6} = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 =
%choiceVAR07 = Do not use
%infotypeVAR07 = objectgroup
ObjectNameList{7} = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = Would you like to calculate the Zernike features for each object (with lots of objects, this can be very slow)?
%choiceVAR08 = Yes
%choiceVAR08 = No
ZernikeChoice = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%%%%%%%%%%%%%%%%%
%%% Features  %%%
%%%%%%%%%%%%%%%%%

if nargin > 1 
    switch varargin{1}
%feature:categories
        case 'categories'
            if nargin == 1 || ismember(varargin{2},ObjectNameList)
                result = { 'AreaShape' };
            else
                result = {};
            end

%feature:measurements
        case 'measurements'
            if ismember(varargin{2},ObjectNameList) && ...
                    strcmp(varargin{3},'AreaShape')
                result = {...
                    'Area','Eccentricity','Solidity','Extent','EulerNumber',...
                    'Perimeter','FormFactor','MajorAxisLength',...
                    'MinorAxisLength','Orientation' };
                if strcmp(ZernikeChoice,'Yes')
                    for i = 0:9
                        result = [result,...
                            arrayfun(@(x) {sprintf(sprintf('Zernike_%d_%%d',i),x)},(0:i)*2+mod(i+2,2))]; %#ok<AGROW>
                    end
                end
            else
                result = {};
            end
        otherwise
            error(['Unhandled category: ',varargin{1}]);
    end
    handles=result;
    return;
end

%%%VariableRevisionNumber = 3

%%% Set up the window for displaying the results
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber);
    CPfigure(handles,'Text',ThisModuleFigureNumber);
    columns = 1;
end

%%% Retrieves the pixel size that the user entered (micrometers per pixel).
PixelSize = str2double(handles.Settings.PixelSize);

SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

%%% START LOOP THROUGH ALL THE OBJECTS
for i = 1:length(ObjectNameList)
    ObjectName = ObjectNameList{i};
    if strcmp(ObjectName,'Do not use')
        continue
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    drawnow

    %%% Retrieves the label matrix image that contains the segmented
    %%% objects which will be measured with this module.
    LabelMatrixImage =  CPretrieveimage(handles,['Segmented', ObjectName],ModuleName,'MustBeGray','DontCheckScale');

    %%%
    %%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
    %%%
    drawnow

    NumObjects = max(LabelMatrixImage(:));
    BasicFeatures = {'Area', 'Eccentricity', 'Solidity', 'Extent', ...
        'EulerNumber', 'Perimeter', 'FormFactor',...
        'MajorAxisLength', 'MinorAxisLength', 'Orientation'};
    if  NumObjects > 0

        %%% Get the basic shape features, excluding FormFactor
        warning('off', 'MATLAB:divideByZero'); %%% Matlab failing atan vs atan2 in regionprops line 672.
        props = regionprops(LabelMatrixImage, BasicFeatures(~strcmp(BasicFeatures,'FormFactor')));
        warning('on', 'MATLAB:divideByZero'); %%% Matlab failing atan vs atan2 in regionprops line 672.
        % Add 1 to perimeter to avoid divide by zero
        FormFactor = (4*pi*cat(1,props.Area)) ./ ((cat(1,props.Perimeter)+1).^2);
        Basic = [cat(1,props.Area)*PixelSize^2,...
            cat(1,props.Eccentricity),...
            cat(1,props.Solidity),...
            cat(1,props.Extent),...
            cat(1,props.EulerNumber),...
            cat(1,props.Perimeter)*PixelSize,...
            FormFactor,...
            cat(1,props.MajorAxisLength)*PixelSize,...
            cat(1,props.MinorAxisLength)*PixelSize,...
            cat(1,props.Orientation)];

        % Save basic shape features.
        for j = 1:length(BasicFeatures)
            handles = CPaddmeasurements(handles, ObjectName, ...
                ['AreaShape_', BasicFeatures{j}], ...
                Basic(:, j));
        end

        if strcmp(ZernikeChoice,'Yes')
            % Get the Zernike features
            [Zernike, ZernikeFeatures] = calculate_zernike(LabelMatrixImage);

            % Save Zernike measurements
            for j=1:length(ZernikeFeatures)
                handles = CPaddmeasurements(handles, ObjectName, ...
                    ZernikeFeatures{j}, Zernike(:,j));
            end
        end
    else
        % If there are no objects, write in an empty
        for j = 1:length(BasicFeatures)
            handles = CPaddmeasurements(handles, ObjectName, ...
                ['AreaShape_', BasicFeatures{j}], []);
        end
        
        if strcmp(ZernikeChoice,'Yes')
            % Here, just getting the feaure names out; I don't care about the
            % values, and the subfunction doesn't error if the label matrix is
            % empty
            [Zernike, ZernikeFeatures] = calculate_zernike(LabelMatrixImage);
            for j = 1:length(ZernikeFeatures)
                handles = CPaddmeasurements(handles, ObjectName, ...
                    ZernikeFeatures{j}, []);
            end
        end
    end

    %%%
    %%% Display measurements
    %%%
    FontSize = handles.Preferences.FontSize;
    if any(findobj == ThisModuleFigureNumber)
        % Remove uicontrols from last cycle
        delete(findobj(ThisModuleFigureNumber,'tag','TextUIControl'));
        
        if SetBeingAnalyzed == handles.Current.StartingImageSet
            delete(findobj('parent',ThisModuleFigureNumber,'string','R'));
            delete(findobj('parent',ThisModuleFigureNumber,'string','G'));
            delete(findobj('parent',ThisModuleFigureNumber,'string','B'));
        end

        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0 0.95 1 0.04],...
            'HorizontalAlignment','center','Backgroundcolor',[.7 .7 .9],'fontname','Helvetica','tag','TextUIControl',...
            'fontsize',FontSize,'fontweight','bold','string',sprintf('Average shape features for cycle #%d',handles.Current.SetBeingAnalyzed),'UserData',handles.Current.SetBeingAnalyzed);

        % Number of objects
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.85 0.25 0.03],...
            'HorizontalAlignment','left','Backgroundcolor',[.7 .7 .9],'fontname','Helvetica','tag','TextUIControl',...
            'fontsize',FontSize,'fontweight','bold','string','Number of objects:','UserData',handles.Current.SetBeingAnalyzed);

        % Text for Basic features
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8 0.25 0.03],...
            'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextUIControl',...
            'fontsize',FontSize,'fontweight','bold','string','Basic features:','UserData',handles.Current.SetBeingAnalyzed);
        for k = 1:length(BasicFeatures)
            uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8-0.04*k 0.25 0.03],...
                'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextUIControl',...
                'fontsize',FontSize,'string',BasicFeatures{k},'UserData',handles.Current.SetBeingAnalyzed);
        end

        if strcmp(ZernikeChoice,'Yes')
            % Text for Zernike features
            uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.35 0.25 0.03],...
                'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextUIControl',...
                'fontsize',FontSize,'fontweight','bold','string','First 5 Zernike features:','UserData',handles.Current.SetBeingAnalyzed);
            
            for k = 1:5 %% Only displaying the first 5 for space considerations
                uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.35-0.04*k 0.25 0.03],...
                    'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextUIControl',...
                    'fontsize',FontSize,'string',ZernikeFeatures{k},'UserData',handles.Current.SetBeingAnalyzed);
            end
        end
        % Second column (numbers)
        % The name of the object image
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.3+0.1*(columns-1) 0.9 0.1 0.03],...
            'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextEachObjUIControl',...
            'fontsize',FontSize,'fontweight','bold','string',ObjectName,'UserData',handles.Current.SetBeingAnalyzed);

        % Number of objects
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.3+0.1*(columns-1) 0.85 0.1 0.03],...
            'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextEachObjUIControl',...
            'fontsize',FontSize,'string',num2str(max(LabelMatrixImage(:))),'UserData',handles.Current.SetBeingAnalyzed);

        % Report features, if there are any.
        % Plot zeros if no objects
        if max(LabelMatrixImage(:)) < 1 
            Basic = zeros(length(ObjectNameList),length(BasicFeatures)+1);
            Zernike = zeros(length(ObjectNameList),5);
        end
        % Basic shape features
        if NumObjects > 0
            for k = 1:size(Basic,2)
                uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.3+0.1*(columns-1) 0.8-0.04*k 0.1 0.03],...
                    'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextEachObjUIControl',...
                    'fontsize',FontSize,'string',sprintf('%0.2f',mean(Basic(:,k))),'UserData',handles.Current.SetBeingAnalyzed);
            end

            if strcmp(ZernikeChoice,'Yes')
                % Zernike shape features
                for k = 1:5
                    uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.3+0.1*(columns-1) 0.35-0.04*k 0.1 0.03],...
                        'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica','tag','TextEachObjUIControl',...
                        'fontsize',FontSize,'string',sprintf('%0.2f',mean(Zernike(:,k))),'UserData',handles.Current.SetBeingAnalyzed);
                end
            end
        end

        % This variable is used to write results in the correct column
        % and to determine the correct window size
        columns = columns + 1;
    end
end

%%%
%%% Subfunction for calculating Zernike
%%%

function [Zernike, ZernikeFeatures] = calculate_zernike(LabelMatrixImage)
NumObjects = max(LabelMatrixImage(:));
% Get index for Zernike functions
Zernikeindex = [];
ZernikeFeatures = {};
for n = 0:9
    for m = 0:n
        if rem(n-m,2) == 0
            Zernikeindex = [Zernikeindex;n m];
            ZernikeFeatures = cat(2,ZernikeFeatures,{sprintf('Zernike_%d_%d',n,m)});
        end
    end
end
lut = construct_lookuptable_Zernike(Zernikeindex);

Zernike = zeros(NumObjects,size(Zernikeindex,1));

for Object = 1:NumObjects
    %%% Calculate Zernike shape features
    [xcord,ycord] = find(LabelMatrixImage==Object);
    %%% It's possible for objects not to have any pixels,
    %%% particularly tertiary objects (such as cytoplasm from
    %%% cells the exact same size as their nucleus).
    if isempty(xcord),
        % no need to create an empty line of data, as that's
        % already done above.
        continue;
    end
    diameter = max((max(xcord)-min(xcord)+1),(max(ycord)-min(ycord)+1));

    if rem(diameter,2)== 0
        % An odd number facilitates implementation
        diameter = diameter + 1;
    end

    % Calculate the Zernike basis functions
    [x,y] = meshgrid(linspace(-1,1,diameter),linspace(-1,1,diameter));
    r = sqrt(x.^2+y.^2);
    phi = atan(y./(x+eps));
    % It is necessary to normalize the bases by area.
    normalization = sum(r(:) <= 1);
    % this happens for diameter == 1
    if (normalization == 0.0),
        normalization = 1.0;
    end

    Zf = zeros(diameter,diameter,size(Zernikeindex,1));

    for k = 1:size(Zernikeindex,1)
        n = Zernikeindex(k,1);
        m = Zernikeindex(k,2); % m = 0,1,2,3,4,5,6,7,8, or 9

        % Optimized
        s_new = zeros(size(x));
        exp_term = exp(sqrt(-1)*m*phi);
        lv_index = [0 : (n-m)/2];
        for i = 1: length(lv_index)
            lv = lv_index(i);
            s_new = s_new + lut(k,i) * r.^(n-2*lv).*exp_term;
        end
        s = s_new;

        s(r>1) = 0;
        Zf(:,:,k) = s / normalization;
    end

    % Get image patch, with offsets to center relative to the Zernike bases
    BWpatch = zeros(diameter, diameter);
    height = max(xcord) - min(xcord) + 1;
    width = max(ycord) - min(ycord) + 1;
    row_offset = floor((diameter - height) / 2) + 1;
    col_offset = floor((diameter - width) / 2) + 1;
    BWpatch(row_offset:(row_offset+height-1), col_offset:(col_offset+width-1)) = (LabelMatrixImage(min(xcord):max(xcord), min(ycord):max(ycord)) == Object);

    % Apply Zernike functions
    try
        Zernike(Object,:) = squeeze(abs(sum(sum(repmat(BWpatch,[1 1 size(Zernikeindex,1)]).*Zf))))';
    catch
        Zernike(Object,:) = 0;
        display(sprintf([ObjectName,' number ',num2str(Object),' was too big to be calculated. Batch Error! (this is included so it can be caught during batch processing without quitting out of the analysis)']))
    end
end

%%%
%%% Subfunctions for optimized Zernike
%%%

%function previousely_calculated_value = lookuptable(lv,m,n)
%previousely_calculated_value = (-1)^lv*fak_table(n-lv)/( fak_table(lv) * fak_table((n+m)/2-lv) * fak_table((n-m)/2-lv));

% Zernikeindex =
%      0     0
%      1     1
%      2     0
%      2     2
%      3     1
%      3     3
%      4     0
%      4     2
%      4     4
%      5     1
%      5     3
%      5     5
%      6     0
%      6     2
%      6     4
%      6     6
%      7     1
%      7     3
%      7     5
%      7     7
%      8     0
%      8     2
%      8     4
%      8     6
%      8     8
%      9     1
%      9     3
%      9     5
%      9     7
%      9     9
function lut = construct_lookuptable_Zernike(Zernikeindex)
for k = 1:size(Zernikeindex,1)
    n = Zernikeindex(k,1);
    m = Zernikeindex(k,2); % m = 0,1,2,3,4,5,6,7,8, or 9
    lv_index = [0 : (n-m)/2];
    for i = 1 : length(lv_index)
        lv = lv_index(i);
        lut(k, i) = (-1)^lv*fak_table(n-lv)/( fak_table(lv) * fak_table((n+m)/2-lv) * fak_table((n-m)/2-lv));
    end
end

function f = fak_table(n)
switch n
    case 0
        f = 1;
    case 1
        f = 1;
    case 2
        f = 2;
    case 3
        f = 6;
    case 4
        f = 24;
    case 5
        f = 120;
    case 6
        f = 720;
    case 7
        f = 5040;
    case 8
        f = 40320;
    case 9
        f = 362880;
    otherwise
        f = NaN; %
end