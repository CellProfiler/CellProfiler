function handles = MeasureOrientation(handles)

% Help for the Measure Orientation module:
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
% Area                    |       1
% Eccentricity            |       2
% Solidity                |       3
% Extent                  |       4
% Euler Number            |       5
% Perimeter               |       6
% Form Factor             |       7
% MajorAxisLength         |       8
% MinorAxisLength         |       9
% Orientation             |       10
%
% Details about how measurements are calculated:
% This module retrieves objects in label matrix format and measures them.
% The label matrix image should be "compacted": that is, each number should
% correspond to an object, with no numbers skipped. So, if some objects
% were discarded from the label matrix image, the image should be converted
% to binary and re-made into a label matrix image before feeding into this
% module.
%
% The following measurements are extracted using the MATLAB regionprops.m
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
% *Euler number - Equal to the number of objects in the image minus the
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
%   Susan Ma
%
% Website: http://www.cellprofiler.org
%
% $Revision: 3261 $

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

%%%VariableRevisionNumber = 2

%%% Set up the window for displaying the results
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber);
    CPfigure(handles,'Text',ThisModuleFigureNumber);
    columns = 1;
end

%%% Retrieves the pixel size that the user entered (micrometers per pixel).
PixelSize = str2double(handles.Settings.PixelSize);

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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    drawnow

    %%% Initialize
    Basic = [];
    BasicFeatures    = {'Area',...
        'Eccentricity',...
        'Solidity',...
        'Extent',...
        'Euler number',...
        'Perimeter',...
        'Form Factor',...
        'MajorAxisLength',...
        'MinorAxisLength'...
        'Orientation'};

    NumObjects = max(LabelMatrixImage(:));
    if  NumObjects > 0

        %%% Get the basic shape features
        props = regionprops(LabelMatrixImage,'Area','Eccentricity','Solidity','Extent','EulerNumber',...
            'MajorAxisLength','MinorAxisLength','Perimeter','Orientation');

        % Form factor
        FormFactor = (4*pi*cat(1,props.Area)) ./ ((cat(1,props.Perimeter)+1).^2);       % Add 1 to perimeter to avoid divide by zero

        % Save basic shape features
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
    else
        Basic = zeros(1,10);
    end

    %%% Save measurements
    handles.Measurements.(ObjectName).AreaShapeFeatures = BasicFeatures;
    handles.Measurements.(ObjectName).AreaShape(handles.Current.SetBeingAnalyzed) = {Basic};

    %%% Report measurements
    FontSize = handles.Preferences.FontSize;
    if any(findobj == ThisModuleFigureNumber)
        if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
            delete(findobj('parent',ThisModuleFigureNumber,'string','R'));
            delete(findobj('parent',ThisModuleFigureNumber,'string','G'));
            delete(findobj('parent',ThisModuleFigureNumber,'string','B'));
        end

        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0 0.95 1 0.04],...
            'HorizontalAlignment','center','Backgroundcolor',[.7 .7 .9],'fontname','Helvetica',...
            'fontsize',FontSize,'fontweight','bold','string',sprintf('Average shape features for cycle #%d',handles.Current.SetBeingAnalyzed),'UserData',handles.Current.SetBeingAnalyzed);

        % Number of objects
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.85 0.25 0.03],...
            'HorizontalAlignment','left','Backgroundcolor',[.7 .7 .9],'fontname','Helvetica',...
            'fontsize',FontSize,'fontweight','bold','string','Number of objects:','UserData',handles.Current.SetBeingAnalyzed);

        % Text for Basic features
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8 0.25 0.03],...
            'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica',...
            'fontsize',FontSize,'fontweight','bold','string','Basic features:','UserData',handles.Current.SetBeingAnalyzed);
        for k = 1:10
            uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8-0.04*k 0.25 0.03],...
                'HorizontalAlignment','left','BackgroundColor',[.7 .7 .9],'fontname','Helvetica',...
                'fontsize',FontSize,'string',BasicFeatures{k},'UserData',handles.Current.SetBeingAnalyzed);
        end

        % The name of the object image
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.3+0.1*(columns-1) 0.9 0.1 0.03],...
            'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica',...
            'fontsize',FontSize,'fontweight','bold','string',ObjectName,'UserData',handles.Current.SetBeingAnalyzed);

        % Number of objects
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.3+0.1*(columns-1) 0.85 0.1 0.03],...
            'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica',...
            'fontsize',FontSize,'string',num2str(max(LabelMatrixImage(:))),'UserData',handles.Current.SetBeingAnalyzed);

        % Report features, if there are any.
        if max(LabelMatrixImage(:)) > 0
            % Basic shape features
            for k = 1:10
                uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.3+0.1*(columns-1) 0.8-0.04*k 0.1 0.03],...
                    'HorizontalAlignment','center','BackgroundColor',[.7 .7 .9],'fontname','Helvetica',...
                    'fontsize',FontSize,'string',sprintf('%0.2f',mean(Basic(:,k))),'UserData',handles.Current.SetBeingAnalyzed);
            end
        end
        % This variable is used to write results in the correct column
        % and to determine the correct window size
        columns = columns + 1;
    end
end

function f = fak(n)
if n==0
    f = 1;
else
    f = prod(1:n);
end