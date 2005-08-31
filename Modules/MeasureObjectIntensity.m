function handles = MeasureObjectIntensity(handles)

% Help for the Measure Object Intensity module:
% Category: Measurement
%
% Given an image with objects identified (e.g. nuclei or cells), this
% module extracts intensity features of each
% object based on a corresponding grayscale image. Measurements are
% recorded for each object.
%
% How it works:
% Retrieves a segmented image, in label matrix format, and a
% corresponding original grayscale image and makes measurements of the
% objects that are segmented in the image. The label matrix image should
% be "compacted": that is, each number should correspond to an object,
% with no numbers skipped.
% So, if some objects were discarded from the label matrix image, the
% image should be converted to binary and re-made into a label matrix
% image before feeding into this module.
%
% See also MEASURETEXTURE, MEASURESHAPE, MEASURECORRELATION,

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
%
% $Revision$


%

drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%



%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the greyscale images you want to measure?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What did you call the segmented objects that you want to measure?
%choiceVAR02 = Do not use
%infotypeVAR02 = objectgroup
ObjectNameList{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = Type / in unused boxes.
%choiceVAR03 = Do not use
%infotypeVAR03 = objectgroup
ObjectNameList{2} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 =
%choiceVAR04 = Do not use
%infotypeVAR04 = objectgroup
ObjectNameList{3} = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%%%VariableRevisionNumber = 1

%%% Set up the window for displaying the results
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber);
    CPfigure(handles,ThisModuleFigureNumber);
    set(ThisModuleFigureNumber,'color',[1 1 1])
    columns = 1;
end

%%% START LOOP THROUGH ALL THE OBJECTS
for i = 1:3
    
    ObjectName = ObjectNameList{i};
    if strcmp(ObjectName,'Do not use') == 1
        continue
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%% Reads (opens) the image you want to analyze and assigns it to a variable,
    %%% "OrigImage".
    fieldname = ['', ImageName];
    %%% Checks whether the image exists in the handles structure.
    if isfield(handles.Pipeline, fieldname) == 0,
        error(['Image processing has been canceled. Prior to running the Measure Intensity module, you must have previously run a module that loads a greyscale image.  You specified in the MeasureObjectIntensity module that the desired image was named ', ImageName, ' which should have produced an image in the handles structure called ', fieldname, '. The Measure Intensity module cannot locate this image.']);
    end
    OrigImage = handles.Pipeline.(fieldname);


    %%% Checks that the original image is two-dimensional (i.e. not a color
    %%% image), which would disrupt several of the image functions.
    if ndims(OrigImage) ~= 2
        s = size(OrigImage);
        if (length(s) == 3 && s(3) == 3)
            OrigImage = OrigImage(:,:,1)+OrigImage(:,:,2)+OrigImage(:,:,3);
        else
            error('Image processing was canceled because the Identify Primary Intensity module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
        end
    end

    %%% Retrieves the label matrix image that contains the segmented objects which
    %%% will be measured with this module.
    fieldname = ['Segmented', ObjectName];
    %%% Checks whether the image exists in the handles structure.
    if isfield(handles.Pipeline, fieldname) == 0,
        error(['Image processing has been canceled. Prior to running the Measure Intensity module, you must have previously run a module that generates an image with the objects identified.  You specified in the Measure Intensity module that the primary objects were named ',ObjectName,' which should have produced an image in the handles structure called ', fieldname, '. The Measure Intensity module cannot locate this image.']);
    end
    LabelMatrixImage = handles.Pipeline.(fieldname);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





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
    props = regionprops(LabelMatrixImage,'PixelIdxList');
    ObjectCount = length(props);
    if ObjectCount > 0

        Basic = zeros(ObjectCount,4);

        [sr sc] = size(LabelMatrixImage);
        for Object = 1:ObjectCount

            %%% Measure basic set of Intensity features
            Basic(Object,1) = sum(OrigImage(props(Object).PixelIdxList));
            Basic(Object,2) = mean(OrigImage(props(Object).PixelIdxList));
            Basic(Object,3) = std(OrigImage(props(Object).PixelIdxList));
            Basic(Object,4) = min(OrigImage(props(Object).PixelIdxList));
            Basic(Object,5) = max(OrigImage(props(Object).PixelIdxList));

            %%% Cut patch so that we don't have to deal with entire image
            [r,c] = ind2sub([sr sc],props(Object).PixelIdxList);
            rmax = min(sr,max(r));
            rmin = max(1,min(r));
            cmax = min(sc,max(c));
            cmin = max(1,min(c));
            BWim   = LabelMatrixImage(rmin:rmax,cmin:cmax) == Object;
            Greyim = OrigImage(rmin:rmax,cmin:cmax);

            % Get perimeter in order to calculate edge features
            perim = bwperim(BWim);
            perim = Greyim(find(perim));
            Basic(Object,6)  = sum(perim);
            Basic(Object,7)  = mean(perim);
            Basic(Object,8)  = std(perim);
            Basic(Object,9)  = min(perim);
            Basic(Object,10) = max(perim);

            % Calculate the Mass displacment (taking the pixelsize into account), which is the distance between
            % the center of gravity in the gray level image and the binary
            % image.
            PixelSize = str2double(handles.Settings.PixelSize);
            BWx = sum([1:size(BWim,2)].*sum(BWim,1))/sum([1:size(BWim,2)]);
            BWy = sum([1:size(BWim,1)]'.*sum(BWim,2))/sum([1:size(BWim,1)]);
            Greyx = sum([1:size(Greyim,2)].*sum(Greyim,1))/sum([1:size(Greyim,2)]);
            Greyy = sum([1:size(Greyim,1)]'.*sum(Greyim,2))/sum([1:size(Greyim,1)]);
            Basic(Object,11) = sqrt((BWx-Greyx)^2+(BWy-Greyy)^2)*PixelSize;
        end
    end
    %%% Save measurements
    handles.Measurements.(ObjectName).(['Intensity_',ImageName,'Features']) = BasicFeatures;
    handles.Measurements.(ObjectName).(['Intensity_',ImageName])(handles.Current.SetBeingAnalyzed) = {Basic};


    %%% Report measurements
    FontSize = handles.Current.FontSize;

    if any(findobj == ThisModuleFigureNumber);
        % This first block writes the same text several times
        % Header
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0 0.95 1 0.04],...
            'HorizontalAlignment','center','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',FontSize,'fontweight','bold','string',sprintf('Average intensity features for image set #%d',handles.Current.SetBeingAnalyzed));

        % Number of objects
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.85 0.3 0.03],...
            'HorizontalAlignment','left','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',FontSize,'fontweight','bold','string','Number of objects:');

        % Text for Basic features
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8 0.3 0.03],...
            'HorizontalAlignment','left','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',FontSize,'fontweight','bold','string','Intensity feature:');
        for k = 1:11
            uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8-0.04*k 0.3 0.03],...
                'HorizontalAlignment','left','BackgroundColor',[1 1 1],'fontname','times',...
                'fontsize',FontSize,'string',BasicFeatures{k});
        end

        % The name of the object image
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.35+0.2*(columns-1) 0.9 0.2 0.03],...
            'HorizontalAlignment','center','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',FontSize,'fontweight','bold','string',ObjectName);

        % Number of objects
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.35+0.2*(columns-1) 0.85 0.2 0.03],...
            'HorizontalAlignment','center','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',FontSize,'string',num2str(ObjectCount));

        if ObjectCount > 0
            % Basic features
            for k = 1:11
                uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.35+0.2*(columns-1) 0.8-0.04*k 0.2 0.03],...
                    'HorizontalAlignment','center','BackgroundColor',[1 1 1],'fontname','times',...
                    'fontsize',FontSize,'string',sprintf('%0.2f',mean(Basic(:,k))));
            end
        end
        % This variable is used to write results in the correct column
        % and to determine the correct window size
        columns = columns + 1;
    end
end
drawnow


