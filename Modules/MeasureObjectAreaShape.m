function handles = MeasureAreaShape(handles)

% Help for the Measure Shape module:
% Category: Measurement
%
% Given an image with objects identified (e.g. nuclei or cells), this
% module extracts area and shape features of each object. Note that shape features
% are only reliable for objects that are inside the image borders.
%
% How it works:
% Retrieves a segmented image, in label matrix format and makes measurements
% of the objects that are segmented in the image. The label matrix image
% should be "compacted": that is, each number should correspond to an object,
% with no numbers skipped. So, if some objects were discarded from the label
% matrix image, the image should be converted to binary and re-made into a
% label matrix image before feeding into this module.
%
% See also MeasureTexture, MeasureIntensity, MeasureCorrelation

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

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%infotypeVAR01 = objectgroup
%textVAR01 = What did you call the segmented objects that you want to measure?
%choiceVAR01 = Do not use
ObjectNameList{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%infotypeVAR02 = objectgroup
%textVAR02 = 
%choiceVAR02 = Do not use
ObjectNameList{2} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%infotypeVAR03 = objectgroup
%textVAR03 = 
%choiceVAR03 = Do not use
ObjectNameList{3} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%%%VariableRevisionNumber = 01


%%% Set up the window for displaying the results
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber);
    CPfigure(handles,ThisModuleFigureNumber);
    set(ThisModuleFigureNumber,'color',[1 1 1])
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

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


    %%% Retrieves the label matrix image that contains the segmented objects which
    %%% will be measured with this module.
    fieldname = ['Segmented', ObjectName];
    %%% Checks whether the image exists in the handles structure.
    if isfield(handles.Pipeline, fieldname) == 0,
        error(['Image processing has been canceled. Prior to running the Measure Shape module, you must have previously run a module that generates an image with the objects identified.  You specified in the Measure Shape module that the primary objects were named ',ObjectName,' which should have produced an image in the handles structure called ', fieldname, '. The Measure Shape module cannot locate this image.']);
    end
    LabelMatrixImage = handles.Pipeline.(fieldname);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% MAKE MEASUREMENTS & SAVE TO HANDLES STRUCTURE %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%% Initialize
    Basic = [];
    BasicFeatures    = {'Area',...
        'Eccentricity',...
        'Solidity',...
        'Extent',...
        'Euler number',...
        'Perimeter',...
        'Form factor',...
        'MajorAxisLength',...
        'MinorAxisLength'};

    % Get index for Zernike functions
    Zernike = [];
    Zernikeindex = [];
    ZernikeFeatures = {};
    for n = 0:9
        for m = 0:n
            if rem(n-m,2) == 0
                Zernikeindex = [Zernikeindex;n m];
                ZernikeFeatures = cat(2,ZernikeFeatures,{sprintf('Zernike%d_%d',n,m)});
            end
        end
    end

    NumObjects = max(LabelMatrixImage(:));
    if  NumObjects> 0

        %%% Calculate Zernike shape features
        % Use ConvexArea to automatically calculate the average equivalent diameter
        % of the objects, and then use this diameter to determine the grid size
        % of the Zernike functions
        tmp = regionprops(LabelMatrixImage,'ConvexArea');
        diameter = floor(sqrt(4/pi*mean(cat(1,tmp.ConvexArea)))+1);
        if rem(diameter,2)== 0, diameter = diameter + 1;end   % An odd number facilitates implementation

        % Calculate the Zernike basis functions
        [x,y] = meshgrid(linspace(-1,1,diameter),linspace(-1,1,diameter));
        r = sqrt(x.^2+y.^2);
        phi = atan(y./(x+eps));
        Zf = zeros(size(x,1),size(x,2),size(Zernikeindex,1));

        for k = 1:size(Zernikeindex,1)
            n = Zernikeindex(k,1);
            m = Zernikeindex(k,2);
            s = zeros(size(x));
            for l = 0:(n-m)/2;
                s  = s + (-1)^l*fak(n-l)/( fak(l) * fak((n+m)/2-l) * fak((n-m)/2-l)) * r.^(n-2*l).*exp(sqrt(-1)*m*phi);
            end
            s(r>1) = 0;
            Zf(:,:,k) = s;
        end

        % Pad the Label image with zeros so that the Zernike
        % features can be calculated also for objects close to
        % the border
        [sr,sc] = size(LabelMatrixImage);
        PaddedLabelMatrixImage = [zeros(diameter,2*diameter+sc);
            zeros(sr,diameter) LabelMatrixImage zeros(sr,diameter)
            zeros(diameter,2*diameter+sc)];

        % Loop over objects to calculate Zernike moments. Center the functions
        % over the centroids of the objects.
        tmp = regionprops(PaddedLabelMatrixImage,'Centroid');
        Centroids = cat(1,tmp.Centroid);
        Zernike = zeros(NumObjects,size(Zernikeindex,1));
        Perimeter = zeros(NumObjects,1);
        for Object = 1:NumObjects

            % Get image patch
            cx = round(Centroids(Object,1));
            cy = round(Centroids(Object,2));
            rmax = round(Centroids(Object,2)+(diameter-1)/2);
            rmin = round(Centroids(Object,2)-(diameter-1)/2);
            cmax = round(Centroids(Object,1)+(diameter-1)/2);
            cmin = round(Centroids(Object,1)-(diameter-1)/2);
            BWpatch   = PaddedLabelMatrixImage(rmin:rmax,cmin:cmax) == Object;

            % Apply Zernike functions
            Zernike(Object,:) = squeeze(abs(sum(sum(repmat(BWpatch,[1 1 size(Zernikeindex,1)]).*Zf))))';

            % Get perimeter for object
            perim = bwperim(BWpatch);
            Perimeter(Object) = sum(perim(:));

        end
     
        %%% Get the basic shape features
        props = regionprops(LabelMatrixImage,'Area','Eccentricity','Solidity','Extent','EulerNumber',...
            'MajorAxisLength','MinorAxisLength');

        % Form factor
        FormFactor = 4*pi*cat(1,props.Area) ./ Perimeter.^2;

        % Save basic shape features
        Basic = [cat(1,props.Area)*PixelSize^2,...
            cat(1,props.Eccentricity),...
            cat(1,props.Solidity),...
            cat(1,props.Extent),...
            cat(1,props.EulerNumber),...
            Perimeter,...
            FormFactor,...
            cat(1,props.MajorAxisLength),...
            cat(1,props.MinorAxisLength)];
    end

    %%% Save measurements
    handles.Measurements.(ObjectName).AreaShapeFeatures = cat(2,BasicFeatures,ZernikeFeatures);
    handles.Measurements.(ObjectName).AreaShape(handles.Current.SetBeingAnalyzed) = {[Basic Zernike]};

    %%% Report measurements
    FontSize = get(0,'UserData');
    if any(findobj == ThisModuleFigureNumber);
        % This first block writes the same text several times
        % Header
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0 0.95 1 0.04],...
            'HorizontalAlignment','center','Backgroundcolor',[1 1 1],'fontname','times',...
            'fontsize',10,'fontweight','bold','string',sprintf('Average shape features for image set #%d',handles.Current.SetBeingAnalyzed));

        % Number of objects
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.85 0.3 0.03],...
            'HorizontalAlignment','left','Backgroundcolor',[1 1 1],'fontname','times',...
            'fontsize',FontSize,'fontweight','bold','string','Number of objects:');

        % Text for Basic features
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8 0.3 0.03],...
            'HorizontalAlignment','left','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',FontSize,'fontweight','bold','string','Basic features:');
        for k = 1:7
            uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8-0.04*k 0.3 0.03],...
                'HorizontalAlignment','left','BackgroundColor',[1 1 1],'fontname','times',...
                'fontsize',FontSize,'string',BasicFeatures{k});
        end

        % Text for Zernike features
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.45 0.3 0.03],...
            'HorizontalAlignment','left','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',FontSize,'fontweight','bold','string','5 first Zernike features:');
        for k = 1:5
            uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.45-0.04*k 0.3 0.03],...
                'HorizontalAlignment','left','BackgroundColor',[1 1 1],'fontname','times',...
                'fontsize',FontSize,'string',ZernikeFeatures{k});
        end


        % The name of the object image
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.3+0.2*(columns-1) 0.9 0.2 0.03],...
            'HorizontalAlignment','center','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',FontSize,'fontweight','bold','string',ObjectName);

        % Number of objects
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.3+0.2*(columns-1) 0.85 0.2 0.03],...
            'HorizontalAlignment','center','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',FontSize,'string',num2str(max(LabelMatrixImage(:))));

        % Report features, if there are any.
        if max(LabelMatrixImage(:)) > 0
            % Basic shape features
            for k = 1:7
                q = uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.3+0.2*(columns-1) 0.8-0.04*k 0.2 0.03],...
                    'HorizontalAlignment','center','BackgroundColor',[1 1 1],'fontname','times',...
                    'fontsize',FontSize,'string',sprintf('%0.2f',mean(Basic(:,k))));
            end

            % Zernike shape features
            for k = 1:5
                q = uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.3+0.2*(columns-1) 0.45-0.04*k 0.2 0.03],...
                    'HorizontalAlignment','center','BackgroundColor',[1 1 1],'fontname','times',...
                    'fontsize',FontSize,'string',sprintf('%0.2f',mean(Zernike(:,k))));
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
