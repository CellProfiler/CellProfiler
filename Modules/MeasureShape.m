function handles = MeasureShape(handles)

% Help for the Measure Shape module:
% Category: Measurement
%
% Given an image with objects identified (e.g. nuclei or cells), this
% module extracts shape features of each object. Note that shape features
% can only be obtained from objects that are inside the image borders.
%
% How it works:
% Retrieves a segmented image, in label matrix format and makes measurements
% of the objects that are segmented in the image. The label matrix image
% should be "compacted": that is, each number should correspond to an object,
% with no numbers skipped. So, if some objects were discarded from the label
% matrix image, the image should be converted to binary and re-made into a
% label matrix image before feeding into this module.
%
% See also MEASURETEXTURE, MEASURECORRELATION,
%
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

% PROGRAMMING NOTE
% HELP:
% The first unbroken block of lines will be extracted as help by
% CellProfiler's 'Help for this analysis module' button as well as Matlab's
% built in 'help' and 'doc' functions at the command line. It will also be
% used to automatically generate a manual page for the module. An example
% image demonstrating the function of the module can also be saved in tif
% format, using the same name as the module, and it will automatically be
% included in the manual page as well.  Follow the convention of: purpose
% of the module, description of the variables and acceptable range for
% each, how it works (technical description), info on which images can be
% saved, and See also CAPITALLETTEROTHERMODULES. The license/author
% information should be separated from the help lines with a blank line so
% that it does not show up in the help displays.  Do not change the
% programming notes in any modules! These are standard across all modules
% for maintenance purposes, so anything module-specific should be kept
% separate.
%
% PROGRAMMING NOTE
% DRAWNOW:
% The 'drawnow' function allows figure windows to be updated and
% buttons to be pushed (like the pause, cancel, help, and view
% buttons).  The 'drawnow' function is sprinkled throughout the code
% so there are plenty of breaks where the figure windows/buttons can
% be interacted with.  This does theoretically slow the computation
% somewhat, so it might be reasonable to remove most of these lines
% when running jobs on a cluster where speed is important.
drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

% PROGRAMMING NOTE
% VARIABLE BOXES AND TEXT:
% The '%textVAR' lines contain the variable descriptions which are
% displayed in the CellProfiler main window next to each variable box.
% This text will wrap appropriately so it can be as long as desired.
% The '%defaultVAR' lines contain the default values which are
% displayed in the variable boxes when the user loads the module.
% The line of code after the textVAR and defaultVAR extracts the value
% that the user has entered from the handles structure and saves it as
% a variable in the workspace of this module with a descriptive
% name. The syntax is important for the %textVAR and %defaultVAR
% lines: be sure there is a space before and after the equals sign and
% also that the capitalization is as shown.
% CellProfiler uses VariableRevisionNumbers to help programmers notify
% users when something significant has changed about the variables.
% For example, if you have switched the position of two variables,
% loading a pipeline made with the old version of the module will not
% behave as expected when using the new version of the module, because
% the settings (variables) will be mixed up. The line should use this
% syntax, with a two digit number for the VariableRevisionNumber:
% '%%%VariableRevisionNumber = 01'  If the module does not have this
% line, the VariableRevisionNumber is assumed to be 00.  This number
% need only be incremented when a change made to the modules will affect
% a user's previously saved settings. There is a revision number at
% the end of the license info at the top of the m-file for revisions
% that do not affect the user's previously saved settings files.

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the segmented objects that you want to measure?
%defaultVAR01 = Nuclei
ObjectNameList{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Type / in unused boxes.
%defaultVAR02 = Cells
ObjectNameList{2} = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 =
%defaultVAR03 = /
ObjectNameList{3} = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%%% To expand for more than 3 objects, just add more lines in groups
%%% of three like those above, then change the line about five lines
%%% down from here (for i = 1:5).

%%%VariableRevisionNumber = 01


%%% Set up the window for displaying the results
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber);
    figure(ThisModuleFigureNumber);
    set(ThisModuleFigureNumber,'color',[1 1 1])
    columns = 1;
end


%%% Retrieves the pixel size that the user entered (micrometers per pixel).
PixelSize = str2double(handles.Settings.PixelSize);


%%% START LOOP THROUGH ALL THE OBJECTS
for i = 1:3
    ObjectName = ObjectNameList{i};
    if strcmp(ObjectName,'/')
        break
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    drawnow


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
    drawnow

    % PROGRAMMING NOTE
    % HANDLES STRUCTURE:
    %       In CellProfiler (and Matlab in general), each independent
    % function (module) has its own workspace and is not able to 'see'
    % variables produced by other modules. For data or images to be shared
    % from one module to the next, they must be saved to what is called
    % the 'handles structure'. This is a variable, whose class is
    % 'structure', and whose name is handles. The contents of the handles
    % structure are printed out at the command line of Matlab using the
    % Tech Diagnosis button. The only variables present in the main
    % handles structure are handles to figures and gui elements.
    % Everything else should be saved in one of the following
    % substructures:
    %
    % handles.Settings:
    %       Everything in handles.Settings is stored when the user uses
    % the Save pipeline button, and these data are loaded into
    % CellProfiler when the user uses the Load pipeline button. This
    % substructure contains all necessary information to re-create a
    % pipeline, including which modules were used (including variable
    % revision numbers), their setting (variables), and the pixel size.
    %   Fields currently in handles.Settings: PixelSize, ModuleNames,
    % VariableValues, NumbersOfVariables, VariableRevisionNumbers.
    %
    % handles.Pipeline:
    %       This substructure is deleted at the beginning of the
    % analysis run (see 'Which substructures are deleted prior to an
    % analysis run?' below). handles.Pipeline is for storing data which
    % must be retrieved by other modules. This data can be overwritten as
    % each image set is processed, or it can be generated once and then
    % retrieved during every subsequent image set's processing, or it can
    % be saved for each image set by saving it according to which image
    % set is being analyzed, depending on how it will be used by other
    % modules. Any module which produces or passes on an image needs to
    % also pass along the original filename of the image, named after the
    % new image name, so that if the SaveImages module attempts to save
    % the resulting image, it can be named by appending text to the
    % original file name.
    %   Example fields in handles.Pipeline: FileListOrigBlue,
    % PathnameOrigBlue, FilenameOrigBlue, OrigBlue (which contains the actual image).
    %
    % handles.Current:
    %       This substructure contains information needed for the main
    % CellProfiler window display and for the various modules to
    % function. It does not contain any module-specific data (which is in
    % handles.Pipeline).
    %   Example fields in handles.Current: NumberOfModules,
    % StartupDirectory, DefaultOutputDirectory, DefaultImageDirectory,
    % FilenamesInImageDir, CellProfilerPathname, ImageToolHelp,
    % DataToolHelp, FigureNumberForModule01, NumberOfImageSets,
    % SetBeingAnalyzed, TimeStarted, CurrentModuleNumber.
    %
    % handles.Preferences:
    %       Everything in handles.Preferences is stored in the file
    % CellProfilerPreferences.mat when the user uses the Set Preferences
    % button. These preferences are loaded upon launching CellProfiler.
    % The PixelSize, DefaultImageDirectory, and DefaultOutputDirectory
    % fields can be changed for the current session by the user using edit
    % boxes in the main CellProfiler window, which changes their values in
    % handles.Current. Therefore, handles.Current is most likely where you
    % should retrieve this information if needed within a module.
    %   Fields currently in handles.Preferences: PixelSize, FontSize,
    % DefaultModuleDirectory, DefaultOutputDirectory,
    % DefaultImageDirectory.
    %
    % handles.Measurements:
    %       Everything in handles.Measurements contains data specific to each
    % image set analyzed for exporting. It is used by the ExportMeanImage
    % and ExportCellByCell data tools. This substructure is deleted at the
    % beginning of the analysis run (see 'Which substructures are deleted
    % prior to an analysis run?' below).
    %    Note that two types of measurements are typically made: Object
    % and Image measurements.  Object measurements have one number for
    % every object in the image (e.g. ObjectArea) and image measurements
    % have one number for the entire image, which could come from one
    % measurement from the entire image (e.g. ImageTotalIntensity), or
    % which could be an aggregate measurement based on individual object
    % measurements (e.g. ImageMeanArea).  Use the appropriate prefix to
    % ensure that your data will be extracted properly. It is likely that
    % Subobject will become a new prefix, when measurements will be
    % collected for objects contained within other objects.
    %       Saving measurements: The data extraction functions of
    % CellProfiler are designed to deal with only one "column" of data per
    % named measurement field. So, for example, instead of creating a
    % field of XY locations stored in pairs, they should be split into a
    % field of X locations and a field of Y locations. It is wise to
    % include the user's input for 'ObjectName' or 'ImageName' as part of
    % the fieldname in the handles structure so that multiple modules can
    % be run and their data will not overwrite each other.
    %   Example fields in handles.Measurements: ImageCountNuclei,
    % ObjectAreaCytoplasm, FilenameOrigBlue, PathnameOrigBlue,
    % TimeElapsed.
    %
    % Which substructures are deleted prior to an analysis run?
    %       Anything stored in handles.Measurements or handles.Pipeline
    % will be deleted at the beginning of the analysis run, whereas
    % anything stored in handles.Settings, handles.Preferences, and
    % handles.Current will be retained from one analysis to the next. It
    % is important to think about which of these data should be deleted at
    % the end of an analysis run because of the way Matlab saves
    % variables: For example, a user might process 12 image sets of nuclei
    % which results in a set of 12 measurements ("ImageTotalNucArea")
    % stored in handles.Measurements. In addition, a processed image of
    % nuclei from the last image set is left in the handles structure
    % ("SegmNucImg"). Now, if the user uses a different module which
    % happens to have the same measurement output name "ImageTotalNucArea"
    % to analyze 4 image sets, the 4 measurements will overwrite the first
    % 4 measurements of the previous analysis, but the remaining 8
    % measurements will still be present. So, the user will end up with 12
    % measurements from the 4 sets. Another potential problem is that if,
    % in the second analysis run, the user runs only a module which
    % depends on the output "SegmNucImg" but does not run a module that
    % produces an image by that name, the module will run just fine: it
    % will just repeatedly use the processed image of nuclei leftover from
    % the last image set, which was left in handles.Pipeline.


    %%% Initialize
    BasicFeatures    = {'Area',...
        'Eccentricity',...
        'Solidity',...
        'Extent',...
        'EulerNumber',...
        'Perimeter',...
        'FormFactor'};

    %%% Get the basic shape features
    props = regionprops(LabelMatrixImage,'Area','Eccentricity','Solidity','Extent','EulerNumber');

    % Perimeter
    perim = bwperim(LabelMatrixImage>0).*LabelMatrixImage;
    perim = perim(:);
    perim = perim(find(perim));
    Perimeter = (hist(perim,[1:max(perim)])*PixelSize)';

    % Form factor
    FormFactor = 4*pi*cat(1,props.Area) ./ Perimeter.^2;

    Basic = [cat(1,props.Area)*PixelSize^2,...
        cat(1,props.Eccentricity),...
        cat(1,props.Solidity),...
        cat(1,props.Extent),...
        cat(1,props.EulerNumber),...
        Perimeter,...
        FormFactor];


    %%% Calculate Zernike shape features

    % Get index for Zernike functions
    index = [];
    ZernikeFeatures = {};
    for n = 0:12
        for m = 0:n
            if rem(n-m,2) == 0
                index = [index;n m];
                ZernikeFeatures = cat(2,ZernikeFeatures,{sprintf('Z%d_%d',n,m)});
            end
        end
    end

    % Use ConvexArea to automatically calculate the average equivalent diameter
    % of the objects, and then use this diameter to determine the grid size
    % of the Zernike functions
    tmp = regionprops(LabelMatrixImage,'ConvexArea');
    diameter = floor(sqrt(4/pi*mean(cat(1,tmp.ConvexArea)))+1);
    if rem(diameter,2)== 0, diameter = diameter + 1;end   % An odd number facilitates implementation

    diameter=50;
    % Calculate the Zernike basis functions
    [x,y] = meshgrid(linspace(-1,1,diameter),linspace(-1,1,diameter));
    r = sqrt(x.^2+y.^2);
    phi = atan(y./(x+eps));
    Zf = zeros(size(x,1),size(x,2),size(index,1));

    for k = 1:size(index,1)
        n = index(k,1);
        m = index(k,2);
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
    Zernike = zeros(size(Centroids,1),size(index,1));
    for Object = 1:size(Centroids,1)

        % Get image patch
        cx = round(Centroids(Object,1));
        cy = round(Centroids(Object,2));
        rmax = round(Centroids(Object,2)+(diameter-1)/2);
        rmin = round(Centroids(Object,2)-(diameter-1)/2);
        cmax = round(Centroids(Object,1)+(diameter-1)/2);
        cmin = round(Centroids(Object,1)-(diameter-1)/2);
        BWpatch   = PaddedLabelMatrixImage(rmin:rmax,cmin:cmax) == Object;

        % Apply Zernike functions
        Zernike(Object,:) = squeeze(abs(sum(sum(repmat(BWpatch,[1 1 size(index,1)]).*Zf))))';

    end

    %%% Save measurements
    for k = 1:length(BasicFeatures)
        handles.Measurements.Shape.Basic.(BasicFeatures{k}).(ObjectName).Object(handles.Current.SetBeingAnalyzed) = {Basic(:,k)};
        handles.Measurements.Shape.Basic.(BasicFeatures{k}).(ObjectName).Image(handles.Current.SetBeingAnalyzed) = ...
            {[mean(Basic(:,k)) median(Basic(:,k)) std(Basic(:,k))]};
    end
    for  k = 1:length(ZernikeFeatures)
        handles.Measurements.Shape.Zernike.(ZernikeFeatures{k}).(ObjectName).Object(handles.Current.SetBeingAnalyzed) = {Zernike(:,k)};
        handles.Measurements.Shape.Zernike.(ZernikeFeatures{k}).(ObjectName).Image(handles.Current.SetBeingAnalyzed) = ...
            {[mean(Zernike(:,k)) median(Zernike(:,k)) std(Zernike(:,k))]};
    end

    %%% Report measurements
    if any(findobj == ThisModuleFigureNumber);
        % This first block writes the same text several times
        % Header
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0 0.95 1 0.04],...
            'HorizontalAlignment','center','Backgroundcolor',[1 1 1],'fontname','times',...
            'fontsize',10,'fontweight','bold','string',sprintf('Average shape features for image set #%d',handles.Current.SetBeingAnalyzed));

        % Number of objects
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.85 0.3 0.03],...
            'HorizontalAlignment','left','Backgroundcolor',[1 1 1],'fontname','times',...
            'fontsize',8,'fontweight','bold','string','Number of objects:');

        % Text for Basic features
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8 0.3 0.03],...
            'HorizontalAlignment','left','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',8,'fontweight','bold','string','Basic features:');
        for k = 1:length(BasicFeatures)
            q = uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.8-0.04*k 0.3 0.03],...
                'HorizontalAlignment','left','BackgroundColor',[1 1 1],'fontname','times',...
                'fontsize',8,'string',BasicFeatures{k});
        end

        % Text for Zernike features
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.45 0.3 0.03],...
            'HorizontalAlignment','left','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',8,'fontweight','bold','string','5 first Zernike features:');
        for k = 1:5
            q = uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.05 0.45-0.04*k 0.3 0.03],...
                'HorizontalAlignment','left','BackgroundColor',[1 1 1],'fontname','times',...
                'fontsize',8,'string',ZernikeFeatures{k});
        end


        % The name of the object image
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.3+0.2*(columns-1) 0.9 0.2 0.03],...
            'HorizontalAlignment','center','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',8,'fontweight','bold','string',ObjectName);

        % Number of objects
        uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.3+0.2*(columns-1) 0.85 0.2 0.03],...
            'HorizontalAlignment','center','BackgroundColor',[1 1 1],'fontname','times',...
            'fontsize',8,'string',num2str(max(LabelMatrixImage(:))));

        % Basic shape features
        for k = 1:length(BasicFeatures)
            q = uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.3+0.2*(columns-1) 0.8-0.04*k 0.2 0.03],...
                'HorizontalAlignment','center','BackgroundColor',[1 1 1],'fontname','times',...
                'fontsize',8,'string',sprintf('%0.2f',mean(Basic(:,k))));
        end

        % Zernike shape features
        for k = 1:5
            q = uicontrol(ThisModuleFigureNumber,'style','text','units','normalized', 'position', [0.3+0.2*(columns-1) 0.45-0.04*k 0.2 0.03],...
                'HorizontalAlignment','center','BackgroundColor',[1 1 1],'fontname','times',...
                'fontsize',8,'string',sprintf('%0.2f',mean(Zernike(:,k))));
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
