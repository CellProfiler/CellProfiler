function handles = IdentifyWellSpots(handles)

% Help for the Identify Well Spots module:
% Category: Object Identification and Modification
%
% This module allows you to identify well spots in a grid. The resulting
% well spots are ordered by columns and then rows, e.g:
% 1 5 9  13
% 2 6 10 14
% 3 7 11 15
% 4 8 12 16
% The module calculates the average radius of all the spots, excluding
% outliers, and redraws all the spots with the same radius.
% If a spot is not present where it should be, the module calculates the
% expected location of the spot and draws it.
% This module should be used after an Identify Primary module to first
% identify objects. Then, a crop module or MeasureAreaShape +
% FilterObjectsAreaShape modules may be needed to eliminate the border or
% misidentified objects.
% See also IDENTIFYPRIMINTENSINTENS, IDENTIFYPRIMSHAPEINTENS, CROP,
% MEASUREAREASHAPE, FILTEROBJECTSAREASHAPE.

%%% Reads the current module number, because this is needed to find 
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the identified objects?
%infotypeVAR01 = objectgroup
PrevObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the objects identified by this module?
%infotypeVAR02 = objectgroup indep
%defaultVAR02 = Spots
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = How many rows of spots?
%defaultVAR03 = 8
Rows = str2double(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = How many columns of spots?
%defaultVAR04 = 12
Cols = str2double(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Enter the radius of a spot in pixels.
%choiceVAR05 = Automatic
Radius = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu custom

%textVAR06 = Enter coordinate of top left corner control spot (X,Y).
%choiceVAR06 = Automatic
LeftCoord = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu custom

%textVAR07 = Enter coordinate of lower right corner control spot (X,Y).
%choiceVAR07 = Automatic
RightCoord = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu custom

%textVAR08 = What do you want to call the image of the outlines of the objects?
%infotypeVAR08 = imagegroup indep
%defaultVAR08 = Do not save
SaveOutlined = char(handles.Settings.VariableValues{CurrentModuleNum,8}); 

%textVAR09 =  What do you want to call the label matrix image?
%infotypeVAR09 = imagegroup indep
%defaultVAR09 = Do not save
SaveColored = char(handles.Settings.VariableValues{CurrentModuleNum,9}); 

%textVAR10 = Do you want to save the label matrix image in RGB or grayscale?
%choiceVAR10 = RGB
%choiceVAR10 = Grayscale
SaveMode = char(handles.Settings.VariableValues{CurrentModuleNum,10}); 
%inputtypeVAR10 = popupmenu

%%%VariableRevisionNumber = 2

FinalLabelMatrixImage = handles.Pipeline.(['Segmented' PrevObjectName]);

props = regionprops(FinalLabelMatrixImage,'Area','Eccentricity');
Area = [props.Area];
Eccentricity = [props.Eccentricity];

tmp = regionprops(FinalLabelMatrixImage,'Centroid');
Location = cat(1,tmp.Centroid);

if strcmp(LeftCoord,'Automatic')
    Leftmost = min(Location(:,1));
    Uppermost = min(Location(:,2));
else
    [Leftmost,Uppermost]=strread(LeftCoord,'%d',2,'delimiter',',');
end
if strcmp(RightCoord,'Automatic')
    Rightmost = max(Location(:,1));
    Lowermost = max(Location(:,2));
else
    [Rightmost,Lowermost]=strread(RightCoord,'%d',2,'delimiter',',');
end

%Assuming the user declared the number of rows and cols
XDiv = (Rightmost - Leftmost)/(Cols - 1);
YDiv = (Lowermost - Uppermost)/(Rows - 1);
%%% Should this not be hard-coded as 8,12???
LocationTable = cell(8,12);
%LocationTable(:,:) = {zeros(1,2)};
XTable = zeros(Rows,Cols);
YTable = zeros(Rows,Cols);
for i=1:Cols
    for j=1:Rows
        LocationTable{j,i} = Location( (Location(:,1) > (Leftmost + (i-1.5)*XDiv)) & (Location(:,1) < (Leftmost + (i-0.5)*XDiv)) & (Location(:,2) > (Uppermost + (j-1.5)*YDiv)) & (Location(:,2) < (Uppermost + (j-0.5)*YDiv)),:);
        if size(LocationTable{j,i},1) > 0
            XTable(j,i) = LocationTable{j,i}(1);
            YTable(j,i) = LocationTable{j,i}(2);
        end
        if size(LocationTable{j,i},1) > 1
            CPmsgbox('More than one spot detected in a grid. Please change settings of the Identify module and/or use MeasureAreaShape and FilterObjectsAreaShape module.');
        end
    end
end

for i=1:Cols
    for j=1:Rows
        if size(LocationTable{j,i},1) == 0
            LocationTable{j,i} = [mean(nonzeros(XTable(:,i))) mean(nonzeros(YTable(j,:)))];
            XTable(j,i) = LocationTable{j,i}(1);
            YTable(j,i) = LocationTable{j,i}(2);
        end
    end
end

if strcmp(Radius,'Automatic')
    MeanArea = mean(Area( Area > (mean(Area) - 2*std(Area)) & Area < (mean(Area) + 2*std(Area))));
    Radius = round((MeanArea/pi)^.5);
else
    Radius = str2double(Radius);
end

OneSpot = getnhood(strel('disk',Radius,0));
Shift1 = round((size(OneSpot,1)-1)/2);
Shift2 = size(OneSpot,1) - Shift1-1;

FinalAlignedMatrixImage = zeros(size(FinalLabelMatrixImage,1),size(FinalLabelMatrixImage,2));
Count=1;
for i=1:Cols
    for j=1:Rows
        FinalAlignedMatrixImage((round(YTable(j,i))-Shift1):(round(YTable(j,i))+Shift2),(round(XTable(j,i))-Shift1):(round(XTable(j,i))+Shift1))=Count*OneSpot;
        Count=Count+1;
    end
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow 

% PROGRAMMING NOTE
% DISPLAYING RESULTS:
% Some calculations produce images that are used only for display or
% for saving to the hard drive, and are not used by downstream
% modules. To speed processing, these calculations are omitted if the
% figure window is closed and the user does not want to save the
% images.

% PROGRAMMING NOTE
% DRAWNOW BEFORE FIGURE COMMAND:
% The "drawnow" function executes any pending figure window-related
% commands.  In general, Matlab does not update figure windows until
% breaks between image analysis modules, or when a few select commands
% are used. "figure" and "drawnow" are two of the commands that allow
% Matlab to pause and carry out any pending figure window- related
% commands (like zooming, or pressing timer pause or cancel buttons or
% pressing a help button.)  If the drawnow command is not used
% immediately prior to the figure(ThisModuleFigureNumber) line, then
% immediately after the figure line executes, the other commands that
% have been waiting are executed in the other windows.  Then, when
% Matlab returns to this module and goes to the subplot line, the
% figure which is active is not necessarily the correct one. This
% results in strange things like the subplots appearing in the timer
% window or in the wrong figure window, or in help dialog boxes.

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);

if any(findobj == ThisModuleFigureNumber) == 1 | strcmpi(SaveColored,'Do not save') ~= 1 | strcmpi(SaveOutlined,'Do not save') ~= 1
    %%% Calculates the ColoredLabelMatrixImage.
    %%% Note that the label2rgb function doesn't work when there are no objects
    %%% in the label matrix image, so there is an "if".
    if sum(sum(FinalAlignedMatrixImage)) >= 1
        ColoredLabelMatrixImage = label2rgb(FinalAlignedMatrixImage, 'jet', 'k', 'shuffle');
    else  ColoredLabelMatrixImage = FinalAlignedMatrixImage;
    end
    %%% Calculates the object outlines.
    %%% Creates the structuring element that will be used for dilation.
    StructuringElement = strel('square',3);
    %%% Converts the FinalLabelMatrixImage to binary.
    FinalBinaryImage = im2bw(FinalAlignedMatrixImage,.5);
    %%% Dilates the FinalBinaryImage by one pixel (8 neighborhood).
    DilatedBinaryImage = imdilate(FinalBinaryImage, StructuringElement);
    %%% Subtracts the FinalBinaryImage from the DilatedBinaryImage,
    %%% which leaves the ObjectOutlines.
    ObjectOutlines = DilatedBinaryImage - FinalBinaryImage;

    drawnow
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1); imagesc(FinalLabelMatrixImage);colormap(gray);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    subplot(2,1,2); imagesc(ColoredLabelMatrixImage); title(['Identified ',ObjectName]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
% handles.Measurements
%      Data extracted from input images are stored in the
% handles.Measurements substructure for exporting or further analysis.
% This substructure is deleted at the beginning of the analysis run
% (see 'Which substructures are deleted prior to an analysis run?'
% below). The Measurements structure is organized in two levels. At
% the first level, directly under handles.Measurements, there are
% substructures (fields) containing measurements of different objects.
% The names of the objects are specified by the user in the Identify
% modules (e.g. 'Cells', 'Nuclei', 'Colonies').  In addition to these
% object fields is a field called 'Image' which contains information
% relating to entire images, such as filenames, thresholds and
% measurements derived from an entire image. That is, the Image field
% contains any features where there is one value for the entire image.
% As an example, the first level might contain the fields
% handles.Measurements.Image, handles.Measurements.Cells and
% handles.Measurements.Nuclei.
%      In the second level, the measurements are stored in matrices 
% with dimension [#objects x #features]. Each measurement module
% writes its own block; for example, the MeasureAreaShape module
% writes shape measurements of 'Cells' in
% handles.Measurements.Cells.AreaShape. An associated cell array of
% dimension [1 x #features] with suffix 'Features' contains the names
% or descriptions of the measurements. The export data tools, e.g.
% ExportData, triggers on this 'Features' suffix. Measurements or data
% that do not follow the convention described above, or that should
% not be exported via the conventional export tools, can thereby be
% stored in the handles.Measurements structure by leaving out the
% '....Features' field. This data will then be invisible to the
% existing export tools.
%      Following is an example where we have measured the area and
% perimeter of 3 cells in the first image and 4 cells in the second
% image. The first column contains the Area measurements and the
% second column contains the Perimeter measurements.  Each row
% contains measurements for a different cell:
% handles.Measurements.Cells.AreaShapeFeatures = {'Area' 'Perimeter'}
% handles.Measurements.Cells.AreaShape{1} = 	40		20
%                                               100		55
%                                              	200		87
% handles.Measurements.Cells.AreaShape{2} = 	130		100
%                                               90		45
%                                               100		67
%                                               45		22
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
% ("SegmNucImg"). Now, if the user uses a different algorithm which
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
    
fieldname = ['Segmented',ObjectName];
handles.Pipeline.(fieldname) = FinalAlignedMatrixImage;
%%% Saves images to the handles structure so they can be saved to the hard
%%% drive, if the user requested.
try
    if ~strcmpi(SaveColored,'Do not save')
        if strcmp(SaveMode,'RGB')
            handles.Pipeline.(SaveColored) = ColoredLabelMatrixImage;
        else
            handles.Pipeline.(SaveColored) = FinalAlignedMatrixImage;
        end
    end
    if ~strcmpi(SaveOutlined,'Do not save')
        handles.Pipeline.(SaveOutlined) = ObjectOutlines;
    end
catch errordlg('The object outlines or colored objects were not calculated by an identify module (possibly because the window is closed) so these images were not saved to the handles structure. The Save Images module will therefore not function on these images. This is just for your information - image processing is still in progress, but the Save Images module will fail if you attempted to save these images.')
end