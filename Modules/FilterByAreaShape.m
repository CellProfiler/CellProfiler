function handles = FilterObjectsAreaShape(handles)

% Help for the Filter Objects by AreaShape module: 
% Category: Object Identification and Modification
%
% This module applies a filter using statistics measured by the 
% MeasureAreaShape module to select objects with desired area or shape. For
% example, it can be used to eliminate objects with a Solidity value below
% a certain threshold.
%
% See also MEASUREAREASHAPE

CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the original image?
%infotypeVAR01 = imagegroup
%inputtypeVAR01 = popupmenu
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What did you call the objects you want to process?
%infotypeVAR02 = objectgroup
%inputtypeVAR02 = popupmenu
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What do you want to call the filtered objects?
%defaultVAR03 = FilteredNuclei
%infotypeVAR03 = objectgroup indep
TargetName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What feature do you want to  use as a filter? Please run this module after MeasureAreaShape module.
%choiceVAR04 = Area
%choiceVAR04 = Eccentricity
%choiceVAR04 = Solidity
%choiceVAR04 = Extent
%choiceVAR04 = Euler Number
%choiceVAR04 = Perimeter
%choiceVAR04 = Form factor
%choiceVAR04 = MajorAxisLength
%choiceVAR04 = MinorAxisLength
%inputtypeVAR04 = popupmenu
FeatureName1 = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Minimum value required:
%choiceVAR05 = 0.5
%choiceVAR05 = Do not use
%inputtypeVAR05 = popupmenu custom
MinValue1 = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Maximum value allowed:
%choiceVAR06 = Do not use
%inputtypeVAR06 = popupmenu custom
MaxValue1 = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Will you want to save the image of the colored objects? It will be saved as ColoredOBJECTNAME
%choiceVAR07 = No
%choiceVAR07 = Yes
%inputtypeVAR07 = popupmenu
SaveColored = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Will you want to save the image of the outlined objects? It will be saved as OutlinedOBJECTNAME
%choiceVAR08 = No
%choiceVAR08 = Yes
%inputtypeVAR08 = popupmenu
SaveOutlined = char(handles.Settings.VariableValues{CurrentModuleNum,8});
OrigImage = handles.Pipeline.(ImageName);
LabelMatrixImage = handles.Pipeline.(['Segmented' ObjectName]);
FilterType = strmatch(FeatureName1, handles.Measurements.(ObjectName).AreaShapeFeatures);
AreaShapeInfo = handles.Measurements.(ObjectName).AreaShape{handles.Current.SetBeingAnalyzed}(:,FilterType);

if strcmp(MinValue1, 'Do not use')
    MinValue1 = -Inf;
else
    MinValue1 = str2double(MinValue1);
end

if strcmp(MaxValue1, 'Do not use')
    MaxValue1 = Inf;
else
    MaxValue1 = str2double(MaxValue1);
end

Filter = find((AreaShapeInfo < MinValue1) | (AreaShapeInfo > MaxValue1));
FinalLabelMatrixImage = LabelMatrixImage;
for i=1:numel(Filter)
    FinalLabelMatrixImage(FinalLabelMatrixImage == Filter(i)) = 0;
end

FinalLabelMatrixImage = bwlabel(FinalLabelMatrixImage);



%%% Calculates the object outlines, which are overlaid on the original
%%% image and displayed in figure subplot (2,2,4).
%%% Creates the structuring element that will be used for dilation.
StructuringElement = strel('square',3);
%%% Converts the FinalLabelMatrixImage to binary.
FinalBinaryImage = im2bw(FinalLabelMatrixImage,.1);
%%% Dilates the FinalBinaryImage by one pixel (8 neighborhood).
DilatedBinaryImage = imdilate(FinalBinaryImage, StructuringElement);
%%% Subtracts the FinalBinaryImage from the DilatedBinaryImage,
%%% which leaves the PrimaryObjectOutlines.
PrimaryObjectOutlines = DilatedBinaryImage - FinalBinaryImage;
%%% Overlays the object outlines on the original image.
ObjectOutlinesOnOrigImage = OrigImage;
%%% Determines the grayscale intensity to use for the cell outlines.
LineIntensity = max(OrigImage(:));
ObjectOutlinesOnOrigImage(PrimaryObjectOutlines == 1) = LineIntensity;

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

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1
    drawnow
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,2,1); imagesc(OrigImage);colormap(gray);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the colored label
    %%% matrix image.
    cmap = jet(max(64,max(LabelMatrixImage(:))));
    subplot(2,2,2); imagesc(label2rgb(LabelMatrixImage, cmap, 'k', 'shuffle')); title(['Segmented ',ObjectName]);
    %%% A subplot of the figure window is set to display the Overlaid image,
    %%% where the maxima are imposed on the inverted original image
    cmap = jet(max(64,max(FinalLabalMatrixImage(:))));
    ColoredLabelMatrixImage = label2rgb(FinalLabelMatrixImage, cmap, 'k', 'shuffle');
    subplot(2,2,3); imagesc(ColoredLabelMatrixImage); title(['Filtered ' ObjectName]);
    %%% A subplot of the figure window is set to display the inverted original
    %%% image with watershed lines drawn to divide up clusters of objects.
    subplot(2,2,4); imagesc(ObjectOutlinesOnOrigImage);colormap(gray); title([TargetName, ' Outlines on Input Image']);
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

handles.Pipeline.(['Segmented' TargetName]) = FinalLabelMatrixImage;

if strcmp(SaveColored,'Yes')
    handles.Pipeline.(['Colored' TargetName]) = ColoredLabelMatrixImage;
end
if strcmp(SaveOutlined,'Yes')
    handles.Pipeline.(['Outlined' TargetName]) = ObjectOutlinesOnOrigImage;
end