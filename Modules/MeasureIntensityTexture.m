function handles = MeasureIntensityTexture(handles)

% Help for the Measure Intensity Texture module:
% Category: Measurement
%
% Given an image with objects identified (e.g. nuclei or cells), this
% module makes measurements of the intensity and texture of each
% object based on a corresponding grayscale image. Measurements are
% recorded for each object, and some population measurements are
% calculated: Mean, Median, Standard Deviation, and in some cases Sum.
% Note that the standard deviation of intensity is a measure of
% texture.  We hope to add other measurements of texture to this
% module.
%
% How it works:
% Retrieves a segmented image, in label matrix format, and a
% corresponding original grayscale image and makes measurements of the
% objects that are segmented in the image. This module differs from
% the MeasureAreaShapeCountLocation module because it lacks
% measurements of shape and area and includes only intensity and
% texture. The label matrix image should be "compacted": that is,
% each number should correspond to an object, with no numbers skipped.
% So, if some objects were discarded from the label matrix image, the
% image should be converted to binary and re-made into a label matrix
% image before feeding into this module.
%
% See also MEASUREAREAOCCUPIED,
% MEASUREAREASHAPECOUNTLOCATION,
% MEASURECORRELATION,
% MEASURETOTALINTENSITY.

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

%textVAR01 = What did you call the greyscale images you want to measure?
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%textVAR02 = What did you call the segmented objects that you want to measure?
%defaultVAR02 = Nuclei
ObjectNameList{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%textVAR03 = Type / in unused boxes.
%defaultVAR03 = Cells
ObjectNameList{2} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%textVAR04 =
%defaultVAR04 = /
ObjectNameList{3} = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%textVAR05 =
%defaultVAR05 = /
ObjectNameList{4} = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%textVAR06 =
%defaultVAR06 = /
ObjectNameList{5} = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%textVAR07 = Measure the fraction of cells with a total intensity greater than or equal to this threshold.  Type N to skip this measurement.
%defaultVAR07 = N
Threshold = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Note: The measurements made by this module will be named based on your entries, e.g. 'OrigBluewithinNuclei', 'OrigBluewithinCells'. Also, it is easy to expand the code for more than 5 objects. See MeasureIntensityTexture.m for details. 

%%% To expand for more than 5 objects, just add more lines in groups
%%% of three like those above, then change the line about five lines
%%% down from here (for i = 1:5).

%%%VariableRevisionNumber = 02

%%% START LOOP THROUGH ALL THE OBJECTS
for i = 1:5
    ObjectName = ObjectNameList{i};
    if strcmp(ObjectName,'/') == 1
        break
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    drawnow

    %%% Reads (opens) the image you want to analyze and assigns it to a variable,
    %%% "OrigImageToBeAnalyzed".
    fieldname = ['', ImageName];
    %%% Checks whether the image exists in the handles structure.
    if isfield(handles.Pipeline, fieldname) == 0,
        error(['Image processing has been canceled. Prior to running the Measure Intensity Texture module, you must have previously run a module that loads a greyscale image.  You specified in the MeasureIntensityTexture module that the desired image was named ', ImageName, ' which should have produced an image in the handles structure called ', fieldname, '. The Measure Intensity Texture module cannot locate this image.']);
    end
    OrigImageToBeAnalyzed = handles.Pipeline.(fieldname);


    %%% Checks that the original image is two-dimensional (i.e. not a color
    %%% image), which would disrupt several of the image functions.
    if ndims(OrigImageToBeAnalyzed) ~= 2
        error('Image processing was canceled because the Measure Intensity Texture module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
    end

    %%% Retrieves the label matrix image that contains the segmented objects which
    %%% will be measured with this module.
    fieldname = ['Segmented', ObjectName];
    %%% Checks whether the image exists in the handles structure.
    if isfield(handles.Pipeline, fieldname) == 0,
        error(['Image processing has been canceled. Prior to running the Measure Intensity Texture module, you must have previously run a module that generates an image with the objects identified.  You specified in the Measure Intensity Texture module that the primary objects were named ',ObjectName,' which should have produced an image in the handles structure called ', fieldname, '. The Measure Intensity Texture module cannot locate this image.']);
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

    %%% The ObjectName is changed to include the Object Name plus the name of
    %%% the grayscale image used, so that the measurements do not overwrite any
    %%% measurements made on the original objects.  For example, if
    %%% measurements were made for the Nuclei using the original blue image and
    %%% this module is being used to measure the intensities, etc. of the
    %%% OrigRed channel at the nuclei, the blue measurements will be called:
    %%% MeanAreaNuclei whereas the red measurements will be called:
    %%% MeanAreaOrigRedWithinNuclei.
    OriginalObjectName = ObjectName;
    ObjectName = strcat(ImageName , 'within', ObjectName);

    %%%
    %%% COUNT
    %%%

    %%% If the objects have already been counted and there are zero
    %%% objects in this image, no measurements are made by this module.
    fieldname = ['ImageCount', OriginalObjectName];
    try ObjectCount = handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed);
        ObjectCount = ObjectCount{1};
    catch
        %%% Counts the number of objects in the label matrix image. This
        %%% does not require that the objects be contiguous. Strange
        %%% results may ensue with non-contiguous objects. Subtracting the
        %%% 1 is necessary because zero (the background) would otherwise
        %%% be counted as an object.
        ObjectCount = length(unique(LabelMatrixImage(:))) - 1;
    end
    if ObjectCount ~= 0
        %%% None of the measurements are made if there are no objects.

        %%% CATCH NAN's -->>
        %%% I am not sure whether this module actually requires this step.
        %%% The Area measurements are only used for this error catching, so
        %%% it's a time-consuming error check.
        Statistics = regionprops(LabelMatrixImage,'Area');
        if sum(isnan(cat(1,Statistics.Area))) ~= 0
            error('Image processing was canceled because there was a problem in the Measure Intensity Texture module. Some of the measurements could not be made.  This might be because some objects had zero area or because some measurements were attempted that were divided by zero. If you want to make measurements despite this problem, remove the 3 lines in the .m file for this module following the line %%% CATCH NANs. This will result in some non-numeric values in the output file, which will be represented as NaN (Not a Number).')
        end

        %%%
        %%% INTEGRATED INTENSITY (TOTAL INTENSITY PER OBJECT)
        %%%
        drawnow

        % The find function (when used as follows) returns the linear index
        % position of all the nonzero elements in the label matrix image.
        ForegroundPixels = find(LabelMatrixImage);
        % The find function (when used as follows) returns the x and y position of
        % the nonzero elements of the label matrix image (which we don't care
        % about), as well as the actual label matrix value at that point (i.e. 1,
        % 2, 3).
        [x,y,LabelValue] = find(LabelMatrixImage);
        % Creates a sparse matrix: Can think of it this way (not sure if I have
        % rows and columns mixed up, but it doesn't matter): each object is a
        % column, identified by the LabelValue, which is really equivalent to the
        % object number.  Each row of the matrix is a position in the original
        % image, identified by linear indexing, so that the number of rows is equal
        % to the linear index value of the last nonzero pixel in the label matrix
        % image.  The value of each cell in this matrix is the intensity value from
        % the original image at that position.
        AllObjectsPixelValues = sparse(ForegroundPixels, LabelValue, OrigImageToBeAnalyzed(ForegroundPixels));
        % Sums all pixel intensity values in each column.
        AlmostIntegratedIntensity = sum(AllObjectsPixelValues(:,:));
        % Converts from sparse to full to end up with one column of numbers.
        IntegratedIntensity = full(AlmostIntegratedIntensity');

        %%% Saves Integrated Intensities to handles structure.
        fieldname = ['ObjectIntegratedIntensity', ObjectName];
        handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {IntegratedIntensity};
        fieldname = ['ImageMeanIntegratedIntensity', ObjectName];
        handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {mean(IntegratedIntensity)};
        fieldname = ['ImageStdevIntegratedIntensity', ObjectName];
        handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {std(IntegratedIntensity)};
        fieldname = ['ImageMedianIntegratedIntensity', ObjectName];
        handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {median(IntegratedIntensity)};
        fieldname = ['ImageSumIntegratedIntensity', ObjectName];
        handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {sum(IntegratedIntensity)};

        %%% Calculates the fraction of cells whose integrated intensity is above the
        %%% user's threshold.
        if strcmp(upper(Threshold),'N') ~= 1
            NumberObjectsAboveThreshold = sum(IntegratedIntensity >= str2double(Threshold));
            TotalNumberObjects = length(IntegratedIntensity);
            FractionObjectsAboveThreshold = NumberObjectsAboveThreshold/TotalNumberObjects;
            fieldname = ['ImageFractionAboveThreshold', ObjectName];
            handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {FractionObjectsAboveThreshold};
        end

        %%%
        %%% MEAN INTENSITY (PER OBJECT)
        %%%
        drawnow

        %%% Finds the locations and labels for different objects.
        ObjectLocations = find(LabelMatrixImage);
        ObjectLabels = LabelMatrixImage(ObjectLocations);
        %%% Creates a sparse matrix with column as label and row as location,
        %%% with a 1 at (A,B) if location A has label B.  Summing the columns
        %%% gives the count of area pixels with a given label.
        Areas1 = full(sum(sparse(ObjectLocations, ObjectLabels, 1)));
        %%% Computes the mean.
        Temp1 = sparse(ObjectLocations, ObjectLabels, OrigImageToBeAnalyzed(ObjectLocations));
        MeanIntensity =  full(sum(Temp1)) ./ Areas1;
        %%% Subtracts the mean from each region.
        Map1 = [0 MeanIntensity];
        try
            OrigImageToBeAnalyzed2 = OrigImageToBeAnalyzed - Map1(LabelMatrixImage + 1);
        catch error('There was a problem in the MeasureIntensityTexture module.  The image to be analyzed is a different size than the image of identified objects.  If the objects were identified from a cropped image, the cropped image should be used by the Measure module.')
        end
        %%% Avoids divide by zero.
        Areas1(Areas1 < 2) = 2;
        drawnow
        %%% Estimates the standard deviation.
        Temp2 = sparse(ObjectLocations, ObjectLabels, OrigImageToBeAnalyzed2(ObjectLocations));
        StDevIntensity = sqrt(full(sum(Temp2.^2)) ./ (Areas1 - 1));
        %%% Converts to a column.
        MeanIntensity = MeanIntensity';
        StDevIntensity = StDevIntensity';
        %%% Saves data to handles structure.
        fieldname = ['ObjectMeanIntensity', ObjectName];
        handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {MeanIntensity};
        fieldname = ['ImageMeanMeanIntensity', ObjectName];
        handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {mean(MeanIntensity)};
        fieldname = ['ImageStdevMeanIntensity', ObjectName];
        handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {std(MeanIntensity)};
        fieldname = ['ImageMedianMeanIntensity', ObjectName];
        handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {median(MeanIntensity)};

        fieldname = ['ObjectStDevIntensity', ObjectName];
        handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {StDevIntensity};
        fieldname = ['ImageMeanStDevIntensity', ObjectName];
        handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {mean(StDevIntensity)};
        fieldname = ['ImageStdevStDevIntensity', ObjectName];
        handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {std(StDevIntensity)};
        fieldname = ['ImageMedianStDevIntensity', ObjectName];
        handles.Measurements.(fieldname)(handles.Current.SetBeingAnalyzed) = {median(StDevIntensity)};

    end % Goes with: if no objects are in the image.

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
    if any(findobj == ThisModuleFigureNumber) == 1;
        figure(ThisModuleFigureNumber);
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(1) = 0;
        newsize(2) = 0;
        if handles.Current.SetBeingAnalyzed == 1 && i == 1
            newsize(3) = originalsize(3)*.5;
            originalsize(3) = originalsize(3)*.5;
            set(ThisModuleFigureNumber, 'position', originalsize);
        end
        displaytexthandle = uicontrol(ThisModuleFigureNumber,'style','text', 'position', newsize,'fontname','fixedwidth','backgroundcolor',[0.7,0.7,0.7]);
        if i == 1
            displaytext =[];
        end
        %%% Note that the number of spaces after each measurement name results in
        %%% the measurement numbers lining up properly when displayed in a fixed
        %%% width font.  Also, it costs less than 0.1 seconds to do all of these
        %%% calculations, so I won't bother to retrieve the already calculated
        %%% means and sums from each measurement's code above.
        %%% Checks whether any objects were found in the image.
        if ObjectCount == 0
            displaytext = strvcat(displaytext,[ObjectName,', Image Set # ',num2str(handles.Current.SetBeingAnalyzed)],... %#ok We want to ignore MLint error checking for this line.
                ['Number of ', OriginalObjectName ,':      zero']);
        else
            displaytext = strvcat(displaytext,[ObjectName,', Image Set # ',num2str(handles.Current.SetBeingAnalyzed)],... %#ok We want to ignore MLint error checking for this line.
                ['MeanIntegratedIntensity:          ', num2str(mean(IntegratedIntensity))],...
                ['MeanMeanIntensity:                ', num2str(mean(MeanIntensity))],...
                ['MeanStDevIntensity:               ', num2str(mean(StDevIntensity))],...
                ['SumIntegratedIntensity:           ', num2str(sum(IntegratedIntensity))]);
            if strcmp(upper(Threshold),'N') ~= 1
                displaytext = strvcat(displaytext,... %#ok We want to ignore MLint error checking for this line.
                    ['Fraction above intensity threshold:', num2str(FractionObjectsAboveThreshold)]);
            end
        end % Goes with: if no objects were in the label matrix image.
        set(displaytexthandle,'string',displaytext)
        drawnow
    end
end
% PROGRAMMING NOTES THAT ARE UNNECESSARY FOR THIS MODULE:
% PROGRAMMING NOTE
% TO TEMPORARILY SHOW IMAGES DURING DEBUGGING:
% figure, imshow(BlurredImage, []), title('BlurredImage')
% TO TEMPORARILY SAVE IMAGES DURING DEBUGGING:
% imwrite(BlurredImage, FileName, FileFormat);
% Note that you may have to alter the format of the image before
% saving.  If the image is not saved correctly, for example, try
% adding the uint8 command:
% imwrite(uint8(BlurredImage), FileName, FileFormat);
% To routinely save images produced by this module, see the help in
% the SaveImages module.

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