function handles = Crop(handles)

% Help for the Crop module: 
% Category: Pre-processing
% 
% Allows the images to be cropped in any shape: 
%
% Rectangular: enter the pixel coordinates for the left, top and
% right, bottom corners, and every image will be cropped at these
% locations.
%
% File: specify the directory and filename of a binary image
% (actually, an image containing only the values 0 and 255) that is
% the exact same starting size as your image, and has zeros for the
% parts you want to remove and 255 for the parts you want to retain.
% This image should contain a contiguous block of 255's, because keep
% in mind that the cropping module will remove rows and columns
% that are completely blank. This image can be created in any image
% program, such as Photoshop.
%
% Ellipse Each: To draw an ellipse on each image, type EE. Each image
% in the set will be opened as CellProfiler cycles through the image
% sets and you will be asked to click five or more points to define an
% ellipse around the part of the image you want to analyze.  More
% points require longer calculations to generate the ellipse shape.
%
% Ellipse All: To draw one ellipse for all images, type EA. In this
% case, you will be asked during the first image set's processing to
% choose an image on which to draw the ellipse you want to use for all
% the images that will be cycled through.  This image need not be one
% that is part of the image set you are analyzing. You will be asked
% to click five or more points to define an ellipse around the part of
% the image you want to analyze.  More points require longer
% calculations to generate the ellipse shape.
%
% Warning: Keep in mind that cropping changes the size of your images,
% which may have unexpected consequences.  For example, identifying
% objects in a cropped image and then trying to measure their
% intensity in the original image will not work because the two images
% are not the same size. As another example, identify primary modules
% ignore objects that touch the outside edge of the image because they
% would be partial objects and therefore not measured properly.
% However, if you crop a round shape, the edge is still officially the
% square edge of the image, and not the round contour, so partial
% objects will be included.
%
% SAVING IMAGES: The cropped images produced by this module can be
% easily saved using the Save Images module, using the name you
% assign. If you want to save other intermediate images, alter the
% code for this module to save those images to the handles structure
% (see the SaveImages module help) and then use the Save Images
% module. You can alter the code of this module to allow saving the cropping
% shape that you have used, so that in future analyses you can use the
% File option.
%
% See also <nothing relevant>.

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
drawnow

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

%textVAR01 = What did you call the image to be cropped?
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call the cropped image?
%defaultVAR02 = CropBlue
CroppedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = For rectangular cropping, type R.  To draw an ellipse on each image, type EE. To draw one ellipse for all images: EA. For any shape (based on an image file), load the shape using the LoadSingleImage module and enter the name you gave the image here. To use the same cropping shape defined by another Crop module within this analysis run, type the name of the other image that was cropped, with the word 'Cropping' prefixed.
%defaultVAR03 = R
Shape = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Rectangular: enter the pixel position for the left (X), top (Y) corner (with comma).
%defaultVAR04 = 1,1
LeftTop = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Rectangular: enter the pixel position for the right (X), bottom (Y) corner (with comma). Enter 'end' for a position if you want to use the maximal bottom or right-most pixel.
%defaultVAR05 = 100,100
RightBottom = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image to be analyzed and assigns it to a variable,
%%% "OrigImage".
fieldname = ['', ImageName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, fieldname)==0,
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Crop module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
OrigImage = handles.Pipeline.(fieldname);

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

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

if strcmpi(Shape, 'EA') == 1 || strcmpi(Shape, 'EE') == 1
    if handles.Current.SetBeingAnalyzed == 1 || strcmp(Shape, 'EE') == 1
        if strcmp(Shape, 'EA') == 1
            %%% The user can choose an image from the pipeline or from the hard drive to use for
            %%% cropping.
            Answer = CPquestdlg('Choose an image to be used for cropping...','Select image','Image file from hard drive','Image from this image set','Image from this image set');
            if strcmp(Answer,'Cancel') == 1
                error('Image processing was canceled by the user in the Crop module.')
            elseif strcmp(Answer,'Image from this image set') == 1
                try ImageToBeCropped = OrigImage;                    
                catch 
                    error('Image processing was canceled because you did not select a valid image to use for cropping in the Crop module.')
                end
            elseif strcmp(Answer,'Image file from hard drive') == 1
                %%% Asks the user to open an image file upon which to draw the
                %%% ellipse.
                %%% Opens a user interface window which retrieves a file name and path
                %%% name for the image to be used as a test image.
                [CroppingFileName,CroppingPathname] = uigetfile(fullfile(handles.Current.DefaultImageDirectory,'.','*'),'Select the image to use for cropping');
                %%% If the user presses "Cancel", the FileName will = 0 and an error
                %%% message results.
                if CroppingFileName == 0
                    error('Image processing was canceled because you did not select an image to use for cropping in the Crop module.')
                else
                    [ImageToBeCropped, handles] = CPimread(fullfile(CroppingPathname,CroppingFileName), handles);
                end
            end
        else ImageToBeCropped = OrigImage;
        end
        %%% Displays the image and asks the user to choose points for the
        %%% ellipse.
        CroppingFigureHandle = figure;
        CroppingImageHandle = imagesc(ImageToBeCropped);
        colormap('gray'); pixval
        title({'Click on 5 or more points to be used to create a cropping ellipse & then press Enter.'; 'Press delete to erase the most recently clicked point.'})
        try imcontrast(CroppingImageHandle); end
        try imcontrast(CroppingImageHandle); end
        [Pre_x,Pre_y] = getpts(CroppingFigureHandle);
        close(CroppingFigureHandle)
        x = Pre_y;
        y = Pre_x;
        drawnow
        %%% Removes bias of the ellipse - to make matrix inversion more
        %%% accurate. (will be added later on) (Not really sure what this
        %%% is doing).
        mean_x = mean(x);
        mean_y = mean(y);
        New_x = x-mean_x;
        New_y = y-mean_y;
        %%% the estimation for the conic equation of the ellipse
        X = [New_x.^2, New_x.*New_y, New_y.^2, New_x, New_y ];
        params = sum(X)/(X'*X);
        masksize = size(ImageToBeCropped);
        [X,Y] = meshgrid(1:masksize(1), 1:masksize(2));
        X = X - mean_x;
        Y = Y - mean_y;
        drawnow
        %%% Produces the BinaryCropImage.
        BinaryCropImage = ((params(1) * (X .* X) + ...
            params(2) * (X .* Y) + ...
            params(3) * (Y .* Y) + ...
            params(4) * X + ...
            params(5) * Y) < 1);
        %%% Need to flip X and Y.
        BinaryCropImage = BinaryCropImage';
        %%% Displays the result in a new figure window.
        % figure;
        % imagesc(BinaryCropImage);title('Cropping Mask')
        % colormap(gray)
        % hold on
        % plot(Pre_x,Pre_y, 'r.')
        % drawnow
        %%% The Binary Crop image is saved to the handles
        %%% structure so it can be used to crop subsequent image sets.
        fieldname = ['Cropping', CroppedImageName];
        handles.Pipeline.(fieldname) = BinaryCropImage;
    end
    %%% See subfunction below.
    [handles, CroppedImage] = CropImageBasedOnMaskInHandles(handles, OrigImage, ['Cropping',CroppedImageName]);

elseif strcmpi(Shape,'R') == 1
    %%% Extracts the top, left, bottom, right pixel positions from the user's
    %%% input.
    LeftTopNumerical = str2num(LeftTop); %#ok We want MLint error checking to ignore this line.
    Left = LeftTopNumerical(1);
    Top = LeftTopNumerical(2);

    try RightBottomNumerical = str2num(RightBottom); %#ok We want MLint error checking to ignore this line.
        try
            Right = RightBottomNumerical(1); %#ok We want MLint error checking to ignore this line.
            %%% If the value is not numerical, then the user selected 'end',
            %%% so we should select the maximal right-most pixel value.
        catch Size = size(OrigImage(:,:,1));
            Right = Size(2);
        end
    catch Size = size(OrigImage(:,:,1));
        Right = Size(2);
    end
    try RightBottomNumerical = str2num(RightBottom); %#ok We want MLint error checking to ignore this line.
        try Bottom = RightBottomNumerical(2); %#ok We want MLint error checking to ignore this line.
            %%% If the value is not numerical, then the user selected 'end',
            %%% so we should select the maximal right-most pixel value.
        catch Size = size(OrigImage(:,:,1));
            Bottom = Size(1);
        end
    catch Size = size(OrigImage(:,:,1));
        Bottom = Size(1);
    end

    if Left == 0 || Right == 0 || Bottom == 0 || Top ==0
        error('There was a problem in the Cropping module. One of the values entered for the rectangular cropping pixel positions was zero: all values must be integers greater than zero.')
    end
    if Left >= Right
        error('There was a problem in the Cropping module. The value entered for the right corner is less than or equal to the value entered for the left corner.')
    end
    if Top >= Bottom
        error('There was a problem in the Cropping module. The value entered for the bottom corner is less than or equal to the value entered for the top corner.')
    end
    try
        CroppedImage = OrigImage(Top:Bottom, Left:Right,:);
    catch error('There was a problem in the Cropping module. The values entered for the rectangular cropping pixel positions are not valid.')
    end

else
    %%% See subfunction below; will work if the user has entered a
    %%% valid image name in the variable box for 'Shape'.
    [handles, CroppedImage] = CropImageBasedOnMaskInHandles(handles, OrigImage, Shape);
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

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        %%% Sets the window to be half as wide as usual.
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 250;
        set(ThisModuleFigureNumber, 'position', newsize);
    end
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
    drawnow
    %%% Activates the appropriate figure window.
    figure(ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1); imagesc(OrigImage);colormap(gray);
    title(['Input Image, Image Set # ',num2str(handles.Current.SetBeingAnalyzed)]);
    %%% A subplot of the figure window is set to display the adjusted
    %%%  image.
    subplot(2,1,2); imagesc(CroppedImage); title('Cropped Image');
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

%%% Saves the adjusted image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(CroppedImageName) = CroppedImage;

function [handles, CroppedImage] = CropImageBasedOnMaskInHandles(handles, OrigImage, CroppedImageName)
%%% Retrieves the Cropping image from the handles structure.
fieldname = [CroppedImageName];
try BinaryCropImage = handles.Pipeline.(fieldname);
    catch error(['You must choose rectangular cropping (R) or cropping from a file (F) or drawing an ellipse (EE or EA), or you must type the name of an image that resulted from a cropping module elsewhere in the image analysis pipeline. Your entry was ', Shape])
end
if size(OrigImage(:,:,1)) ~= size(BinaryCropImage(:,:,1))
    error('Image processing was canceled because an image you wanted to analyze is not the same size as the image used for cropping in the Crop module.  The pixel dimensions must be identical.')
end
%%% Sets pixels in the original image to zero if those pixels are zero in
%%% the binary image file.
PrelimCroppedImage = OrigImage;
ImagePixels = size(PrelimCroppedImage,1)*size(PrelimCroppedImage,2);
for Channel = 1:size(PrelimCroppedImage,3),
    PrelimCroppedImage((Channel-1)*ImagePixels + find(BinaryCropImage == 0)) = 0;
end
drawnow
%%% Removes Rows and Columns that are completely blank.
ColumnTotals = sum(BinaryCropImage,1);
RowTotals = sum(BinaryCropImage,2)';
warning off all
ColumnsToDelete = ~logical(ColumnTotals);
RowsToDelete = ~logical(RowTotals);
warning on all
drawnow
CroppedImage = PrelimCroppedImage;
CroppedImage(:,ColumnsToDelete,:) = [];
CroppedImage(RowsToDelete,:,:) = [];
%%% The Binary Crop Mask image is saved to the handles
%%% structure so it can be used in subsequent image sets to
%%% show which parts of the image were cropped (this will be used
%%% by CPgraythresh).
BinaryCropMaskImage = BinaryCropImage;
BinaryCropMaskImage(:,ColumnsToDelete,:) = [];
BinaryCropMaskImage(RowsToDelete,:,:) = [];
fieldname = ['CropMask', CroppedImageName];
handles.Pipeline.(fieldname) = BinaryCropMaskImage;