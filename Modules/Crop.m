function handles = AlgCrop(handles)

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
% in mind that the cropping algorithm will remove rows and columns
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

% The contents of this file are subject to the Mozilla Public License Version 
% 1.1 (the "License"); you may not use this file except in compliance with 
% the License. You may obtain a copy of the License at 
% http://www.mozilla.org/MPL/
% 
% Software distributed under the License is distributed on an "AS IS" basis,
% WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
% for the specific language governing rights and limitations under the
% License.
% 
% 
% The Original Code is the Crop module.
% 
% The Initial Developer of the Original Code is
% Whitehead Institute for Biomedical Research
% Portions created by the Initial Developer are Copyright (C) 2003,2004
% the Initial Developer. All Rights Reserved.
% 
% Contributor(s):
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$

% PROGRAMMING NOTE
% HELP:
% The first unbroken block of lines will be extracted as help by
% CellProfiler's 'Help for this analysis module' button as well as
% Matlab's built in 'help' and 'doc' functions at the command line. It
% will also be used to automatically generate a manual page for the
% module. An example image demonstrating the function of the module
% can also be saved in tif format, using the same name as the
% algorithm (minus Alg), and it will automatically be included in the
% manual page as well.  Follow the convention of: purpose of the
% module, description of the variables and acceptable range for each,
% how it works (technical description), info on which images can be 
% saved, and See also CAPITALLETTEROTHERALGORITHMS. The license/author
% information should be separated from the help lines with a blank
% line so that it does not show up in the help displays.  Do not
% change the programming notes in any modules! These are standard
% across all modules for maintenance purposes, so anything
% module-specific should be kept separate.

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
% The '%textVAR' lines contain the text which is displayed in the GUI
% next to each variable box. The '%defaultVAR' lines contain the
% default values which are displayed in the variable boxes when the
% user loads the algorithm. The line of code after the textVAR and
% defaultVAR extracts the value that the user has entered from the
% handles structure and saves it as a variable in the workspace of
% this algorithm with a descriptive name. The syntax is important for
% the %textVAR and %defaultVAR lines: be sure there is a space before
% and after the equals sign and also that the capitalization is as
% shown.  Don't allow the text to wrap around to another line; the
% second line will not be displayed.  If you need more space to
% describe a variable, you can refer the user to the help file, or you
% can put text in the %textVAR line above or below the one of
% interest, and do not include a %defaultVAR line so that the variable
% edit box for that variable will not be displayed; the text will
% still be displayed. CellProfiler is currently being restructured to
% handle more than 11 variable boxes. Keep in mind that you can have
% several inputs into the same box: for example, a box could be
% designed to receive two numbers separated by a comma, as long as you
% write a little extraction algorithm that separates the input into
% two distinct variables.  Any extraction algorithms like this should
% be within the VARIABLES section of the code, at the end.

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;
CurrentAlgorithmNum = str2double(handles.currentalgorithm);

%textVAR01 = What did you call the image to be cropped?
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What do you want to call the cropped image?
%defaultVAR02 = CropBlue
CroppedImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = For rectangular cropping, type R. For any shape (based on an image File), type F
%textVAR04 = To draw an ellipse on each image, type EE; draw one ellipse for all images: EA
%defaultVAR04 = R
Shape = upper(char(handles.Settings.Vvariable{CurrentAlgorithmNum,4}));

%textVAR05 = Rectangular: enter the pixel position for the left (X), top (Y) corner (with comma)
%defaultVAR05 = 1,1
LeftTop = char(handles.Settings.Vvariable{CurrentAlgorithmNum,5});

%textVAR06 = Rectangular: enter the pixel position for the right (X), bottom (Y) corner (with comma)
%defaultVAR06 = 100,100
RightBottom = char(handles.Settings.Vvariable{CurrentAlgorithmNum,6});

%textVAR10 = File-based cropping: To crop to another shape, type the location and file name of 
%textVAR11 = the binary image to guide the cropping (Zero values will be removed).  Type carefully!  
%defaultVAR11 = /
BinaryCropImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,11});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image to be used for cropping and assigns it to a
%%% variable. There is an error catching mechanism in case the image cannot
%%% be found or opened.
if Shape == 'F'
    try
    BinaryCropImage = imread(BinaryCropImageName);
    catch error(['Image processing was canceled because the Crop module could not find the binary cropping image.  It was supposed to be here: ', BinaryCropImageName, ', but a readable image does not exist at that location or with that name.  Perhaps there is a typo.'])
    end
end

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

        
%%% For cropping from a file image, checks whether the crop image is the
%%% same size as the image to be processed.
if Shape == 'F'
    if size(OrigImage(:,:,1)) ~= size(BinaryCropImage)
        error('Image processing was canceled because the binary image selected for cropping in the Crop module is not the same size as the image to be cropped.  The pixel dimensions must be identical.')
    end 
end

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

if Shape == 'F'
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

elseif Shape == 'R'
    %%% Extracts the top, left, bottom, right pixel positions from the user's
    %%% input.
    LeftTopNumerical = str2num(LeftTop); %#ok We want MLint error checking to ignore this line.
    Left = LeftTopNumerical(1);
    Top = LeftTopNumerical(2);
    RightBottomNumerical = str2num(RightBottom); %#ok We want MLint error checking to ignore this line.
    Right = RightBottomNumerical(1);
    Bottom = RightBottomNumerical(2);
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

elseif strcmp(Shape, 'EA') == 1 || strcmp(Shape, 'EE') == 1
    if handles.setbeinganalyzed == 1 || strcmp(Shape, 'EE') == 1
        if strcmp(Shape, 'EA') == 1
            %%% Asks the user to open an image file upon which to draw the
            %%% ellipse.
            CurrentDirectory = cd;
            Directory = handles.Vpathname;
            cd(Directory)
            %%% Opens a user interface window which retrieves a file name and path
            %%% name for the image to be used as a test image.
            [CroppingFileName,CroppingPathname] = uigetfile('*.*','Select the image to use for cropping');
            %%% If the user presses "Cancel", the FileName will = 0 and an error
            %%% message results.
            if CroppingFileName == 0
                error('Image processing was canceled because you did not select an image to use for cropping in the Crop module.')
            else
                ImageToBeCropped = imread([CroppingPathname,'/',CroppingFileName]);
            end
            cd(CurrentDirectory)
        else ImageToBeCropped = OrigImage;
        end
        %%% Displays the image and asks the user to choose points for the
        %%% ellipse.
        CroppingFigureHandle = figure;
        imshow(ImageToBeCropped), colormap('gray');pixval
        title('Click on 5 or more points to be used to create a cropping ellipse & then press Enter.')
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
        figure;
        imagesc(BinaryCropImage);title('Cropping Mask')
        colormap(gray)
        hold on
        plot(Pre_x,Pre_y, 'r.')
        drawnow
        %%% The Binary Crop image is saved to the handles
        %%% structure so it can be used to crop subsequent image sets.
        fieldname = ['Cropping', CroppedImageName];
        handles.Pipeline.(fieldname) = BinaryCropImage;
    end
    %%% Retrieves previously selected cropping ellipse from handles
    %%% structure.
    fieldname = ['Cropping', CroppedImageName];
    BinaryCropImage = handles.Pipeline.(fieldname);
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
    %%% Removes Rows and Columns that are completely blank.
    ColumnTotals = sum(BinaryCropImage,1);
    RowTotals = sum(BinaryCropImage,2)';
    warning off all
    ColumnsToDelete = ~logical(ColumnTotals);
    RowsToDelete = ~logical(RowTotals);
    warning on all
    CroppedImage = PrelimCroppedImage;
    drawnow
    CroppedImage(:,ColumnsToDelete,:) = [];
    drawnow
    CroppedImage(RowsToDelete,:,:) = [];
else error('You must choose rectangular cropping (R) or cropping from a file (F) or drawing an ellipse (EE or EA) to use the Crop module.')
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

% PROGRAMMING NOTE
% DISPLAYING RESULTS:
% Each module checks whether its figure is open before calculating
% images that are for display only. This is done by examining all the
% figure handles for one whose handle is equal to the assigned figure
% number for this algorithm. If the figure is not open, everything
% between the "if" and "end" is ignored (to speed execution), so do
% not do any important calculations here. Otherwise an error message
% will be produced if the user has closed the window but you have
% attempted to access data that was supposed to be produced by this
% part of the code. If you plan to save images which are normally
% produced for display only, the corresponding lines should be moved
% outside this if statement.

fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
if any(findobj == ThisAlgFigureNumber) == 1;
    %%% Sets the window to be half as wide as usual.
    originalsize = get(ThisAlgFigureNumber, 'position');
    newsize = originalsize;
    newsize(3) = 250;
    set(ThisAlgFigureNumber, 'position', newsize);
% PROGRAMMING NOTE
% DRAWNOW BEFORE FIGURE COMMAND:
% The "drawnow" function executes any pending figure window-related
% commands.  In general, Matlab does not update figure windows until
% breaks between image analysis modules, or when a few select commands
% are used. "figure" and "drawnow" are two of the commands that allow
% Matlab to pause and carry out any pending figure window- related
% commands (like zooming, or pressing timer pause or cancel buttons or
% pressing a help button.)  If the drawnow command is not used
% immediately prior to the figure(ThisAlgFigureNumber) line, then
% immediately after the figure line executes, the other commands that
% have been waiting are executed in the other windows.  Then, when
% Matlab returns to this module and goes to the subplot line, the
% figure which is active is not necessarily the correct one. This
% results in strange things like the subplots appearing in the timer
% window or in the wrong figure window, or in help dialog boxes.
    drawnow
    %%% Activates the appropriate figure window.
    figure(ThisAlgFigureNumber);
    %%% A subplot of the figure window is set to display the original image.
    subplot(2,1,1); imagesc(OrigImage);colormap(gray);
    title(['Input Image, Image Set # ',num2str(handles.setbeinganalyzed)]);
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
% 'structure', and whose name is handles. Data which should be saved
% to the handles structure within each module includes: any images,
% data or measurements which are to be eventually saved to the hard
% drive (either in an output file, or using the SaveImages module) or
% which are to be used by a later module in the analysis pipeline. Any
% module which produces or passes on an image needs to also pass along
% the original filename of the image, named after the new image name,
% so that if the SaveImages module attempts to save the resulting
% image, it can be named by appending text to the original file name.
% handles.Pipeline is for storing data which must be retrieved by other modules.
% This data can be overwritten as each image set is processed, or it
% can be generated once and then retrieved during every subsequent image
% set's processing, or it can be saved for each image set by
% saving it according to which image set is being analyzed.
%       Anything stored in handles.Measurements or handles.Pipeline
% will be deleted at the end of the analysis run, whereas anything
% stored in handles.Settings will be retained from one analysis to the
% next. It is important to think about which of these data should be
% deleted at the end of an analysis run because of the way Matlab
% saves variables: For example, a user might process 12 image sets of
% nuclei which results in a set of 12 measurements ("TotalNucArea")
% stored in the handles structure. In addition, a processed image of
% nuclei from the last image set is left in the handles structure
% ("SegmNucImg"). Now, if the user uses a different algorithm which
% happens to have the same measurement output name "TotalNucArea" to
% analyze 4 image sets, the 4 measurements will overwrite the first 4
% measurements of the previous analysis, but the remaining 8
% measurements will still be present. So, the user will end up with 12
% measurements from the 4 sets. Another potential problem is that if,
% in the second analysis run, the user runs only an algorithm which
% depends on the output "SegmNucImg" but does not run an algorithm
% that produces an image by that name, the algorithm will run just
% fine: it will just repeatedly use the processed image of nuclei
% leftover from the last image set, which was left in the handles
% structure ("SegmNucImg").
%       Note that two types of measurements are typically made: Object
% and Image measurements.  Object measurements have one number for
% every object in the image (e.g. ObjectArea) and image measurements
% have one number for the entire image, which could come from one
% measurement from the entire image (e.g. ImageTotalIntensity), or
% which could be an aggregate measurement based on individual object
% measurements (e.g. ImageMeanArea).  Use the appropriate prefix to
% ensure that your data will be extracted properly.
%       Saving measurements: The data extraction functions of
% CellProfiler are designed to deal with only one "column" of data per
% named measurement field. So, for example, instead of creating a
% field of XY locations stored in pairs, they should be split into a field
% of X locations and a field of Y locations. Measurements must be
% stored in double format, because the extraction part of the program
% is designed to deal with that type of array only, not cell or
% structure arrays. It is wise to include the user's input for
% 'ObjectName' as part of the fieldname in the handles structure so
% that multiple modules can be run and their data will not overwrite
% each other.
%       Extracting measurements: handles.Measurements.CenterXNuclei{1}(2) gives
% the X position for the second object in the first image.
% handles.Measurements.AreaNuclei{2}(1) gives the area of the first object in
% the second image.

%%% Saves the adjusted image to the handles structure so it can be used by
%%% subsequent algorithms.
handles.Pipeline.(CroppedImageName) = CroppedImage;

%%% Determines the filename of the image to be analyzed.
fieldname = ['Filename', ImageName];
FileName = handles.Pipeline.(fieldname)(handles.setbeinganalyzed);
%%% Saves the original file name to the handles structure in a
%%% field named after the adjusted image name.
fieldname = ['Filename', CroppedImageName];
handles.Pipeline.(fieldname)(handles.setbeinganalyzed) = FileName;