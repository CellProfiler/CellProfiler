function handles = AlgLoadImagesOrder(handles)

% Help for the Load Images Order module:
% Category: File Handling
%
% Tells CellProfiler where to retrieve images and gives each image a
% meaningful name for the other modules to access.
%
% If more than four images per set must be loaded, more than one Load
% Images Order module can be run sequentially. Running more than one
% of these modules also allows images to be retrieved from different
% folders.  If you want to load all images in a directory, the number
% of images per set can be set to 1.
% 
% Load Images Order is useful when images are present in a repeating
% order, like DAPI, FITC, Red, DAPI, FITC, Red, and so on, where
% images are selected based on how many images are in each set and
% what position within each set a particular color is located (e.g.
% three images per set, DAPI is always first).  By contrast, Load
% Images Text is used to load images that have a particular piece of
% text in the name.
%
% You may have folders within the directory that is being searched;
% they will be ignored by this module.
%
% SAVING IMAGES: The images loaded by this module can be easily saved
% using the Save Images module, using the name you assign (e.g.
% OrigBlue).  In the Save Images module, the images can be saved in a
% different format, allowing this module to function as a file format
% converter.
%
% See also ALGLOADIMAGESTEXT.

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

%textVAR01 = The images to be loaded are located in what position in each set? (1,2,3,...)
%defaultVAR01 = 1
NumberInSet1 = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What do you want to call these images?
%defaultVAR02 = OrigBlue
ImageName1 = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = The images to be loaded are located in what position in each set? (1,2,3,...)
%defaultVAR03 = 0
NumberInSet2 = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});

%textVAR04 = What do you want to call these images?
%defaultVAR04 = OrigGreen
ImageName2 = char(handles.Settings.Vvariable{CurrentAlgorithmNum,4});

%textVAR05 = The images to be loaded are located in what position in each set? (1,2,3,...)
%defaultVAR05 = 0
NumberInSet3 = char(handles.Settings.Vvariable{CurrentAlgorithmNum,5});

%textVAR06 = What do you want to call these images?
%defaultVAR06 = OrigRed
ImageName3 = char(handles.Settings.Vvariable{CurrentAlgorithmNum,6});

%textVAR07 = The images to be loaded are located in what position in each set? (1,2,3,...)
%defaultVAR07 = 0
NumberInSet4 = char(handles.Settings.Vvariable{CurrentAlgorithmNum,7});

%textVAR08 = What do you want to call these images?
%defaultVAR08 = OrigOther1
ImageName4 = char(handles.Settings.Vvariable{CurrentAlgorithmNum,8});

%textVAR09 = How many images are there in each set (i.e. each field of view)?
%defaultVAR09 = 3
ImagesPerSet = char(handles.Settings.Vvariable{CurrentAlgorithmNum,9});

%textVAR10 = Type the file format of the images
%defaultVAR10 = tif
FileFormat = char(handles.Settings.Vvariable{CurrentAlgorithmNum,10});

%textVAR11 = Carefully type the directory path name where the images to be loaded are located#LongBox#
%defaultVAR11 = Default Directory - leave this text to retrieve images from the directory specified in STEP1
Pathname = char(handles.Settings.Vvariable{CurrentAlgorithmNum,11});

%textVAR12 = Analyze All Subdirectories (Y or N)?
%defaultVAR12 = N
AnalyzeSubDir = char(handles.Settings.Vvariable{CurrentAlgorithmNum,12});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determines which set is being analyzed.
SetBeingAnalyzed = handles.setbeinganalyzed;
ImagesPerSet = str2double(ImagesPerSet);
if strncmp(Pathname, 'Default', 7) == 1
    Pathname = handles.Vpathname;
end
SpecifiedPathname = Pathname;
%%% If the user left boxes blank, sets the values to 0.
if isempty(NumberInSet1) == 1
    NumberInSet1 = '0';
end
if isempty(NumberInSet2) == 1
    NumberInSet2 = '0';
end
if isempty(NumberInSet3) == 1
    NumberInSet3 = '0';
end
if isempty(NumberInSet4) == 1
    NumberInSet4 = '0';
end
%%% Stores the text the user entered into cell arrays.
NumberInSet{1} = str2double(NumberInSet1);
NumberInSet{2} = str2double(NumberInSet2);
NumberInSet{3} = str2double(NumberInSet3);
NumberInSet{4} = str2double(NumberInSet4);
%%% Checks whether the position in set exceeds the number per set.
Max12 = max(NumberInSet{1}, NumberInSet{2});
Max34 = max(NumberInSet{3}, NumberInSet{4});
Max1234 = max(Max12, Max34);
if ImagesPerSet < Max1234
    error(['Image processing was canceled during the Load Images Order module because the position of one of the image types within each image set exceeds the number of images per set that you entered (', num2str(ImagesPerSet), ').'])
end
ImageName{1} = ImageName1;
ImageName{2} = ImageName2;
ImageName{3} = ImageName3;
ImageName{4} = ImageName4;
%%% Determines the current directory so the module can switch back at the
%%% end.
CurrentDirectory = cd;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIRST IMAGE SET FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%% Extracting the list of files to be analyzed occurs only the first time
%%% through this module.
if SetBeingAnalyzed == 1
    %%% Checks whether the file format the user entered is readable by Matlab.
    IsFormat = imformats(FileFormat);
    if isempty(IsFormat) == 1
        %%% Checks if the image is a DIB image file.
        if strcmp(upper(FileFormat),'DIB') == 1
            Answers = inputdlg({'Enter the width of the images in pixels','Enter the height of the images in pixels','Enter the bit depth of the camera','Enter the number of channels'},'Enter DIB file information',1,{'512','512','12','1'});
            handles.Pipeline.DIBwidth = str2double(Answers{1});
            handles.Pipeline.DIBheight = str2double(Answers{2});
            handles.Pipeline.DIBbitdepth = str2double(Answers{3});
            handles.Pipeline.DIBchannels = str2double(Answers{4});
        else
            error('The image file type entered in the Load Images Order module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.')
        end
    end
    %%% For all 4 image slots, extracts the file names.
    for n = 1:4
        %%% Checks whether the two variables required have been entered by
        %%% the user.
        if NumberInSet{n} ~= 0 && isempty(ImageName{n}) == 0
            %%% If a directory was typed in, retrieves the filenames
            %%% from the chosen directory.
            if exist(SpecifiedPathname) ~= 7
                error('Image processing was canceled because the directory typed into the Load Images Order module does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.')
            else
                [handles, FileNames] = RetrieveImageFileNames(handles, SpecifiedPathname,AnalyzeSubDir);
                if SetBeingAnalyzed == 1
                    %%% Determines the number of image sets to be analyzed.
                    NumberOfImageSets = fix(length(FileNames)/ImagesPerSet);
                    handles.Vnumberimagesets = NumberOfImageSets;
                else NumberOfImageSets = handles.Vnumberimagesets;
                end
                %%% Loops through the names in the FileNames listing,
                %%% creating a new list of files.
                for i = 1:NumberOfImageSets
                    Number = (i - 1) .* ImagesPerSet + NumberInSet{n};
                    FileList(i) = FileNames(Number);
                end
                %%% Saves the File Lists and Path Names to the handles structure.
                fieldname = ['FileList', ImageName{n}];
                handles.Pipeline.(fieldname) = FileList;
                fieldname = ['Pathname', ImageName{n}];
                handles.Pipeline.(fieldname) = SpecifiedPathname;
                clear FileList

            end
        end
    end  % Goes with: for n = 1:4
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOADING IMAGES EACH TIME %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for n = 1:4
    %%% This try/catch will catch any problems in the load images module.
    try
        if NumberInSet{n} ~= 0 && isempty(ImageName{n}) == 0
            %%% Determines which image to analyze.
            fieldname = ['FileList', ImageName{n}];
            FileList = handles.Pipeline.(fieldname);
            %%% Determines the file name of the image you want to analyze.
            CurrentFileName = FileList(SetBeingAnalyzed);
            %%% Determines the directory to switch to.
            fieldname = ['Pathname', ImageName{n}];
            Pathname = handles.Pipeline.(fieldname);
            %%% Switches to the directory
            try
                cd(Pathname);
            catch error(['Could not CD to ' Pathname]);
            end;
            %%% Handles a non-Matlab readable file format.
            if isfield(handles.Pipeline, 'DIBwidth') == 1
                %%% Opens this non-Matlab readable file format.
                Width = handles.Pipeline.DIBwidth;
                Height = handles.Pipeline.DIBheight;
                Channels = handles.Pipeline.DIBchannels;
                BitDepth = handles.Pipeline.DIBbitdepth;
                fid = fopen(char(CurrentFileName), 'r');
                if (fid == -1),
                    error(['The file ', char(CurrentFileName), ' could not be opened. CellProfiler attempted to open it in DIB file format.']);
                end
                fread(fid, 52, 'uchar');
                LoadedImage = zeros(Height,Width,Channels);
                for c=1:Channels,
                    [Data, Count] = fread(fid, Width * Height, 'uint16', 0, 'l');
                    if Count < (Width * Height),
                        fclose(fid);
                        error(['End-of-file encountered while reading ', char(CurrentFileName), '. Have you entered the proper size and number of channels for these images?']);
                    end
                    LoadedImage(:,:,c) = reshape(Data, [Width Height])' / (2^BitDepth - 1);
                end
                fclose(fid);
            else
                %%% Opens Matlab-readable file formats.
                try
                    %%% Read (open) the image you want to analyze and assign it to a variable,
                    %%% "LoadedImage".
                    LoadedImage = im2double(imread(char(CurrentFileName),FileFormat));
                catch error(['Image processing was canceled because the Load Images Order module could not load the image "', char(CurrentFileName), '" in directory "', pwd, '" which you specified is in "', FileFormat, '" file format.  The error message was "', lasterr, '"'])
                end
            end
            %%% Saves the original image file name to the handles.Pipeline structure.
            %%% The field is named appropriately based on the user's input, and will
            %%% be deleted at the end of the analysis batch.
            fieldname = ['Filename', ImageName{n}];
            handles.Pipeline.(fieldname)(SetBeingAnalyzed) = CurrentFileName;
            %%% Also saved to the handles.Measurements structure for reference in output files.
            handles.Measurements.(fieldname)(SetBeingAnalyzed) = CurrentFileName;
            %%% Saves the loaded image to the handles structure.  The field is named
            %%% appropriately based on the user's input.
            handles.Pipeline.(ImageName{n}) = LoadedImage;
        end
    catch ErrorMessage = lasterr;
        ErrorNumber(1) = {'first'};
        ErrorNumber(2) = {'second'};
        ErrorNumber(3) = {'third'};
        ErrorNumber(4) = {'fourth'};
        error(['An error occurred when trying to load the ', ErrorNumber{n}, ' set of images using the Load Images Order module. Please check the settings. A common problem is that there are non-image files in the directory you are trying to analyze. Matlab says the problem is: ', ErrorMessage])
    end % Goes with: catch
end
%%% Changes back to the original directory.
cd(CurrentDirectory)

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

%%%%%%%%%%%%%%%%%%%%
%%% FIGURE WINDOW %%%
%%%%%%%%%%%%%%%%%%%%
drawnow

if SetBeingAnalyzed == 1
    %%% The figure window display is unnecessary for this module, so the figure
    %%% window is closed the first time through the module.
    %%% Determines the figure number.
    fieldname = ['figurealgorithm',CurrentAlgorithm];
    ThisAlgFigureNumber = handles.(fieldname);
    %%% If the window is open, it is closed.
    if any(findobj == ThisAlgFigureNumber) == 1;
        close(ThisAlgFigureNumber)
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION TO RETRIEVE FILE NAMES %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [handles, FileNames] = RetrieveImageFileNames(handles, Pathname,recurse)
%%% Lists all the contents of that path into a structure which includes the
%%% name of each object as well as whether the object is a file or
%%% directory.
FilesAndDirsStructure = dir(Pathname);
%%% Puts the names of each object into a list.
FileAndDirNames = sortrows({FilesAndDirsStructure.name}');
%%% Puts the logical value of whether each object is a directory into a list.
LogicalIsDirectory = [FilesAndDirsStructure.isdir];
%%% Eliminates directories from the list of file names.
FileNamesNoDir = FileAndDirNames(~LogicalIsDirectory);
if isempty(FileNamesNoDir) == 1
    errordlg('There are no files in the chosen directory')
    handles.Vfilenames = [];
else
    %%% Makes a logical array that marks with a "1" all file names that start
    %%% with a period (hidden files):
    DiscardLogical1 = strncmp(FileNamesNoDir,'.',1);
    %%% Makes logical arrays that mark with a "1" all file names that have
    %%% particular suffixes (mat, m, m~, and frk). The dollar sign indicates
    %%% that the pattern must be at the end of the string in order to count as
    %%% matching.  The first line of each set finds the suffix and marks its
    %%% location in a cell array with the index of where that suffix begins;
    %%% the third line converts this cell array of numbers into a logical
    %%% array of 1's and 0's.   cellfun only works on arrays of class 'cell',
    %%% so there is a check to make sure the class is appropriate.  When there
    %%% are very few files in the directory (I think just one), the class is
    %%% not cell for some reason.
    DiscardLogical2Pre = regexpi(FileNamesNoDir, '.mat$','once');
    if strcmp(class(DiscardLogical2Pre), 'cell') == 1
        DiscardLogical2 = cellfun('prodofsize',DiscardLogical2Pre);
    else DiscardLogical2 = [];
    end
    DiscardLogical3Pre = regexpi(FileNamesNoDir, '.m$','once');
    if strcmp(class(DiscardLogical3Pre), 'cell') == 1
        DiscardLogical3 = cellfun('prodofsize',DiscardLogical3Pre);
    else DiscardLogical3 = [];
    end
    DiscardLogical4Pre = regexpi(FileNamesNoDir, '.m~$','once');
    if strcmp(class(DiscardLogical4Pre), 'cell') == 1
        DiscardLogical4 = cellfun('prodofsize',DiscardLogical4Pre);
    else DiscardLogical4 = [];
    end
    DiscardLogical5Pre = regexpi(FileNamesNoDir, '.frk$','once');
    if strcmp(class(DiscardLogical5Pre), 'cell') == 1
        DiscardLogical5 = cellfun('prodofsize',DiscardLogical5Pre);
    else DiscardLogical5 = [];
    end
    %%% Combines all of the DiscardLogical arrays into one.
    DiscardLogical = DiscardLogical1 | DiscardLogical2 | DiscardLogical3 | DiscardLogical4 | DiscardLogical5;
    %%% Eliminates filenames to be discarded.
    if isempty(DiscardLogical) == 1
        FileNames = FileNamesNoDir;
    else FileNames = FileNamesNoDir(~DiscardLogical);
    end

    if(strcmp(upper(recurse),'Y'))
        DirNamesNoFiles = FileAndDirNames(LogicalIsDirectory);
        DiscardLogical1Dir = strncmp(DirNamesNoFiles,'.',1);
        DirNames = DirNamesNoFiles(~DiscardLogical1Dir);
        if (length(DirNames) > 0)
            for i=1:length(DirNames),
                [handles, MoreFileNames] = RetrieveImageFileNames(handles, [Pathname '\' char(DirNames(i))], recurse);
                for j = 1:length(MoreFileNames)
                    MoreFileNames{j} = [char(DirNames(i)) '\' char(MoreFileNames(j))];
                end
                FileNames(end+1:end+length(MoreFileNames)) = MoreFileNames(1:end);
            end
        end
    end

        
    %%% Checks whether any files are left.
    if isempty(FileNames) == 1
        errordlg('There are no image files in the chosen directory')
    end
end

% PROGRAM NOTES THAT ARE UNNECESSARY FOR THIS MODULE:
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