function handles = AlgLoadImagesOrder(handles)

% Help for the Load Images Order module:
%
% Tells CellProfiler where to retrieve images and gives each image a
% meaningful name for the other modules to access.

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
% See also AlgLoadImagesText.

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
% The Original Code is the Load Images Order module.
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

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

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

%textVAR11 = Carefully type the directory path name where the images to be loaded are located
%defaultVAR11 = Default Directory - leave this text to retrieve images from the directory specified in STEP1
PathName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,11});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Determines which set is being analyzed.
SetBeingAnalyzed = handles.setbeinganalyzed;
ImagesPerSet = str2double(ImagesPerSet);
SpecifiedPathName = PathName;
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

%%% Extracting the list of files to be analyzed occurs only the first time
%%% through this module.
if SetBeingAnalyzed == 1
    %%% Checks whether the file format the user entered is readable by Matlab.
    IsFormat = imformats(FileFormat);
    if isempty(IsFormat) == 1
        %%% Checks if the image is a DIB image file.
        if strcmp(upper(FileFormat),'DIB') == 1
            Answers = inputdlg({'Enter the width of the images in pixels','Enter the height of the images in pixels','Enter the bit depth of the camera','Enter the number of channels'},'Enter DIB file information',1,{'512','512','12','1'});
            handles.dOTDIBwidth = str2double(Answers{1});
            handles.dOTDIBheight = str2double(Answers{2});
            handles.dOTDIBbitdepth = str2double(Answers{3});
            handles.dOTDIBchannels = str2double(Answers{4});
        else
            error('The image file type entered in the Load Images Order module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.')
        end
    end
    %%% For all 4 image slots, exracts the file names.
    for n = 1:4
        %%% Checks whether the two variables required have been entered by
        %%% the user.
        if NumberInSet{n} ~= 0 && isempty(ImageName{n}) == 0
            if strncmp(SpecifiedPathName, 'Default', 7) == 1
                PathName = handles.Vpathname;
                FileNames = handles.Vfilenames;
                if SetBeingAnalyzed == 1
                    if length(handles.Vfilenames) < ImagesPerSet
                        error(['In the Load Images Order module, you specified that there are ', num2str(ImagesPerSet),' images per set, but only ', num2str(length(handles.Vfilenames)), ' were found in the chosen directory. Please check the settings.'])    
                    end
                    %%% Determines the number of image sets to be analyzed.
                    NumberOfImageSets = fix(length(handles.Vfilenames)/ImagesPerSet);
                    %%% Checks whether another load images module has
                    %%% already recorded a number of image sets.  If it
                    %%% has, it will not be set at the default of 1.  Then,
                    %%% it checks whether the number already stored as the
                    %%% number of image sets is equal to the number of
                    %%% image sets that this module has found.  If not, an
                    %%% error message is generated. Note: this will not
                    %%% catch the case where the number of image sets
                    %%% detected by this module is more than 1 and another
                    %%% module has detected only one image set, since there
                    %%% is no way to tell whether the 1 stored in
                    %%% handles.Vnumberimagesets is the default value or a
                    %%% value determined by another image-loading module.
                    if handles.Vnumberimagesets ~= 1;
                        if handles.Vnumberimagesets ~= NumberOfImageSets
                        error(['The number of image sets loaded by the Load Images Order module (', num2str(NumberOfImageSets),') does not equal the number of image sets loaded by another image-loading module (', num2str(handles.Vnumberimagesets), '). Please check the settings.'])    
                        end
                    end
                    %%% Stores the number of image sets in the
                    %%% handles structure.
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
                fieldname = ['dOTFileList', ImageName{n}];
                handles.(fieldname) = FileList;
                fieldname = ['dOTPathName', ImageName{n}];
                handles.(fieldname) = PathName;
                clear FileList
            else
                %%% If a directory was typed in, retrieves the filenames
                %%% from the chosen directory.
                if exist(SpecifiedPathName,'var') ~= 7
                    error('Image processing was canceled because the directory typed into the Load Images Order module does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.')
                else [handles, FileNames] = RetrieveImageFileNames(handles, SpecifiedPathName);
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
                    fieldname = ['dOTFileList', ImageName{n}];
                    handles.(fieldname) = FileList;
                    fieldname = ['dOTPathName', ImageName{n}];
                    handles.(fieldname) = PathName;
                    clear FileList
                end
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
            fieldname = ['dOTFileList', ImageName{n}];
            FileList = handles.(fieldname);
            %%% Determines the file name of the image you want to analyze.
            CurrentFileName = FileList(SetBeingAnalyzed);
            %%% Determines the directory to switch to.
            if (~ isfield(handles, 'parallel_machines') || SetBeingAnalyzed == 1),
                fieldname = ['dOTPathName', ImageName{n}];
                PathName = handles.(fieldname);
            else
                PathName = handles.RemoteImagePathName;
            end
            %%% Switches to the directory
            try
                cd(PathName);
            catch error(['Could not CD to ' PathName]);
            end;
            %%% Handles a non-Matlab readable file format.
            if isfield(handles, 'dOTDIBwidth') == 1
                %%% Opens this non-Matlab readable file format.
                Width = handles.dOTDIBwidth;
                Height = handles.dOTDIBheight;
                Channels = handles.dOTDIBchannels;
                BitDepth = handles.dOTDIBbitdepth;
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
            %%% Saves the original image file name to the handles structure.  The field
            %%% is named
            %%% appropriately based on the user's input, with the 'dOT' prefix added so
            %%% that this field will be deleted at the end of the analysis batch.
            fieldname = ['dOTFilename', ImageName{n}];
            handles.(fieldname)(SetBeingAnalyzed) = CurrentFileName;
            %%% Saves the loaded image to the handles structure.The field is named
            %%% appropriately based on the user's input.The prefix 'dOT' is added to
            %%% the beginning of the measurement name so that this field will be
            %%% deleted at the end of the analysis batch.
            fieldname = ['dOT',ImageName{n}];
            handles.(fieldname) = LoadedImage;
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

function [handles, FileNames] = RetrieveImageFileNames(handles, PathName)
%%% Lists all the contents of that path into a structure which includes the
%%% name of each object as well as whether the object is a file or
%%% directory.
FilesAndDirsStructure = dir(PathName);
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
    %%% Checks whether any files are left.
    if isempty(FileNames) == 1
        errordlg('There are no image files in the chosen directory')
    end
end