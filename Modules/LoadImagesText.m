function handles = AlgLoadImagesText(handles)

% Help for Load Images Text module:
% 
% This module is required to load images from the hard drive into a
% format recognizable by CellProfiler.  The images are given a
% meaningful name, which is then used by subsequent modules to retrieve
% the proper image.  If more than four images per set must be loaded,
% more than one Load Images Order module can be run sequentially. Running
% more than one of these modules also allows images to be retrieved from
% different folders.
%  
% This module is different from the Load Images Order module because
% Load Images Text can be used to load images that are not in a defined
% order.  That is, Load Images Order is useful when images are present
% in a repeating order, like DAPI, FITC, Red, DAPI, FITC, Red, and so
% on, where images are selected based on how many images are in each
% set and what position within each set a particular color is located
% (e.g. three images per set, DAPI is always first).  Load Images Text
% is used instead to load images that have a particular piece of text
% in the name.
% 
% You may have folders within the directory that is being searched, but
% these folders must not contain the text you are searching for or an
% error will result.

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
% The Original Code is the Load Images Text module.
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

%textVAR01 = Type the text that this set of images has in common
%defaultVAR01 = DAPI
TextToFind1 = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What do you want to call these images?
%defaultVAR02 = OrigBlue
ImageName1 = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = Type the text that this set of images has in common
%defaultVAR03 = /
TextToFind2 = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});

%textVAR04 = What do you want to call these images?
%defaultVAR04 = /
ImageName2 = char(handles.Settings.Vvariable{CurrentAlgorithmNum,4});

%textVAR05 = Type the text that this set of images has in common
%defaultVAR05 = /
TextToFind3 = char(handles.Settings.Vvariable{CurrentAlgorithmNum,5});

%textVAR06 = What do you want to call these images?
%defaultVAR06 = /
ImageName3 = char(handles.Settings.Vvariable{CurrentAlgorithmNum,6});

%textVAR07 = Type the text that this set of images has in common
%defaultVAR07 = /
TextToFind4 = char(handles.Settings.Vvariable{CurrentAlgorithmNum,7});

%textVAR08 = What do you want to call these images?
%defaultVAR08 = /
ImageName4 = char(handles.Settings.Vvariable{CurrentAlgorithmNum,8});

%textVAR09 = If an image slot is not being used, type a slash  /  in the box.
%textVAR10 = Type the file format of the images
%defaultVAR10 = tif
FileFormat = char(handles.Settings.Vvariable{CurrentAlgorithmNum,10});

%textVAR11 = Carefully type the directory path name where the images to be loaded are located
%defaultVAR11 = Default Directory - leave this text to retrieve images from the directory specified in STEP1
TypedPathName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,11});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the current directory so the module can switch back at the
%%% end.
CurrentDirectory = cd;
%%% Determines which image set is being analyzed.
SetBeingAnalyzed = handles.setbeinganalyzed;
%%% Stores the text the user entered into cell arrays.
TextToFind{1} = TextToFind1;
TextToFind{2} = TextToFind2;
TextToFind{3} = TextToFind3;
TextToFind{4} = TextToFind4;
ImageName{1} = ImageName1;
ImageName{2} = ImageName2;
ImageName{3} = ImageName3;
ImageName{4} = ImageName4;

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
            error('The image file type entered in the Load Images Text module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.')
        end
    end
    %%% If the user did not enter any data in the first slot (they put
    %%% a slash in either box), no images are retrieved.
    if strcmp(TextToFind{1}, '/') == 1 || strcmp(ImageName{1}, '/') == 1
        error('Image processing was canceled because the first image slot in the Load Images Text module was left blank.')
    end
    %%% For all 4 image slots, extracts the file names.
    for n = 1:4
        %%% Checks whether the two variables required have been entered by
        %%% the user.
        if strcmp(TextToFind{n}, '/') == 0 && strcmp(ImageName{n}, '/') == 0
            if strncmp(TypedPathName, 'Default', 7) == 1
                FileNames = handles.Vfilenames;
                PathName = handles.Vpathname;
                cd(PathName)
                %%% Loops through the names in the FileNames listing, looking for the text
                %%% of interest.  Creates the array Match which contains the numbers of the
                %%% file names that match.
                Count = 1;
                if exist('Match','var') ~= 0
                    clear('Match')
                end 
                for i=1:length(FileNames),
                    if findstr(char(FileNames(i)), char(TextToFind(n))),
                        Match(Count) = i;
                        Count = Count + 1;
                    end
                end
                if exist('Match','var') == 0
                    error(['Image processing was canceled because no image files containing the text you specified (', char(TextToFind(n)), ') were found in the directory you specified: ', PathName, '.'])
                end
                %%% Creates the File List by extracting the names of files
                %%% that matched the text of interest.
                FileList{n} = FileNames(Match);
            else
                %%% If a directory was typed in, retrieves the filenames
                %%% from the chosen directory.
                if exist(TypedPathName,'var') ~= 7
                    error('Image processing was canceled because the directory typed into the Load Images Text module does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.')
                else
                    PathName = TypedPathName;
                    %%% Lists the contents of the chosen directory.
                    DirectoryListing = dir(PathName);
                    %%% Loops through the names in the Directory listing, looking for the text
                    %%% of interest.  Creates the array Match which contains the numbers of the
                    %%% file names that match.
                    Count = 1;
                    if exist('Match','var') ~= 0
                        clear('Match')
                    end 
                    for i=1:length(DirectoryListing),
                        if findstr(DirectoryListing(i).name, char(TextToFind(n))),
                            Match(Count) = i;
                            Count = Count + 1;
                        end
                    end
                    if exist('Match','var') == 0
                        error(['Image processing was canceled because no image files containing the text you specified (', char(TextToFind(n)), ') were found in the directory you specified: ', PathName, '.'])
                    end
                    %%% The File List is created by extracting only the names of files (not the
                    %%% directory or other information stored in the Directory Listing) and
                    %%% only those files that matched the text of interest.
                    FileList{n} = {DirectoryListing(Match).name};
                end % Goes with: if exist - if the directory typed in exists error checking.
            end % Goes with: if strncmp
            %%% Saves the File Lists and Path Names to the handles structure.
            fieldname = ['dOTFileList', ImageName{n}];
            handles.(fieldname) = FileList{n};
            fieldname = ['dOTPathName', ImageName{n}];
            handles.(fieldname) = PathName;
            NumberOfFiles{n} = num2str(length(FileList{n}));
        end % Goes with: if isempty
    end  % Goes with: for i = 1:5
    %%% Determines which slots are empty.  None should be zero, because there is
    %%% an error check for that when looping through n = 1:5.
    for g = 1: length(NumberOfFiles)
        LogicalSlotsToBeDeleted(g) =  isempty(NumberOfFiles{g});
    end
    %%% Removes the empty slots from both the Number of Files array and the
    %%% Image Name array.
    NumberOfFiles = NumberOfFiles(~LogicalSlotsToBeDeleted);
    ImageName2 = ImageName(~LogicalSlotsToBeDeleted);
    %%% Determines how many unique numbers of files there are.  If all the image
    %%% types have loaded the same number of images, there should only be one
    %%% unique number, which is the number of image sets.
    UniqueNumbers = unique(NumberOfFiles);
    %%% If NumberOfFiles is not all the same number at each position, generate an error.
    if length(UniqueNumbers) ~= 1
        CharImageName = char(ImageName2);
        CharNumberOfFiles = char(NumberOfFiles);
        Number = length(CharNumberOfFiles);
        for f = 1:Number
            SpacesArray(f,:) = ':     ';
        end
        PreErrorText = cat(2, CharImageName, SpacesArray);
        ErrorText = cat(2, PreErrorText, CharNumberOfFiles);
        msgbox(ErrorText)
        error('In the Load Images Text module, the number of images identified for each image type is not equal.  In the window under this box you will see how many images have been found for each image type.')
    end
    NumberOfImageSets = str2double(UniqueNumbers{1});
    %%% Checks whether another load images module has already recorded a
    %%% number of image sets.  If it has, it will not be set at the default
    %%% of 1.  Then, it checks whether the number already stored as the
    %%% number of image sets is equal to the number of image sets that this
    %%% module has found.  If not, an error message is generated. Note:
    %%% this will not catch the case where the number of image sets
    %%% detected by this module is more than 1 and another module has
    %%% detected only one image set, since there is no way to tell whether
    %%% the 1 stored in handles.Vnumberimagesets is the default value or a
    %%% value determined by another image-loading module.
    if handles.Vnumberimagesets ~= 1;
        if handles.Vnumberimagesets ~= NumberOfImageSets
            error(['The number of image sets loaded by the Load Images Text module (', num2str(NumberOfImageSets),') does not equal the number of image sets loaded by another image-loading module (', num2str(handles.Vnumberimagesets), '). Please check the settings.'])    
        end
    end
    handles.Vnumberimagesets = NumberOfImageSets;
end % Goes with: if SetBeingAnalyzed == 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOADING IMAGES EACH TIME %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for n = 1:4
    %%% This try/catch will catch any problems in the load images module.
    try
        if strcmp(TextToFind{n}, '/') == 0 && strcmp(ImageName{n}, '/') == 0
            %%% The following runs every time through this module (i.e. for
            %%% every image set).
            %%% Determines which image to analyze.
            fieldname = ['dOTFileList', ImageName{n}];
            FileList = handles.(fieldname);
            %%% Determines the file name of the image you want to analyze.
            CurrentFileName = FileList(SetBeingAnalyzed);
            %%% Determines the directory to switch to.
            if (~ isfield(handles, 'parallel_machines')),
                fieldname = ['dOTPathName', ImageName{n}];
                PathName = handles.(fieldname);
            else
                PathName = handles.RemoteImagePathName;
            end
            %%% Switches to the directory
            cd(PathName);
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
                    LoadedImage = im2double(imread(char(CurrentFileName),FileFormat));
                catch error(['Image processing was canceled because the Load Images Text module could not load the image "', char(CurrentFileName), '" which you specified is in "', FileFormat, '" file format.'])
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
        error(['An error occurred when trying to load the ', ErrorNumber{n}, ' set of images using the Load Images Text module. Please check the settings. A common problem is that there are non-image files in the directory you are trying to analyze, or that the image file is not in the format you specified: ', FileFormat, '. Matlab says the problem is: ', ErrorMessage])
    end % Goes with: catch
end
%%% Changes back to the original directory.
cd(CurrentDirectory)

%%%%%%%%%%%%%%%%%%%%
%%% FIGURE WINDOW %%%
%%%%%%%%%%%%%%%%%%%%

if SetBeingAnalyzed == 1
    %%% The figure window display is unnecessary for this module, so the figure
    %%% window is closed the first time through the module.
    %%% Determines the figure number.
    fieldname = ['figurealgorithm',CurrentAlgorithm];
    ThisAlgFigureNumber = handles.(fieldname);
    %%% Closes the window if it is open.
    if any(findobj == ThisAlgFigureNumber) == 1;
        close(ThisAlgFigureNumber)
    end
end