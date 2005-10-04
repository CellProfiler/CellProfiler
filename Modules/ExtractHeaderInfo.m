function handles = ExtractHeaderInfo(handles)

% Help for the Extract Header Info module:
% Category: File Processing
%
% This module was written for an old version of CellProfiler and may
% not be functional anymore, but it serves as an example of how to
% extract header info from an unusual file format.  These are images
% acquired using ISee software from an automated microscope.
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

drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

%%% Reads the current module number, because this is needed to find 
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What do you want to call the images saved in the first location?
%defaultVAR01 = CFP
%infotypeVAR01 = imagegroup indep
FirstImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call the images saved in the third location?
%defaultVAR02 = DAPI
%infotypeVAR02 = imagegroup indep
ThirdImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What do you want to call the images saved in the fifth location?
%defaultVAR03 = YFP
%infotypeVAR03 = imagegroup indep
FifthImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%pathnametextVAR04 = Enter the directory path name where the images are saved.  Type period (.) for default directory.
PathName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% determines the set number being analyzed
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;
ImagesPerSet = 5;
SpecifiedPathName = PathName;

%%% Stores the text the user entered into cell arrays.
NumberInSet{1} = 1;
NumberInSet{3} = 3;
NumberInSet{5} = 5;

ImageName{1} = FirstImageName;
ImageName{3} = ThirdImageName;
ImageName{5} = FifthImageName;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIRST IMAGE SET FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Extracting the list of files to be analyzed occurs only the first time
%%% through this module.
if SetBeingAnalyzed == 1
    %%% For all 3 image slots, the file names are extracted.
    for n = 1:2:5
        if strncmp(SpecifiedPathName, 'Default', 7) == 1
            PathName = handles.Current.DefaultImageDirectory;
            FileNames = handles.Current.FilenamesInImageDir;
            if SetBeingAnalyzed == 1
                %%% Determines the number of image sets to be analyzed.
                NumberOfImageSets = fix(length(FileNames)/ImagesPerSet);
                %%% The number of image sets is stored in the
                %%% handles structure.
                handles.Current.NumberOfImageSets = NumberOfImageSets;
            else NumberOfImageSets = handles.Current.NumberOfImageSets;
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
            %%% If a directory was typed in, the filenames are retrieved
            %%% from the chosen directory.
            if exist(SpecifiedPathName) ~= 7
                error('Image processing was canceled because the directory typed into the Extract Header Info module does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.')
            else [handles, FileNames] = RetrieveImageFileNames(handles, SpecifiedPathName);
                if SetBeingAnalyzed == 1
                    %%% Determines the number of image sets to be analyzed.
                    NumberOfImageSets = fix(length(FileNames)/ImagesPerSet);
                    handles.Current.NumberOfImageSets = NumberOfImageSets;
                else NumberOfImageSets = handles.Current.NumberOfImageSets;
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
        
    end  % Goes with: for n = 1:2:5
    %%% Update the handles structure.
    guidata(gcbo, handles);    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOADING IMAGES EACH TIME %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for n = 1:2:5
    %%% This try/catch will catch any problems in the EHI module.
    try 
        %%% Determine which image to analyze.
        fieldname = ['dOTFileList', ImageName{n}];
        FileList = handles.(fieldname);
        %%% Determine the file name of the image you want to analyze.
        CurrentFileName = FileList(SetBeingAnalyzed);
        %%% Determine the directory to switch to.
        fieldname = ['dOTPathName', ImageName{n}];
        PathName = handles.(fieldname);
        try
            %%% run the header info function on the loaded image
            [ExpTime, ExpNum, WorldXYZ, TimeDate] = ExtractHeaderInfo(fullfile(PathName,char(CurrentFileName)));
        catch error(['You encountered an error during the subfunction "ExtractHeaderInfo".  Not a good thing.'])
        end
        %%% Converts the WorldXYZ data into three separate variables
        WorldXYZChar = char(WorldXYZ);
        equals = '=';
        equalslocations = strfind(WorldXYZChar,equals);
        WorldX = WorldXYZChar(3:(equalslocations(2)-2));
        WorldY = WorldXYZChar((equalslocations(2)+1):(equalslocations(3)-2));
        WorldZ = WorldXYZChar((equalslocations(3)+1):end);
        %%% Saves the original image file name to the handles structure.  The field
        %%% is named appropriately based on the user's input, with the
        %%% 'dOT' prefix added so that this field will be deleted at
        %%% the end of the analysis batch.
        fieldname = ['dOTFilename', ImageName{n}];
        handles.(fieldname)(SetBeingAnalyzed) = CurrentFileName;
        %%% Saves the extracted header information to the handles
        %%% structure, naming it with dMT because there is a new one
        %%% for each image but it must be deleted from the handles
        %%% structure anyway.  The field name comes from the user's
        %%% input.
        fieldname = ['dMTExpTime',ImageName{n}];
        handles.(fieldname)(SetBeingAnalyzed) = {(ExpTime)};
        fieldname = ['dMTWorldX',ImageName{n}];
        handles.(fieldname)(SetBeingAnalyzed) = {(WorldX)};
        fieldname = ['dMTWorldY',ImageName{n}];
        handles.(fieldname)(SetBeingAnalyzed) = {(WorldY)};
        fieldname = ['dMTWorldZ',ImageName{n}];
        handles.(fieldname)(SetBeingAnalyzed) = {(WorldZ)};
        fieldname = ['dMTTimeDate',ImageName{n}];
        handles.(fieldname)(SetBeingAnalyzed) = {(TimeDate)};
    catch ErrorMessage = lasterr;
        ErrorNumber(1) = {'first'};
        ErrorNumber(2) = {'second'};
        ErrorNumber(3) = {'third'};
        error(['An error occurred when trying to load the ', ErrorNumber{n}, ' set of images using the Extract Header Information module. Please check the settings. A common problem is that there are non-image files in the directory you are trying to analyze. Matlab says the problem is: ', ErrorMessage])
    end % Goes with: catch
end

%%% Update the handles structure.
guidata(gcbo, handles);

%%%%%%%%%%%%%%%%%%%%
%%% FIGURE WINDOW %%%
%%%%%%%%%%%%%%%%%%%%

if SetBeingAnalyzed == handles.Current.StartingImageSet
    %%% The figure window display is unnecessary for this module, so the figure
    %%% window is closed the first time through the module.
    %%% Determines the figure number.
    fieldname = ['FigureNumberForModule',CurrentModule];
    ThisModuleFigureNumber = handles.Current.(fieldname);
    %%% If the window is open, it is closed.
    if any(findobj == ThisModuleFigureNumber) == 1;
        close(ThisModuleFigureNumber)
    end
end

%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%

function [exp_time_ms, exp_num, worldxyz, timedate] = ExtractHeaderInfo(filename)

fid = fopen(filename, 'r', 'l');

fseek(fid, 28, 'bof');
exp_time_ms = fread(fid, 1, 'int32');
exp_num = fread(fid, 1, 'int32');

fseek(fid, 44, 'bof');
timedate = char(fread(fid, 36, 'char')');

fseek(fid, 260, 'bof');
worldxyz = char(fread(fid, 36, 'char')');

fclose(fid);



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
    handles.Current.FilenamesInImageDir = [];
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
