function handles = LoadImagesOrder(handles)

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
% You may have subfolders within the folder that is being searched, but the
% names of the folders themselves must not contain the text you are
% searching for or an error will result.
%
% SAVING IMAGES: The images loaded by this module can be easily saved
% using the Save Images module, using the name you assign (e.g.
% OrigBlue).  In the Save Images module, the images can be saved in a
% different format, allowing this module to function as a file format
% converter.
%
% See also LOADIMAGESTEXT.

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

%textVAR01 = The images to be loaded are located in what position in each set?
%choiceVAR01 = 1
%choiceVAR01 = 2
%choiceVAR01 = 3
%choiceVAR01 = 4
NumberInSet{1} = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,1}));
%inputtypeVAR01 = popupmenu


%textVAR02 = What do you want to call these images?
%infotypeVAR02 = imagegroup indep
%defaultVAR02 = OrigBlue
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = The images to be loaded are located in what position in each set?
%choiceVAR03 = N/A
%choiceVAR03 = 1
%choiceVAR03 = 2
%choiceVAR03 = 3
%choiceVAR03 = 4
NumberInSet{2} = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,3}));
%inputtypeVAR03 = popupmenu

%textVAR04 = What do you want to call these images?
%infotypeVAR04 = imagegroup indep
%defaultVAR04 = OrigGreen
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = The images to be loaded are located in what position in each set?
%choiceVAR05 = N/A
%choiceVAR05 = 1
%choiceVAR05 = 2
%choiceVAR05 = 3
%choiceVAR05 = 4
NumberInSet{3} = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,5}));
%inputtypeVAR05 = popupmenu

%textVAR06 = What do you want to call these images?
%infotypeVAR06 = imagegroup indep
%defaultVAR06 = OrigRed
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = The images to be loaded are located in what position in each set?
%choiceVAR07 = N/A
%choiceVAR07 = 1
%choiceVAR07 = 2
%choiceVAR07 = 3
%choiceVAR07 = 4
NumberInSet{4} = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,7}));
%inputtypeVAR07 = popupmenu

%textVAR08 = What do you want to call these images?
%infotypeVAR08 = imagegroup indep
%defaultVAR08 = OrigOther
ImageName{4} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = How many images are there in each set (i.e. each field of view)?
%choiceVAR09 = 1
%choiceVAR09 = 2
%choiceVAR09 = 3
%choiceVAR09 = 4
ImagesPerSet = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,9}));
%inputtypeVAR09 = popupmenu

%textVAR10 = Type the file format of the images
%choiceVAR10 = tif
%choiceVAR10 = bmp
%choiceVAR10 = gif
%choiceVAR10 = jpg
%choiceVAR10 = mat
%choiceVAR10 = DIB
FileFormat = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu

%textVAR11 = Analyze all subdirectories within the selected directory?
%choiceVAR11 = No
%choiceVAR11 = Yes
AnalyzeSubDir = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu

%pathnametextVAR12 = Enter the path name to the folder where the images to be loaded are located.
Pathname = char(handles.Settings.VariableValues{CurrentModuleNum,12});

%%%VariableRevisionNumber = 5

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines which image set is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

%%% Remove slashes entries with N/A or no filename from the input,
%%% i.e., store only valid entries
tmp1 = {};
tmp2 = {};
for n = 1:length(ImageName)
    if ~isempty(NumberInSet{n}) && ~strcmp(ImageName{n}, '/') && ~isempty(ImageName{n})
        tmp1{end+1} = NumberInSet{n};
        tmp2{end+1} = ImageName{n};
    end
end
NumberInSet = tmp1;
ImageName = tmp2;

%%% Checks whether the position in set exceeds the number per set.
if ImagesPerSet < max([NumberInSet{:}])
    error(['Image processing was canceled during the Load Images Order module because the position of one of the image types within each image set exceeds the number of images per set that you entered (', num2str(ImagesPerSet), ').'])
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIRST IMAGE SET FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%% Extracting the list of files to be analyzed occurs only the first time
%%% through this module.
if SetBeingAnalyzed == 1

    %%% Get the pathname and check that it exists
    if strcmp(Pathname, '.')
        Pathname = handles.Current.DefaultImageDirectory;
    end
    SpecifiedPathname = Pathname;
    if ~exist(SpecifiedPathname,'dir')
        errordlg(['Image processing was canceled because the directory "',SpecifiedPathname,'" does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.'])
    end

    % Get all filenames in the specified directory wich contains the specified extension (e.g., .tif, .jpg, .DIB).
    % Note that there is no check that the extensions actually is the last part of the filename.
    FileNames = CPretrieveMediaFileNames(SpecifiedPathname,['.',FileFormat],AnalyzeSubDir,'Regular','Image');

    %%% Checks whether any files have been specified.
    if isempty(FileNames)
        errordlg(['Image processing was canceled because there are no image files in the chosen directory (or subdirectories, if you requested them to be analyzed as well), according to the Load Images Order module.'])
    end

    %%% Determines the number of image sets to be analyzed.
    NumberOfImageSets = length(FileNames)/ImagesPerSet;
    if rem(NumberOfImageSets,1) ~= 0
        errordlg(sprintf('Image processing was canceled because because the number of image files (%d) found in the specified directory is not a multiple of the number of images per set (%d), according to the Load Images Order module.',length(FileNames),ImagesPerSet));
    end
    handles.Current.NumberOfImageSets = NumberOfImageSets;

    %%% For all valid entries, write list of image files to handles structure
    for n = 1:length(ImageName)
        % Get the list of filenames
        FileList = FileNames(NumberInSet{n}:ImagesPerSet:end);

        %%% Saves the File Lists and Path Names to the handles structure.
        fieldname = ['FileList', ImageName{n}];
        handles.Pipeline.(fieldname) = FileList;
        fieldname = ['Pathname', ImageName{n}];
        handles.Pipeline.(fieldname) = SpecifiedPathname;
    end
    clear FileNames
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOADING IMAGES EACH TIME %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for n = 1:length(ImageName)
    %%% This try/catch will catch any problems in the load images module.
    try
        %%% Determines which image to analyze.
        fieldname = ['FileList', ImageName{n}];
        FileList = handles.Pipeline.(fieldname);
        %%% Determines the file name of the image you want to analyze.
        CurrentFileName = FileList(SetBeingAnalyzed);
        %%% Determines the directory to switch to.
        fieldname = ['Pathname', ImageName{n}];
        Pathname = handles.Pipeline.(fieldname);
        [LoadedImage, handles] = CPimread(fullfile(Pathname,CurrentFileName{1}), handles);
        %%% Saves the original image file name to the handles
        %%% structure.  The field is named appropriately based on
        %%% the user's input, in the Pipeline substructure so that
        %%% this field will be deleted at the end of the analysis
        %%% batch.
        fieldname = ['Filename', ImageName{n}];
        handles.Pipeline.(fieldname)(SetBeingAnalyzed) = CurrentFileName;
        handles.Pipeline.(ImageName{n}) = LoadedImage;
    catch ErrorMessage = lasterr;
        ErrorNumber = {'first','second','third','fourth'};
        error(['An error occurred when trying to load the ', ErrorNumber{n}, ' set of images using the Load Images Order module. Please check the settings. A common problem is that there are non-image files in the directory you are trying to analyze. Matlab says the problem is: ', ErrorMessage])
    end % Goes with: catch

    % Create a cell array with the filenames
    FileNames(n) = CurrentFileName(1);
end

%%% -- Save to the handles.Measurements structure for reference in output files --------------- %%%
%%% NOTE: The structure for filenames and pathnames will be a cell array of cell arrays

%%% First, fix feature names and the pathname
PathNames = cell(1,length(ImageName));
FileNamesText = cell(1,length(ImageName));
PathNamesText = cell(1,length(ImageName));
for n = 1:length(ImageName)
    PathNames{n} = Pathname;
    FileNamesText{n} = ['Filename ', ImageName{n}];
    PathNamesText{n} = ['Path ', ImageName{n}];
end

%%% Since there may be several load modules in the pipeline which all write to the
%%% handles.Measurements.Image.FileName field, we have store filenames in an "appending" style.
%%% Here we check if any of the modules above the current module in the pipline has written to
%%% handles.Measurements.Image.Filenames. Then we should append the current filenames and path
%%% names to the already written ones.
if  isfield(handles,'Measurements') && isfield(handles.Measurements,'Image') &&...
        length(handles.Measurements.Image.FileNames) == SetBeingAnalyzed
    % Get existing file/path names. Returns a cell array of names
    ExistingFileNamesText = handles.Measurements.Image.FileNamesText;
    ExistingFileNames     = handles.Measurements.Image.FileNames{SetBeingAnalyzed};
    ExistingPathNamesText = handles.Measurements.Image.PathNamesText;
    ExistingPathNames     = handles.Measurements.Image.PathNames{SetBeingAnalyzed};

    % Append current file names to existing file names
    FileNamesText = cat(2,ExistingFileNamesText,FileNamesText);
    FileNames     = cat(2,ExistingFileNames,FileNames);
    PathNamesText = cat(2,ExistingPathNamesText,PathNamesText);
    PathNames     = cat(2,ExistingPathNames,PathNames);
end

%%% Write to the handles.Measurements.Image structure
handles.Measurements.Image.FileNamesText                   = FileNamesText;
handles.Measurements.Image.FileNames(SetBeingAnalyzed)         = {FileNames};
handles.Measurements.Image.PathNamesText                   = PathNamesText;
handles.Measurements.Image.PathNames(SetBeingAnalyzed)         = {PathNames};
%%% ------------------------------------------------------------------------------------------------ %%%




%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%

if SetBeingAnalyzed == handles.Current.StartingImageSet
    %%% The figure window display is unnecessary for this module, so the figure
    %%% window is closed the first time through the module.
    %%% Determines the figure number.
    fieldname = ['FigureNumberForModule',CurrentModule];
    ThisModuleFigureNumber = handles.Current.(fieldname);
    %%% Closes the window if it is open.
    if any(findobj == ThisModuleFigureNumber) == 1;
        close(ThisModuleFigureNumber)
    end
    drawnow
end


% PROGRAM NOTES THAT ARE UNNECESSARY FOR THIS MODULE:



