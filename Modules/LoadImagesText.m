function handles = LoadImagesText(handles)

% Help for the Load Images Text module:
% Category: File Processing
%
% Tells CellProfiler where to retrieve images and gives each image a
% meaningful name for the other modules to access.
%
% If more than four images per set must be loaded, more than one Load
% Images Text module can be run sequentially. Running more than one of
% these modules also allows images to be retrieved from different
% folders. If you want to load all images in a directory, you can
% enter the file extension as the text for which to search.
%
% This module is different from the Load Images Order module because
% Load Images Text can be used to load images that are not in a
% defined order.  That is, Load Images Order is useful when images are
% present in a repeating order, like DAPI, FITC, Red, DAPI, FITC, Red,
% and so on, where images are selected based on how many images are in
% each set and what position within each set a particular color is
% located (e.g. three images per set, DAPI is always first).  Load
% Images Text is used instead to load images that have a particular
% piece of text in the name.
%
% You have the option of matching text exactly, or using regular
% expressions to match text. For example, typing image[12]dapi in the box
% asking for text in common and typing R in the Exact/Regular expression
% box will select any file containing the digit 1 or 2 immediately in
% between the text 'image' and 'dapi'.
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
% See also LOADIMAGESORDER.

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

%textVAR01 = Type the text that this set of images has in common
%defaultVAR01 = DAPI
TextToFind{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call these images?
%infotypeVAR02 = imagegroup indep
%defaultVAR02 = OrigBlue
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Type the text that this set of images has in common
%defaultVAR03 = /
TextToFind{2} = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What do you want to call these images?
%infotypeVAR04 = imagegroup indep
%defaultVAR04 = /
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = Type the text that this set of images has in common
%defaultVAR05 = /
TextToFind{3} = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = What do you want to call these images?
%infotypeVAR06 = imagegroup indep
%defaultVAR06 = /
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Type the text that this set of images has in common
%defaultVAR07 = /
TextToFind{4} = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = What do you want to call these images?
%infotypeVAR08 = imagegroup indep
%defaultVAR08 = /
ImageName{4} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = If an image slot above is not being used, type a slash  /  in the box. Do you want to match the text exactly or use regular expressions?
%choiceVAR09 = Exact
%choiceVAR09 = Regular expressions
ExactOrRegExp = char(handles.Settings.VariableValues{CurrentModuleNum,9});
ExactOrRegExp = ExactOrRegExp(1);
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

%textVAR11 = Analyze all subdirectories within the selected directory (Y or N)?
%choiceVAR11 = No
%choiceVAR11 = Yes
AnalyzeSubDir = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu

%pathnametextVAR12 = Enter the path name to the folder where the images to be loaded are located.
Pathname = char(handles.Settings.VariableValues{CurrentModuleNum,12});

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines which image set is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

%%% Remove slashes '/' from the input
tmp1 = {};
tmp2 = {};
for n = 1:4
    if ~strcmp(TextToFind{n}, '/') && ~strcmp(ImageName{n}, '/')
        tmp1{end+1} = TextToFind{n};
        tmp2{end+1} = ImageName{n};
    end
end
TextToFind = tmp1;
ImageName = tmp2;


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
        error(['Image processing was canceled because the directory "',SpecifiedPathname,'" does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.'])
    end

    %%% Extract the file names
    for n = 1:length(ImageName)
        FileList = CPretrieveMediaFileNames(SpecifiedPathname,char(TextToFind(n)),AnalyzeSubDir(1), ExactOrRegExp,'Image');
        %%% Checks whether any files are left.
        if isempty(FileList)
            error(['Image processing was canceled because there are no image files with the text "', TextToFind{n}, '" in the chosen directory (or subdirectories, if you requested them to be analyzed as well), according to the LoadImagesText module.'])
        end
        %%% Saves the File Lists and Path Names to the handles structure.
        fieldname = ['FileList', ImageName{n}];
        handles.Pipeline.(fieldname) = FileList;
        fieldname = ['Pathname', ImageName{n}];
        handles.Pipeline.(fieldname) = SpecifiedPathname;

        NumberOfFiles{n} = num2str(length(FileList)); %#ok We want to ignore MLint error checking for this line.
        clear FileList % Prevents confusion when loading this value later, for each image set.
    end

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
        CPmsgbox(ErrorText)
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
    %%% the 1 stored in handles.Current.NumberOfImageSets is the default value or a
    %%% value determined by another image-loading module.
    if handles.Current.NumberOfImageSets ~= 1;
        if handles.Current.NumberOfImageSets ~= NumberOfImageSets
            error(['The number of image sets loaded by the Load Images Text module (', num2str(NumberOfImageSets),') does not equal the number of image sets loaded by another image-loading module (', num2str(handles.Current.NumberOfImageSets), '). Please check the settings.'])
        end
    end
    handles.Current.NumberOfImageSets = NumberOfImageSets;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% LOADING IMAGES EACH TIME %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for n = 1:length(ImageName)
    %%% This try/catch will catch any problems in the load images module.
    try
        %%% The following runs every time through this module (i.e. for every image set).
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
        error(['An error occurred when trying to load the ', ErrorNumber{n}, ' set of images using the Load Images Text module. Please check the settings. A common problem is that there are non-image files in the directory you are trying to analyze, or that the image file is not in the format you specified: ', FileFormat, '. Matlab says the problem is: ', ErrorMessage])
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


