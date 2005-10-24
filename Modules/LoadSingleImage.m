function handles = LoadSingleImage(handles)

% Help for the Load Single Image module:
% Category: File Processing
%
% Tells CellProfiler where to retrieve a single image and gives the
% image a meaningful name for the other modules to access.  The module
% only functions the first time through the pipeline, and thereafter
% the image is accessible to all subsequent image sets being
% processed. This is particularly useful for loading an image like the
% Illumination correction image to be used by the CorrectIllumDivide
% module.
%
% Relative pathnames can be used: e.g. enter ../Imagetobeloaded.tif as
% the name of the file you would like to load and leave the image
% directory set to the default image directory in order to load the
% image from the directory one above the default image directory.
%
% SAVING IMAGES: The images loaded by this module can be easily saved
% using the Save Images module, using the name you assign (e.g.
% OrigBlue).  In the Save Images module, the images can be saved in a
% different format, allowing this module to function as a file format
% converter.
%
% See also LOADIMAGESORDER and LOADIMAGESORDERTEXT.

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
%   Ola Friman     <friman@bwh.harvard.edu>
%   Steve Lowe     <stevelowe@alum.mit.edu>
%   Joo Han Chang  <joohan.chang@gmail.com>
%   Colin Clarke   <colinc@mit.edu>
%   Mike Lamprecht <mrl@wi.mit.edu>
%   Susan Ma       <xuefang_ma@wi.mit.edu>
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

%filenametextVAR01 = Type the name of the image file you want to load (include the extension, like .tif)
TextToFind{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call that image?
%defaultVAR02 = OrigBlue
%infotypeVAR02 = imagegroup indep
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%filenametextVAR03 = Type the name of the image file you want to load (include the extension, like .tif)
TextToFind{2} = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What do you want to call that image?
%defaultVAR04 = /
%infotypeVAR04 = imagegroup indep
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%filenametextVAR05 = Type the name of the image file you want to load (include the extension, like .tif)
TextToFind{3} = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = What do you want to call that image?
%defaultVAR06 = /
%infotypeVAR06 = imagegroup indep
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%filenametextVAR07 = Type the name of the image file you want to load (include the extension, like .tif)
TextToFind{4} = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = What do you want to call that image?
%defaultVAR08 = /
%infotypeVAR08 = imagegroup indep
ImageName{4} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = If an image slot is not being used, type a slash  /  in the box.

%textVAR10 = Type the file format of the images
%choiceVAR10 = mat
%choiceVAR10 = bmp
%choiceVAR10 = gif
%choiceVAR10 = jpg
%choiceVAR10 = tif
%choiceVAR10 = DIB
FileFormat = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu

%pathnametextVAR11 = Enter the path name to the folder where the images to be loaded are located.  Type period (.) for default image directory.
Pathname = char(handles.Settings.VariableValues{CurrentModuleNum,11});

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

%%% Get the pathname and check that it exists
if strncmp(Pathname,'.',1)
    if length(Pathname) == 1
        Pathname = handles.Current.DefaultImageDirectory;
    else
        Pathname = fullfile(handles.Current.DefaultImageDirectory,Pathname(2:end));
    end
end
SpecifiedPathname = Pathname;
if ~exist(SpecifiedPathname,'dir')
    error(['Image processing was canceled because the directory "',SpecifiedPathname,'" does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.'])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIRST IMAGE SET FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


for n = 1:length(ImageName)
    %%% This try/catch will catch any problems in the load images module.
    try
        CurrentFileName = TextToFind{n};
        %%% The following runs every time through this module (i.e. for
        %%% every image set).
        %%% Saves the original image file name to the handles
        %%% structure.  The field is named appropriately based on
        %%% the user's input, in the Pipeline substructure so that
        %%% this field will be deleted at the end of the analysis
        %%% batch.
        fieldname = ['Filename', ImageName{n}];
        handles.Pipeline.(fieldname) = CurrentFileName;
        fieldname = ['Pathname', ImageName{n}];
        handles.Pipeline.(fieldname) =  Pathname;

        FileAndPathname = fullfile(Pathname, CurrentFileName);
        if strcmpi(FileFormat,'mat')
            try
                StructureLoadedImage = load(FileAndPathname);
            catch error(['CellProfiler was unable to load ',FileAndPathname,'. The file may be corrupt.']);
            end
            LoadedImage = StructureLoadedImage.Image;
        else [LoadedImage, handles] = CPimread(FileAndPathname,handles);
        end
        %%% Saves the image to the handles structure.
        handles.Pipeline.(ImageName{n}) = LoadedImage;

    catch ErrorMessage = lasterr;
        ErrorNumber = {'first','second','third','fourth'};
        error(['An error occurred when trying to load the ', ErrorNumber{n}, ' set of images using the Load Single Image module. Please check the settings. A common problem is that there are non-image files in the directory you are trying to analyze, or that the image file is not in the format you specified: ', FileFormat, '. Matlab says the problem is: ', ErrorMessage])
    end % Goes with: catch

    % Create a cell array with the filenames
    FileNames(n) = {CurrentFileName};
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
handles.Measurements.Image.FileNames(SetBeingAnalyzed)     = {FileNames};
handles.Measurements.Image.PathNamesText                   = PathNamesText;
handles.Measurements.Image.PathNames(SetBeingAnalyzed)     = {PathNames};
%%% ------------------------------------------------------------------------------------------------ %%%





%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%

%%% The figure window display is unnecessary for this module, so the figure
%%% window is closed the first time through the module.
%%% Determines the figure number.
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
%%% Closes the window if it is open.
if any(findobj == ThisModuleFigureNumber) == 1;
    close(ThisModuleFigureNumber)
end

% PROGRAM NOTES THAT ARE UNNECESSARY FOR THIS MODULE:



