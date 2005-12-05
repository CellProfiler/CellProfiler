function handles = LoadSingleImage(handles)

% Help for the Load Single Image module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Loads a single image, which will be used for all image cycles.
% *************************************************************************
% Note: for most purposes, you will probably want to use the Load Images
% module, not this one.
%
% Tells CellProfiler where to retrieve a single image and gives the image a
% meaningful name for the other modules to access.  The module only
% functions the first time through the pipeline, and thereafter the image
% is accessible to all subsequent cycles being processed. This is
% particularly useful for loading an image like an Illumination correction
% image to be used by the CorrectIllumination_Apply module. Note: Actually,
% you can load four 'single' images using this module.
%
% Relative pathnames can be used. For example, on the Mac platform you
% could leave the folder where images are to be loaded as '.' to choose the
% default image folder, and then enter ../Imagetobeloaded.tif as the name
% of the file you would like to load in order to load the image from the
% directory one above the default image directory. Or, you could type
% .../AnotherSubfolder (note the three periods: the first is interpreted as
% a standin for the default image folder) as the folder from which images
% are to be loaded and enter the filename as Imagetobeloaded.tif to load an
% image from a different subfolder of the parent of the default image
% folder.
%
% If more than four single images must be loaded, more than one Load Single
% Image module can be run sequentially. Running more than one of these
% modules also allows images to be retrieved from different folders.
%
% See also LoadImages.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = This module loads one image for *all* cycles that will be processed. Normally, use the Load Images module to load new sets of images during each cycle of processing.

%pathnametextVAR02 = Enter the path name to the folder where the images to be loaded are located.  Type period (.) for the default image folder.
Pathname = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%filenametextVAR03 = What image file do you want to load? Include the extension, like .tif
TextToFind{1} = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What do you want to call that image?
%defaultVAR04 = OrigBlue
%infotypeVAR04 = imagegroup indep
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%filenametextVAR05 = What image file do you want to load? Include the extension, like .tif
TextToFind{2} = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = What do you want to call that image?
%defaultVAR06 = Do not load
%infotypeVAR06 = imagegroup indep
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%filenametextVAR07 = What image file do you want to load? Include the extension, like .tif
TextToFind{3} = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = What do you want to call that image?
%defaultVAR08 = Do not load
%infotypeVAR08 = imagegroup indep
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%filenametextVAR09 = What image file do you want to load? Include the extension, like .tif
TextToFind{4} = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = What do you want to call that image?
%defaultVAR10 = Do not load
%infotypeVAR10 = imagegroup indep
ImageName{4} = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = If an image slot is not being used, type a slash  /  in the box.

%textVAR11 = If your file names do not have extensions, choose the file format of the images (Note, this doesn't work currently)
%choiceVAR11 = mat
%choiceVAR11 = bmp
%choiceVAR11 = gif
%choiceVAR11 = jpg
%choiceVAR11 = tif
%choiceVAR11 = DIB
FileFormat = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines which cycle is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

%%% Remove slashes '/' from the input
tmp1 = {};
tmp2 = {};
for n = 1:4
    if ~strcmp(TextToFind{n}, 'NO FILE LOADED') && ~strcmp(ImageName{n}, 'Do not load')
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
    error(['Image processing was canceled in the ', ModuleName, ' module because the directory "',SpecifiedPathname,'" does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with / (for Mac/Unix) or \ (for PC).'])
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIRST CYCLE FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if isempty(ImageName)
    error(['Image processing was canceled in the ', ModuleName, ' module because you have not chosen any images to load.'])
end

for n = 1:length(ImageName)
    %%% This try/catch will catch any problems in the load images module.
    try
        CurrentFileName = TextToFind{n};
        %%% The following runs every time through this module (i.e. for
        %%% every cycle).
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
            try StructureLoadedImage = load(FileAndPathname);
                LoadedImage = StructureLoadedImage.Image;
            catch
                try [LoadedImage, handles] = CPimread(FileAndPathname,handles);
                catch
                    error(['Image processing was canceled in the ', ModuleName, ' module because an error occurred when trying to load this image: ',FileAndPathname,'. The file may be corrupt.']);
                end
            end
        else [LoadedImage, handles] = CPimread(FileAndPathname,handles);
        end
        %%% Saves the image to the handles structure.
        handles.Pipeline.(ImageName{n}) = LoadedImage;

    catch ErrorMessage = lasterr;
        ErrorNumber = {'first','second','third','fourth'};
        error(['Image processing was canceled in the ', ModuleName, ' module because an error occurred when trying to load the ', ErrorNumber{n}, ' set of images. Please check the settings. A common problem is that there are non-image files in the directory you are trying to analyze, or that the image file is not in the format you specified: ', FileFormat, '. Matlab says the problem is: ', ErrorMessage])
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
if  isfield(handles,'Measurements') && isfield(handles.Measurements,'Image') && isfield(handles.Measurements.Image,'FileNames')
    if length(handles.Measurements.Image.FileNames) == SetBeingAnalyzed
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
end

%%% Write to the handles.Measurements.Image structure
handles.Measurements.Image.FileNamesText                   = FileNamesText;
handles.Measurements.Image.FileNames(SetBeingAnalyzed)     = {FileNames};
handles.Measurements.Image.PathNamesText                   = PathNamesText;
handles.Measurements.Image.PathNames(SetBeingAnalyzed)     = {PathNames};

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The figure window display is unnecessary for this module, so the figure
%%% window is closed the first time through the module.
%%% Determines the figure number.
ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
%%% Closes the window if it is open.
if any(findobj == ThisModuleFigureNumber)
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure('','NarrowText')
    end
    for n = 1:length(ImageName)
        drawnow
        %%% Activates the appropriate figure window.
        CPfigure(handles,'Text',ThisModuleFigureNumber);
        if iscell(ImageName)
            TextString = [ImageName{n},': ',FileNames{n}];
        else
            TextString = [ImageName,': ',FileNames];
        end
        uicontrol('style','text','units','normalized','fontsize',handles.Preferences.FontSize,'HorizontalAlignment','left','string',TextString,'position',[.05 .85-(n-1)*.15 .95 .1],'BackgroundColor',[.7 .7 .9])
    end
end