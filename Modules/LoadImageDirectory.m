function handles = LoadImageDirectory(handles)

% Help for the LoadImageDirectory module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Makes a projection either by averaging or taking the maximum pixel value
% at each pixel position of a number of images organized by directory
%
% *************************************************************************
%
% This module combines a set of images by averaging or by taking the maximum
% pixel intensity at each pixel position. When this module is used to 
% average a Z-stack (3-D image stack), this process is known as making 
% a projection.
%
% Settings:
%
% * Enter the pathname to the folders containing the images:
%   This is the base directory that contains each of the directories
%   to be combined into a projection. Relative paths are from the image
%   folder: for instance "." operates on the directories in the image folder.
%
% * Enter the text that the folders have in common:
%   Enter some portion of the directory name that's common to all
%   directories for this type of image. For instance, "DNA-A01" and
%   "DNA-A02" can be selected by entering "DNA". Enter "*" to take all
%   directories.
%
% * Analyze all subfolders within each folder:
%   If yes, the module will read image files in subfolders of each
%   folder it looks at. If no, it will only read image files out of
%   the named folder.
% 
% * How do you want to load these files:
%   Exact match - if the text below matches some part of the text in the
%                 image file name, the file will be accepted.
%   Regular expressions - use a regular expression to match the image
%                         file name (see LoadImages for details).
%
% * What kind of projection would you like to make?:
%   If you choose Average, the average pixel intensity at each pixel
%   position will be used to created the final image.  If you choose
%   Maximum, the maximum pixel value at each pixel position will be used to
%   created the final image.
% * What do you want to call the projected image?:
%   Give a name to the resulting image, which could be used in subsequent
%   modules.
%
% Measurements:
% Image / DirectoryName - the name of the directory from which we took
%                         the image files for this image
%       / PathName      - the path to that directory
%
% See also LoadImages

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = Enter the pathname to the folders containing images:
%defaultVAR01 = .
Pathname = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Enter the text that the folders have in common:
%defaultVAR02 = *
DirTextToFind = char(handles.Settings.VariableValues{CurrentModuleNum,2});
if strcmp(DirTextToFind,'*')
    DirTextToFind = '';
end

%textVAR03 = Analyze all subfolders within each folder?
%choiceVAR03 = No
%choiceVAR03 = Yes
AnalyzeSubDir = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = How do you want to load these files?
%choiceVAR04 = Exact match
%choiceVAR04 = Regular expressions
ExactOrRegExp = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Enter the text that one type of file has in common:
%defaultVAR05 = *
FileTextToFind = char(handles.Settings.VariableValues{CurrentModuleNum,5});
if strcmp(FileTextToFind,'*')
    FileTextToFind = '';
end;

%textVAR06 = What kind of projection would you like to make?
%choiceVAR06 = Average
%choiceVAR06 = Maximum
ProjectionType = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 = Would you like to check images for a QCFlag (loaded earlier in a load text module) and bypass images possessing a flag in calculation of the illumination correction function?
%choiceVAR07 = No
%choiceVAR07 = Yes
%infotypeVAR07 = imagegroup indep
CheckForQC = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = What did you call the loaded text that contains the QCFlag? (This is only used for QCFlag=YES)
%defaultVAR08 = QCFlag
QCFileName = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = How many images are in each directory you are loading? (This is only used for QCFlag=YES)
%defaultVAR09 = 288
NumEachDirTotal = str2double(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = How many channels are there each directory you are loading? (This is only used for QCFlag=YES)
%defaultVAR10 = 3
NumChannels = str2double(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = What do you want to call the projected image?
%defaultVAR11 = ProjectedBlue
%infotypeVAR11 = imagegroup indep
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

dir_fieldname = ['DirectoryList', ImageName];
path_fieldname = ['Pathname', ImageName];
if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
    %%%
    %%% First time through - collect all directory names
    %%%
    
    %%% Get the pathname and check that it exists
    if strncmp(Pathname,'.',1)
        if length(Pathname) == 1
            Pathname = handles.Current.DefaultImageDirectory;
        else
	    % If the pathname start with '.', interpret it relative to
            % the default image dir.
            Pathname = fullfile(handles.Current.DefaultImageDirectory,Pathname(2:end));
        end
    end
    if ~exist(Pathname,'dir')
        error(['Image processing was canceled in the ', ModuleName, ' module because the directory "',Pathname,'" does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.'])
    end
    DirectoryList = CPdir(Pathname);
    DirMatchesName = arrayfun(@(x) ~isempty(strfind(x.name,DirTextToFind)), DirectoryList);
    IsADir = [DirectoryList.isdir];
    DirectoryList = DirectoryList(DirMatchesName(:) & IsADir(:));
    DirectoryNames = arrayfun(@(x) {x.name},DirectoryList);
    handles.Pipeline.(dir_fieldname) = DirectoryNames;
    handles.Pipeline.(path_fieldname) = Pathname;
    NumberOfImageSets = length(DirectoryNames);
    if NumberOfImageSets == 0
        error(['Image processing was canceled in the ', ModuleName, ' module because no directories were found']);
    end
    %%% Checks whether another load images module has already recorded a
    %%% number of cycles.  If it has, it will not be set at the default
    %%% of 1.  Then, it checks whether the number already stored as the
    %%% number of cycles is equal to the number of cycles that this
    %%% module has found.  If not, an error message is generated. Note:
    %%% this will not catch the case where the number of cycles
    %%% detected by this module is more than 1 and another module has
    %%% detected only one cycle, since there is no way to tell whether
    %%% the 1 stored in handles.Current.NumberOfImageSets is the default value or a
    %%% value determined by another image-loading module.
    if handles.Current.NumberOfImageSets ~= 1;
        if handles.Current.NumberOfImageSets ~= NumberOfImageSets
            error(['Image processing was canceled in the ', ModuleName, ' module because the number of image cycles loaded (', num2str(NumberOfImageSets),') does not equal the number of image cycles loaded by another image-loading module (', num2str(handles.Current.NumberOfImageSets), '). Please check the settings.'])
        end
    end
    handles.Current.NumberOfImageSets = NumberOfImageSets;
end
%%%
%%% Handle the list of files
%%%
DirPaths = handles.Pipeline.(dir_fieldname);
DirPath = fullfile(handles.Pipeline.(path_fieldname),...
                   DirPaths{handles.Current.SetBeingAnalyzed});
[handles,FileList] = CPretrievemediafilenames(handles, DirPath,FileTextToFind,AnalyzeSubDir(1), ExactOrRegExp,'Image');
if isempty(FileList)
    error(['Image processing was canceled in the ', ModuleName, ' module because no files were found in the ', DirPath, ' directory.']);
end

switch CheckForQC
    case 'No'
        for i=1:length(FileList)
            Image = CPimread(fullfile(DirPath,FileList{i}));
            if i == 1
                ProjectionImage = zeros(size(Image));
            end
            if strcmp(ProjectionType,'Average')
                ProjectionImage = ProjectionImage + Image;
            else
                ProjectionImage = max(ProjectionImage, Image);
            end
        end
        if strcmp(ProjectionType,'Average')
            ProjectionImage = ProjectionImage / length(FileList);
        end
    case 'Yes' 
        for i = 1:handles.Current.NumberOfImageSets
            QCFlagDataName = CPjoinstrings('LoadedText',QCFileName);
            if ~isfield(handles.Measurements.Image,QCFlagDataName)
                error('You asked to check the incoming images for a QCFlag, but the named of the loaded text file is incorrect.')
            end
            QCFlagData = str2double(handles.Measurements.Image.(QCFlagDataName));
            NumEachDir = NumEachDirTotal/NumChannels;
            TotalPerChannel= NumEachDir*(handles.Current.NumberOfImageSets);
            TotalPerChannel_supplied = length(QCFlagData);
            if TotalPerChannel ~= TotalPerChannel_supplied
                error('You have specified a number of images per directory and number of channels that does not match the QCFlagData you supplied.')
            end
            QCFlagData_perdir{i} = QCFlagData((NumEachDir*i-(NumEachDir-1)):NumEachDir*i);
        end
        for i = 1:length(FileList)
            QCFlagData = QCFlagData_perdir{handles.Current.SetBeingAnalyzed};
            if QCFlagData(i) == 0
                Image = CPimread(fullfile(DirPath,char(FileList(i))));
            else
                ImageSize = CPimread(fullfile(DirPath,char(FileList(i))));
                Image = zeros(size(ImageSize));
            end
            if i == 1 
                ProjectionImage = zeros(size(Image));
                ImageNumSkipped = 0;
            end
            ImageOrEmpty = find(Image, 1);
            logical = isempty(ImageOrEmpty);
            if logical == 1
                %disp(fullfile(DirPath,char(FileList(i))))
                ImageNumSkipped = ImageNumSkipped +1;
                %continue
            end
            if strcmp(ProjectionType,'Average')
                ProjectionImage = ProjectionImage + Image;
            else 
                ProjectionImage = max(ProjectionImage,Image);
            end
        end
        if strcmp(ProjectionType,'Average')
            ProjectionImage = ProjectionImage/(length(FileList)-ImageNumSkipped);
        end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(ProjectionImage,'OneByOne',ThisModuleFigureNumber)
    end
    hAx = subplot(1,1,1,'Parent',ThisModuleFigureNumber);
    CPimagesc(ProjectionImage,handles,hAx);
    title(hAx,['Projected Image # ', num2str(handles.Current.SetBeingAnalyzed)]);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the averaged image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(ImageName) = ProjectionImage;
handles = CPaddmeasurements(handles, 'Image', ...
				['FileName_', ImageName], DirPath);
handles = CPaddmeasurements(handles, 'Image', ...
				['PathName_', ImageName], ...
				handles.Pipeline.(path_fieldname));
