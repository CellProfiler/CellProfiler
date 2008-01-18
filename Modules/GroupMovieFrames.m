function handles = GroupMovieFrames(handles)

% Help for the Load Images module:
% Category: File Processing
%
% SHORT DESCRIPTION:
%
% GroupMovieFrames handle a movie to group movie frames to be processed
% within a cycle. The position of a frame within a group can be specified
% with its ImageName to be used downstream. 
%
% Each loaded movie frame will be treated as an individual image with its
% own ImageName.

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
% $Revision: 4861 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the movie you want to group?
%infotypeVAR01 = imagegroup
MovieName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = How many frames per cycle do you want extract?
%defaultVAR02 = 1
NumGroupFrames = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%--
%textVAR03 = 1. What's the position of the frame (within the group) that you want to load?
%defaultVAR03 = 1
Position{1} = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What do you want to call this frame within CellProfiler?
%defaultVAR04 = OrigDAPI
%infotypeVAR04 = imagegroup indep
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%--
%textVAR05 = 2. What's the position of the frame (within the group) that you want to load?
%defaultVAR05 = /
Position{2} = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = What do you want to call this frame within CellProfiler?
%defaultVAR06 = /
%infotypeVAR06 = imagegroup indep
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%--
%textVAR07 = 3. What's the position of the frame (within the group) that you want to load?
%defaultVAR07 = /
Position{3} = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = What do you want to call this frame within CellProfiler?
%defaultVAR08 = /
%infotypeVAR08 = imagegroup indep
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%--
%textVAR09 = 4. What's the position of the frame (within the group) that you want to load?
%defaultVAR09 = /
Position{4} = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = What do you want to call this frame within CellProfiler?
%defaultVAR10 = /
%infotypeVAR10 = imagegroup indep
ImageName{4} = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%--
%textVAR11 = 5. What's the position of the frame (within the group) that you want to load?
%defaultVAR11 = /
Position{5} = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%textVAR12 = What do you want to call this frame within CellProfiler?
%defaultVAR12 = /
%infotypeVAR12 = imagegroup indep
ImageName{5} = char(handles.Settings.VariableValues{CurrentModuleNum,12});

%--
%textVAR13 = 6. What's the position of the frame (within the group) that you want to load?
%defaultVAR13 = /
Position{6} = char(handles.Settings.VariableValues{CurrentModuleNum,13});

%textVAR14 = What do you want to call this frame within CellProfiler?
%defaultVAR14 = /
%infotypeVAR14 = imagegroup indep
ImageName{6} = char(handles.Settings.VariableValues{CurrentModuleNum,14});

%--
%textVAR15 = 7. What's the position of the frame (within the group) that you want to load?
%defaultVAR15 = /
Position{7} = char(handles.Settings.VariableValues{CurrentModuleNum,15});

%textVAR16 = What do you want to call this frame within CellProfiler?
%defaultVAR16 = /
%infotypeVAR16 = imagegroup indep
ImageName{7} = char(handles.Settings.VariableValues{CurrentModuleNum,16});

%--
%textVAR17 = 8. What's the position of the frame (within the group) that you want to load?
%defaultVAR17 = /
Position{8} = char(handles.Settings.VariableValues{CurrentModuleNum,17});

%textVAR18 = What do you want to call this frame within CellProfiler?
%defaultVAR18 = /
%infotypeVAR18 = imagegroup indep
ImageName{8} = char(handles.Settings.VariableValues{CurrentModuleNum,18});

%--
%textVAR19 = 9. What's the position of the frame (within the group) that you want to load?
%defaultVAR19 = /
Position{9} = char(handles.Settings.VariableValues{CurrentModuleNum,19});

%textVAR20 = What do you want to call this frame within CellProfiler?
%defaultVAR20 = /
%infotypeVAR20 = imagegroup indep
ImageName{9} = char(handles.Settings.VariableValues{CurrentModuleNum,20});

%--
%textVAR21 = 10. What's the position of the frame (within the group) that you want to load?
%defaultVAR21 = /
Position{10} = char(handles.Settings.VariableValues{CurrentModuleNum,21});

%textVAR22 = What do you want to call this frame within CellProfiler?
%defaultVAR22 = /
%infotypeVAR22 = imagegroup indep
ImageName{10} = char(handles.Settings.VariableValues{CurrentModuleNum,22});

%--
%textVAR23 = 11. What's the position of the frame (within the group) that you want to load?
%defaultVAR23 = /
Position{11} = char(handles.Settings.VariableValues{CurrentModuleNum,23});

%textVAR24 = What do you want to call this frame within CellProfiler?
%defaultVAR24 = /
%infotypeVAR24 = imagegroup indep
ImageName{11} = char(handles.Settings.VariableValues{CurrentModuleNum,24});

%--
%textVAR25 = 12. What's the position of the frame (within the group) that you want to load?
%defaultVAR25 = /
Position{12} = char(handles.Settings.VariableValues{CurrentModuleNum,25});

%textVAR26 = What do you want to call this frame within CellProfiler?
%defaultVAR26 = /
%infotypeVAR26 = imagegroup indep
ImageName{12} = char(handles.Settings.VariableValues{CurrentModuleNum,26});

%--
%textVAR27 = 13. What's the position of the frame (within the group) that you want to load?
%defaultVAR27 = /
Position{13} = char(handles.Settings.VariableValues{CurrentModuleNum,27});

%textVAR28 = What do you want to call this frame within CellProfiler?
%defaultVAR28 = /
%infotypeVAR28 = imagegroup indep
ImageName{13} = char(handles.Settings.VariableValues{CurrentModuleNum,28});

%--
%textVAR29 = 14. What's the position of the frame (within the group) that you want to load?
%defaultVAR29 = /
Position{14} = char(handles.Settings.VariableValues{CurrentModuleNum,29});

%textVAR30 = What do you want to call this frame within CellProfiler?
%defaultVAR30 = /
%infotypeVAR30 = imagegroup indep
ImageName{14} = char(handles.Settings.VariableValues{CurrentModuleNum,30});

%--
%textVAR31 = 15. What's the position of the frame (within the group) that you want to load?
%defaultVAR31 = /
Position{15} = char(handles.Settings.VariableValues{CurrentModuleNum,31});

%textVAR32 = What do you want to call this frame within CellProfiler?
%defaultVAR32 = /
%infotypeVAR32 = imagegroup indep
ImageName{15} = char(handles.Settings.VariableValues{CurrentModuleNum,32});

%--
%textVAR33 = 16. What's the position of the frame (within the group) that you want to load?
%defaultVAR33 = /
Position{16} = char(handles.Settings.VariableValues{CurrentModuleNum,33});

%textVAR34 = What do you want to call this frame within CellProfiler?
%defaultVAR34 = /
%infotypeVAR34 = imagegroup indep
ImageName{16} = char(handles.Settings.VariableValues{CurrentModuleNum,34});

%--
%textVAR35 = 17. What's the position of the frame (within the group) that you want to load?
%defaultVAR35 = /
Position{17} = char(handles.Settings.VariableValues{CurrentModuleNum,35});

%textVAR36 = What do you want to call this frame within CellProfiler?
%defaultVAR36 = /
%infotypeVAR36 = imagegroup indep
ImageName{17} = char(handles.Settings.VariableValues{CurrentModuleNum,36});

%--
%textVAR37 = 18. What's the position of the frame (within the group) that you want to load?
%defaultVAR37 = /
Position{18} = char(handles.Settings.VariableValues{CurrentModuleNum,37});

%textVAR38 = What do you want to call this frame within CellProfiler?
%defaultVAR38 = /
%infotypeVAR38 = imagegroup indep
ImageName{18} = char(handles.Settings.VariableValues{CurrentModuleNum,38});

%--
%textVAR39 = 19. What's the position of the frame (within the group) that you want to load?
%defaultVAR39 = /
Position{19} = char(handles.Settings.VariableValues{CurrentModuleNum,39});

%textVAR40 = What do you want to call this frame within CellProfiler?
%defaultVAR40 = /
%infotypeVAR40 = imagegroup indep
ImageName{19} = char(handles.Settings.VariableValues{CurrentModuleNum,40});

%--
%textVAR41 = 20. What's the position of the frame (within the group) that you want to load?
%defaultVAR41 = /
Position{20} = char(handles.Settings.VariableValues{CurrentModuleNum,41});

%textVAR42 = What do you want to call this frame within CellProfiler?
%defaultVAR42 = /
%infotypeVAR42 = imagegroup indep
ImageName{20} = char(handles.Settings.VariableValues{CurrentModuleNum,42});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% GROUP LOADING FROM MOVIE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

nGroupFrames = str2num(NumGroupFrames);

%%% Determines which cycle is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

%%% Reset the max number of cycles here 
if SetBeingAnalyzed == 1  
    handles.Current.NumberOfImageSets = floor(handles.Current.NumberOfImageSets/nGroupFrames);
end

%%% Remove slashes entries with N/A or no filename from the input,
%%% i.e., store only valid entries
tmp2 = {};
for n = 1:14
    if ~strcmp(Position{n}, '/') || ~strcmp(ImageName{n}, '/')
        tmp2{end+1} = ImageName{n};
    end
end
ImageName = tmp2;

for n = 1:length(ImageName) 
    %%% This try/catch will catch any problems in the load images module.
    try
        %%% The following runs every time through this module (i.e. for every cycle).
        %%% Determines which movie to analyze.
        fieldname = ['FileList', MovieName];
        FileList = handles.Pipeline.(fieldname);
        %%% Determines the file name of the movie you want to analyze.
        CurrentFileName = FileList(:,(SetBeingAnalyzed-1)*nGroupFrames+str2num(Position{n}));
        %%% Determines the directory to switch to.
        fieldname = ['Pathname', MovieName];
        Pathname = handles.Pipeline.(fieldname);
        fieldname = ['FileFormat', MovieName];
        FileFormat = handles.Pipeline.(fieldname);         

        if strcmpi(FileFormat,'avi movies') == 1
            %%%If you do not subtract 1 from the index, as specified 
            %%%in aviread.m, the  movie will fail to load.  However,
            %%%the first frame will fail if the index=0.  
            IndexLocation=(cell2mat(CurrentFileName(2)));
            NumberOfImageSets = handles.Current.NumberOfImageSets;
            if (cell2mat(CurrentFileName(2)) ~= NumberOfImageSets)
                LoadedRawImage = aviread(fullfile(Pathname, char(CurrentFileName(1))), (IndexLocation));
                LoadedImage = im2double(LoadedRawImage.cdata);
            else
                LoadedRawImage = aviread(fullfile(Pathname, char(CurrentFileName(1))), (IndexLocation-1));
                LoadedImage = im2double(LoadedRawImage.cdata);
            end
        elseif strcmpi(FileFormat,'stk movies') == 1
            LoadedRawImage = tiffread(fullfile(Pathname, char(CurrentFileName(1))), cell2mat(CurrentFileName(2)));
            LoadedImage = im2double(LoadedRawImage.data);
        elseif (strcmpi(FileFormat,'tif,tiff,flex movies') == 1)
            LoadedRawImage = imread(fullfile(Pathname, char(CurrentFileName(1))), cell2mat(CurrentFileName(2)));
            LoadedImage = im2double(LoadedRawImage);                
        end
        %%% Saves the original movie file name to the handles
        %%% structure.  The field is named appropriately based on
        %%% the user's input, in the Pipeline substructure so that
        %%% this field will be deleted at the end of the analysis
        %%% batch.
        fieldname = ['Filename', ImageName{n}];
        [SubdirectoryPathName,BareFileName,ext] = fileparts(char(CurrentFileName(1))); %#ok Ignore MLint
        CurrentFileNameWithFrame = [BareFileName, '_', num2str(cell2mat(CurrentFileName(2))),ext];
        %%% Saves the loaded image to the handles structure.  The field is named
        %%% appropriately based on the user's input, and put into the Pipeline
        %%% substructure so it will be deleted at the end of the analysis batch.
        handles.Pipeline.(fieldname)(SetBeingAnalyzed) = {CurrentFileNameWithFrame};
        handles.Pipeline.(ImageName{n}) = LoadedImage;
    catch ErrorMessage = lasterr;
        ErrorNumber = {'first','second','third'};
        error(['Image processing was canceled in the ', ModuleName, ' module because an error occurred when trying to load the ', ErrorNumber{n}, ' frame of the group. Please check the settings. Matlab says the problem is: ', ErrorMessage])
    end % Goes with: catch
    FileNames(n) = {CurrentFileNameWithFrame};
end
       
%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure('','NarrowText',ThisModuleFigureNumber)
    end
    for n = 1:length(ImageName)
        %%% Activates the appropriate figure window.
        currentfig = CPfigure(handles,'Text',ThisModuleFigureNumber);
        if iscell(ImageName)
            TextString = [ImageName{n},': ',FileNames{n}];
        else
            TextString = [ImageName,': ',FileNames];
        end
        uicontrol(currentfig,'style','text','units','normalized','fontsize',handles.Preferences.FontSize,'HorizontalAlignment','left','string',TextString,'position',[.05 .85-(n-1)*.15 .95 .1],'BackgroundColor',[.7 .7 .9])
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% NOTE: The structure for filenames and pathnames will be a cell array of cell arrays

%%% First, fix feature names and the pathname
PathNames = cell(1,length(ImageName));
FileNamesText = cell(1,length(ImageName));
PathNamesText = cell(1,length(ImageName));
for n = 1:length(ImageName)
    PathNames{n} = Pathname;
    FileNamesText{n} = [ImageName{n}];
    PathNamesText{n} = [ImageName{n}];
end

%%% Since there may be several load/save modules in the pipeline which all
%%% write to the handles.Measurements.Image.FileName field, we store
%%% filenames in an "appending" style. Here we check if any of the modules
%%% above the current module in the pipeline has written to
%%% handles.Measurements.Image.Filenames. Then we should append the current
%%% filenames and path names to the already written ones. If this is the
%%% first module to put anything into the handles.Measurements.Image
%%% structure, then this section is skipped and the FileNamesText fields
%%% are created with their initial entry coming from this module.

if  isfield(handles,'Measurements') && isfield(handles.Measurements,'Image') &&...
        isfield(handles.Measurements.Image,'FileNames') && length(handles.Measurements.Image.FileNames) == SetBeingAnalyzed
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