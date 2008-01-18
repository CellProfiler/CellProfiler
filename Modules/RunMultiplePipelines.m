function handles = RunMultiplePipelines(handles)

% Help for the RunMultiplePipelines module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Add Short Description here
% *************************************************************************
%
% Add help here
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
% $Revision: 4202 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = This module will run all PIPE.mat files in the default image directory. The output will be saved in the default image directory.

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%
%%% PROCESSING %%%
%%%%%%%%%%%%%%%%%%
drawnow

if (handles.Current.SetBeingAnalyzed ~= 1)
    return
end


FilesAndDirsStructure = dir(handles.Current.DefaultImageDirectory);
%%% Puts the names of each object into a list.
FileAndDirNames = sortrows({FilesAndDirsStructure.name}');
%%% Puts the logical value of whether each object is a directory into a list.
LogicalIsDirectory = [FilesAndDirsStructure.isdir];
%%% Saves the File and Directory List separately.
handles.Current.PipelineDirectories.Directories = FileAndDirNames(LogicalIsDirectory) ;
DirectoryListing = handles.Current.PipelineDirectories.Directories;
%%% Eliminates directories from the list of file names.
FileNamesNoDir = FileAndDirNames(~LogicalIsDirectory);
handles.Current.PipelineDirectories.FileListing = FileNamesNoDir;
i=3;
for i=3:size(handles.Current.PipelineDirectories.Directories); %%%blah ok to here
%%% First, we will go through this directory and store all subdirectories
%%% and filenames to the appropriate place.
    if ispc;
    slashdirection = '\';
    else
    slashdirection = '/';
    end
FullDirectoryName = strcat(handles.Current.DefaultImageDirectory,slashdirection,DirectoryListing(5));
FullDirectory = char(FullDirectoryName);
cd (FullDirectory);
FileAndSubdirStructure = dir(FullDirectory);
FileAndSubDirNames = sortrows({FileAndSubdirStructure.name}');
LogicalIsSubDirectory = [FileAndSubdirStructure.isdir];
handles.Current.PipelineDirectories.Subdirectories = FileAndSubDirNames(LogicalIsSubDirectory);
FileNamesNoSubDir = FileAndSubDirNames(~LogicalIsSubDirectory);
%%% Store all filenames for the subdirectory.
handles.Current.PipelineDirectories.Subdirectories.FileNames =FileNamesNoSubDir;

%%% Now, we will try to load each pipeline listed in FileListing.  Then,
%%% change the Default Image and Output directories to be the same location
%%% as the current directory.
k=1;
for k=1:FileNamesNoSubDir(1);
try
    if strfind(handles.Current.PipelineDirectories.FileListing(i),'PIPE');
        %%% If the file contains PIPE load it as a pipeline.
        eventdata= [DirectoryListing, handles.Current.PipelineDirectories.FileListing(5), errFlg];
        LoadPipeline_Callback(hObject,eventdata,handles);
    else
        %%% Otherwise, clear the Pipeline.  
        handles.Settings.ModuleNames = {};
        handles.Settings.VariableValues = {};
        handles.Settings.VariableInfoTypes = {};
        handles.Settings.VariableRevisionNumbers = [];
        handles.Settings.ModuleRevisionNumbers = [];
        delete(get(handles.variablepanel,'children'));
        set(handles.slider1,'visible','off');
        handles.VariableBox = {};
        handles.VariableDescription = {};
        set(handles.ModulePipelineListBox,'Value',1);
        handles.Settings.NumbersOfVariables = [];
        handles.Current.NumberOfModules = 0;
        contents = {'No Modules Loaded'};
        set(handles.ModulePipelineListBox,'String',contents);
        guidata(hObject,handles);
    end    
catch
    errFlg = 1;
end

if (errFlg ~= 0)
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not load any pipelines.']);
end
        eventdata= [DirectoryListing, handles.Current.PipelineDirectories.FileListing(5), errFlg];
        LoadPipeline_Callback(hObject,eventdata,handles);
try
    [filepath, filename, errFlg, updatedhandles] = callback(gcbo,[],guidata(gcbo));
catch
    errFlg = 1;
end

if (errFlg ~= 0)
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not initialize pipeline.']);
end
try
    importhandles = load(fullfile(filepath,filename));
catch
    error(['Image processing was canceled in the ', ModuleName, ' module because it could not load from file ' ...
        fullfile(filepath,filename),'.']);
end
% save figure properties
ScreenSize = get(0,'ScreenSize');
ScreenWidth = ScreenSize(3);
ScreenHeight = ScreenSize(4);
fig = handles.Current.(['FigureNumberForModule',CurrentModule]);
if ~isempty(fig)
    try
        close(fig); % Close the Restart figure
    end
end

handles.Settings = updatedhandles.Settings;
handles.Pipeline = importhandles.handles.Pipeline;
handles.Measurements = importhandles.handles.Measurements;
handles.Current = importhandles.handles.Current;
handles.VariableBox = updatedhandles.VariableBox;
handles.VariableDescription = updatedhandles.VariableDescription;
handles.Current.StartingImageSet = handles.Current.SetBeingAnalyzed + 1;
handles.Current.CurrentModuleNumber = '01';
handles.Preferences.DisplayWindows = importhandles.handles.Preferences.DisplayWindows;

%%% Reassign figures handles and open figure windows
for i=1:handles.Current.NumberOfModules;
    if iscellstr(handles.Settings.ModuleNames(i))
        if handles.Preferences.DisplayWindows(i)
            handles.Current.(['FigureNumberForModule' TwoDigitString(i)]) = ...
                CPfigure(handles,'','name',[char(handles.Settings.ModuleNames(i)), ' Display, cycle # '],...
                'Position',[(ScreenWidth*((i-1)/12)) (ScreenHeight-522) 560 442]);
        end
    end
end
end
continue;
%%%At the end of this cycle, it will start back at the top of the for-loop.
end

guidata(gcbo,handles);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
handles.Current.PipelineDirectories={};
end

%%% SUBFUNCTION %%%
function twodigit = TwoDigitString(val)
%TwoDigitString is a function like num2str(int) but it returns a two digit
%representation of a string for our purposes.
if ((val > 99) || (val < 0)),
    error(['TwoDigitString: Can''t convert ' num2str(val) ' to a 2 digit number']);
end
twodigit = sprintf('%02d', val);
return;
end

function LoadThisPIPE(hObject, eventdata, handles) 
ListOfTools = handles.Current.ModulesFilenames;
ToolsHelpSubfunction(handles, 'Modules', ListOfTools)
end