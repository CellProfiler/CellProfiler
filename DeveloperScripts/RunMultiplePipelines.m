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

%textVAR01 = This module will run all PIPE.mat files in the default image directory. The output will be saved in the default image directory. You
%must start Technical Diagnosis prior to running this module (File >> Tech Diagnosis ). 
%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%
%%% PROCESSING %%%
%%%%%%%%%%%%%%%%%%
drawnow

if (handles.Current.SetBeingAnalyzed ~= 1)
    return
end

    
cd(handles.Current.DefaultImageDirectory) 

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

for i=3:size(handles.Current.PipelineDirectories.Directories);
%%% First, we will go through this directory and store all subdirectories
%%% and filenames to the appropriate place.
    SkipDirectory=0;
    if findstr(pwd,'.')
        SkipDirectory = 1;
        cd ..
    elseif SkipDirectory == 0;
        if ispc;
            slashdirection = '\';
        else
            slashdirection = '/';
        end
    FullDirectoryName = strcat(handles.Current.DefaultImageDirectory,slashdirection,DirectoryListing(i));
    SettingsPathname = char(FullDirectoryName);
        if findstr(SettingsPathname,'.')
            continue
        else
            cd (SettingsPathname);
        end
    handles.Current.DefaultImageDirectory = SettingsPathname;
    handles.Current.DefaultOutputDirectory = SettingsPathname; 
    FileAndSubdirStructure = dir(SettingsPathname);
    FileAndSubDirNames = sortrows({FileAndSubdirStructure.name}');
    LogicalIsSubDirectory = [FileAndSubdirStructure.isdir];
    handles.Current.PipelineDirectories.Subdirectories = FileAndSubDirNames(LogicalIsSubDirectory);
    FileNamesNoSubDir = FileAndSubDirNames(~LogicalIsSubDirectory);
    %%% Store all filenames for the subdirectory.
    handles.Current.PipelineDirectories.Subdirectories.FileNames = FileNamesNoSubDir;
    %%% Now, we will try to load each pipeline listed in FileListing.  Then,
    %%% change the Default Image and Output directories to be the same
    %%% location
    %%% as the current directory.
    for k=1:size(handles.Current.PipelineDirectories.Subdirectories.FileNames{1})
            errFlg =0;
            handles.Current.PipelineDirectories.Subdirectories.FileNames = FileNamesNoSubDir;
            FileNameContainsPIPE = strfind(FileNamesNoSubDir(k), 'PIPE');
            if FileNameContainsPIPE{1} > 0;
                k
                %%% If the file contains PIPE load it as a pipeline.
                SettingsFileName= char(FileNamesNoSubDir(k));
                eventdata.SettingsPathname = SettingsPathname;
                eventdata.SettingsFileName = SettingsFileName;
                LoadPipeline_Callback(hObject,eventdata,handles);%%%blah ok to here
                % Now, the pipeline needs to be run.
                %LoadedSettings = load(fullfile(SettingsPathname,SettingsFileName))
                guidata(gcbo, handles);
                AnalyzeImagesButton_Callback(hObject, eventdata, handles);

            else
                %%% Otherwise, clear the Pipeline.
%                 handles.Settings.ModuleNames = {};
%                 handles.Settings.VariableValues = {};
%                 handles.Settings.VariableInfoTypes = {};
%                 handles.Settings.VariableRevisionNumbers = [];
%                 handles.Settings.ModuleRevisionNumbers = [];
%                 delete(get(handles.variablepanel,'children'));
%                 set(handles.slider1,'visible','off');
%                 handles.VariableBox = {};
%                 handles.VariableDescription = {};
%                 set(handles.ModulePipelineListBox,'Value',1);
%                 handles.Settings.NumbersOfVariables = [];
%                 handles.Current.NumberOfModules = 0;
%                 contents = {'No Modules Loaded'};
%                 set(handles.ModulePipelineListBox,'String',contents);
                %guidata(hObject,handles);
                k = k+1;
            end
            cd ..
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
[ScreenWidth,ScreenHeight] = CPscreensize;
fig = handles.Current.(['FigureNumberForModule',CurrentModule]);
if ~isempty(fig)
    try
        close(fig); % Close the Restart figure

    end
    end

    handles.Current.DefaultImageDirectory = pwd;
    handles.Current.DefaultOutputDirectory = pwd; 
    end
 
%%%At the end of this cycle, it will start back at the top of the for-loop.
end


