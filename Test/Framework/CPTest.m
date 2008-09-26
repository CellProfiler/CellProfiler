function results=CPTest(Directory)

% Test CellProfiler by loading pipelines and comparing results.
%
% CPTest() - uses the current directory + 'Test' as the root,
%            assuming that the current directory is trunk/CellProfiler
% CPTest(Directory) - Directory is the root directory for the tests.
%
% CPTest searches subdirectories recursively for the following files
% per directory:
% CellProfilerPreferences.mat - the contents of handles.Preferences.
% PIPE.mat - the pipeline to run
% ExpectedOUT.mat - a skeleton OUT.mat file that holds the expected
%                   values residing in handles when done.
% Subdirectory names must be useable as field names.
%
%
% The test passes if there is one field in the generated OUT file's
% handles structure (recursively descending) per field in ExpectedOUT.mat
%
if nargin < 1
    Directory=fullfile(pwd,'Test');
end
listing = dir(Directory);
directories = listing([listing.isdir] & ~strncmp({listing.name},'.',1));
RequiredFiles = { 'CellProfilerPreferences.mat','PIPE.mat',...
    'ExpectedOUT.mat' };
[ignore,name]=fileparts(Directory);
results = struct(...
    'Directory',Directory,...
    'Name',name,...
    'failures',[]);
if all(ismember(RequiredFiles, {listing.name}))
    results = DoTest(Directory,results);
end
if ~isempty(directories)
    results.Subdirectory = cell(1,length(directories));
end

for i = 1:length(directories)
    entry=directories(i);
    try
        results.Subdirectory{i}=CPTest(fullfile(Directory,entry.name));
    catch x
        results.failures = [results.failures, x];
    end
end

%%%%%%%%%%%%%%
%%% DoTest %%%
%%%%%%%%%%%%%%
%
% Having found a directory with a test, run that test.
function results=DoTest(Directory,results)
results.Directory=Directory;
handles = struct(...
    'Current',struct(...
        'NumberOfImageSets',1,...
        'SetBeingAnalyzed',1,...
        'SaveOutputHowOften',1,...
        'TimeStarted',datestr(now),...
        'StartingImageSet',1,...
        'StartupDirectory',pwd),...
    'Measurements',struct(),...
    'Pipeline',struct());
old_path = path;
try
    %%%
    %%% Load the preferences file
    %%%
    PreferencesFilename=fullfile(Directory,'CellProfilerPreferences.mat');
    LoadedPreferences = load(PreferencesFilename);
    if ~ isfield(LoadedPreferences,'SavedPreferences')
        throw(MException('CPTest:BadPreferencesFile',[PreferencesFilename, ' does not contain a SavedPreferences structure']));
    end
    handles.Preferences=LoadedPreferences.SavedPreferences;
    handles.Current.DefaultOutputDirectory = handles.Preferences.DefaultOutputDirectory;
    handles.Current.DefaultImageDirectory = handles.Preferences.DefaultImageDirectory;
    if isfield(LoadedPreferences.SavedPreferences,'DefaultModuleDirectory')
        addpath(LoadedPreferences.SavedPreferences.DefaultModuleDirectory);
    end
    %%%
    %%% Load the pipeline file
    %%%
    handles = LoadPipeline(handles,Directory);
    %%%
    %%% Run the pipeline
    %%%
    [handles, results] = RunPipeline(handles,results);
    %%%
    %%% Compare the results
    %%%
    ExpectedOutFilename = fullfile(Directory,'ExpectedOUT.mat');
    ExpectedOut = load(ExpectedOutFilename);
    if ~isfield(ExpectedOut,'handles')
        throw(MException('CPTest:NoHandles',[ExpectedOutFilename, ' did not contain a handles structure']));
    end
    results = CPTestCompare(struct('handles',handles),ExpectedOut,results);
catch x
    results.failures = [results.failures,x];
end
path(old_path);
%%%
%%% Run the pipeline in handles.Settings, capturing exceptions
%%% in the results.
%%%
function [handles,results] = RunPipeline(handles,results)

while handles.Current.SetBeingAnalyzed <= handles.Current.NumberOfImageSets
    NumberofWindows = 0;
    SlotNumber = 1;
    while SlotNumber <= handles.Current.NumberOfModules
        %%% If a module is not chosen in this slot, continue on to the next.
        ModuleNumberAsString = CPtwodigitstring(SlotNumber);
        ModuleName = char(handles.Settings.ModuleNames(SlotNumber));
        if ~iscellstr(handles.Settings.ModuleNames(SlotNumber))
        else
            %%% Saves the current module number in the handles
            %%% structure.
            handles.Current.CurrentModuleNumber = ModuleNumberAsString;
            %%% The try/catch/end set catches any errors that occur during the
            %%% running of module 1, notifies the user, breaks out of the image
            %%% analysis loop, and completes the refreshing
            %%% process.
            try
                if handles.Current.SetBeingAnalyzed == 1
                    if (~isfield(handles.Preferences,'DisplayWindows')) ||...
                        handles.Preferences.DisplayWindows(SlotNumber) == 0
                        handles.Current.(['FigureNumberForModule' CPtwodigitstring(SlotNumber)]) = ceil(max(findobj))+1;
                    else
                        NumberofWindows = NumberofWindows+1;
                        if iscellstr(handles.Settings.ModuleNames(SlotNumber))
                            LeftPos = ScreenWidth*rem((NumberofWindows-1),12)/12;
                            handles.Current.(['FigureNumberForModule' CPtwodigitstring(SlotNumber)]) = ...
                                CPfigure(handles,'','name',[char(handles.Settings.ModuleNames(SlotNumber)), ' Display, cycle # '],...
                                'Position',[LeftPos (ScreenHeight-522) 560 442]);
                        end
                        TempFigHandle = handles.Current.(['FigureNumberForModule' CPtwodigitstring(SlotNumber)]);
                        if exist('FigHandleList','var')
                            if any(TempFigHandle == FigHandleList)
                                for z = 1:length(FigHandleList)
                                    if TempFigHandle == FigHandleList(z)
                                        handles.Current.(['FigureNumberForModule' CPtwodigitstring(z)]) = ceil(max(findobj))+z;
                                    end
                                end
                            end
                        end
                        FigHandleList(SlotNumber) = handles.Current.(['FigureNumberForModule' CPtwodigitstring(SlotNumber)]); %#ok
                    end
                end
                %%% Runs the appropriate module, with the handles structure as an
                %%% input argument and as the output
                %%% argument.
                handles = feval(ModuleName,handles);
                %%% We apparently ran the module successfully, so record a Zero (unless we're restarting)
                if ~ strcmp(ModuleName, 'Restart')
                    handles = CPaddmeasurements(handles,'Image',CPjoinstrings('ModuleError',[CPtwodigitstring(SlotNumber),ModuleName]),0);
                end
            catch x
                results.ModuleError = struct(...
                    'Exception', x,...
                    'ModuleNumber',SlotNumber,...
                    'ModuleName',ModuleName,...
                    'SetBeingAnalyzed',handles.Current.SetBeingAnalyzed);
                return
            end
        end
        SlotNumber = SlotNumber+1;
    end
    handles.Current.SetBeingAnalyzed = handles.Current.SetBeingAnalyzed+1;
end

function handles=LoadPipeline(handles,Directory)
%%%
%%% Load the pipeline file
%%% Fix up modules based on version # (checks whether the fix can be done)
PipelineFilename=fullfile(Directory,'PIPE.mat');
Pipeline = load(PipelineFilename);
if ~ isfield(Pipeline,'Settings')
    throw(MException('CPTest:BadPipelineFile',[PipelineFilename, ' does not contain a Settings field']));
end
handles.Settings = Pipeline.Settings;
handles.Current.NumberOfModules = length(handles.Settings.ModuleNames);
for ModuleNum=1:length(handles.Settings.ModuleNames)
    if ~isempty(handles.Settings.ModuleNames)
        handles = FixModule(handles,ModuleNum);
    end
end

function handles=FixModule(handles,ModuleNum)
%%%
%%% Scan the file for the version # and run CPImportPreviousModuleSettings
%%% to possibly update the module. Check whether the update was possible.
%%%
ModuleName = handles.Settings.ModuleNames{ModuleNum};
if isfield(handles.Settings,'VariableRevisionNumbers')
    SavedVarRevNum = handles.Settings.VariableRevisionNumbers(ModuleNum);
else
    SavedVarRevNum = 0;
end
[Settings, SavedVarRevNum,IsModuleModified,NeedsPlaceholderUpdateMsg,ModuleName]=...
    CPImportPreviousModuleSettings(handles.Settings,ModuleName,ModuleNum,0,SavedVarRevNum);
%%%
%%% At this point, the new module name must be the name of a module along the path.
%%% We can read the module and extract the # of variables and the revision %
ModuleFileName = fullfile(handles.Preferences.DefaultModuleDirectory,[ModuleName,'.m']);
version = 0;
VariableCount = 0;
fid=fopen(ModuleFileName,'r');
while true
    line = fgetl(fid);
    if ~ischar(line)
        break
    end
    [tmpVersion, count, errmsg, nextindex] = ...
        sscanf(line,'%%%%%%VariableRevisionNumber = %d');
    if count == 1
        version = tmpVersion;
        break
    else
        [tokens,match]=regexp(line,'^%(text|choice|inputtype|default)VAR([0-9]+)','tokens','match');
        if ~isempty(match)
            var=str2num(tokens{1}{2});
            VariableCount=max([var,VariableCount]);
        end
    end
end
fclose(fid);
%%%
%%% Make sure the version #s and # of variables match. If not
%%% 1) ***best*** Update CPimportPreviousModuleSettings to make the test work
%%% 2) or change the test
%%% 3) or remove the test
if SavedVarRevNum ~= version
    throw(MException('CPTest:WrongVersionNumber',sprintf('Module # %d (%s) has version # %d in the pipeline file. Current verison # is %d',ModuleNum, ModuleName, SavedVarRevNum,version)));
end
if VariableCount ~= Settings.NumbersOfVariables(ModuleNum)
    throw(MException('CPTest:WrongNumberOfVariables',sprintf('Module # %d (%s) has %d variables in the pipeline file. Current # is %d',ModuleNum, ModuleName, Settings.NumberOfVariables(ModuleNum),VariableCount)));
end
handles.Settings = Settings;