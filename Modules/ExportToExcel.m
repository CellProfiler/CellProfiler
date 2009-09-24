function handles = ExportToExcel(handles)

% Help for the ExportToExcel module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Exports measurements into a tab-delimited text file which can be opened
% in Excel or other spreadsheet programs.
% *************************************************************************
% Note: this module is beta-version and has not been thoroughly checked.
%
% The data will be converted to a tab-delimited text file which can be read
% by Excel, another spreadsheet program, or a text editor. The file is
% stored in the default output folder.
%
% This module performs the same function as the data tool, Export Data.
% Please refer to the help for ExportData for more details.
%
% Settings:
%
% Enter the directory where the Excel files are to be saved. 
% If you used the FileNameMetadata module, metadata tokens may be used 
% here. If the directory does not exist, it will be created.
%
% What prefix should be used to name the Excel files? 
% Here you can choose what to prepend to the output file. If you choose 
% "Do not use", the output filename will be prepended. If you choose
% a prefix, the file will become PREFIX_<ObjectName>.xls. If you used 
% FileNameMetadata, metadata tokens may be used here.

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

% PyCP notes
% (0) Most importantly, we eventually should just have a single ExportData
% module, right? With the option of Excel/spreadsheet (perhaps both tab-delimited and comma-delimited) vs. Oracle DB vs.
% MySQLDB. This would replace both ExportToExcel and ExportToDatabase.
% (1) There should be something in the help explaining Image vs Experiment
% vs Objects... because it seems confusing to ask, Which objects do you
% want to export? and then offer Image and Experiment as choices.
% Actually, I think we should say "What data do you want to export?" then
% list Experiment-wide data, Image data, and then Individual object data.
% If they choose the latter, the list of objects becomes available. There
% should be "Add another" buttons at the level of the first question (Expt
% vs Image vs Object) so they can add infinitely many. There should also be
% an option as to whether to export each object to a new separate file or
% to the same file as the previous object - right? because although many
% times there is precise concordance between dift objects (same # of nuclei
% as cells as cytoplasms) this is sometimes not the case (when we have
% multiple individual speckles inside each nucleus) and currently the
% ExportToExcel module spits them all into separate files whereas
% ExportToDatabase spits them all into a single file. Neither behavior is
% always what people want. If spitting into a single file, there should be
% a check to ensure that the # of objects matches.
%
% (2) Perhaps it should also be made clear that each selection you make
% will create a separate tab-delimited file, and give you the option to
% name these.
% (3) Having all the blank selection menus under the initial
% question has always looked visually unappealing to me; if there is a way
% in Python to allow the user to click a "+" button and simply add another
% object to export, I think that would be preferable to an arbitrary number
% of blank objects. (Agreed --Anne)

drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = Which objects do you want to export?
%infotypeVAR01 = objectgroup
%choiceVAR01 = Image
%choiceVAR01 = Experiment
Object{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 =
%infotypeVAR02 = objectgroup
%choiceVAR02 = Do not use
%choiceVAR02 = Image
%choiceVAR02 = Experiment
Object{2} = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 =
%infotypeVAR03 = objectgroup
%choiceVAR03 = Do not use
%choiceVAR03 = Image
%choiceVAR03 = Experiment
Object{3} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 =
%infotypeVAR04 = objectgroup
%choiceVAR04 = Do not use
%choiceVAR04 = Image
%choiceVAR04 = Experiment
Object{4} = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 =
%infotypeVAR05 = objectgroup
%choiceVAR05 = Do not use
%choiceVAR05 = Image
%choiceVAR05 = Experiment
Object{5} = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 =
%infotypeVAR06 = objectgroup
%choiceVAR06 = Do not use
%choiceVAR06 = Image
%choiceVAR06 = Experiment
Object{6} = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 =
%infotypeVAR07 = objectgroup
%choiceVAR07 = Do not use
%choiceVAR07 = Image
%choiceVAR07 = Experiment
Object{7} = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 =
%infotypeVAR08 = objectgroup
%choiceVAR08 = Do not use
%choiceVAR08 = Image
%choiceVAR08 = Experiment
Object{8} = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%pathnametextVAR09 = Enter the directory where the Excel files are to be saved. Type period (.) to use the default output folder or ampersand (&) for the default input folder. If a FileNameMetadata module was used, metadata tokens may be used here. If this directory does not exist, it will be created automatically.
%defaultVAR09 = .
FileDirectory = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = What prefix should be used to name the Excel files? An underscore will be added to the end of the prefix automatically. Metadata tokens may be used here. Use "Do not use" to prepend the Output filename to the file.
%defaultVAR10 = Do not use
FilePrefix = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

Object(strcmp(Object,'Do not use')) = [];

% If creating batch files, warn that this module only works if the jobs are
% submitted as one batch
if strcmp(handles.Settings.ModuleNames{handles.Current.NumberOfModules},'CreateBatchFiles') && ~isfield(handles.Current, 'BatchInfo'),
    msg = ['You are creating batch file(s) for a cluster run. Please note that ',mfilename,' can only work on the cluster if the jobs are submitted as a single batch, since measurements cannot be compiled from multiple batches.'];
    if isfield(handles.Current, 'BatchInfo'),
        warning(msg);   % If a batch run, print to text (no dialogs allowed)
    else
        CPwarndlg(msg); % If on local machine, create dialog box with the warning
    end
end

% If not a batch run, save the output file name from the uicontrol so it
% can used to name the Excel file (during a batch run, the uicontrol can't
% be accessed)
if ~isfield(handles.Current, 'BatchInfo'),
    handles.Pipeline.OutputFileName = get(handles.OutputFileNameEditBox,'string');
end

isImageGroups = isfield(handles.Pipeline,'ImageGroupFields');
if ~isImageGroups
    SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;
    NumberOfImageSets = handles.Current.NumberOfImageSets;
else
    SetBeingAnalyzed = handles.Pipeline.GroupFileList{handles.Pipeline.CurrentImageGroupID}.SetBeingAnalyzed;
    NumberOfImageSets = handles.Pipeline.GroupFileList{handles.Pipeline.CurrentImageGroupID}.NumberOfImageSets;
end

if SetBeingAnalyzed == NumberOfImageSets
    
    FileDirectory = CPreplacemetadata(handles,FileDirectory);
    if strncmp(FileDirectory,'.',1)
        PathName = fullfile(handles.Current.DefaultOutputDirectory, strrep(strrep(FileDirectory(2:end),'/',filesep),'\',filesep),'');
    elseif strncmp(FileDirectory, '&', 1)
        PathName = handles.Measurements.Image.(['PathName_', ImageFileName]);
        if iscell(PathName), PathName = PathName{SetBeingAnalyzed}; end
    else
        PathName = FileDirectory;
    end
    
    if ~isdir(PathName)
        success = mkdir(PathName);
        if ~success
            error(['Image processing was canceled in the ', ModuleName, ' module because the specified directory "', PathName, '" does not exist.']);
        end
    end
    ExportInfo.ObjectNames = unique(Object);
    ExportInfo.MeasurementExtension = '.xls';
    % Substitute filename metadata tokens into FilePrefix (if found)
    if strcmpi(FilePrefix,'Do not use')
        ExportInfo.MeasurementFilename = handles.Pipeline.OutputFileName;
    else
        ExportInfo.MeasurementFilename = CPreplacemetadata(handles,FilePrefix);
    end
    ExportInfo.IgnoreNaN = 1;
    ExportInfo.SwapRowsColumnInfo = 'No';
    ExportInfo.DataParameter = 'mean';
    ExportInfo.isBatchRun = isfield(handles.Current, 'BatchInfo');

    % If the user wants to export based on a group, group the measurements
    if isImageGroups
        % Get the relevant indices for the filelist
        idx = handles.Pipeline.GroupFileListIDs == handles.Pipeline.CurrentImageGroupID;
        idx = idx(1:handles.Current.SetBeingAnalyzed);  % Truncate to appropriate length
        
        % Extract only those Measurements for the current group
        handles_MeasurementsOnly.Measurements = handles.Measurements;
        ObjectName = fieldnames(handles_MeasurementsOnly.Measurements);
        for j = 1:length(ObjectName),
            if ~strcmp(ObjectName{j}, 'Experiment')
                FeatureName = fieldnames(handles_MeasurementsOnly.Measurements.(ObjectName{j}));
                for k = 1:length(FeatureName),
					if isempty(regexp(FeatureName{k},'^ModuleError','once'))
						handles_MeasurementsOnly.Measurements.(ObjectName{j}).(FeatureName{k})(~idx) = [];
					else
						% The ModuleError field hasn't filled in all the 
						% element yet, so we have to fill it in ourselves
                        handles_MeasurementsOnly.Measurements.(ObjectName{j}).(FeatureName{k}) = repmat({0},[1 length(find(idx))]);
					end
                end
            end
        end
        % Write the Measurement subset out
        CPwritemeasurements(handles_MeasurementsOnly,ExportInfo,PathName);
    else
        % Export the whole thing
        CPwritemeasurements(handles,ExportInfo,PathName);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The figure window display is unnecessary for this module, so it is
%%% closed during the starting image cycle.
CPclosefigure(handles,CurrentModule)