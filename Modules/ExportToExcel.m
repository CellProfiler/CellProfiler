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
% Do you want to create the input image subdirectory structure in the  
% output directory?
% If the input images are located in subdirectories (such that you used 
% "Analyze all subfolders within the selected folder" in LoadImages), you 
% can re-create the subdirectory structure in the default output directory.

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

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);%#ok Ignore MLint

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

%textVAR09 = Do you want to create subdirectories in the default output directory to match the input image directory structure?
%choiceVAR09 = No
%choiceVAR09 = Yes
CreateSubdirectories = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow
tmp = {};
for n = 1:8
    if ~strcmp(Object{n}, 'Do not use')
        tmp{end+1} = Object{n};
    end
end
Object = tmp;

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
        
if handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets
    RawPathname = handles.Current.DefaultOutputDirectory;
    ExportInfo.ObjectNames = unique(Object);
    ExportInfo.MeasurementExtension = '.xls';
    ExportInfo.MeasurementFilename = handles.Pipeline.OutputFileName;
    ExportInfo.IgnoreNaN = 1;
    ExportInfo.SwapRowsColumnInfo = 'No';
    ExportInfo.DataParameter = 'mean';
    
    % If the user wants to add subdirectories, alter the output path accordingly
    if strncmpi(CreateSubdirectories,'y',1)
        fn = fieldnames(handles.Measurements.Image);
        % We will be pulling the pathname list from handles.Pipeline but we 
        % need an object to reference first. We obtain this from
        % handles.Measurement.Image
        % ASSUMPTION: The needed object was created by LoadImages/LoadSingleImage
        % which (a) produces a field with 'FileName_' at the beginning and 
        % (b) the first such field is sufficient (i.e, all other image 
        % objects come from this directory tree as well)
        prefix = 'FileName_';
        idx = find(~cellfun(@isempty,regexp(fn,['^',prefix])),1);
        ImageFileName = fn{idx}((length(prefix)+1):end);
        
        % Pull the input filelist from handles.Pipeline and parse for
        % subdirs
        fn = handles.Pipeline.(['FileList', ImageFileName]);
        PathStr = cellfun(@fileparts,fn,'UniformOutput',false);
        uniquePathStr = unique(PathStr);
        
        % Process each subdir
        for i = 1:length(uniquePathStr),
            SubDir = uniquePathStr{i};
            SubRawPathName = fullfile(RawPathname, SubDir,'');
            if ~isdir(SubRawPathName),    % If the directory doesn't already exist, create it
                success = mkdir(SubRawPathName);
                if ~success, error(['Image processing was canceled in the ', ModuleName, ' module because the specified subdirectory "', SubRawPathName, '" could not be created.']); end
            end

            % Extract only those Measurements for the current subdirectory
            idx = strcmp(PathStr,uniquePathStr{i});
            handles_MeasurementsOnly.Measurements = handles.Measurements;
            ObjectName = fieldnames(handles_MeasurementsOnly.Measurements);
            for j = 1:length(ObjectName),
                if ~strcmp(ObjectName{j}, 'Experiment')
                    FeatureName = fieldnames(handles_MeasurementsOnly.Measurements.(ObjectName{j}));
                    for k = 1:length(FeatureName),
                        if isempty(regexp(FeatureName{k},[ModuleName,'$'],'once'))
                            handles_MeasurementsOnly.Measurements.(ObjectName{j}).(FeatureName{k})(~idx) = [];
                        else
                            % The ModuleError field for ExportToExcel hasn't
                            % filled in the last element yet, so we have to
                            % fill it in ourselves
                            handles_MeasurementsOnly.Measurements.(ObjectName{j}).(FeatureName{k}) = repmat({0},[1 length(find(idx))]);
                        end
                    end
                end
            end
            % Write the Measurement subset out
            CPwritemeasurements(handles_MeasurementsOnly,ExportInfo,SubRawPathName);
        end
    else
        % Export the whole thing
        CPwritemeasurements(handles,ExportInfo,RawPathname);
    end
    
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The figure window display is unnecessary for this module, so it is
%%% closed during the starting image cycle.
CPclosefigure(handles,CurrentModule)