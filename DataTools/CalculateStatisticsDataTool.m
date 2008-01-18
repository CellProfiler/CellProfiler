function CalculateStatisticsDataTool(handles)

% Help for the Calculate Statistics data tool module:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Calculates measures of assay quality (V and Z' factors) and dose response
% data (EC50) for all measured features made from images.
% *************************************************************************
% Note: this tool is beta-version and has not been thoroughly checked.
%
% See the help for the CalculateStatistics module for information on the
% settings for this data tool and how to use it.

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
% $Revision: 4711 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
% drawnow

ModuleName = 'CalculateStatisticsDataTool';

[FileName, Pathname] = CPuigetfile('*.mat', 'Select the raw measurements file',handles.Current.DefaultOutputDirectory);
if FileName == 0, return, end   %% CPuigetfile canceled

%% Override the DefaultOutputDirectory since the 'load' command below
%% may have the denoted the path as on the cluster if CreatebatchFiles 
%% was used to generate the raw measurements file
origDefaultOutputDirectory = handles.Preferences.DefaultOutputDirectory;
origDefaultImageDirectory = handles.Preferences.DefaultImageDirectory;
%%% Load the specified CellProfiler output file
try
    load(fullfile(Pathname, FileName));
catch
    CPerrordlg('Selected file is not a CellProfiler or MATLAB file (it does not have the extension .mat).')
    return
end

promptLoop = 0;
while promptLoop == 0
    oldLoadTextModNum = find(strcmp(handles.Settings.ModuleNames,'LoadText'), 1);
    if ~isempty(oldLoadTextModNum) %% If user already used a LoadText Module in their original pipeline
        %% *NOTE* IF LoadText changes the order or composition of its
        %% queries, the '1' and '3' below may need to be changed!
        TextFileName = handles.Settings.VariableValues{oldLoadTextModNum,1};
        DataName = handles.Settings.VariableValues{oldLoadTextModNum,2};
        TextFilePathname = handles.Settings.VariableValues{oldLoadTextModNum,3};
        
        Answers = CPinputdlg({'Would you like to log-transform the grouping values before attempting to fit a sigmoid curve? (Yes/No)', ...
            'Enter the filename to save the plotted dose response data for each feature as an interactive figure in the default output folder (.fig extension will be automatically added). To skip saving figures, enter ''Do not save'''}, ...
            'Operation', 1, {'Yes', 'Do not save'});
        if isempty(Answers), return, end %% Inputdlg canceled
        LogOrLinear = Answers{1};
        FigureName = Answers{2};

    else 
        %% If no Loadtext Module used previously
        [TextFileName,TextFilePathname] = CPuigetfile('*.txt', ...
            'Select the text file with the grouping values you loaded for each image cycle', ...
            handles.Current.DefaultImageDirectory);
        if TextFileName == 0, return, end %% CPuigetfile canceled

        Answers = CPinputdlg({'What name would you like to give this data (what column heading)? You can leave this empty if a LoadText Module was already run',...
            'Would you like to log-transform the grouping values before attempting to fit a sigmoid curve? (Yes/No)', ...
            'Enter the filename to save the plotted dose response data for each feature as an interactive figure in the default output folder (.fig extension will be automatically added). To skip saving figures, enter ''Do not save'''}, ...
            'Operation', 1, {'positives','Yes', 'Do not save'});
        if isempty(Answers), return, end %% Inputdlg canceled
        DataName = Answers{1};
        LogOrLinear = Answers{2};
        FigureName = Answers{3};
    end

    %% Check 'DataName'
    if isempty(DataName) && ~isfield(handles.Measurements.Image,DataName)
        uiwait(CPerrordlg('Error: there was a problem with your choice of grouping values'));
        continue
    end
    
    %% Check 'LogOrLinear'
    if ~strcmpi(LogOrLinear, 'yes') &&  ~strcmpi(LogOrLinear, 'y') && ~strcmpi(LogOrLinear, 'no') && ~strcmpi(LogOrLinear, 'n') && ~strcmpi(LogOrLinear, '/')
        uiwait(CPerrordlg('Error: there was a problem with your choice for log10'));
        continue
    elseif strcmpi(LogOrLinear, 'no') || strcmpi(LogOrLinear, 'n')
        LogOrLinear = '/';
    end

    promptLoop = 1;
end
    
%% This section below is similar to AddData.m, which also calls LoadText.m

%% Save temp values that LoadText needs
tempVarValues=handles.Settings.VariableValues;
tempCurrentField = handles.Current;
%% Change handles that LoadText requires
handles.Settings.VariableValues{1,1}=TextFileName;
handles.Settings.VariableValues{1,2}=DataName;
handles.Settings.VariableValues{1,3}=TextFilePathname;
handles.Current.CurrentModuleNumber='01';
handles.Current.SetBeingAnalyzed=1;
handles.Current.DefaultImageDirectory = origDefaultImageDirectory; %% In case cluster path is different
%% Load Text
handles = LoadText(handles);
%% Return previous values
handles.Settings.VariableValues=tempVarValues;
handles.Current=tempCurrentField;

Answer = CPinputdlg({'What do you want to call the output file with statistics?'},...
    'Calculate Statistics DataTool',1,{'StatsOUT.mat'});
if isempty(Answer), return, end
OutputFileName = Answer{1};

%% Override the DefaultOutputDirectory since the 'load' command above
%% may have the denoted the path as on the cluster if CreatebatchFiles 
%% was used to generate the raw measurements file
handles.Current.DefaultOutputDirectory = origDefaultOutputDirectory;
handles.Current.DefaultImageDirectory = origDefaultImageDirectory;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Checks whether the user has the Image Processing Toolbox.
LicenseStats = license('test','statistics_toolbox');
if LicenseStats ~= 1
    CPwarndlg('It appears that you do not have a license for the Statistics Toolbox of Matlab.  You will be able to calculate V and Z'' factors, but not EC50 values. Typing ''ver'' or ''license'' at the Matlab command line may provide more information about your current license situation.');
end

handles = CPcalculateStatistics(handles,DataName,LogOrLinear,FigureName,ModuleName,LicenseStats); %#ok<NASGU>

%%% Save the updated CellProfiler output file
try
    save(fullfile(Pathname, OutputFileName),'handles');
    CPmsgbox(['Updated ',OutputFileName,' successfully saved.'])
catch
    CPwarndlg(['Could not save updated ',OutputFileName,' file.']);
end