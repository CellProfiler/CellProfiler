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
% $Revision$

%%%%%%%%%%%%%
% VARIABLES %
%%%%%%%%%%%%%
% drawnow

ModuleName = 'CalculateStatisticsDataTool';

[FileName, Pathname] = CPuigetfile('*.mat', 'Select the raw measurements file',handles.Current.DefaultOutputDirectory);
if FileName == 0, return, end   %% CPuigetfile canceled

% Override the DefaultOutputDirectory since the 'load' command below
% may have the denoted the path as on the cluster if CreatebatchFiles 
% was used to generate the raw measurements file
origDefaultOutputDirectory = handles.Current.DefaultOutputDirectory;
origDefaultImageDirectory = handles.Current.DefaultImageDirectory;
%%% Load the specified CellProfiler output file
try
    MsgBoxLoad = CPmsgbox(['Loading file into ' ModuleName '.  Please wait...']);
    temp = load(fullfile(Pathname, FileName));
    handles = CP_convert_old_measurements(temp.handles);
catch
    CPerrordlg(['Unable to load file ''', fullfile(Pathname, FileName), ''' (file does not exist or is not a CellProfiler output file).'])
    close(MsgBoxLoad)
    return
end
close(MsgBoxLoad)


ValidGroupings = false;

while ~ValidGroupings,
    % Is there already dosage data available in the measurements?
    if any(strncmpi(fieldnames(handles.Measurements.Image), 'LoadedText', 10)),
        PreloadedFeatures = get_postfixes(fieldnames(handles.Measurements.Image), 'LoadedText');
        [Selection, ok] = CPlistdlg('ListString', PreloadedFeatures, 'ListSize', [300 400],...
            'Name', 'Select loaded data',...
            'PromptString', 'Choose data preoloaded by LoadText, or select ''Other file''.',...
            'CancelString', 'Other file',...
            'SelectionMode', 'single');

        if ok,
            ValidGroupings = true;
            FeatureName = PreloadedFeatures{Selection};

            Answers = CPinputdlg({'Would you like to log-transform the grouping values before attempting to fit a sigmoid curve? (Yes/No)', ...
                    'Enter the filename to save the plotted dose response data for each feature as an interactive figure in the default output folder (.fig extension will be automatically added). To skip saving figures, enter ''Do not use'''}, ...
                'Operation', 1, {'Yes', 'Do not use'});

            if isempty(Answers), return, end %% Inputdlg canceled

            Logarithmic = Answers{1};
            FigureName = Answers{2};

            break;
        end
    else PreloadedFeatures = '';
    end

    % If no preloaded data, or user selected 'Other file'...
    [TextFileName,TextFilePathname] = CPuigetfile('*.txt', ...
        'Select the text file with the grouping values for each image cycle', ...
        handles.Current.DefaultImageDirectory);
    if TextFileName == 0, return, end %% CPuigetfile cancelled

    Answers = CPinputdlg({'What name would you like to give this data (what column heading)?',...
            'Would you like to log-transform the grouping values before attempting to fit a sigmoid curve? (Yes/No)', ...
            'Enter the filename to save the plotted dose response data for each feature as an interactive figure in the default output folder (.fig extension will be automatically added). To skip saving figures, enter ''Do not use'''}, ...
        'Operation', 1, {'positives','Yes', 'Do not use'});

    if isempty(Answers), return, end %% Inputdlg cancelled
    
    FeatureName = Answers{1};
    Logarithmic = Answers{2};
    FigureName = Answers{3};

    % Check 'Logarithmic'
    if ~ any(strcmpi({'yes', 'y', 'no', 'n', '/'}, Logarithmic)),
        uiwait(CPerrordlg('Error: there was a problem with your choice for whether to log-transform data.'));
        continue
    elseif any(strcmpi({'no', 'n'}, Logarithmic)),
        Logarithmic = '/';
    end
    
    % Check if the user used the same name as a previous LoadText module
    if any(strcmp(PreloadedFeatures, FeatureName)),
        Replace = CPquestdlg(['A feature named ''', FeatureName, ''' already exists in the measurements.  Do you want to replace it?'], 'Existing feature', 'Yes', 'No', 'Cancel', 'Yes');
        if strcmp(Replace, 'No'),
            %%% jump back to the top of the loop. - or should this jump back to just below the CPuigetfile?
            continue;
        end
        if strcmp(Replace, 'Cancel'), % cancelled.
            return
        end

        %%% remove the conflicting measurement
        handles.Measurements.Image = rmfield(handles.Measurements.Image, CPjoinstrings('LoadedText', FeatureName));
    end

    % Save temp values that LoadText needs
    tempVarValues=handles.Settings.VariableValues;
    tempCurrent = handles.Current;
    % Change handles that LoadText requires.
    % Note that DataTools are denoted as Module #1
    handles.Settings.VariableValues{1,1}=TextFileName;
    handles.Settings.VariableValues{1,2}=FeatureName;
    handles.Settings.VariableValues{1,3}=TextFilePathname;
    
    handles.Current.CurrentModuleNumber='01';
    handles.Current.SetBeingAnalyzed=1;
    handles.Current.DefaultImageDirectory = origDefaultImageDirectory; %% In case cluster path is different
    % Load Text
    handles = LoadText(handles);
    % Return previous values
    handles.Settings.VariableValues=tempVarValues;
    handles.Current=tempCurrent;

    % success...
    ValidGroupings = true;
end
    

Answer = CPinputdlg({'What do you want to call the output file with statistics?'},...
    'Calculate Statistics DataTool',1,{'StatsOUT.mat'});
if isempty(Answer), return, end
OutputFileName = Answer{1};

% Override the DefaultOutputDirectory since the 'load' command above
% may have the denoted the path as on the cluster if CreatebatchFiles 
% was used to generate the raw measurements file
handles.Current.DefaultOutputDirectory = origDefaultOutputDirectory;
handles.Current.DefaultImageDirectory = origDefaultImageDirectory;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PRELIMINARY CALCULATIONS %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Checks whether the user has the Statistics Toolbox.
LicenseStats = license('test','statistics_toolbox');
if LicenseStats ~= 1
    CPwarndlg('It appears that you do not have a license for the Statistics Toolbox of Matlab.  You will be able to calculate V and Z'' factors, but not EC50 values. Typing ''ver'' or ''license'' at the Matlab command line may provide more information about your current license situation.');
end

handles = CPcalculateStatistics(handles,CPjoinstrings('LoadedText', FeatureName),Logarithmic,FigureName,ModuleName,LicenseStats); %#ok<NASGU>

% Save the updated CellProfiler output file
try
    save(fullfile(Pathname, OutputFileName),'handles');
    CPmsgbox(['Updated ',OutputFileName,' successfully saved.']);
catch
    CPwarndlg(['Could not save updated ',OutputFileName,' file.']);
end


function postfixes = get_postfixes(names, prefix)
postfixes = regexp(names, [prefix '_(.*)'], 'tokens', 'once');
postfixes(cellfun('isempty', postfixes)) = [];
postfixes = cellfun(@(x) x{1}, postfixes, 'UniformOutput', false);
