function handles = ClearData(handles)

% Help for the Clear Data tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Removes information/measurements from an output file.
% *************************************************************************
% Note: this tool is beta-version and has not been thoroughly checked.
%
% This tool lets the user remove a measurement or data field from a
% CellProfiler output file. The same measurement can be removed from
% several files.
%
% After executing this option, CelProfiler will ask the user to specify the
% output file(s) from which to remove data from. The user will then specify
% which data to clear. In most cases, the data to be cleared will be data
% providing information about an object.

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

%%% Get the pathname and let user select the files he wants
[Pathname, SelectedFiles] = CPselectoutputfiles(handles);

%%% Check if cancel button pressed
if ~iscellstr(SelectedFiles)
    return
end

%%% Load the first specified CellProfiler output file so we can choose the
%%% feature to be removed.
%%% Load the specified CellProfiler output file
try
    temp = load(fullfile(Pathname, SelectedFiles{1}));
    handles = CP_convert_old_measurements(temp.handles);
catch
    CPerrordlg(['Unable to load file ''', fullfile(Pathname, SelectedFiles{1}), ''' (possibly not a CellProfiler output file).'])
    return
end

[MeasureObject,Measurefieldname] = CPgetfeature(handles,false);

%%% If Cancel button pressed
if isempty(MeasureObject),return,end

%%% Ask the user if he really wants to clear the selected feature
Confirmation = CPquestdlg(['Are you sure you want to delete the selected feature? (', MeasureObject, ' ', Measurefieldname, ')'],'Confirmation','Yes','Cancel','Cancel');
if strcmp(Confirmation,'Cancel')
    return
end



%%% Loop over the selected files and remove the selected feature
%%% An cell array is used to indicated any errors in the processing
errors = cell(length(SelectedFiles),1);
for FileNbr = 1:length(SelectedFiles)

    %%% Load the specified CellProfiler output file
    try
        temp = load(fullfile(Pathname, SelectedFiles{FileNbr}));
        handles = CP_convert_old_measurements(temp.handles);
    catch
        errors{FileNbr} = [fullfile(Pathname, SelectedFiles{FileNbr}),' is not a CellProfiler or MATLAB file (it does not have the extension .mat)'];
        return
    end

    handles.Measurements.(MeasureObject) = rmfield(handles.Measurements.(MeasureObject), Measurefieldname);

    %%% Save the updated CellProfiler output file
    try
        save(fullfile(Pathname, SelectedFiles{FileNbr}),'handles');
    catch
        errors{FileNbr} = ['Could not save updated ',fullfile(Pathname, SelectedFiles{FileNbr}),' file.'];
        continue
    end
end

%%% Finished, display success or warning windows if we failed for some data set
error_index = find(~cellfun('isempty',errors));
if isempty(error_index)
    CPmsgbox('Data successfully deleted.');
else
    %%% Show a warning dialog box for each error
    for k = 1:length(error_index)
        CPwarndlg(errors{error_index(k)},'Clear Data failure');
    end
end
