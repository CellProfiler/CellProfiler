function handles = ClearData(handles)

% Help for the Clear Data tool:
% Category: Data Tools
%
% This module has not yet been documented. The Clear Sample Info
% button allows deleting any list of sample info, specified by its
% heading, from the handles structure.
%
% See also ADDDATA VIEWDATA.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$

ExistingOrMemory = CPquestdlg('Do you want to delete sample info or data in an existing output file or do you want to delete the sample info or data stored in memory to be placed into future output files?', 'Delete Sample Info', 'Existing', 'Memory', 'Cancel', 'Existing');
if strcmp(ExistingOrMemory, 'Cancel') == 1 | isempty(ExistingOrMemory) ==1
    %%% Allows canceling.
    return
elseif strcmp(ExistingOrMemory, 'Memory') == 1
    %%% Checks whether any headings are loaded yet.
    Fieldnames = fieldnames(handles.Measurements);
    ImportedFieldnames = Fieldnames(strncmp(Fieldnames,'Imported',8) == 1);
    if isempty(ImportedFieldnames) == 1
        errordlg('No sample info has been loaded.')
    else
        %%% Opens a filenameslistbox which displays the list of headings so that one can be
        %%% selected.  The OK button has been assigned to mean "Delete".
        [Selected,Action] = listdlg('ListString',ImportedFieldnames, 'ListSize', [300 600],...
            'Name','Current sample info loaded',...
            'PromptString','Select the sample descriptions you would like to delete',...
            'OKString','Delete','CancelString','Cancel','SelectionMode','single');
        %%% Extracts the actual heading name.
        SelectedFieldName = ImportedFieldnames(Selected);
        % Action = 1 if the user pressed the OK (DELETE) button.  If they pressed
        % the cancel button or closed the window Action == 0 and nothing happens.
        if Action == 1
            %%% Delete the selected heading (with its contents, the sample data)
            %%% from the structure.
            handles.Measurements = rmfield(handles.Measurements,SelectedFieldName);
            %%% Handles structure is updated
            guidata(gcbo,handles)
            h = CPmsgbox(['The sample info was successfully deleted from memory']);
        end
        %%% This end goes with the error-detecting - "Do you have any sample info
        %%% loaded?"
    end
elseif strcmp(ExistingOrMemory, 'Existing') == 1
[fOutName,pOutName] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'MATLABBUG11432TP','*.mat'),'Choose the output file');
    %%% Allows canceling.
    if fOutName == 0
        return
    else
        try OutputFile = load(fullfile(pOutName,fOutName));
        catch error('Sorry, the file could not be loaded for some reason.')
        end
    end
    %%% Checks whether any sample info is contained within the file.
    Fieldnames = fieldnames(OutputFile.handles.Measurements);
    ImportedFieldnames = Fieldnames(strncmp(Fieldnames,'Imported',8) == 1 | strncmp(Fieldnames,'Image',5) == 1);
    if isempty(ImportedFieldnames) == 1
        errordlg('The output file you selected does not contain any sample info or data. It would be in a field called handles.Measurements, and would be prefixed with either ''Image'' or ''Imported''.')
    else
        %%% Opens a filenameslistbox which displays the list of headings so that one can be
        %%% selected.  The OK button has been assigned to mean "Delete".
        [Selected,Action] = listdlg('ListString',ImportedFieldnames, 'ListSize', [300 600],...
            'Name','Current sample info loaded',...
            'PromptString','Select the sample descriptions you would like to delete',...
            'OKString','Delete','CancelString','Cancel','SelectionMode','single');
        %%% Extracts the actual heading name.
        SelectedFieldName = ImportedFieldnames(Selected);
        % Action = 1 if the user pressed the OK (DELETE) button.  If they pressed
        % the cancel button or closed the window Action == 0 and nothing happens.
        if Action == 1
            %%% Delete the selected heading (with its contents, the sample data)
            %%% from the structure.
            OutputFile.handles.Measurements = rmfield(OutputFile.handles.Measurements,SelectedFieldName);
            %%% Saves the output file with this new sample info.
            save(fullfile(pOutName,fOutName),'-struct','OutputFile');
            h = CPmsgbox(['The sample info was successfully deleted from the output file']);
        end
        %%% This end goes with the error-detecting - "Do you have any sample info
        %%% loaded?"
    end
end