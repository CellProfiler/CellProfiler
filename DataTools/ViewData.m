function handles = ViewData(handles)

% Help for the View Data tool:
% Category: Data Tools
%
% This module has not yet been documented. It
% allows viewing any list of sample info or data, specified by its heading,
% taken from the handles structure of an output file or existing memory.
%
% See also CLEARDATA VIEWDATA.

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

ExistingOrMemory = CPquestdlg('Do you want to view sample info or data in an existing output file or do you want to view the sample info or data stored in memory to be placed into future output files?', 'View Sample Info', 'Existing', 'Memory', 'Cancel', 'Existing');
if strcmp(ExistingOrMemory, 'Cancel') == 1 | isempty(ExistingOrMemory) ==1
    %%% Allows canceling.
    return
elseif strcmp(ExistingOrMemory, 'Memory') == 1
    %%% Checks whether any headings are loaded yet.
    Fieldnames = fieldnames(handles.Measurements);
    ImportedFieldnames = Fieldnames(strncmp(Fieldnames,'Imported',8) == 1);
    if isempty(ImportedFieldnames) == 1
        errordlg('No sample info or data is currently stored in memory.')
        %%% Opens a filenameslistbox which displays the list of headings so that one can be
        %%% selected.  The OK button has been assigned to mean "View".
    else
        [Selected,Action] = listdlg('ListString',ImportedFieldnames, 'ListSize', [300 600],...
            'Name','Current sample info loaded',...
            'PromptString','Select the sample descriptions you would like to view.',...
            'OKString','View','CancelString','Cancel','SelectionMode','single');

        %%% Extracts the actual heading name.
        SelectedFieldName = ImportedFieldnames(Selected);

        % Action = 1 if the user pressed the OK (VIEW) button.  If they pressed
        % the cancel button or closed the window Action == 0.
        if Action == 1
            CPtextdisplaybox(ListToShow, char(SelectedFieldName))
            %%% The OK buttons within this window don't do anything.
        else
            %%% If the user pressed "cancel" or closes the window, Action = 0, so
            %%% nothing happens.
        end
        %%% This "end" goes with the "isempty" if no sample info is loaded.
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
    %%% Checks whether any sample info is contained within the file. Some
    %%% old output files may not have the 'Measurements'
    %%% substructure, so we check for that field first.
    if isfield(OutputFile.handles,'Measurements') == 1
        Fieldnames = fieldnames(OutputFile.handles.Measurements);
        ImportedFieldnames = Fieldnames(strncmp(Fieldnames,'Imported',8) == 1 | strncmp(Fieldnames,'Image',5) == 1 | strncmp(Fieldnames,'Filename',8) == 1 );
    else ImportedFieldnames = [];
    end
    if isempty(ImportedFieldnames) == 1
        errordlg('The output file you selected does not contain any sample info or data. It would be in a field called handles.Measurements, and would be prefixed with either ''Image'' or ''Imported''.')
        %%% Opens a filenameslistbox which displays the list of headings so that one can be
        %%% selected.  The OK button has been assigned to mean "View".
    else
        [Selected,Action] = listdlg('ListString',ImportedFieldnames, 'ListSize', [300 600],...
            'Name','Current sample info loaded',...
            'PromptString','Select the sample descriptions you would like to view.',...
            'OKString','View','CancelString','Cancel','SelectionMode','single');

        %%% Extracts the actual heading name.
        SelectedFieldName = ImportedFieldnames(Selected);

        % Action = 1 if the user pressed the OK (VIEW) button.  If they pressed
        % the cancel button or closed the window Action == 0.
        if Action == 1
            try
                ListToShow = OutputFile.handles.Measurements.(char(SelectedFieldName));
                if strcmp(class(ListToShow{1}),'double') == 1
                    ListToShow = sprintf('%d\n',cell2mat(ListToShow));
                end
                CPtextdisplaybox(ListToShow, char(SelectedFieldName))
                %%% The OK buttons within this window don't do anything.
            catch errordlg('Sorry, there was an error displaying this sample info or data. This function may not yet work properly on mixed numerical and text data.')
            end
        else
            %%% If the user pressed "cancel" or closes the window, Action = 0, so
            %%% nothing happens.
        end
        %%% This "end" goes with the "isempty" if no sample info is loaded.
    end
end