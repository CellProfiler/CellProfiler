function [ObjectTypename,FeatureName] = CPgetfeature(handles,ExcludeImageMeasurements,AllowObjectNumbers)

% This function takes the user through three list dialogs where a specific
% feature is chosen. It is possible to go back and forth between the list
% dialogs. The chosen feature can be identified via the output variables as
% handles.Measurements.(ObjectTypename).(FeatureType){FeatureNbr} Empty
% variables will be returned if the cancel button is pressed.
%
% The input variable 'suffix' is a cell array containing strings of
% suffixes to look for. 

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

if nargin < 2
    ExcludeImageMeasurements = false;
end

if nargin < 3
    AllowObjectNumbers = false;
end

%%% Quick check if it seems to be a CellProfiler file or not
if ~isfield(handles,'Measurements')
    error('The selected file does not contain any measurements.')
end

%%% Extract the Object names the handles structure.
ObjectNames = fieldnames(handles.Measurements);

if ExcludeImageMeasurements
    ObjectNames(strcmp(ObjectNames, 'Image')) = [];
end

%%% Error detection.
if isempty(ObjectNames)
    error('No measurements were found in the selected file.')
end

dlgno = 1;                            % This variable keeps track of which list dialog is shown

while dlgno < 4
    switch dlgno
        case 1
            [Selection, ok] = CPlistdlg('ListString',ObjectNames,'ListSize', [300 400],...
                'Name','Select object type',...
                'PromptString','Choose an object type',...
                'CancelString','Cancel',...
                'SelectionMode','single');
            if ok == 0
                ObjectTypename = [];FeatureName = [];
                return
            end
            ObjectTypename = ObjectNames{Selection};
            dlgno = 2;                      % Indicates that the next dialog box is to be shown next

        case 2
            % Get feature prefixes for this object
            FeatureTypes = get_prefixes(fieldnames(handles.Measurements.(ObjectTypename)));
            if (AllowObjectNumbers),
                FeatureTypes{end + 1} = 'Object Number';
            end
            [Selection, ok] = CPlistdlg('ListString',FeatureTypes, 'ListSize', [300 400],...
                'Name','Select measurement',...
                'PromptString',['Choose a feature type for ', ObjectTypename],...
                'CancelString','Back',...
                'SelectionMode','single');
            if ok == 0
                dlgno = 1;                  % Back button pressed, go back one step in the menu system
            else
                FeatureType = FeatureTypes{Selection};
                if strcmp(FeatureType, 'Object Number'),
                    %%% exit
                    FeatureName = 'Object Number';
                    dlgno = 4;
                else
                    dlgno = 3;                  % Indicates that the next dialog box is to be shown next
                end
            end
        case 3
            % Get features for this selection
            Features = get_postfixes(fieldnames(handles.Measurements.(ObjectTypename)), FeatureType);
            [Selection, ok] = CPlistdlg('ListString',Features, 'ListSize', [300 400],...
                'Name','Select measurement',...
                'PromptString',['Choose a feature of type ',ObjectTypename ' / ' FeatureType],...
                'CancelString','Back',...
                'SelectionMode','single');
            if ok == 0
                dlgno = 2;                  % Back button pressed, go back one step in the menu system
            else
                FeatureName = CPjoinstrings(FeatureType, Features{Selection});
                dlgno = 4;                  % dlgno = 4 will exit the while-loop
            end
    end
end


function prefixes = get_prefixes(names)
prefixes = cellfun(@(x) x{1}, regexp(names, '^([^_])+', 'tokens', 'once'), 'UniformOutput', false);
idx = 1;
while idx < length(prefixes)
    duplicates = strmatch(prefixes{idx}, prefixes);
    prefixes(duplicates(2:end)) = [];
    idx = idx + 1;
end


function postfixes = get_postfixes(names, prefix)
postfixes = regexp(names, [prefix '_(.*)'], 'tokens', 'once');
postfixes(cellfun('isempty', postfixes)) = [];
postfixes = cellfun(@(x) x{1}, postfixes, 'UniformOutput', false);
