function [ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(handles)
%
%   This function takes the user through three list dialogs where a
%   specific feature is chosen. It is possible to go back and forth
%   between the list dialogs. The chosen feature can be identified
%   via the output variables
%


%%% Extract the fieldnames of measurements from the handles structure.
MeasFieldnames = fieldnames(handles.Measurements);

%%% Error detection.
if isempty(MeasFieldnames)
    errordlg('No measurements were found.')
    ObjectTypename = [];FeatureType = [];FeatureNo = [];
    return
end

dlgno = 1;                            % This variable keeps track of which list dialog is shown
while dlgno < 4
    switch dlgno
        case 1
            [Selection, ok] = listdlg('ListString',MeasFieldnames, 'ListSize', [300 400],...
                'Name','Select measurement',...
                'PromptString','Choose an object type',...
                'CancelString','Cancel',...
                'SelectionMode','single');
            if ok == 0
                ObjectTypename = [];FeatureType = [];FeatureNo = [];
                return
            end
            ObjectTypename = MeasFieldnames{Selection};

            % Get all fields with the suffix 'Features'. These fields
            % contain measurements.
            FeatureTypes = fieldnames(handles.Measurements.(ObjectTypename));
            tmp = {};
            for k = 1:length(FeatureTypes)
                if ~isempty(strfind(FeatureTypes{k},'Features'))
                    tmp{end+1} = FeatureTypes{k}(1:end-8);    % Remove the 'Features' suffix
                end
            end
            FeatureTypes = tmp;
            dlgno = 2;                      % Indicates that the next dialog box is to be shown next
        case 2
            [Selection, ok] = listdlg('ListString',FeatureTypes, 'ListSize', [300 400],...
                'Name','Select measurement',...
                'PromptString',['Choose a feature type for ', ObjectTypename],...
                'CancelString','Back',...
                'SelectionMode','single');
            if ok == 0
                dlgno = 1;                  % Back button pressed, go back one step in the menu system
            else
                FeatureType = FeatureTypes{Selection};
                Features = handles.Measurements.(ObjectTypename).([FeatureType 'Features']);
                dlgno = 3;                  % Indicates that the next dialog box is to be shown next
            end
        case 3
            [Selection, ok] = listdlg('ListString',Features, 'ListSize', [300 400],...
                'Name','Select measurement',...
                'PromptString',['Choose a ',FeatureType,' of ', ObjectTypename],...
                'CancelString','Back',...
                'SelectionMode','single');
            if ok == 0
                dlgno = 2;                  % Back button pressed, go back one step in the menu system
            else
                FeatureNo = Selection;
                dlgno = 4;                  % dlgno = 4 will exit the while-loop
            end
    end
end