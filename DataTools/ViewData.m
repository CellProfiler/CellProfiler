function ViewData(handles)

% Help for the View Data tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Displays data or measurements from a CellProfiler output file. This is
% displayed after the user specifies which output file and which
% measurements to extract data from.
% *************************************************************************
% Note: this tool is beta-version and has not been thoroughly checked.
%
% This tool views any text data or measurements that have been stored in a
% CellProfiler output file. It can be useful to check that any text data
% added with the AddData tool is associated with the correct image sets.

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

% Ask the user to choose the file from which to extract measurements.
[FileName, Pathname] = CPuigetfile('*.mat', 'Select the raw measurements file',handles.Current.DefaultOutputDirectory);
if FileName == 0
    return
end

% Quick check if it seems to be a CellProfiler file or not
s = whos('-file',fullfile(Pathname, FileName));
if ~any(strcmp('handles',cellstr(cat(1,s.name)))),
    CPerrordlg('Selected file is not a CellProfiler output file.')
    return
end

% Load the specified CellProfiler output file
try
    load(fullfile(Pathname, FileName));
catch
    CPerrordlg('Selected file is not a CellProfiler or MATLAB file (it does not have the extension .mat).')
    return
end

% Try to convert features
handles = CP_convert_old_measurements(handles);

FinalOK = 0;
while FinalOK == 0
    %%% Let the user select which feature to view
    try
        [ObjectTypename,FeatureType] = CPgetfeature(handles,0);
    catch
        ErrorMessage = lasterr;
        CPerrordlg(['An error occurred in the ViewData Data Tool. ' ErrorMessage(30:end)]);
        return
    end
    if isempty(ObjectTypename),return,end

    %%% Generate a cell array with strings to display
    NbrOfImageSets = length(handles.Measurements.(ObjectTypename).(FeatureType));
    TextToDisplay = cell(NbrOfImageSets,1);
    
    filenames = fieldnames(handles.Measurements.Image);
    idx = ~cellfun(@isempty,strfind(filenames,'FileName'));
    fileswithmeasurements = handles.Measurements.Image.(char(filenames(find(idx,1))));
    if NbrOfImageSets ~= length(fileswithmeasurements)
        CPerrordlg('There is an unequal number of measures to the number of image files. This has not yet been supported.');
    end
    
    filenames = filenames(idx,:);
    filenamelist = {}; for i = 1:length(filenames),filenamelist = cat(1,filenamelist,handles.Measurements.Image.(filenames{i})); end
    for ImageSet = 1:NbrOfImageSets
        if ~isempty(handles.Measurements.(ObjectTypename).(FeatureType){ImageSet}),
            info = handles.Measurements.(ObjectTypename).(FeatureType){ImageSet};
        else
            info = 'No objects identified';
        end
        if isnumeric(info), info = num2str(mean(info)); end     % Numeric data
        if iscell(info), info = char(info); end                 % Text data
        try
            TextToDisplay{ImageSet} = sprintf('Cycle #%d, %s:     %s',...
                ImageSet,...
                filenamelist{1,ImageSet},...
                info(1,:));
            if size(info,1) > 1,    % If there is a vector of text data
                for j = 2:size(info,1),
                    TextToDisplay{ImageSet} = strvcat(1,TextToDisplay{ImageSet},...
                        sprintf('                 %s',info(j,:)));
                end
            end
        catch
            CPerrordlg('Use the data tool MergeOutputFiles or ConvertBatchFiles to convert the data first');
            return;
        end
    end

    %%% Produce an infostring that explains what is displayed
    if strfind(FeatureType,'Text')
        InfoString = 'Cycle #,  <filename>:     <text>';
    elseif strcmp(ObjectTypename,'Image')
        InfoString = 'Cycle #,  <filename>:     <value>';
    else
        InfoString = 'Cycle #,  <filename>:     <mean value>';
    end
    TextToDisplay = cat(1,{InfoString},{''},TextToDisplay);

    % Display data in a list dialog box
    [Selection, FinalOK] = listdlg('ListString',TextToDisplay, 'ListSize', [600 200],...
        'Name',['Information for ',FeatureType],...
        'PromptString','Press ''Back'' to select another information entry.',...
        'CancelString','Back',...
        'SelectionMode','single');
end