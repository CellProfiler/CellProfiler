function ExportLocations(handles)

% Help for the Export Locations tool:
% Category: Data Tools

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision: 2950 $

%%% Ask the user to choose the file from which to extract
%%% measurements. The window opens in the default output directory.
[RawFileName, RawPathname] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'.','*.mat'),'Select the raw measurements file');
%%% Allows canceling.
if RawFileName == 0
    return
end
load(fullfile(RawPathname, RawFileName));

if ~exist('handles','var')
    error('This is not a CellProfiler output file.');
end

try FontSize = handles.Preferences.FontSize;
    %%% We used to store the font size in Current, so this line makes old
    %%% output files compatible. Shouldn't be necessary with any files made
    %%% after November 15th, 2006.
catch FontSize = handles.Current.FontSize;
end

%%% Quick check if it seems to be a CellProfiler file or not
if ~isfield(handles,'Measurements')
    errordlg('The selected file does not contain any measurements.')
    return
end

%%% Extract the fieldnames of measurements from the handles structure.
MeasFieldnames = fieldnames(handles.Measurements);
for i=1:length(handles.Measurements)
    if strcmp(MeasFieldnames{i},'Image')
        MeasFieldnames(i) = [];
    end
end

[Selection, ok] = listdlg('ListString',MeasFieldnames,'ListSize', [300 400],...
    'Name','Select measurement',...
    'PromptString','Which object do you want to export locations from?',...
    'CancelString','Cancel',...
    'SelectionMode','single');

if ok == 0
    return
end

ObjectTypename = MeasFieldnames{Selection};

if isfield(handles.Measurements.(ObjectTypename),'Location')
    Locations = handles.Measurements.(ObjectTypename).Location;
else
    error('The object you have chosen does not have location measurements.')
end

for ImageNumber = 1:length(Locations)
    filename = [ObjectTypename,'_Locations_Image_',num2str(ImageNumber),'.csv'];
    fid = fopen(fullfile(handles.Current.DefaultOutputDirectory,filename),'w');
    if fid == -1
        error(sprintf('Cannot create the output file %s. There might be another program using a file with the same name.',filename));
    end
    for ObjectNumber = 1:size(Locations{ImageNumber},1)
        fprintf(fid,[Locations{ImageNumber}(ObjectNumber,1),'\t',Locations{ImageNumber}(ObjectNumber,2),'\n']);
    end
    fclose(fid)
end