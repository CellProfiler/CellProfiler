function ExportLocations(handles)

% Help for the Export Locations tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Exports center locations of objects. Specialty function for creating a
% locations list for microscopy image acquisition of gridded spots.
% *************************************************************************
% Useful for creating a locations list for microscope.

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
    CPerrordlg('This is not a CellProfiler output file.');
    return
end

%%% Quick check if it seems to be a CellProfiler file or not
if ~isfield(handles,'Measurements')
    CPerrordlg('The selected file does not contain any measurements.')
    return
end

%%% Extract the fieldnames of measurements from the handles structure.
MeasFieldnames = fieldnames(handles.Measurements);
for i=1:length(handles.Measurements)
    if strcmp(MeasFieldnames{i},'Image')
        MeasFieldnames(i) = [];
    end
    if isempty(MeasFieldnames)
        CPerrordlg('The output file you have chosen does not have location measurements.');
        return
    end
end

[Selection, ok] = listdlg('ListString',MeasFieldnames,'ListSize', [300 200],...
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
    CPerrordlg('The object you have chosen does not have location measurements.');
    return
end

AcceptableAnswers = 0;
while AcceptableAnswers == 0
    Prompts{1} = 'What do you want the x,y location of the first object to be (in final units)?';
    Prompts{2} = 'How many units per pixel (e.g. microns)?';

    Defaults{1} = '0,0';
    Defaults{2} = '10';

    Answers = inputdlg(Prompts(1:2),'Export Locations Settings',1,Defaults(1:2),'on');

    FirstObjectLocation = Answers{1};
    comma = strfind(FirstObjectLocation,',');
    commaerrortext = 'The x,y location must be entered in the following format: 0,0';
    if length(comma) > 1
        uiwait(CPwarndlg(commaerrortext));
        continue
    else
        FirstSpotX = str2double(FirstObjectLocation(1:comma-1));
        FirstSpotY = str2double(FirstObjectLocation(comma+1:end));
        if isempty(FirstSpotX) || isempty(FirstSpotY)
            uiwait(CPwarndlg(commaerrortext));
            continue
        end
    end

    PixelUnits = str2double(Answers{2});
    if isempty(PixelUnits)
        uiwait(CPwarndlg('Units per pixel must be a valid number.'));
        continue
    end

    AcceptableAnswers = 1;
end

for ImageNumber = 1:length(Locations)
    filename = [ObjectTypename,'_Locations_Image_',num2str(ImageNumber),'.csv'];
    fid = fopen(fullfile(handles.Current.DefaultOutputDirectory,filename),'w');
    if fid == -1
        CPerrordlg(sprintf('Cannot create the output file %s. There might be another program using a file with the same name.',filename));
        return
    end
    FixedLocations = Locations{ImageNumber}*PixelUnits;
    FixedLocations(:,1) = FixedLocations(:,1) - (FixedLocations(1,1)-FirstSpotX);
    FixedLocations(:,2) = FixedLocations(:,2) - (FixedLocations(1,2)-FirstSpotY);
    for ObjectNumber = 1:size(Locations{ImageNumber},1)
        fprintf(fid,[num2str(round(FixedLocations(ObjectNumber,1))),',',num2str(round(FixedLocations(ObjectNumber,2))),'\n']);
    end
    fclose(fid);
end