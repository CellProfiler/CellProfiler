function handles=FileNameMetadata(handles, varargin)

% Help for the File Name Metadata module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Captures metadata such as plate name, well column and well row from
% the filename of an image file.
% *************************************************************************
%
% This module uses regular expressions to capture metadata such as
% plate name, and well position from an image's file name. The captured
% metadata is stored in the image's measurements under the "Metadata"
% category.
%
% Variables:
% What did you call the image?
%     This is the name entered in LoadImages. This module will use the
%     file name of this image as the source of its metadata.
%
% Enter the regular expression to use to capture the fields:
%     The regular expression syntax can be used to name different parts
%     of your expression. The syntax for this is (?<FIELDNAME>expr) to
%     extract whatever matches "expr" and assign it to the measurement,
%     FIELDNAME for the image.
%     For instance, a researcher uses plate names composed of two
%     capital letters followed by five numbers, then appends the
%     well name to this, separated  by an underbar: "TE12345_A05.tif"
%     The following regular expression will capture the plate, well
%     row and well column in the fields, "Plate","WellRow" and "WellCol":
%         ^(?<Plate>[A-Z]{2}[0-9]{5})_(?<WellRow>[A-H])(?<WellCol>[0-9]+)
%         1    2        3      4     5     6       7        8        9
%  1. "^"           Only start at beginning of the file name
%  2. "(?<Plate>"   Name the captured field, "Plate"
%  3. "[A-Z]{2}     First, capture exactly two letters between A and Z
%  4. "[0-9]{5}     Also capture exactly five digits
%  5. "_"           Discard the underbar separating plate from well
%  6. "(?<WellRow>" Name the captured field, "WellRow"
%  7. "[A-H]"       Capture exactly one letter between A and H
%  8. "(?<WellCol>" Name the captured field, "WellCol"
%  9. "[0-9]+"      Capture as many digits as follow
%
% See also LoadImages module for regular expression format

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

% MBray 2009_03_20: Comments on variables for pyCP upgrade
%
% Recommended variable order (setting, followed by current variable in MATLAB CP)
% This module needs to be folded into LoadImages, so we have access to the 
% metadata from the very beginning (such as for image confirmation). In
% that case, the settings would be:
% (1) What is the regular expression to capture the fields in the image filename? (RegularExpressionFilename)
% (2) What is the regular expression to capture the fields in the image pathname? (RegularExpressionPathname)
%
% Both (1) and (2) need 'Do not use' defaults to allow user to use one or
% the other or both.

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%

drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = For the filename, enter the regular expression to use to capture the fields. Type "Do not use" to ignore.
%defaultVAR02 = ^(?<Plate>.+)_(?<WellRow>[A-P])(?<WellColumn>[0-9]{1,2})_(?<Site>[0-9])
RegularExpressionFilename = char(handles.Settings.VariableValues{CurrentModuleNum,2});

% Capture the field names inside the structure, (?<fieldname>
if ~strcmpi(RegularExpressionFilename,'Do not use')
    FileFieldNames = regexp(RegularExpressionFilename,'\(\?[<](?<token>.+?)[>]','tokens');
    FileFieldNames = [FileFieldNames{:}];
else
    FileFieldNames = [];
end

%textVAR03 = For the pathname, enter the regular expression to use to capture the fields. Separate each directory-specific field using vertical lines (i.e, | ). Type "Do not use" to ignore.
%defaultVAR03 = Do not use
RegularExpressionPathname = char(handles.Settings.VariableValues{CurrentModuleNum,3});

% Capture the field names inside the structure, (?<fieldname>
if ~strcmpi(RegularExpressionPathname,'Do not use')
    PathFieldNames = regexp(RegularExpressionPathname,'\(\?[<](?<token>.+?)[>]','tokens');
    PathFieldNames = [PathFieldNames{:}];
else
    PathFieldNames = [];
end

%%%%%%%%%%%%%%%%
%%% FEATURES %%%
%%%%%%%%%%%%%%%%

if nargin > 1 
    switch varargin{1}
%feature:categories
        case 'categories'
            if nargin == 2 && strcmp(varargin{2},'Image')
                result = { 'Metadata' };
            else
                result = {};
            end
%feature:measurements
        case 'measurements'
            result = {};
            if nargin >= 3 &&...
                strcmp(varargin{3},'Metadata') &&...
                ismember(varargin{2},'Image')
                result = FieldNames;
            end
        otherwise
            error(['Unhandled category: ',varargin{1}]);
    end
    handles=result;
    return;
end

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%
%%% ANALYSIS %%%
%%%%%%%%%%%%%%%%
FileNameField = ['FileName_',ImageName];

if ~isfield(handles.Measurements,'Image')
    error([ 'Image processing was canceled in the ', ModuleName, ' module. There are no image measurements.']);
end
if ~isfield(handles.Measurements.Image,FileNameField)
    error([ 'Image processing was canceled in the ', ModuleName, ' module. ',ImageName,' has no file name measurement (maybe you did not use LoadImages to create it?)']);
end

SetIndex = handles.Current.SetBeingAnalyzed;
FileOrPathName = [handles.Measurements.Image.(['PathName_',ImageName]){SetIndex} filesep handles.Measurements.Image.(FileNameField){SetIndex}];
[PathName,FileName] = fileparts(FileOrPathName);

Metadata = [];
if ~isempty(PathFieldNames)
    if strcmp(filesep,'\')
        RegularExpressionPathname = strrep(RegularExpressionPathname,'|',[filesep filesep]);
    else
        RegularExpressionPathname = strrep(RegularExpressionPathname,'|',filesep);
    end
    Metadata = cat(1,Metadata, regexp(PathName,RegularExpressionPathname,'names'));
end

if isempty(Metadata) && ~isempty(PathFieldNames)
    error([ 'Image processing was canceled in the ', ModuleName, ' module. The path "',PathName,'" doesn''t match the regular expression "',RegularExpressionPathname,'"']);
end

if ~isempty(FileFieldNames)
    s2 = regexp(FileName,RegularExpressionFilename,'names');
    f = fieldnames(s2);
    for i = 1:length(f)
        Metadata.(f{i}) = s2.(f{i});
    end
end

if isempty(Metadata) && ~isempty(FileFieldNames)
    error([ 'Image processing was canceled in the ', ModuleName, ' module. The file name, "',FileName,'" doesn''t match the regular expression "',RegularExpressionFilename,'"']);
end

FieldNames = [PathFieldNames,FileFieldNames];
if isfield(Metadata,'WellRow') && isfield(Metadata,'WellColumn');
    Metadata.Well = [Metadata.WellRow Metadata.WellColumn];
    FieldNames{length(FieldNames)+1} = 'Well';
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);

if any(findobj == ThisModuleFigureNumber);
    FontSize = handles.Preferences.FontSize;

    figure_position = get(ThisModuleFigureNumber,'Position');
    w = figure_position(3);
    h = figure_position(4);
    LineHeight = 20;
    FieldNameX = 10;
    FieldNameWidth = w/3;
    ValueX = FieldNameX*2+FieldNameWidth;
    ValueWidth = FieldNameWidth;
    for i = 1:length(FieldNames)
        h = h - LineHeight;
        uicontrol(ThisModuleFigureNumber,...
                  'style','text',...
                  'position', [FieldNameX,h,FieldNameWidth,LineHeight],...
                  'HorizontalAlignment','right',...
                  'fontname','Helvetica',...
                  'fontsize',FontSize,...
                  'fontweight','bold',...
                  'string',FieldNames{i});
        uicontrol(ThisModuleFigureNumber,...
                  'style','text',...
                  'position', [ValueX,h,ValueWidth,LineHeight],...
                  'HorizontalAlignment','left',...
                  'fontname','Helvetica',...
                  'fontsize',FontSize,...
                  'fontweight','normal',...
                  'string',Metadata.(FieldNames{i}));
    end
end
drawnow

%%%%%%%%%%%%%%%%%%%%
%%% MEASUREMENTS %%%
%%%%%%%%%%%%%%%%%%%%

for i = 1:length(FieldNames)
    value = Metadata.(FieldNames{i});
    if isempty(value), value = ''; end
    handles = CPaddmeasurements(handles, 'Image', ['Metadata_',FieldNames{i}],value);
end
