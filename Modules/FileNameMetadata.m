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
% When entering fields for the pathname, because slashs are platform-
% dependdent and are escape characters in regexp, use a vertical line ('|')
% to separate the direcrories, like this:
%   (?<rootdir>)|(?<subdir1>)|(?<subdir2>)....
% For instance, if an experimental run is given a unique directory name,
% the following expression will capture the directory name from the path:
%   .*|(?<Run>.*)$
% This captures the immediate directory containing the image file in the 
% token "Run", ignoring earlier directories in the path.
%
% If you want to group the images according to a set of tokens, enter the
% fields here, separated by commas. Type "Do not use" to ignore.
% If you want to group image files by a particular regexp token field,
% enter the fields (not the tokens) you want to group by
% here. For example, using the above examples, entering "Run, Plate" will
% create groups containing images that share the same Run and the same
% Plate fields. This is especially useful if you want to group all plates 
% together for an illumination correction calculation, rather than running
% the correction pipeline on each directory containing a plate separately.
%
% To use the grouping functionality, you must place this module immediately
% after any LoadImage modules and before any subsequent modules that might
% make use of tokens.
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
ImageName = CPreplacemetadata(handles,ImageName);
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

%textVAR04 = If you want to group the images according to a set of tokens, enter the fields here, separated by commas. Type "Do not use" to ignore.
%defaultVAR04 = Do not use
FieldsToGroupBy = char(handles.Settings.VariableValues{CurrentModuleNum,4});
if ~strcmpi(FieldsToGroupBy, 'Do not use'),
    FieldsToGroupBy = strtrim(strread(FieldsToGroupBy,'%s','delimiter',','));
else
    FieldsToGroupBy = [];
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

%%%VariableRevisionNumber = 4

%%%%%%%%%%%%%%%%
%%% ANALYSIS %%%
%%%%%%%%%%%%%%%%
if find(strcmp(handles.Settings.ModuleNames,'LoadImages'),1,'last') > find(strcmp(handles.Settings.ModuleNames,ModuleName),1,'first')
    error(['Image processing was canceled in the ', ModuleName,' module. ',ModuleName,' must be placed immediately after the last LoadImage module and before any subsequent modules that may make use of tokens.']);
end

FileNameField = ['FileName_',ImageName];

if ~isfield(handles.Measurements,'Image')
    error([ 'Image processing was canceled in the ', ModuleName, ' module. There are no image measurements.']);
end
if ~isfield(handles.Measurements.Image,FileNameField)
    error([ 'Image processing was canceled in the ', ModuleName, ' module. ',ImageName,' has no file name measurement (maybe you did not use LoadImages to create it?)']);
end

SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;
FileOrPathName = fullfile(handles.Measurements.Image.(['PathName_',ImageName]){SetBeingAnalyzed}, handles.Measurements.Image.(FileNameField){SetBeingAnalyzed});
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
% if Row and Column exist, create Well from them
if isfield(Metadata,'WellRow') && isfield(Metadata,'WellColumn');
    % If Column is a number, make sure it's 0-padded
    lpadcolnum = num2str(str2num(Metadata.WellColumn),'%02d');
    if isempty(lpadcolnum), lpadcolnum = Metadata.WellColumn; end
    Metadata.Well = [Metadata.WellRow lpadcolnum];
    FieldNames{length(FieldNames)+1} = 'Well';
    % Add 'Well' to available metadata list if needed later
    if any(strcmp(FileFieldNames,'WellRow')) || any(strcmp(FileFieldNames,'WellColumn'))
        FileFieldNames{length(FileFieldNames)+1} = 'Well';
    end
    if any(strcmp(PathFieldNames,'WellRow')) || any(strcmp(PathFieldNames,'WellColumn'))
        PathFieldNames{length(PathFieldNames)+1} = 'Well';
    end
end

% Place current token values in handles.Pipeline
for i = 1:length(FieldNames)
    if isempty(Metadata.(FieldNames{i}))
        value = ''; 
    else
        value = Metadata.(FieldNames{i});
    end
    handles.Pipeline.CurrentMetadata.(FieldNames{i}) = value;
end
    
% If groups are being defined, set up the handles.Pipeline structure
% appropriately. The updated structure has the following fields:
% GroupFileList: One for each group which contains;
%   SetBeingAnalyzed: The image being analyzed in the group
%   NumberOfImageSets: The total number of images in the group
% GroupFileListIDs: A vector the same length of handles.Pipeline.FileList*
%   where the index corresponds to the group number an image belongs to
% GroupIDs: 
% CurrentImageGroupID: Index of the current group being analyzed
% ImageGroupFields: The metadata used to group the images

if ~isempty(FieldsToGroupBy)
    if handles.Current.SetBeingAnalyzed == 1 
        % Find the strings corresponding to metadata fields
        % Some caution is needed here because during image confirmation, some
        % of the fields may be ''. So I need to check all the FileLists since
        % at least one of them will have all files represented
        % TODO: This is not true if only 1 channel is present, so I need to
        % correct that
        prefix = 'FileList';
        fn = fieldnames(handles.Pipeline);
        AllImageNames = strrep(fn(strncmp(fn,prefix,length(prefix))),prefix,'');
        [IndivPathnames,IndivFileNames] = deal(cell(1,length(AllImageNames)));
        for i = 1:length(AllImageNames)
            % Construct full path/filename so we can properly split it
            % apart
            f = handles.Pipeline.(['FileList',AllImageNames{i}]);
            p = repmat({[handles.Pipeline.(['Pathname',AllImageNames{i}]),filesep]},[1 length(f)]); % Append slash to take care of cases where file is empty
            [IndivPathnames{i},IndivFileNames{i}] = cellfun(@fileparts,cellfun(@fullfile,p,f,'UniformOutput',false),'UniformOutput',false);
            IndivPathnames{i}(cellfun(@isempty,f)) = {''};
        end

        % Assign the path metadata an ID number
        if ~isempty(PathFieldNames)
            s1 = cell(1,length(AllImageNames));
            for i = 1:length(AllImageNames)
                s1{i} = regexp(IndivPathnames{i},RegularExpressionPathname,'names','once');
            end
            s1 = cat(1,s1{:})';
            idx = reshape(~cellfun(@isempty,s1(:)),size(s1));
            s1 = cat(1,s1{:,find(all(idx,1),1,'first')});
            PathFieldsToGroupBy = FieldsToGroupBy(ismember(FieldsToGroupBy,PathFieldNames));
            path_idstr = cell(size(s1,1),length(PathFieldsToGroupBy));
            [path_idstr{:}] = deal('');
            PathID = zeros(size(s1,1),length(PathFieldsToGroupBy));
            for i = 1:length(PathFieldsToGroupBy),
                path_idstr(:,i) = cellstr(strvcat(s1(:).(PathFieldsToGroupBy{i})));
                [ignore,idx,PathID(:,i)] = group2index(path_idstr(:,i));
            end
        else
            path_idstr = cell(length(handles.Pipeline.([prefix,ImageName])),1);
            [path_idstr{:}] = deal('');
            PathID = [];
        end

        % Assign the file metadata an ID number
        if ~isempty(FileFieldNames)
            s2 = cell(1,length(AllImageNames));
            for i = 1:length(AllImageNames)
                s2{i} = regexp(IndivFileNames{i},RegularExpressionFilename,'names','once');
            end
            s2 = cat(1,s2{:})';
            idx = reshape(~cellfun(@isempty,s2(:)),size(s2));
            s2 = cat(1,s2{:,find(all(idx,1),1,'first')});
            FileFieldsToGroupBy = FieldsToGroupBy(ismember(FieldsToGroupBy,FileFieldNames));
            file_idstr = cell(size(s2,1),length(FileFieldsToGroupBy));
            [file_idstr{:}] = deal('');
            FileID = zeros(size(s2,1),length(FileFieldsToGroupBy));
            for i = 1:length(FileFieldsToGroupBy),
                file_idstr(:,i) = cellstr(strvcat(s2(:).(FileFieldsToGroupBy{i})));
                [ignore,idx,FileID(:,i)] = group2index(file_idstr(:,i));
            end
        else
            file_idstr = cell(length(handles.Pipeline.([prefix,ImageName])),1);
            [file_idstr{:}] = deal('');
            FileID = [];
        end

        % Determine the valid combinations of the path/file fields
        [ignore,idx] = unique([PathID FileID],'rows');
        PathFileIDs = [path_idstr(idx,:) file_idstr(idx,:)];

        % Pull the filelist into separate structures, one for each unique
        % combination
        handles.Pipeline.GroupFileListIDs = zeros(1,length(FileID));
        for i = 1:size(PathFileIDs,1)
            idx = all(ismember([path_idstr file_idstr],PathFileIDs(i,:)),2);
            for j = 1:length(AllImageNames),
                handles.Pipeline.GroupFileList{i}.(['FileList',AllImageNames{j}]) = handles.Pipeline.(['FileList',AllImageNames{j}])(idx);
                handles.Pipeline.GroupFileList{i}.(['Pathname',AllImageNames{j}]) = handles.Pipeline.(['Pathname',AllImageNames{j}]);
            end
            handles.Pipeline.GroupFileList{i}.Fields = PathFileIDs(i,~all(cellfun(@isempty,PathFileIDs),1));
            handles.Pipeline.GroupFileList{i}.SetBeingAnalyzed = 1;
            handles.Pipeline.GroupFileList{i}.NumberOfImageSets = length(handles.Pipeline.GroupFileList{i}.(['FileList',AllImageNames{1}]));
            handles.Pipeline.GroupFileListIDs(idx) = i;
        end
        PathFileIDs(:,all(cellfun(@isempty,PathFileIDs),1)) = [];
        handles.Pipeline.GroupIDs = PathFileIDs;

        % Since LoadImages and FileNameMetadata are separated, the initial
        % images are saved to handles.Pipeline before this module gets to group
        % them. So copy the images from LoadImages into the grouping structure
        % (CPaddimages will take of this for all other cycles)
        idxID = cell(1,length(FieldsToGroupBy));
        for i = 1:length(FieldsToGroupBy)
            idxID{i} = Metadata.(FieldsToGroupBy{i});
        end
        idx = find(all(ismember(handles.Pipeline.GroupIDs,idxID),2));
        for i = 1:length(AllImageNames)
            handles.Pipeline.GroupFileList{idx}.(AllImageNames{i}) = handles.Pipeline.(AllImageNames{i});
        end

        % Set a few last variables in handles
        handles.Pipeline.CurrentImageGroupID = idx;
        handles.Pipeline.ImageGroupFields = FieldsToGroupBy;
        handles.Current.NumberOfImageGroups = length(handles.Pipeline.GroupFileList);
        
        % Lastly, in case the groups are not contigiuous in the original
        % FileList, re-order the FileLists to make it so (required for
        % processing in CPCluster)
        [newIDlist,idx] = sort(handles.Pipeline.GroupFileListIDs);
        for i = 1:length(AllImageNames)
            handles.Pipeline.(['FileList',AllImageNames{i}]) = handles.Pipeline.(['FileList',AllImageNames{i}])(idx);
        end
        handles.Pipeline.GroupFileListIDs = newIDlist;
    else
        % If grouping fields have been created, set the current group
        % number (This will not be true until FileNameMetadata has
        % been processed once)
        idxID = cell(1,length(handles.Pipeline.ImageGroupFields));
        for i = 1:length(handles.Pipeline.ImageGroupFields)
            idxID{i} = handles.Pipeline.CurrentMetadata.(handles.Pipeline.ImageGroupFields{i});
        end
        newImageGroupID = find(all(ismember(handles.Pipeline.GroupIDs,idxID),2));

        % Determine the current image being used within the group. 
        % NB: An unfortunate side-effect separating of LoadImages and 
        % FileNameMetadata is that if the ImageGroupID changes, LoadImages has
        % already placed the images in the old ImageGroupID. So I need to
        % import the images here.
        if newImageGroupID ~= handles.Pipeline.CurrentImageGroupID,
            handles.Pipeline.GroupFileList{newImageGroupID}.SetBeingAnalyzed = 1;
            prefix = 'FileList';
            fn = fieldnames(handles.Pipeline);
            AllImageNames = strrep(fn(strncmp(fn,prefix,length(prefix))),prefix,'');
            for j = 1:length(AllImageNames),
                    handles.Pipeline.GroupFileList{newImageGroupID}.(AllImageNames{j}) = ...
                        handles.Pipeline.GroupFileList{handles.Pipeline.CurrentImageGroupID}.(AllImageNames{j});
            end
            handles.Pipeline.CurrentImageGroupID = newImageGroupID;
        else
            % I think I should be able to just incrememt SetBeingAnalyzed,
            % but to be safe I'm going to check the filenames
            idx = handles.Pipeline.CurrentImageGroupID;
            if ~isempty(handles.Pipeline.(['Filename',ImageName]){end})
                handles.Pipeline.GroupFileList{idx}.SetBeingAnalyzed = ...
                    find(ismember(handles.Pipeline.GroupFileList{idx}.(['FileList',ImageName]),handles.Pipeline.(['Filename',ImageName])(end)));
            else    % Unless the filename is empty. Then just increment
                handles.Pipeline.GroupFileList{idx}.SetBeingAnalyzed = handles.Pipeline.GroupFileList{idx}.SetBeingAnalyzed + 1;
            end
        end
    end

    % A final check, in case the filename is '': If image groups exist, fill in
    % as much as the metadata as possible from the group definition
    if isempty(handles.Pipeline.(['Filename',ImageName]){end})
        for i = 1:length(FieldNames)
            handles.Pipeline.CurrentMetadata.(FieldNames{i}) = '';  % To clear any erroneous metadata
        end
        idx = handles.Pipeline.CurrentImageGroupID;
        for i = 1:length(handles.Pipeline.ImageGroupFields)
            handles.Pipeline.CurrentMetadata.(handles.Pipeline.ImageGroupFields{i}) = handles.Pipeline.GroupFileList{idx}.Fields{i};
        end
    end
else
    handles.Current.NumberOfImageGroups = 1;
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
    if isempty(Metadata.(FieldNames{i}))
        value = ''; 
    else
        value = Metadata.(FieldNames{i});
    end
    handles = CPaddmeasurements(handles, 'Image', CPjoinstrings('Metadata',FieldNames{i}),value);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - GROUP2INDEX
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Create index vector from a grouping variable. Taken from GRP2IDX in the
% Stats toolbox
function [b,i,j] = group2index(s)
% Same as UNIQUE but orders result:
%    if iscell(s), preserve original order
%    otherwise use numeric order

s = s(:);
s = s(end:-1:1);
[b,i,j] = unique(s);     % b=unique group names
i = length(s) + 1 - i; % make sure this is the first instance
isort = i;  
if (~iscell(s))  
   if (any(isnan(b)))  % remove multiple NaNs; put one at the end
      nans = isnan(b);
      b = [b(~nans); NaN];
      x = find(isnan(s));
      i = [i(~nans); x(1)];
      j(isnan(s)) = length(b);
   end
   isort = b;          % sort based on numeric values
   if any(isnan(isort))
      isort(isnan(isort)) = max(isort) + 1;
   end
end

[is, f] = sort(isort); % sort according to the right criterion
b = b(f,:);

[fs, ff] = sort(f);    % rearrange j also
j = ff(j);
j = j(end:-1:1);
if (~iscell(b))        % make sure b is a cell array of strings
   b = cellstr(strjust(num2str(b), 'left'));
end