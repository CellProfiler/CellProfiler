function handles = ExportSQL(handles)

% Help for the ExportSQl tool:
% Category: Data Tools
%
% This tool exports measurements from one or several CellProfiler
% output files to delimited text files. It also creates an SQL
% script that puts the measurements into an SQL database.
%
% Current known limitations and things to consider:
%
% - No check is performed that the selected files actually are CellProfiler
%   output files. .
%
% - No check is performed that the selected files are compatible, i.e.
%   were produced with the same pipeline of modules.
%
% - The tool only works with standard CellProfiler output files, not
%   batch output files.
%
% - Image sets are numbered according to the order they are written by
%   this tool. This numbering may not be consistent with the order they
%   were processed, e.g. on the cluster. This can be fixed by adding an
%   extra field in for example handles.Measurements.GeneralInfo
%
% - There are dots '.' in some of the measurement names described in the
%   SQL script. Is this a problem?
%
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


%%% Clear the handles structure that is sent to this function.
oldhandles = handles;
clear handles


%%% ---------------------------------------------------------------------- %%%
%%%                     Initial file handling                              %%%
%%% ---------------------------------------------------------------------- %%%
%%% Let the user select one output file to indicate the directory
[ExampleFile, DataPath] = uigetfile('*.mat','Select one CellProfiler output file');
if ~DataPath;handles = oldhandles;return;end                                     % CellProfiler expects a handles structure as output

%%% Get all files with .mat extension in the chosen directory.
%%% If the selected file name contains an 'OUT', it is assumed
%%% that all interesting files contain an 'OUT'.
AllFiles = dir(DataPath);                                                        % Get all file names in the chosen directory
AllFiles = {AllFiles.name};
Files = AllFiles(~cellfun('isempty',strfind(AllFiles,'.mat')));                  % Keep files that has and .mat extension
if strfind(ExampleFile,'OUT')
    Files = Files(~cellfun('isempty',strfind(Files,'OUT')));                     % Keep files with an 'OUT' in the name
end

%%% Let the user select the files to be exported
[selection,ok] = listdlg('liststring',Files,'name','Export SQL',...
    'PromptString','Select files to export. Use Ctrl+Click or Shift+Click.','listsize',[300 500]);
if ~ok,handles = oldhandles;return,end
CellProfilerDataFileNames = Files(selection);

%%% Ask for database name and name of SQL script
answer = inputdlg({'Database to use:','SQL script name:'},'Export SQL',1,{'','SQLScript.SQL'});
if isempty(answer),handles = oldhandles;return;end
DatabaseName = answer{1};
SQLScriptFileName = fullfile(DataPath,answer{2});
if isempty(DatabaseName) | isempty(SQLScriptFileName)
    errordlg('A database name and an SQL script name must be specified!');
    handles = oldhandles;
    return
end

%%% ------------------------------------------------------------------------------ %%%
%%% Get list of measurement and file names (from the first CellProfiler data file) %%%
%%% ------------------------------------------------------------------------------ %%%
load(fullfile(DataPath,CellProfilerDataFileNames{1}));
try% Load handles structure from the first data file
    FilenameFields = fieldnames(handles.Measurements.GeneralInfo);                % Get the fields where the filenames are stored
catch
    errordlg('The output file does not contain a field called Measurements.');
    return;
end

FilenameFields = FilenameFields(~cellfun('isempty',strfind(FilenameFields,'Filename')));            %                       .

handles.Measurements = rmfield(handles.Measurements,'GeneralInfo');               % Remove the GeneralInfo field, only interested in the measurements
MeasurementNames = {};                                                            % Initialize cell array with measurement names
NbrOfDataBlocks = 0;
ObjectNames = fieldnames(handles.Measurements);                                   % Get object names, e.g. Nuclei, Cells, Cytoplasm,...
for i = 1:length(ObjectNames)                                                     % Loop over the objects
    AllFields = fieldnames(handles.Measurements.(ObjectNames{i}));                % Get all existing fields for this object
    index = ~cellfun('isempty',strfind(AllFields,'Features'));                    % The fields with 'Features' in the name contain the measurement names
    FeatureFields = AllFields(index);                                             % Get them ...
    for j = 1:length(FeatureFields)                                               % Loop over these fields to add measurements to the cell arrays
        NbrOfDataBlocks = NbrOfDataBlocks + 1;
        FeatureNames = handles.Measurements.(ObjectNames{i}).(FeatureFields{j});  % Cell array with measurement names
        Prefix = sprintf('%s_%s_',ObjectNames{i},FeatureFields{j}(1:end-8));      % E.g. Cells_Shape_, Nuclei_Texture_OrigGreen,..
        for k = 1:length(FeatureNames)                                            % Loop over the measurement names...
            FeatureName = FeatureNames{k};                                        %
            FeatureNames{k} = [Prefix FeatureName(~isspace(FeatureName))];        % ... to add prefix and remove spaces in the names
        end
        MeasurementNames = cat(2,MeasurementNames,FeatureNames);                  % Add the measurement names to the cell array
    end
end

%%% ------------------------------------------------------------------------------------ %%%
%%% Generate SQL data filenames by removing a potential 'OUT' and appending '_SQLData.SQL'.
%%% ------------------------------------------------------------------------------------ %%%
SQLDataFileNames = cell(length(CellProfilerDataFileNames),1);
SQLMeanDataFileNames = cell(length(CellProfilerDataFileNames),1);
for k = 1:length(CellProfilerDataFileNames)
    filename = CellProfilerDataFileNames{k};
    index = strfind(filename,'OUT');
    if ~isempty(index)
        filename = [filename(1:index-1),filename(index+3:end)];
    end
    SQLDataFileNames{k} = [filename(1:end-4),'_SQLData.SQL'];
    SQLMeanDataFileNames{k} = [filename(1:end-4),'_SQLMeanData.SQL'];
end


%%% ------------------------------------------------ %%%
%%%            Create the SQL script                 %%%
%%% ------------------------------------------------ %%%
% Open the file for writing
SQLScriptFid = fopen(SQLScriptFileName, 'wt');
if SQLScriptFid == -1, error(['Could not open ' SQLScriptFileName ' for writing.']); end
fprintf(SQLScriptFid, 'USE %s;\n', DatabaseName);

% Write script-lines for mean data (i.e. average object data)
fprintf(SQLScriptFid, 'DROP TABLE IF EXISTS MeanData;\n');                                       % Create a table for mean data
fprintf(SQLScriptFid, 'CREATE TABLE MeanData (ImageSetNo INTEGER PRIMARY KEY');                  % Add ImageSetNo data entry in the table
for i = 1:length(MeasurementNames),
    fprintf(SQLScriptFid, ', %s FLOAT', MeasurementNames{i});                                    % Write the measurement names
end
for i = 1:length(FilenameFields),
    fprintf(SQLScriptFid, ', %s CHAR(50)', FilenameFields{i});                                   % Write the filename fields names
end
fprintf(SQLScriptFid,');\n');
for i = 1:length(SQLDataFileNames)
    fprintf(SQLScriptFid, 'LOAD DATA LOCAL INFILE ''%s'' REPLACE INTO TABLE MeanData FIELDS TERMINATED BY ''|'';\n', fullfile(DataPath,SQLDataFileNames{i}));
end
fprintf('\n');

% Write script-lines for object data
fprintf(SQLScriptFid, 'DROP TABLE IF EXISTS ObjectData;\n');                                     % Create table for object data
fprintf(SQLScriptFid, 'CREATE TABLE ObjectData (ImageSetNo INTEGER, ObjectNo INTEGER');          % Add ImageSetNo and ObjectNo entries in the table
for i = 1:length(MeasurementNames),
    fprintf(SQLScriptFid, ', %s FLOAT', MeasurementNames{i});                                    % Write the measurement names
end
fprintf(SQLScriptFid, ', PRIMARY KEY (ImageSetNo, ObjectNo));\n');
for i = 1:length(SQLMeanDataFileNames)
    fprintf(SQLScriptFid, 'LOAD DATA LOCAL INFILE ''%s'' REPLACE INTO TABLE ObjectData FIELDS TERMINATED BY ''|'';\n', fullfile(DataPath,SQLMeanDataFileNames{i}));
end
fclose(SQLScriptFid);

%%% ---------------------------------------------------------------- %%%
%%% Create the data files. One SQL file will be written for          %%%
%%% each CellProfiler output file.                                   %%%
%%% ---------------------------------------------------------------- %%%
waitbarhandle = waitbar(0,'Exporting SQL files');drawnow
GlobalImageSetNo = 0;                              % This variable keeps track of the total number of image sets over all files
for fileno = 1:length(CellProfilerDataFileNames)

    %%% Open SQL files for writing
    SQLDataFid = fopen(fullfile(DataPath,SQLDataFileNames{fileno}), 'wt');
    if SQLDataFid == -1, fclose(SQLScriptFid); error(['Could not open ' SQLDataFileNames{fileno} ' for writing.']);end
    SQLMeanDataFid = fopen(fullfile(DataPath,SQLMeanDataFileNames{fileno}), 'wt');
    if SQLMeanDataFid == -1, fclose(SQLScriptFid); error(['Could not open ' SQLMeanDataFileNames{fileno} ' for writing.']);end

    %%% Load CellProfiler output file
    load(fullfile(DataPath,CellProfilerDataFileNames{fileno}));

    %%% Get number of processed image sets (this is a bit clumsy)
    fields = fieldnames(handles.Measurements.GeneralInfo);
    index = find(~cellfun('isempty',strfind(fields,'Threshold')));
    NbrOfImageSets = length(handles.Measurements.GeneralInfo.(fields{index(1)}));

    %%% Loop over the image sets and write the data into the SQL data files
    for ImageSetNo = 1:NbrOfImageSets

        %%% Update the total number of image sets than have been exported
        %%% ImageSetNo is the image set number in the current CellProfiler
        %%% output file.
        GlobalImageSetNo = GlobalImageSetNo + 1;

        %%% Get the measurements. It's important that this is done in the
        %%% same way as for the measurement names above to get the
        %%% measurements in the same order. The Measurements variable will
        %%% be a cell array with measurement blocks.
        MaxNbrOfObjects = 0;                                                              % Variable to keep track of max number of measurements
        BlockNo = 1;                                                                      % Block counter
        Measurements = cell(1,NbrOfDataBlocks);                                           % Initialize cell array with measurements
        ObjectNames = fieldnames(handles.Measurements);                                   % Get object names, e.g. Nuclei, Cells, Cytoplasm,...
        ObjectNames = ObjectNames(cellfun('isempty',strfind(ObjectNames,'GeneralInfo'))); % Remove GeneralInfo field
        for i = 1:length(ObjectNames)                                                     % Loop over the objects
            AllFields = fieldnames(handles.Measurements.(ObjectNames{i}));                % Get all existing fields for this object
            index = ~cellfun('isempty',strfind(AllFields,'Features'));                    % The fields with 'Features' in the name contain the measurement names
            FeatureFields = AllFields(index);                                             % Get them ...
            for j = 1:length(FeatureFields)
                Measurements(BlockNo) = handles.Measurements.(ObjectNames{i}).(FeatureFields{j}(1:end-8))(ImageSetNo);
                if size(Measurements{BlockNo},1) > MaxNbrOfObjects              % Check for max number of measurements
                    MaxNbrOfObjects = size(Measurements{BlockNo},1);
                end
                BlockNo = BlockNo + 1;                                          % Update BlockNo
            end
        end

        %%% Write mean measurements to the SQL file. The first piece of code
        %%% makes a cell array with one mean measurement in each cell. This
        %%% cell array is then interleaved with delimiters '|', and finally converted
        %%% to a string and written as one row in the MeanDataSQL file.
        tmp = {};
        for BlockNo = 1:NbrOfDataBlocks
            tmp = cat(2,tmp,cellstr(num2str(mean(Measurements{BlockNo},1)','%g'))');
        end
        str = cell(2*length(MeasurementNames),1);
        str(1:2:end) = {'|'};                                             % Interleave with delimiters
        str(2:2:end) = tmp;
        fprintf(SQLMeanDataFid,'%d',GlobalImageSetNo);                    % Write the image set number
        fprintf(SQLMeanDataFid,sprintf('%s',cat(2,str{:})));              % Write mean measurements
        for i = 1:length(FilenameFields)                                  % Write image file names
            fprintf(SQLMeanDataFid,'|%s',handles.Measurements.GeneralInfo.(FilenameFields{i}){ImageSetNo});
        end
        fprintf(SQLMeanDataFid,'\n');

        %%% Write object measurements to SQL file. The procedure is similar to the one
        %%% above, but now we have to write one row per object. Also, it might for example happen
        %%% that there is a different number of Nuclei objects and Cell objects. In such case blanks
        %%% must be filled in.
        for ObjectNo = 1:MaxNbrOfObjects
            tmp = cell(1,length(MeasurementNames));
            index = 1;
            for BlockNo = 1:NbrOfDataBlocks
                BlockSize = size(Measurements{BlockNo},2);
                if size(Measurements{BlockNo},1) < ObjectNo
                    tmp(index:index+BlockSize-1) = cell(1,BlockSize);               % Fill in with blanks
                else
                    tmp(index:index+BlockSize-1) = cellstr(num2str(Measurements{BlockNo}(ObjectNo,:)','%g'));  % Create cell array with measurements
                end
                index = index + BlockSize;
            end
            str = cell(2*length(MeasurementNames),1);                     % Interleave with delimiters
            str(1:2:end) = {'|'};                                         %             .
            str(2:2:end) = tmp;                                           %             .
            fprintf(SQLDataFid,'%d|%d',GlobalImageSetNo,ObjectNo);        % Write the image set number and object number
            fprintf(SQLDataFid,sprintf('%s\n',cat(2,str{:})));            % Write measurements
        end
    end
    fclose(SQLMeanDataFid);
    fclose(SQLDataFid);
    waitbar(fileno/length(CellProfilerDataFileNames),waitbarhandle);drawnow
end

%%% Done!!
close(waitbarhandle)
CPmsgbox('Exporting is completed.')
