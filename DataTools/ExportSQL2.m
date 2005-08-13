function ExportSQL2(handles)

% Help for the ExportSQl tool:
% Category: Data Tools
%
% This tool exports measurements from one or several CellProfiler
% output files to delimited text files. It also creates an SQL
% script that puts the measurements into an SQL database. This function
% is also called from the WriteSQLFiles module.
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
%   extra feature field in handles.Measurements.Image
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

%%% Determine if this function is called from the WriteSQLFiles module
%%% In this case the required input data can be found in the handles structure
if isfield(handles,'DatabaseName')
    ModuleCall = 1;
    DatabaseName = handles.DatabaseName;
    DataPath = handles.DataPath;
    CellProfilerDataFileNames = {get(handles.OutputFileNameEditBox,'string')};

    %%% Generate SQL by removing the 'OUT' and appending '_SQLScript.SQL'.
    filename = get(handles.OutputFileNameEditBox,'string');
    index = strfind(filename,'OUT');
    if ~isempty(index)
        filename = [filename(1:index-1),filename(index+3:end)];
    end
    SQLScriptFileName = fullfile(DataPath,[filename(1:end-4),'_SQLScript.SQL']);
else
    % This function is not called from WriteSQLFiles but from the Data tool menu
    ModuleCall = 0;

    %%% Let the user select one output file to indicate the directory and how the
    %%% filename is constructed
    if exist(handles.Current.DefaultOutputDirectory, 'dir')
        [ExampleFile, DataPath] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'.','*.mat'),'Select one CellProfiler output file');
    else
        [ExampleFile, DataPath] = uigetfile('*.mat','Select one CellProfiler output file');
    end
    if ~DataPath;return;end


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
    if ~ok,return,end
    CellProfilerDataFileNames = Files(selection);

    %%% Ask for database name and name of SQL script
    answer = inputdlg({'Database to use:','SQL script name:'},'Export SQL',1,{'Default','SQLScript.SQL'});
    if isempty(answer),return;end
    DatabaseName = answer{1};
    SQLScriptFileName = fullfile(DataPath,answer{2});
    if isempty(DatabaseName) | isempty(SQLScriptFileName)
        errordlg('A database name and an SQL script name must be specified!');
        return
    end
end

SQLScriptFid = fopen(SQLScriptFileName, 'wt');
if SQLScriptFid == -1, error(['Could not open ' SQLScriptFileName ' for writing.']); end
fprintf(SQLScriptFid, 'USE %s;\n', DatabaseName);

GlobalImageSetNo = 0;                                                                % This variable is used for numbering the image sets when several input files are exported
waitbarhandle = waitbar(0,'Exporting SQL files');
drawnow         

for FileNo = 1:length(CellProfilerDataFileNames)

    % If called as a Data tool, load handles structure from file
    if ~ModuleCall,load(fullfile(DataPath,CellProfilerDataFileNames{FileNo}));end

    % Get the object types, e.g. 'Image', 'Cells', 'Nuclei',...
    try
        ObjectTypes = fieldnames(handles.Measurements);
    catch
        errordlg('The output file does not contain a field called Measurements');
        return;
    end

    for ObjectTypeNo = 1:length(ObjectTypes)                                          % Loop over the objects

        % Get the object type in a variable for convenience
        ObjectType = ObjectTypes{ObjectTypeNo};

        % Update waitbar
        done = (FileNo - 1)*length(ObjectTypes) + ObjectTypeNo;
        total = length(CellProfilerDataFileNames)*length(ObjectTypes);
        waitbar(done/total,waitbarhandle);drawnow


        %%% For each image set, construct a big matrix of size [NbrOfObjects x Total number of features]
        %%% and corresponding cell array of size [1 x Total number of features] with the measurement names.
        MeasurementNames = {};
        Measurements = {};
        AllFields = fieldnames(handles.Measurements.(ObjectType));     % Get all existing fields for this object
        for j = 1:length(AllFields)
            if ~isempty(strfind(AllFields{j},'Features'))              % Found a field with feature names

                % Get the corresponding measurements
                CellArray = handles.Measurements.(ObjectType).(AllFields{j}(1:end-8));

                % Concatenate measurements
                if length(Measurements) == 0
                    Measurements = CellArray;                             % The first measurement structure encounterered
                else
                    for k = 1:length(CellArray)                           % Loop over the image sets and concatenate
                        Measurements{k} = cat(2,Measurements{k},CellArray{k});
                    end
                end

                % Fix and concatenate measurement names
                FeatureNames = handles.Measurements.(ObjectType).(AllFields{j});      % Cell array with measurement names
                Prefix = sprintf('%s_%s_',ObjectType,AllFields{j}(1:end-8));          % E.g. Cells_Shape_, Nuclei_Texture_OrigGreen,..
                for k = 1:length(FeatureNames)                                            % Loop over the measurement names...
                    FeatureName = FeatureNames{k};                                        %
                    FeatureNames{k} = [Prefix FeatureName(~isspace(FeatureName))];        % ... to add prefix and remove spaces in the names
                end
                MeasurementNames = cat(2,MeasurementNames,FeatureNames);                  % Add the measurement names to the cell array
            end
        end

        %%% Get the fields where the filenames are stored, these must be written to the SQL database in order
        %%% to identify the image sets
        FilenameFields = fieldnames(handles.Measurements.Image);
        FilenameFields = FilenameFields(~cellfun('isempty',strfind(FilenameFields,'Filename')));

        %%% Generate SQL data filenames by removing a potential 'OUT' and appending '_SQLData.SQL'.
        filename = CellProfilerDataFileNames{FileNo};
        index = strfind(filename,'OUT');
        if ~isempty(index)
            filename = [filename(1:index-1),filename(index+3:end)];
        end
        SQLDataFileName = [filename(1:end-4),'_',ObjectTypes{ObjectTypeNo},'_SQLData.SQL'];
        SQLMeanDataFileName = [filename(1:end-4),'_',ObjectTypes{ObjectTypeNo},'_SQLMeanData.SQL'];

        % Write lines in the SQL script for mean data (i.e. average object data)
        % Add ImageSetNo and ObjectNo entries in the table
        % Write lines for object data
        fprintf(SQLScriptFid, 'CREATE TABLE IF NOT EXISTS %s (ImageSetNo INTEGER, ObjectNo INTEGER',ObjectType);
        for k = 1:length(MeasurementNames),
            fprintf(SQLScriptFid, ', %s FLOAT', MeasurementNames{k});
        end
        fprintf(SQLScriptFid, ', PRIMARY KEY (ImageSetNo, ObjectNo));\n');
        fprintf(SQLScriptFid, 'LOAD DATA LOCAL INFILE ''%s'' REPLACE INTO TABLE %s FIELDS TERMINATED BY ''|'';\n', fullfile(DataPath,SQLMeanDataFileName),ObjectType);

        % Write lines for average data
        fprintf(SQLScriptFid, 'CREATE TABLE IF NOT EXISTS Mean%s (ImageSetNo INTEGER PRIMARY KEY',ObjectType);
        for k = 1:length(MeasurementNames),
            fprintf(SQLScriptFid, ', %s FLOAT', MeasurementNames{k});
        end
        for k = 1:length(FilenameFields),
            fprintf(SQLScriptFid, ', %s CHAR(50)', FilenameFields{k});                                   % Write the filename fields names
        end
        fprintf(SQLScriptFid,');\n');
        fprintf(SQLScriptFid, 'LOAD DATA LOCAL INFILE ''%s'' REPLACE INTO TABLE Mean%s FIELDS TERMINATED BY ''|'';\n', fullfile(DataPath,SQLDataFileName),ObjectType);

        %%% Open the SQL data files for writing
        SQLDataFid = fopen(fullfile(DataPath,SQLDataFileName), 'wt');
        if SQLDataFid == -1, fclose(SQLScriptFid); error(['Could not open ' SQLDataFileName ' for writing.']);end
        SQLMeanDataFid = fopen(fullfile(DataPath,SQLMeanDataFileName), 'wt');
        if SQLMeanDataFid == -1, fclose(SQLScriptFid); error(['Could not open ' SQLMeanDataFileName ' for writing.']);end


        %%% Loop over the image sets and write the data into the SQL data files
        for ImageSetNo = 1:length(Measurements)

            % Calculate mean data
            AverageData = mean(Measurements{ImageSetNo},1);

            % Create a cell array where every other cell contains
            % the delimiter '|'
            str = cell(2*length(AverageData),1);
            str(1:2:end) = {'|'};

            % Write the object data
            for ObjectNo = 1:size(Measurements{ImageSetNo},1)
                str(2:2:end) = cellstr(num2str(Measurements{ImageSetNo}(ObjectNo,:)','%g'));  % Create cell array with measurements
                fprintf(SQLDataFid,'%d|%d',GlobalImageSetNo+ImageSetNo,ObjectNo);        % Write the image set number and object number
                fprintf(SQLDataFid,sprintf('%s\n',cat(2,str{:})));            % Write measurements
            end

            % Write the average data
            str(2:2:end) = cellstr(num2str(AverageData','%g'));
            fprintf(SQLMeanDataFid,'%d',GlobalImageSetNo+ImageSetNo);                    % Write the image set number
            fprintf(SQLMeanDataFid,sprintf('%s',cat(2,str{:})));              % Write mean measurements
            for k = 1:length(FilenameFields)                                  % Write image file names
                fprintf(SQLMeanDataFid,'|%s',handles.Measurements.Image.(FilenameFields{k}){ImageSetNo});
            end
            fprintf(SQLMeanDataFid,'\n');

        end % End loop over image sets
        fclose(SQLDataFid);
        fclose(SQLMeanDataFid);
    end % End loop over object type, e.g. 'Cells', 'Nuclei', 'Image',...
    GlobalImageSetNo = GlobalImageSetNo + length(Measurements);
end % End loop over data files

%%% Done, let the user know if this function was called as a data tool and restore the handles structure
close(waitbarhandle);
if ~ModuleCall
    CPmsgbox('Exporting is completed.')
end
fclose(SQLScriptFid);

