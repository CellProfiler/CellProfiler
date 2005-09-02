function handles = BatchOutputImportMySQL(handles)

% Help for the Batch Ouput Import to MySQL module:
% Category: File Processing
%
% This module creates SQL files from batch run output files.
%
% It does not make sense to run this module in conjunction with other
% modules.  It should be the only module in the pipeline.
%
% See also: CREATEBATCHSCRIPTS.

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

drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%pathnametextVAR01 = What is the path to the directory where the batch files were saved?
BatchPath = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What was the prefix of the batch files?
%defaultVAR02 = Batch_
BatchFilePrefix = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What is the name of the database to use?
%defaultVAR03 = Slide07
DatabaseName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%%%VariableRevisionNumber = 01
% The variables have changed for this module.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if strcmp(BatchPath, '.') == 1
    BatchPath = handles.Current.DefaultOutputDirectory;
end

%%% If this isn't the first cycle, we're probably running on the
%%% cluster, and should just continue.
if (handles.Current.SetBeingAnalyzed > 1),
    return;
end

%%% Load the data file
BatchData = load(fullfile(BatchPath,[BatchFilePrefix 'data.mat']));


%%% Create the SQL script
SQLMainFileName = sprintf('%s/%sdata.SQL', BatchPath, BatchFilePrefix);
SQLMainFile = fopen(SQLMainFileName, 'wt');
if SQLMainFile == -1,
    error(['Could not open ' SQLMainFileName ' for writing.']);
end
fprintf(SQLMainFile, 'USE %s;\n', DatabaseName);

% temp fix
BatchData.handles.Measurements.ImageThresholdNuclei = BatchData.handles.Measurements.GeneralInfo.ImageThresholdNuclei;
BatchData.handles.Measurements.ImageThresholdCells = BatchData.handles.Measurements.GeneralInfo.ImageThresholdCells;
BatchData.handles.Measurements = rmfield(BatchData.handles.Measurements, 'GeneralInfo');

%%% get the list of measurements
Fieldnames = fieldnames(BatchData.handles.Measurements);

%%% create tables
ImageFieldNames = Fieldnames(strncmp(Fieldnames, 'Image', 5));
fprintf(SQLMainFile, 'DROP TABLE IF EXISTS spots;\n');
fprintf(SQLMainFile, 'CREATE TABLE spots (spotnumber INTEGER PRIMARY KEY');
for i = 1:length(ImageFieldNames),
    fprintf(SQLMainFile, ', %s FLOAT', char(ImageFieldNames{i}));
end
OtherFieldNames = Fieldnames((~ strncmp(Fieldnames, 'Image', 5)) & (~ strncmp(Fieldnames, 'Object', 6)) & (~ strncmp(Fieldnames, 'Pathname', 8)));
for i = 1:length(OtherFieldNames),
    fprintf(SQLMainFile, ', %s CHAR(50)', char(OtherFieldNames{i}));
end
% Should also handle imported headings
% HeadingFieldNames = BatchData.handles.Measurements.headings
fprintf(SQLMainFile, ');\n');

ObjectFieldNames = Fieldnames(strncmp(Fieldnames, 'Object', 6));
fprintf(SQLMainFile, 'DROP TABLE IF EXISTS cells;\n');
fprintf(SQLMainFile, 'CREATE TABLE cells (spotnumber integer, cellnumber integer');
for i = 1:length(ObjectFieldNames),
    fprintf(SQLMainFile, ', %s FLOAT', char(ObjectFieldNames{i}));
end
fprintf(SQLMainFile, ', PRIMARY KEY (spotnumber, cellnumber));\n');

%%% write a data file for the first spot
SQLSubFileName = sprintf('%s/%s1_to_1.Image.SQL', BatchPath, BatchFilePrefix);
SQLSubFile = fopen(SQLSubFileName, 'wt');
if SQLSubFile == -1,
    fclose(SQLMainFile);
    error(['Could not open ' SQLSubFileName ' for writing.']);
end
fprintf(SQLSubFile, '1');
for i = 1:length(ImageFieldNames),
    fprintf(SQLSubFile, '|%d', BatchData.handles.Measurements.(char(ImageFieldNames{i})){1});
end
for i = 1:length(OtherFieldNames),
    fprintf(SQLSubFile, '|%s', BatchData.handles.Measurements.(char(OtherFieldNames{i})){1});
end
fprintf(SQLSubFile, '\n');
fclose(SQLSubFile);
SQLSubFileName = sprintf('%s1_to_1.Image.SQL', BatchFilePrefix);
fprintf(SQLMainFile, 'LOAD DATA LOCAL INFILE ''%s'' REPLACE INTO TABLE spots FIELDS TERMINATED BY ''|'';\n', SQLSubFileName);

if (length(ObjectFieldNames) > 0),
    SQLSubFileName = sprintf('%s/%s1_to_1.Object.SQL', BatchPath, BatchFilePrefix);
    SQLSubFile = fopen(SQLSubFileName, 'wt');
    if SQLSubFile == -1,
        fclose(SQLMainFile);
        error(['Could not open ' SQLSubFileName ' for writing.']);
    end
    for cellcount = 1:length(BatchData.handles.Measurements.(char(ObjectFieldNames{1})){1}),
        fprintf(SQLSubFile, '1|%d', cellcount);
        for i = 1:length(ObjectFieldNames),
            msr = BatchData.handles.Measurements.(char(ObjectFieldNames{i})){1};
            fprintf(SQLSubFile, '|%d', msr(cellcount));
        end
        fprintf(SQLSubFile, '\n');
    end
    fclose(SQLSubFile);
    SQLSubFileName = sprintf('%s1_to_1.Object.SQL', BatchFilePrefix);
    fprintf(SQLMainFile, 'LOAD DATA LOCAL INFILE ''%s'' REPLACE INTO TABLE cells FIELDS TERMINATED BY ''|'';\n', SQLSubFileName);
end


%%% Write files for the other batches
FileList = dir(BatchPath);
Matches = ~ cellfun('isempty', regexp({FileList.name}, ['^' BatchFilePrefix '[0-9]+_to_[0-9]+_OUT.mat$']));
FileList = FileList(Matches);

WaitbarHandle = waitbar(0,'Writing SQL files');
for filenum = 1:length(FileList),
    SubsetData = load(fullfile(BatchPath,FileList(filenum).name));

    SubsetData.handles.Measurements.ImageThresholdNuclei = SubsetData.handles.Measurements.GeneralInfo.ImageThresholdNuclei;
    SubsetData.handles.Measurements.ImageThresholdCells = SubsetData.handles.Measurements.GeneralInfo.ImageThresholdCells;
    SubsetData.handles.Measurements = rmfield(SubsetData.handles.Measurements, 'GeneralInfo');

    if (isfield(SubsetData.handles, 'BatchError')),
        fclose(SQLMainFile);
        error(['Error writing SQL data from batch file output.  File ' BatchPath '/' FileList(i).name ' encountered an error during batch processing.  The error was ' SubsetData.handles.BatchError '.  Please re-run that batch file.']);
    end

    matches = regexp(FileList(filenum).name, '[0-9]+', 'match');
    lo = str2num(matches{end-1});
    hi = str2num(matches{end});


    SubSetMeasurements = SubsetData.handles.Measurements;
    SQLSubFileName = sprintf('%s/%s%d_to_%d.Image.SQL', BatchPath, BatchFilePrefix, lo, hi);
    SQLSubFile = fopen(SQLSubFileName, 'wt');
    if SQLSubFile == -1,
        fclose(SQLMainFile);
        error(['Could not open ' SQLSubFileName ' for writing.']);
    end

    for spotnum = lo:hi,
        %%% write a data file for the spotnum-th spot
        fprintf(SQLSubFile, '%d', spotnum);
        for i = 1:length(ImageFieldNames),
            if (length(SubSetMeasurements.(char(ImageFieldNames{i}))) >= spotnum),
                fprintf(SQLSubFile, '|%d', SubSetMeasurements.(char(ImageFieldNames{i})){spotnum});
            else
                fprintf(SQLSubFile, '|');
            end
        end
        for i = 1:length(OtherFieldNames),
            if (length(SubSetMeasurements.(char(OtherFieldNames{i}))) >= spotnum),
                fprintf(SQLSubFile, '|%s', SubSetMeasurements.(char(OtherFieldNames{i})){spotnum});
            else
                fprintf(SQLSubFile, '|');
            end
        end
        fprintf(SQLSubFile, '\n');
    end
    fclose(SQLSubFile);
    SQLSubFileName = sprintf('%s%d_to_%d.Image.SQL', BatchFilePrefix, lo, hi);
    fprintf(SQLMainFile, 'LOAD DATA LOCAL INFILE ''%s'' REPLACE INTO TABLE spots FIELDS TERMINATED BY ''|'';\n', SQLSubFileName);


    if (length(ObjectFieldNames) > 0),
        SQLSubFileName = sprintf('%s/%s%d_to_%d.Object.SQL', BatchPath, BatchFilePrefix, lo, hi);
        SQLSubFile = fopen(SQLSubFileName, 'wt');
        if SQLSubFile == -1,
            fclose(SQLMainFile);
            error(['Could not open ' SQLSubFileName ' for writing.']);
        end
        for spotnum = lo:hi,
            if (length(SubSetMeasurements.(char(ObjectFieldNames{1}))) >= spotnum),
                for cellcount = 1:length(SubSetMeasurements.(char(ObjectFieldNames{1})){spotnum}),
                    fprintf(SQLSubFile, '%d|%d', spotnum, cellcount);
                    for i = 1:length(ObjectFieldNames),
                        fprintf(SQLSubFile, '|%d', SubSetMeasurements.(char(ObjectFieldNames{i})){spotnum}(cellcount));
                    end
                    fprintf(SQLSubFile, '\n');
                end
            end
        end
        fclose(SQLSubFile);
        SQLSubFileName = sprintf('%s%d_to_%d.Object.SQL', BatchFilePrefix, lo, hi);
        fprintf(SQLMainFile, 'LOAD DATA LOCAL INFILE ''%s'' REPLACE INTO TABLE cells FIELDS TERMINATED BY ''|'';\n', SQLSubFileName);
    end

    waitbar(filenum/length(FileList), WaitbarHandle);
end

fclose(SQLMainFile);
close(WaitbarHandle);

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%

%%% The figure window display is unnecessary for this module, so the figure
%%% window is closed if it was previously open.
%%% Determines the figure number.
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
%%% If the window is open, it is closed.
if any(findobj == ThisModuleFigureNumber) == 1;
    delete(ThisModuleFigureNumber)
end