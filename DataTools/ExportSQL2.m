function ExportSQL2(handles)

% Help for the ExportSQl tool:
% Category: Data Tools
%
% This tool exports measurements from one or several CellProfiler
% output files to delimited text files. It also creates an SQL
% script that puts the measurements into an SQL database. This tool calls CPConvertSQL 
% function to do the actuall exporting, which is same as WriteSQL module.
%
% Current known limitations and things to consider:
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


    % This function is not called from WriteSQLFiles ModuleCall but from the Data tool menu
    ModuleCall = 0;

    %%% Let the user select one output file to indicate the directory and how the
    %%% filename is constructed
    if exist(handles.Current.DefaultOutputDirectory, 'dir')
        [ExampleFile, DataPath] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'.','*.mat'),'Select one CellProfiler output file');
    else
        [ExampleFile, DataPath] = uigetfile('*.mat','Select one CellProfiler output file');
    end
    if ~DataPath;return;end
     %CurrentDataPath = handles.Current.DefaultOutputDirectory %current dir

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
    answer = inputdlg({'Database to use:','SQL script name:'},'Export SQL',1,{'Default','SQLScript_'});
    if isempty(answer),return;end
    DatabaseName = answer{1};
    SQLScriptFileName = answer{2};%fullfile(DataPath,answer{2});
    if isempty(DatabaseName) | isempty(SQLScriptFileName)
        errordlg('A database name and an SQL script name must be specified!');
        return
    end
%end

%check whether have write permission in current dir
SQLScriptFid = fopen(SQLScriptFileName, 'wt');
if SQLScriptFid == -1, error(['Could not open ' SQLScriptFileName ' for writing.']); end
 

%display waiting bar
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

    %get the output file's prefix as .sql file's fileprefix, so if mutliple
    %files are selected the sql files will be named differently, even the
    %out put files are exactly the same except the file names
    filename = CellProfilerDataFileNames{FileNo};
    index = strfind(filename,'.');
    if ~isempty(index)
        filename = [filename(1:index-1)]%,everything before . will be kept

    end



    for ObjectTypeNo = 1:length(ObjectTypes)    % Loop over the objects

        % Get the object type in a variable for convenience
        ObjectType = ObjectTypes{ObjectTypeNo};

        % Update waitbar
        done = (FileNo - 1)*length(ObjectTypes) + ObjectTypeNo;
        total = length(CellProfilerDataFileNames)*length(ObjectTypes);
        waitbar(done/total,waitbarhandle);drawnow

    end %endloop over objtype
    %figure out the first and last set
    if isfield(handles.Measurements, 'BatchInfo'),
        FirstSet = handles.Measurements.BatchInfo.Start;
        LastSet = handles.Measurements.BatchInfo.End;
    else
        FirstSet = 1;
        LastSet = length(handles.Measurements.Image.FileNames);
    end


    %%% This is necessary to make sure the export works with the last
    %%% set.  Otherwise, the TimeElapsed array is missing the last
    %%% element.  The corresponding 'tic' is in CellProfiler.m.

    handles.Measurements.Image.TimeElapsed{handles.Current.SetBeingAnalyzed} = toc;
    % for calling from data tool, no tableprefix is asked from user, leave
    % it as blank
    CPConvertSQL(handles, DataPath, [filename,SQLScriptFileName], DatabaseName,'',FirstSet, LastSet);

end % End loop over data files

%%% Done, let the user know if this function was called as a data tool and restore the handles structure
close(waitbarhandle);
if ~ModuleCall
    CPmsgbox('Exporting is completed.')
end
fclose(SQLScriptFid);
