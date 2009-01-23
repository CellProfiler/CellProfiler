function handles = CreateBatchFiles(handles)

% Help for the Create Batch Files module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Produces text files which allow individual batches of images to be
% processed separately on a cluster of computers.
% *************************************************************************
%
% This module creates a set of files that can be submitted in parallel to a
% cluster for faster processing. This module should be placed at the end of
% an image processing pipeline.
%
% Before using this module, you should read Help -> General Help ->
% Batch Processing. That help file also will instruct you on how to
% actually run the batch files that are created by this module.
%
% Settings:
% Other Paths: The last two settings allow changing the paths between
% local and cluster computers. For example, when starting with a PC
% computer and going to a Linux machine, the path may be the same
% except the first notation:
%
% PC:    \\remoteserver1\cluster\project
% Linux: /remoteserver2/cluster/project
%
% In this case, for the local machine you would type "\\remoteserver1" and
% for the remote machine you would type "/remoteserver2". As of now, this
% is hardcoded to always use Linux and Macintosh style slashes (/).
%
% If your input image folder and output folder are located on different 
% machines, you can specify the input image paths followed by the output 
% path separated by a comma.
%
% Note: This module produces a Batch_data.mat file. This contains the
% first image set's measurements plus information about the processing
% that each batch file needs access to in order to initialize the
% processing.  See the BatchRunner.py, CPCluster.py, and CPCluster.m
% files for how this information is used.  As many clusters use
% different configurations for batch control, compiled versus
% interpreted Matlab, access paths, etc. it will probably be necessary
% to use those files as guides for a locally customized solution.
% BatchRunner.py requires Python 2.5.2 and the module scipy 0.6 to be installed. 
%
% See also MergeOutputFiles, GSBatchProcessing.

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

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%pathnametextVAR01 = What is the path to the folder where the batch control file (Batch_data.mat) will be saved? Leave a period (.) to use the default output folder.
%defaultVAR01 = .
BatchSavePath = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%pathnametextVAR02 = If pathnames are specified differently between the local and cluster machines, enter that part of the pathname from the LOCAL MACHINE'S perspective, omitting trailing slashes. Otherwise, leave a period (.). If your image and output folder paths are different, enter the input then the output path separated by a comma.
%defaultVAR02 = .
OldPathname = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%pathnametextVAR03 = If pathnames are specified differently between the local and cluster machines, enter that part of the pathname from the CLUSTER MACHINES' perspective, omitting trailing slashes. Otherwise, leave a period (.). See above for entering multiple paths.
%defaultVAR03 = .
NewPathname = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Note: This module must be the last one in the analysis pipeline.

%%%VariableRevisionNumber = 8

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

% If this isn't the first cycle, we are running on the
% cluster, and should just continue.
if (handles.Current.SetBeingAnalyzed > 1) || isfield(handles.Current, 'BatchInfo')
    return;
end
if handles.Current.NumberOfImageSets == 1 
   CPwarndlg(['Warning: No batch scripts have been written because ',...
       'you have scheduled only one cycle to be processed and that cycle is already complete.']);
   return;
end    
if strncmp(BatchSavePath, '.',1)
    if length(BatchSavePath) == 1
        BatchSavePath = handles.Current.DefaultOutputDirectory;
    else
        BatchSavePath = fullfile(handles.Current.DefaultOutputDirectory,BatchSavePath(2:end));
    end
end

% Check that Batch_data.mat does not already exist
PathAndFileName = fullfile(BatchSavePath, 'Batch_data.mat');
if exist(PathAndFileName,'file') == 2
    button = CPquestdlg('Batch_data.mat already exists in the output directory.  Are you sure you want to overwrite Batch_data.mat?',...
        'Overwrite Batch_data');
    if ~strcmp(button,'Yes')
        
        set(handles.timertexthandle,'string','Canceling after current module')
        CPmsgbox(['Image processing was canceled in the ', ModuleName, ' module at your request.'])
        CPclosefigure(handles,CurrentModule)
        return

    end
end


% Checks that this is the last module in the analysis path.
if (CurrentModuleNum ~= handles.Current.NumberOfModules),
    error(['Image processing was canceled because ', ModuleName, ' must be the last module in the pipeline.']);
end

% Saves a copy of the handles structure to revert back to later. The
% altered handles must be saved using the variable name 'handles'
% because the save function will not allow us to save a variable
% under a different name.
PreservedHandles = handles;
 
% Changes parts of several pathnames if the user has
% specified that parts of the pathname are named differently from
% the perspective of the local computer vs. the cluster
% machines

% Parse OldPathname, checking for comma separating multiple paths
if ~strcmp(OldPathname,strtok(OldPathname,',')),
    [firstpathstr,secondpathstr] = strtok(OldPathname,',');
    s = cell(1,2); s{1} = strtrim(firstpathstr); s{2} = strtrim(secondpathstr(2:end));
    OldPathname = s;
    if ~strcmp(NewPathname,strtok(NewPathname,',')),
        [firstpathstr,secondpathstr] = strtok(NewPathname,',');
        s = cell(1,2); s{1} = strtrim(firstpathstr); s{2} = strtrim(secondpathstr(2:end));
        NewPathname = s;
    else
        error(['Processing was canceled in the ', ModuleName, 'module because if there are two pathnames specified for the local machine, there need to be two pathnames for the cluster machine.']);
    end
else    % If no comma, one path only. Duplicate to both strings
    s = cell(1,2); [s{:}] = deal(strtrim(OldPathname));
    OldPathname = s;
    s = cell(1,2); [s{:}] = deal(strtrim(NewPathname));
    NewPathname = s;
end

if ~any(strcmp(OldPathname, '.'))
    % Changes the default output and image pathnames
    if ~strcmp(OldPathname{1}, '.'),
        NewDefaultImageDirectory = strrep(strrep(handles.Current.DefaultImageDirectory,OldPathname{1},NewPathname{1}),'\','/');
        handles.Current.DefaultImageDirectory = NewDefaultImageDirectory;
    end
    if ~strcmp(OldPathname{2}, '.'),
        NewDefaultOutputDirectory = strrep(strrep(handles.Current.DefaultOutputDirectory,OldPathname{2},NewPathname{2}),'\','/');
        handles.Current.DefaultOutputDirectory = NewDefaultOutputDirectory;
    end
    
    % Replaces \ with / in all image filenames (only relevant for PCs)
    Fieldnames = fieldnames(handles.Pipeline);
    FileListFieldNames = Fieldnames(strncmp(Fieldnames, 'FileList', 8));
    for i = 1:length(FileListFieldNames)
        if ndims(handles.Pipeline.(FileListFieldNames{i})) == 2
            %% Assumed to be FLEX file, with filename in the 1st row of
            %% cell array
            for j = 1:size(handles.Pipeline.(FileListFieldNames{i}),2)
                handles.Pipeline.(FileListFieldNames{i}){1,j} = strrep(handles.Pipeline.(FileListFieldNames{i}){1,j},'\','/');
            end
        else
            handles.Pipeline.(FileListFieldNames{i}) = strrep(handles.Pipeline.(FileListFieldNames{i}),'\','/');
        end
    end

    % Deal with input paths that have already been saved to the handles
    % (a) handles.Pipelines
    Fieldnames = fieldnames(handles.Pipeline);
    PathFieldnames = Fieldnames(strncmpi(Fieldnames,'pathname',8));
    for i = 1:length(PathFieldnames),
        handles.Pipeline.(PathFieldnames{i}) = strrep(strrep(handles.Pipeline.(PathFieldnames{i}),OldPathname{1},NewPathname{1}),'\','/');
        handles.Pipeline.(PathFieldnames{i}) = strrep(strrep(handles.Pipeline.(PathFieldnames{i}),OldPathname{2},NewPathname{2}),'\','/');
    end
    % (b) handles.Measurements.Image
    Fieldnames = fieldnames(handles.Measurements.Image);
    PathFieldnames = Fieldnames(strncmpi(Fieldnames,'pathname',8));
    for i = 1:length(PathFieldnames),
        handles.Measurements.Image.(PathFieldnames{i}) = strrep(strrep(handles.Measurements.Image.(PathFieldnames{i}),OldPathname{1},NewPathname{1}),'\','/');
        handles.Measurements.Image.(PathFieldnames{i}) = strrep(strrep(handles.Measurements.Image.(PathFieldnames{i}),OldPathname{2},NewPathname{2}),'\','/');
    end
end

% % Make sure ModuleError has same number of elements as
% % ModuleErrorFeatures
% handles.Measurements.Image.ModuleError{handles.Current.SetBeingAnalyzed}(1,CurrentModuleNum) = 0;
% I'm not entirely sure why we need to record that there were no module
% errors in the CreateBatchFiles module here rather than in
% CellProfiler.m, but will go ahead and update the old code to the new
% CPaddmeasurements format:
handles = CPaddmeasurements(handles,'Image',CPjoinstrings('ModuleError',[CPtwodigitstring(CurrentModuleNum),ModuleName]),0);

% Python can't load function pointers
handles = rmfield(handles, 'FunctionHandles');

% Saves the altered handles in a file which the user will feed to
% the remote machines.
save(PathAndFileName, 'handles');

% Reverts to the preserved handles.  (Probably not necessary, but simpler.)
handles = PreservedHandles;

CPhelpdlg('Batch files have been written.  This analysis pipeline will now stop.  You should submit the batch files for processing on your cluster. See Help > General Help > BatchProcessing for more information.', 'BatchFilesDialog');

% This is the first cycle, so this is the first time seeing this
% module.  It should cause a cancel so no further processing is done
% on this machine.
set(handles.timertexthandle,'string','Cancel')

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

% The figure window display is unnecessary for this module, so it is
% closed during the starting image cycle.
CPclosefigure(handles,CurrentModule)
