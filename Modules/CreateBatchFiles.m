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
% Note: This module produces a Batch_data.mat file. This contains the
% first image set's measurements plus information about the processing
% that each batch file needs access to in order to initialize the
% processing.  See the BatchRunner.py, CPCluster.py, and CPCluster.m
% files for how this information is used.  As many clusters use
% different configurations for batch control, compiled versus
% interpreted Matlab, access paths, etc. it will probably be necessary
% to use those files as guides for a locally customized solution.
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

%pathnametextVAR02 = If pathnames are specified differently between the local and cluster machines, enter that part of the pathname from the local machine's perspective, omitting trailing slashes. Otherwise, leave a period (.)
%defaultVAR02 = .
OldPathname = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%pathnametextVAR03 = If pathnames are specified differently between the local and cluster machines, enter that part of the pathname from the cluster machines' perspective, omitting trailing slashes. Otherwise, leave a period (.)
%defaultVAR03 = .
NewPathname = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Note: This module must be the last one in the analysis pipeline.

%%%VariableRevisionNumber = 8

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% If this isn't the first cycle, we are running on the
%%% cluster, and should just continue.
if (handles.Current.SetBeingAnalyzed > 1)
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

%%% Checks that this is the last module in the analysis path.
if (CurrentModuleNum ~= handles.Current.NumberOfModules),
    error(['Image processing was canceled because ', ModuleName, ' must be the last module in the pipeline.']);
end

%%% Saves a copy of the handles structure to revert back to later. The
%%% altered handles must be saved using the variable name 'handles'
%%% because the save function will not allow us to save a variable
%%% under a different name.
PreservedHandles = handles;
 
%%% Changes parts of several pathnames if the user has
%%% specified that parts of the pathname are named differently from
%%% the perspective of the local computer vs. the cluster
%%% machines.
if ~ strcmp(OldPathname, '.')
    %%% Changes the default output and image pathnames.
    NewDefaultOutputDirectory = strrep(strrep(handles.Current.DefaultOutputDirectory,OldPathname,NewPathname),'\','/');
    handles.Current.DefaultOutputDirectory = NewDefaultOutputDirectory;
    NewDefaultImageDirectory = strrep(strrep(handles.Current.DefaultImageDirectory,OldPathname,NewPathname),'\','/');
    handles.Current.DefaultImageDirectory = NewDefaultImageDirectory;

    %%% Replaces \ with / in all image filenames (PC only)
    Fields=fieldnames(handles.Pipeline);
    for i = 1:length(Fields)
        if strncmp(Fields{i}, 'FileList', 8)
            FieldName = Fields{i};
            handles.Pipeline.(FieldName)=strrep(handles.Pipeline.(FieldName),'\','/');
        end
    end

    %%% Deal with Pathnames
    Fieldnames = fieldnames(handles.Pipeline);
    PathFieldnames = Fieldnames(strncmp(Fieldnames,'Pathname',8)==1);
    for i = 1:length(PathFieldnames),
        handles.Pipeline.(PathFieldnames{i}) = strrep(strrep(handles.Pipeline.(PathFieldnames{i}),OldPathname,NewPathname),'\','/');
    end
end

% %%% Make sure ModuleError has same number of elements as
% %%% ModuleErrorFeatures
% handles.Measurements.Image.ModuleError{handles.Current.SetBeingAnalyzed}(1,CurrentModuleNum) = 0;
%%% I'm not entirely sure why we need to record that there were no module
%%% errors in the CreateBatchFiles module here rather than in
%%% CellProfiler.m, but will go ahead and update the old code to the new
%%% CPaddmeasurements format:
handles = CPaddmeasurements(handles,'Image',CPjoinstrings('ModuleError',[CPtwodigitstring(CurrentModuleNum),ModuleName]),0);

%%% Python can't load function pointers
handles = rmfield(handles, 'FunctionHandles');

%%% Saves the altered handles in a file which the user will feed to
%%% the remote machines.
PathAndFileName = fullfile(BatchSavePath, ['Batch_data.mat']);
save(PathAndFileName, 'handles');

%%% Reverts to the preserved handles.  (Probably not necessary, but simpler.)
handles = PreservedHandles;

CPhelpdlg('Batch files have been written.  This analysis pipeline will now stop.  You should submit the batch files for processing on your cluster. See Help > General Help > BatchProcessing for more information.', 'BatchFilesDialog');

%%% This is the first cycle, so this is the first time seeing this
%%% module.  It should cause a cancel so no further processing is done
%%% on this machine.
set(handles.timertexthandle,'string','Cancel')

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The figure window display is unnecessary for this module, so it is
%%% closed during the starting image cycle.
CPclosefigure(handles,CurrentModule)
