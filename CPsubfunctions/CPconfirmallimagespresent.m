function handles = CPconfirmallimagespresent(handles,TextToFind,ImageName,ExactOrRegExp,SaveOutputFile)

% Given a directory (or subdirectory structure) of input images, find which
% images (or directories) for a representative channel are not matched up 
% with the other channels. The output is (currently) a revised
% handles.Pipeline.FileList* structure with the missing filenames replaced with
% an empty string.
% Inputs are named after the corresponding cell variables in LoadImages.
% SaveOutputFile can be 'y' or 'n' depending on whether you want to produce
% an output file.

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Determines which cycle is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;
IsBatchSubmission = isfield(handles.Current,'BatchInfo');

if SetBeingAnalyzed ~= 1, return; end

% Extract the file names
fn = fieldnames(handles.Pipeline);
prefix = 'filelist';
fn = fn(strncmpi(fn,prefix,length(prefix)));
if ~iscell(fn), fn = {fn}; end

[AllPathnames,IndivPathnames,IndivFileNames,IndivFileExtensions,idxIndivPaths] = deal(cell(1,length(ImageName)));

for i = 1:length(ImageName)
    % ASSUMPTION: Channels are located in the same directory (i.e.
    % IndivPathnames is the same for all channels)
    [IndivPathnames{i},IndivFileNames{i},IndivFileExtensions{i}] = cellfun(@fileparts,handles.Pipeline.(fn{i}),'UniformOutput',false);
    [AllPathnames{i},ignore,idxIndivPaths{i}] = unique(IndivPathnames{i});
end

% First, check if directories in which channel images were located are identical. If
% not, keep track of the different ones for each channel (e.g., if a
% channel image is found in only some directories and not in others)
uniquePaths = unique(cat(2,AllPathnames{:}));
UnmatchedDirectories = cellfun(@setdiff,repmat({uniquePaths},[1 length(AllPathnames)]),AllPathnames,'UniformOutput',false);

% Second, check if the images in each directory/subdirectory match up by
% channel
FileNamesForEachChannel = cell(length(idxIndivPaths),length(uniquePaths));
NewFileList = cell(length(uniquePaths),1);
[UnmatchedFilenames,DuplicateFilenames] = deal(cell(1,length(uniquePaths)));
for m = 1:length(uniquePaths)
    FileNamesForChannelN = cell(1,length(idxIndivPaths));
    for n = 1:length(ImageName)
        % FileNamesForEachChannel{channel}{subdirectory}: Cell array of strings
        FileNamesForEachChannel{n}{m} = IndivFileNames{n}(idxIndivPaths{n} == m);

        % Find the position of the channel text in the filenames for
        % each subdirectory
        TextToFindIdx = unique(cell2mat(regexpi(FileNamesForEachChannel{n}{m},TextToFind{n},'once')));
        % If the position is the same for all...
        if isscalar(TextToFindIdx)
            %... drop the filename text after the channel text and use the 
            % remainder for comparision
            % ASSUMPTION: If TextToFind is a regular expression, the text
            % in common precedes any alphabetic, numeric, or underscore
            % characters in the search string
            if strcmp(ExactOrRegExp,'R'),
                TextToFindIdx = TextToFindIdx + regexp(TextToFind{n},'\W','once');
            end
            % ASSUMPTION: Files from same system share common prefix during
            % the same run
            FileNamesForChannelN{n} = strvcat(FileNamesForEachChannel{n}{m});
            FileNamesForChannelN{n} = FileNamesForChannelN{n}(:,1:TextToFindIdx-1);
        else
            %... otherwise, error
            error(['The specified text for ',ImageName{n},' is not located at a consistent position within the filenames in directory ',uniquePaths{m}]);
        end
    end
    
    % TODO: How should this information be used downstream?
    % Three ways to handle this:
    % (1) Trim the images from the FileList structure. However, the user might
    %   want to know that that images have gone "missing"
    % (2) Keep the images in the FileList structure and set the siblings in the
    %   corresponding FileLists to []. This will insure the FileList lengths
    %   match. However, the downstream modules will need to check for this, 
    %   probably by modifying CPretrieveimages to return a 0 or NaN image for
    %   the missing one.
    % (3) Set a QC flag for the images w/o siblings to be used for filtering 
    %   later. Still have the problem of the FileList being different lengths
    %
    % Right now, I've decided on (2), with the option of outputing a text file.
    % (3) can probably be folded into (2).

    % Combine all the filename prefixes to find what the "master list"
    % should look like
    AllFileNamesForChannelN = [];
    for n = 1:length(ImageName),
        cellFileNamesForChannelN = cellstr(FileNamesForChannelN{n});
        AllFileNamesForChannelN = union(cellFileNamesForChannelN,AllFileNamesForChannelN);
        
        % Look for images with duplicate prefixes
        [ignore,idx] = unique(cellFileNamesForChannelN);
        idxDuplicate = setdiff(1:length(cellFileNamesForChannelN),idx);
        if ~isempty(idxDuplicate)
            DuplicateFilenames{m} = cat(1,DuplicateFilenames{m},cat(2,cellFileNamesForChannelN(idxDuplicate),{n}));
        end
    end
    
    % Copy the filenames into the new list, leaving [] in place of missing
    % files
    % TODO: How to process the duplicate files similarly? Especially when
    % we don't know which file is the "right" one.
    NewFileList{m} = cell(length(ImageName),length(AllFileNamesForChannelN));
    [NewFileList{m}{:}] = deal('');
    for n = 1:length(ImageName),
        [idxFileList,locFileList] = ismember(AllFileNamesForChannelN,cellstr(FileNamesForChannelN{n}));
        FullFilenames = cellfun(@fullfile,IndivPathnames{n}(idxIndivPaths{n} == m),...
                                cellfun(@strcat,IndivFileNames{n}(idxIndivPaths{n} == m),IndivFileExtensions{n}(idxIndivPaths{n} == m),'UniformOutput',false),'UniformOutput',false); 
        NewFileList{m}(n,idxFileList) = FullFilenames(locFileList(idxFileList));
    end
    
    IsFileMissing = cellfun(@isempty,NewFileList{m});
    idxMissingFiles = any(IsFileMissing,1);
    for n = find(idxMissingFiles)
        UnmatchedFilenames{m} = cat(1,UnmatchedFilenames{m},cat(2,cellstr(AllFileNamesForChannelN(n,:)),{find(IsFileMissing(:,n))'}));
    end
end

% Saves the new filelist to the handles structure
for m = 1:length(ImageName),
    handles.Pipeline.(fn{m}) = [];
    for n = 1:length(uniquePaths),
        handles.Pipeline.(fn{m}) = cat(2,handles.Pipeline.(fn{m}), NewFileList{n}(m,:));
    end
end

% Create dialog box with results
TextString{1} = ['Image directory: ',handles.Current.DefaultImageDirectory];
TextString{end+1} = '';

% List upmatched directories
TextString{end+1} = 'Unmatched directories found:';
if all(cellfun(@isempty,UnmatchedDirectories))
    TextString{end+1} = '  None';
else
    for n = 1:length(UnmatchedDirectories)
        for m = 1:size(UnmatchedDirectories{n},1),
            TextString{end+1} = ['    ',UnmatchedDirectories{n}{m,:}];
        end
    end
end

TextString{end+1} = '';

% List duplicate filenames
TextString{end+1} = 'Duplicate filenames found: (Prefix: Channel)';
if cellfun(@isempty,DuplicateFilenames)
    TextString{end+1} = '  None';
else
    for n = 1:length(DuplicateFilenames)
        if ~isempty(DuplicateFilenames{n}),
            if ~isempty(uniquePaths{n})
                TextString{end+1} = ['  Subdirectory: ',uniquePaths{n}];
            end
            for m = 1:size(DuplicateFilenames{n},1),
                TextString{end+1} = ['    ',DuplicateFilenames{n}{m,1},':  ',num2str(DuplicateFilenames{n}{m,2})];
            end
        end
    end
end

TextString{end+1} = '';

% List unmatched filenames
TextString{end+1} = 'Unmatched filenames found: (Prefix: Channel)';
if cellfun(@isempty,UnmatchedFilenames)
    TextString{end+1} = '  None';
else
    for n = 1:length(UnmatchedFilenames)
        if ~isempty(UnmatchedFilenames{n}),
            if ~isempty(uniquePaths{n})
                TextString{end+1} = ['  Subdirectory: ',uniquePaths{n}];
            end
            for m = 1:size(UnmatchedFilenames{n},1),
                TextString{end+1} = ['    ',UnmatchedFilenames{n}{m,1},':  ',num2str(UnmatchedFilenames{n}{m,2})];
            end
        end
    end
end

TextString{end+1} = '';
TextString{end+1} = 'If there are duplicate images, you should halt the pipeline, examine the files and remove the duplicates.';
TextString{end+1} = 'If there are unmatched images, placeholders have been inserted for the missing files and pipeline execution will continue. However, there will be no measurements made for the missing image.';

if ~IsBatchSubmission
    CPwarndlg(TextString,'Image check for missing or duplicate files','replace');
end
    
% Output file if desired
if strncmpi(SaveOutputFile,'y',1),
    OutputPathname = handles.Current.DefaultOutputDirectory;
    OutputFilename = [mfilename,'_output'];
    OutputExtension = '.txt';
       
    fid = fopen(fullfile(OutputPathname,[OutputFilename OutputExtension]),'at+');
    if fid > 0,
        fprintf(fid,'%s\n',['Output of ',mfilename]);
        fprintf(fid,'%s\n','%%%%%%%%%%%%%%%%%%%%%%%%');
        for i = 1:length(TextString)
            fprintf(fid,'%s\n',TextString{i});
        end
        fclose(fid);
    else
        error([ModuleName,': Failed to open the output file for writing']);
    end
end