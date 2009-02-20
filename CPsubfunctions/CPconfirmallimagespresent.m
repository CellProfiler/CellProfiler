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
isBatchSubmission = isfield(handles.Current,'BatchInfo');

if SetBeingAnalyzed ~= 1, return; end

% Make sure a regular expression is input
WarningDlgBoxTitle = 'Quality control for missing or duplicate image files';
WarningDlgBoxBoilerplate =  'Execution of the pipeline will continue but the image set will not be checked.';
if ~strcmp(ExactOrRegExp,'R')
    msg = ['You must specify "Text-Regular Expressions" to check image sets. ',WarningDlgBoxBoilerplate];
    if isBatchSubmission
        warning(msg);
    else
        CPwarndlg(msg, WarningDlgBoxTitle,'replace');
    end
    return;
end

% Make sure either named or unnamed tokens are being used
isTokenPresent = true;
for n = 1:numel(TextToFind)
    isTokenPresent = isTokenPresent & ~(isempty(regexp(TextToFind{n},'\(\[(?<token>.+?)\]','tokens','once')) & ...
                                        isempty(regexp(TextToFind{n},'\(\?[<](?<token>.+?)[>]','tokens','once')));
end
if ~isTokenPresent   % No tokens are present
    msg = ['Tokens must be used in all regular expressions in order to check image sets. ',WarningDlgBoxBoilerplate];
    if isBatchSubmission
        warning(msg);
    else
        CPwarndlg(msg,WarningDlgBoxTitle,'replace');
    end
    return;
end

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
% channel image is found in only some directories and not in others) and
% remove them from consideration
uniquePaths = unique(cat(2,AllPathnames{:}));
UnmatchedDirectories = cellfun(@setdiff,repmat({uniquePaths},[1 length(AllPathnames)]),AllPathnames,'UniformOutput',false);
if ~isempty(cat(2,UnmatchedDirectories{:}))
    pathstoremove = cat(2,UnmatchedDirectories{:});
    for i = 1:length(ImageName)
        idx = find(ismember(AllPathnames{i},pathstoremove));
        AllPathnames{i}(idx) = [];
        idx = ismember(idxIndivPaths{i},idx);
        IndivPathnames{i}(idx) = [];
        IndivFileNames{i}(idx) = [];
        IndivFileExtensions{i}(idx) = [];
        idxIndivPaths{i}(idx) = [];
        [ignore,ignore,idxIndivPaths{i}] = unique(idxIndivPaths{i});
    end
    uniquePaths = setdiff(uniquePaths,pathstoremove);
end

% Second, for those directories which do have all channels represented,
% check if the images in each directory/subdirectory match up by channel

% TODO: THE FOLLWOWING SECTION WOULD BENEFIT BY HAVING ACCESS TO
% THE FILENAME METADATA TOKENS (I.E. NAMED TOKENS)

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
        [tokens,tokenExtents] = regexpi(FileNamesForEachChannel{n}{m},TextToFind{n},'tokens','tokenExtents','once');
        tokenExtents = cat(1,tokenExtents{:});
        StartingIndex = unique(tokenExtents(:,1)); EndingIndex = unique(tokenExtents(:,2));
        
        % If the position is the same for all...
        if isscalar(StartingIndex)
            %... drop the filename text after the token text and use the 
            % remainder for comparision.
            % The token can be used to distinguish filename irregualrities.
            % For multichannel images, the token is used to
            % distinguish the channel, so the token must be removed to 
            % identify mismatches and duplicates. Filename mismatches
            % are not defined for single-channel images but duplicates are
            % identified based on the token, so it must be retained.
            % ASSUMPTION: The token is the last component of the filename so
            % we capture the string up and/or including the token. If the
            % filename metadata is incorporated, this operation can be made
            % more general and this assumption can be dropped
            if numel(ImageName) > 1         % Multi-channel: 
                idx = StartingIndex - 1;    % Up to the beginning of the token
            else                            % Single-channel:
                idx = EndingIndex;          % Include the token
            end
            FileNamesForChannelN{n} = strvcat(FileNamesForEachChannel{n}{m});
            FileNamesForChannelN{n} = FileNamesForChannelN{n}(:,1:idx);
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
        
        % Look for images with duplicate prefixes, and if so, keep the first
        [ignore,idx] = unique(cellFileNamesForChannelN);
        idxDuplicate = setdiff(1:length(cellFileNamesForChannelN),idx);
        if ~isempty(idxDuplicate)
            DuplicateFilenames{m} = cat(1,DuplicateFilenames{m},...
                                          cat(2,cellFileNamesForChannelN(idxDuplicate),...
                                                num2cell(repmat(n,[length(idxDuplicate) 1]))));
        end
    end
    
    % Copy the filenames into the new list, leaving [] in place of missing
    % files....
    NewFileList{m} = cell(length(ImageName),length(AllFileNamesForChannelN));
    [NewFileList{m}{:}] = deal('');
    for n = 1:length(ImageName),
        [idxFileList,locFileList] = ismember(AllFileNamesForChannelN,cellstr(FileNamesForChannelN{n}));
        idxPathlist = idxIndivPaths{n} == m;
        FullFilenames = cellfun(@fullfile,IndivPathnames{n}(idxPathlist),...
                                cellfun(@strcat,IndivFileNames{n}(idxPathlist),IndivFileExtensions{n}(idxPathlist),...
                                        'UniformOutput',false),...
                                'UniformOutput',false); 
        NewFileList{m}(n,idxFileList) = FullFilenames(locFileList(idxFileList));
    end
    
    IsFileMissing = cellfun(@isempty,NewFileList{m});
    idxMissingFiles = any(IsFileMissing,1);
    for n = find(idxMissingFiles)
        UnmatchedFilenames{m} = cat(1,UnmatchedFilenames{m},...
                                      cat(2, cellstr(AllFileNamesForChannelN(n,:)),...
                                             {find(~IsFileMissing(:,n))'}));
    end

    % ... check whether the unmatched images are corrupt...
    if ~isempty(UnmatchedFilenames{m}),
        NewFileList{m} = FindAndReplaceCorruptFilesInFilelist(handles,NewFileList{m},UnmatchedFilenames{m},m,FileNamesForChannelN,idxIndivPaths,IndivPathnames,IndivFileNames,IndivFileExtensions,fn,prefix);
    end
    
    % ... and removing duplicate files, also by checking integrity.
    % ASSUMPTION: A duplicate file means that one of them is corrupted,
    % which seems to be the case on HCS systems
    if ~isempty(DuplicateFilenames{m}),
        NewFileList{m} = FindAndReplaceCorruptFilesInFilelist(handles,NewFileList{m},DuplicateFilenames{m},m,FileNamesForChannelN,idxIndivPaths,IndivPathnames,IndivFileNames,IndivFileExtensions,fn,prefix);
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
TextString{1} = ['Output of ',mfilename,': ',datestr(now)];
TextString{2} = '----------------------------------------';
TextString{3} = ['Image directory: ',handles.Current.DefaultImageDirectory];
TextString{4} = '';

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
TextString{end+1} = 'If there are duplicate images, the integrity of the duplicates are checked and the first "good" image is used, if any. If there are none, the images are treated as missing.';
TextString{end+1} = 'If there are unmatched images, placeholders are inserted for the missing files and pipeline execution will continue. However, there will be no measurements made for the missing images.';

if isBatchSubmission
    warning(char(TextString)');
else
    CPwarndlg(TextString(3:end),WarningDlgBoxTitle,'replace');
end
    
% Output file if desired
if strncmpi(SaveOutputFile,'y',1),
    OutputPathname = handles.Current.DefaultOutputDirectory;
    OutputFilename = [mfilename,'_output'];
    OutputExtension = '.txt';
       
    fid = fopen(fullfile(OutputPathname,[OutputFilename OutputExtension]),'wt+');
    if fid > 0,
        for i = 1:length(TextString)
            fprintf(fid,'%s\n',TextString{i});
        end
        fclose(fid);
    else
        error([ModuleName,': Failed to open the output file for writing']);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION - FindAndReplaceCorruptFilesInFilelist
function NewFileList = FindAndReplaceCorruptFilesInFilelist(handles,NewFileList,FlaggedFilenames,idxUniquePath,FileNamesForChannelN,idxIndivPaths,IndivPathnames,IndivFileNames,IndivFileExtensions,FileListFieldnames,FileListPrefix)

for n = 1:size(FlaggedFilenames,1)
    channel = FlaggedFilenames{n,2};
    % Find the full names of the duplicate images
    idxFileList = ismember(cellstr(FileNamesForChannelN{channel}),FlaggedFilenames{n,1});
    idxPathlist = idxIndivPaths{channel} == idxUniquePath;
    FullFilenames = cellfun(@fullfile,IndivPathnames{channel}(idxPathlist),...
                            cellfun(@strcat,IndivFileNames{channel}(idxPathlist),IndivFileExtensions{channel}(idxPathlist),...
                                    'UniformOutput',false),...
                            'UniformOutput',false);
    FlaggedFileList = FullFilenames(idxFileList); 

    % Check whether the mismatch is corrupt by attempting an imread
    isImageCorrupt = false(1,length(FlaggedFileList));
    for k = 1:length(FlaggedFileList),
        try
            CPimread([handles.Pipeline.(['Pathname',FileListFieldnames{channel}(length(FileListPrefix)+1:end)]),FlaggedFileList{k}]);
        catch
            isImageCorrupt(k) = true;
        end
    end

    % Remove corrupt files from the new FileList, replacing them
    % with the first file(s) that pass the test, or [] if none of them pass
    % TODO: A more intelligent way to do this substitution
    if any(~isImageCorrupt) && length(isImageCorrupt) > 1
        NewFileList(channel,ismember(NewFileList(channel,:),FlaggedFileList(isImageCorrupt))) = ...
                FlaggedFileList(find(~isImageCorrupt,length(find(isImageCorrupt))));
    else
        NewFileList(channel,ismember(NewFileList(channel,:),FlaggedFileList(isImageCorrupt))) = {''};
    end
end