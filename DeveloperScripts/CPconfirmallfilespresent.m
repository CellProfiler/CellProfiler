function [handles,UnmatchedDirectories,DuplicateFilenames,UnmatchedFilenames] = CPconfirmallfilespresent(handles,TextToFind,ImageName)

% Given a directory (or subdirectory structure) of input images, find which
% images (or directories) for a representative channel are not matched up 
% with the other channels. The output is (currently) a revised
% handles.Pipeline.FileList* structure with the missing files replaced with
% [].
% Inputs are named after the corresponding cell variables in LoadImages.

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
NewFileList = cell(length(uniquePaths),1);
[UnmatchedFilenames,DuplicateFilenames] = deal(cell(1,length(uniquePaths)));
for m = 1:length(uniquePaths)
    FileNamesForChannelN = cell(1,length(idxIndivPaths));
    for n = 1:length(ImageName)
        % FileNamesForEachChannel: Cell array of strings for each
        % channel/subdirectory
        FileNamesForEachChannel = cellfun(@strcat,IndivFileNames{n}(idxIndivPaths{n} == m),IndivFileExtensions{n}(idxIndivPaths{n} == m),'UniformOutput',false);
        
        % Find the position of the channel text in the filenames for
        % each subdirectory
        TextToFindIdx = unique(cell2mat(regexpi(FileNamesForEachChannel,TextToFind{n},'once')));
        % If the position is the same for all...
        if isscalar(TextToFindIdx)
            %... drop the filename text after the channel text and use the 
            % remainder for comparision
            % ASSUMPTION: Files from same system share common prefix during
            % the same run
            FileNamesForChannelN{n} = strvcat(FileNamesForEachChannel);
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