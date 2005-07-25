function FileNames = CPretrieveMediaFileNames(Pathname, TextToFind, recurse, ExactOrRegExp, ImageOrMovie)
%%% Lists all the contents of that path into a structure which includes the
%%% name of each object as well as whether the object is a file or
%%% directory.
FilesAndDirsStructure = dir(Pathname);
%%% Puts the names of each object into a list.
FileAndDirNames = sortrows({FilesAndDirsStructure.name}');
%%% Puts the logical value of whether each object is a directory into a list.
LogicalIsDirectory = [FilesAndDirsStructure.isdir];
%%% Eliminates directories from the list of file names.
FileNamesNoDir = FileAndDirNames(~LogicalIsDirectory);
if isempty(FileNamesNoDir) == 1
    FileNames = [];
else
    %%% Makes a logical array that marks with a "1" all file names that start
    %%% with a period (hidden files):
    DiscardLogical1 = strncmp(FileNamesNoDir,'.',1);
    
    if strncmpi(ImageOrMovie,'I',1)
        formats = imformats;
        MediaExtensions = CPimread;
    elseif strncmpi(ImageOrMovie,'M',1)
        MediaExtensions = {'avi' 'stk'};
    else
        error('You have selected an invalid entry for ImageOrMovie.  It can only be something that starts with an M or I.');
    end
        
    DiscardsByExtension = zeros(size(FileNamesNoDir));
    for i = [1:length(DiscardsByExtension)]
        DiscardsByExtension(i) = ~any(strcmpi(FileNamesNoDir{i}(end-2:end),MediaExtensions));
    end
   
    %%% Combines all of the DiscardLogical arrays into one.
    DiscardLogical = DiscardLogical1 | DiscardsByExtension;
    %%% Eliminates filenames to be discarded.
    if isempty(DiscardLogical) == 1
        NotYetTextMatchedFileNames = FileNamesNoDir;
    else NotYetTextMatchedFileNames = FileNamesNoDir(~DiscardLogical);
    end

    %%% Loops through the names in the Directory listing, looking for the text
    %%% of interest.  Creates the array Match which contains the numbers of the
    %%% file names that match.
    FileNames = cell(0);
    Count = 1;
    for i=1:length(NotYetTextMatchedFileNames)
        if ~isempty(TextToFind)
            if strncmpi(ExactOrRegExp,'E',1)
                if ~isempty(findstr(char(NotYetTextMatchedFileNames(i)), TextToFind))||isempty(TextToFind)
                    FileNames{Count} = char(NotYetTextMatchedFileNames(i));
                    Count = Count + 1;
                end
            elseif strncmpi(ExactOrRegExp,'R',1)
                if ~isempty(regexp(char(NotYetTextMatchedFileNames(i)), TextToFind))||isempty(TextToFind)
                    FileNames{Count} = char(NotYetTextMatchedFileNames(i));
                    Count = Count + 1;
                end
            end
        else
            FileNames{Count} = char(NotYetTextMatchedFileNames(i));
            Count = Count + 1;
        end 
    end
end
if(strncmpi(recurse,'Y',1))
    DirNamesNoFiles = FileAndDirNames(LogicalIsDirectory);
    DiscardLogical1Dir = strncmp(DirNamesNoFiles,'.',1);
    DirNames = DirNamesNoFiles(~DiscardLogical1Dir);
    if (length(DirNames) > 0)
        for i=1:length(DirNames),
            MoreFileNames = CPretrieveMediaFileNames(fullfile(Pathname, char(DirNames(i))), TextToFind, recurse, ExactOrRegExp, ImageOrMovie);
            for j = 1:length(MoreFileNames)
                MoreFileNames{j} = fullfile(char(DirNames(i)), char(MoreFileNames(j)));
            end
            if isempty(FileNames) == 1
                FileNames = MoreFileNames;
            else
                FileNames(end+1:end+length(MoreFileNames)) = MoreFileNames(1:end);
            end
        end
    end
end