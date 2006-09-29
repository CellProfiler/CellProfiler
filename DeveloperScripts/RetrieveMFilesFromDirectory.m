function MFileNames = RetrieveMFilesFromDirectory(PathName)
%%% Retrieves every file in the current directory that ends in ".m"
FilesAndDirsStructure = dir(PathName);
FileAndDirNames = sortrows({FilesAndDirsStructure.name}');
LogicalIsDirectory = [FilesAndDirsStructure.isdir];
FileNamesNoDir = FileAndDirNames(~LogicalIsDirectory);
MFileNames = cell(0);
for i = 1:length(FileNamesNoDir),
    if strncmp(FileNamesNoDir{i}(end-1:end),'.m',2) == 1,
        MFileNames(length(MFileNames)+1) = {fullfile(PathName,FileNamesNoDir{i}(1:end-2))};
    end
end