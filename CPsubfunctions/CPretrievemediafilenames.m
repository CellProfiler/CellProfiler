function [handles,FileNames] = CPretrievemediafilenames(handles, Pathname, TextToFind, recurse, ExactOrRegExp, ImageOrMovie)
%                                                               (handles, pathname,'','No','Exact','Both')
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

if strncmpi(recurse,'S',1) && ~isfield(handles.Pipeline,'PathNameSubFolders')
    SelectDirType = CPquestdlg('Do you want to select directories using a dialog box, or entering a text list of directories?','Choose Image Subfolder','Dialog','Text','Cancel','Dialog');
    switch SelectDirType
        case 'Dialog'
            idx = 1;
            More = 'Yes';
            while strcmp(More,'Yes')
                SubDirectory = CPuigetdir(Pathname,'Choose Image Subfolder');
                if SubDirectory == 0 %% User hit Cancel in uigetdir window
                    error('Processing was stopped because the user chose Cancel');
                end

                % Check for duplicate directories
                if idx > 1 && any(strcmp(SubDirectory, Directories))
                    CPmsgbox(['This directory ' SubDirectory ' has already been selected.  Please choose another.'])
                    continue
                end

                Directories{idx} = SubDirectory;

                % Feedback for directories chosen
                CPmsgbox(char(Directories), 'SubDirectories chosen:','help','replace');

                % Make sure the selected directories all lie under the Default Image path
                % Here, characters are treated like numbers in order to find where the 
                % difference lies between the DefaultImage path and the selected
                % path.
                str = strvcat(Pathname,Directories{idx});
                offset = find(str(1,:)-str(2,:),1,'first');
                if offset < length(Pathname),
                    CPwarndlg('All selected director(ies) must be located under the Default image folder. Please choose a different directory, or Cancel and then change the Default Image directory appropriately and try again.','Error in directory selection','replace');
                else
                  idx = idx + 1;
                end
                More = CPquestdlg('Do you want to choose another directory?');
            end
            switch More
                case 'Cancel'
                    error('Processing was stopped because the user chose Cancel');
                case 'No'
                    DirectoryText = strcat(char(Directories),pathsep); DirectoryText = DirectoryText'; DirectoryText = DirectoryText(:)';
                    OutputPathname = handles.Current.DefaultOutputDirectory;
                    OutputFilename = [mfilename,'_directorylist'];
                    OutputExtension = '.txt';
                    fid = fopen(fullfile(OutputPathname,[OutputFilename OutputExtension]),'wt+');
                    if fid > 0,
                        fprintf(fid,'%s',DirectoryText);
                        fclose(fid);
                    else
                        error([mfilename,': Failed to open the output file for writing']);
                    end
                    CPmsgbox(['The directory list you entered was saved as ',[OutputFilename OutputExtension],' in the Default Output directory. Use this file if you want to process the same folders using the Text directory selection option']);
            end
        case 'Text'
            Directories = CPinputdlg(['Enter the text string with the directories. Each path should be separated with a "',pathsep,'"'],'Enter Image Subfolders');
            if isempty(Directories)
                error('Processing was stopped because the user chose Cancel or entered no text.');
            end
            Directories = strread(Directories{:},'%s','delimiter',['',pathsep,'']);
        case 'Cancel'
            error('Processing was stopped because the user chose Cancel');
    end
    
    % Recurse selected subdirectories
    SelectedDirectoryTree = []; 
    for i = 1:length(Directories), 
        SelectedDirectoryTree = cat(1,SelectedDirectoryTree,CPgetdirectorytree(Directories{i})); 
    end; 
    [handles.Pipeline.PathNameSubFolders,Directories] = deal(SelectedDirectoryTree(:)');
    
elseif strncmpi(recurse,'Y',1) && ~isfield(handles.Pipeline,'PathNameSubFolders')
    % CPselectdirectories is still too slow for a lot of subfolders on
    % bcb_image
    Directories = CPgetdirectorytree(Pathname);
    %         Directories= CPselectdirectories(Directories);
    handles.Pipeline.PathNameSubFolders = Directories;
    
elseif isfield(handles.Pipeline,'PathNameSubFolders') && ~strcmpi(ImageOrMovie,'Both')
    % This is only used for multiple channels, so that Directories is set to PathNameSubFolders after the first channel
    % The 'Both' protection is used because CellProfiler.m, which just happens to be the only function that uses the 'Both' argument, 
    % runs a DefaultImageDirectory check which calls this subfn and
    % occurs before handles.Pipeline is cleared, so that PathNameSubFolders is still hanging around even after Analyze Images is clicked.
    Directories = handles.Pipeline.PathNameSubFolders;
else
    Directories = {Pathname};
end

FileNames = cell(0);
Count = 1;

% Add feedback
if strncmpi(recurse,'S',1) && isfield(handles.Pipeline,'PathNameSubFolders')
    hWait = CPwaitbar(0,'Loading files from multiple subdirectories, once for each channel...');
end

for i=1:length(Directories)
    % Lists all the contents of that path into a structure which includes the
    % name of each object as well as whether the object is a file or
    % directory.
    FilesAndDirsStructure = CPdir(Directories{i});
    % Puts the names of each object into a list.
    FileAndDirNames = sortrows({FilesAndDirsStructure.name}');
    % Puts the logical value of whether each object is a directory into a list.
    LogicalIsDirectory = [FilesAndDirsStructure.isdir];
    % Eliminates directories from the list of file names.
    FileNamesNoDir = FileAndDirNames(~LogicalIsDirectory);
    if isempty(FileNamesNoDir) == 1
        continue;
    else
        % Makes a logical array that marks with a "1" all file names that start
        % with a period (hidden files):
        DiscardLogical1 = strncmp(FileNamesNoDir,'.',1);

        if strncmpi(ImageOrMovie,'I',1)
            MediaExtensions = CPimread;
        elseif strncmpi(ImageOrMovie,'M',1)
            MediaExtensions = {'avi' 'stk' 'tif' 'tiff' 'flex'};
        elseif strncmpi(ImageOrMovie,'B',1)
            MediaExtensions = CPimread;
            MediaExtensions = [MediaExtensions, {'avi'}, {'stk'}, {'tif'} {'tiff'} {'flex'}];
        else
            error('You have selected an invalid entry for ImageOrMovie.  It can only be something that starts with an M (Movie) or I (Image) or B (Both).');
        end

        DiscardsByExtension = zeros(size(FileNamesNoDir));
        for j = 1:length(DiscardsByExtension)
            if ~isempty(strfind(FileNamesNoDir{j},'.'))
                %fails on four-letter extensions
                %DiscardsByExtension(i) = ~any(strcmpi(FileNamesNoDir{i}(end-2:end),MediaExtensions));

                [pathstr, name, ext] = fileparts([Pathname FileNamesNoDir{j}]);
                ext = strrep(ext, '.', '');
                DiscardsByExtension(j) = ~any(strcmpi(ext,MediaExtensions));
            end
        end

        % Combines all of the DiscardLogical arrays into one.
        DiscardLogical = DiscardLogical1 | DiscardsByExtension;
        % Eliminates filenames to be discarded.
        if isempty(DiscardLogical)
            NotYetTextMatchedFileNames = FileNamesNoDir;
        else NotYetTextMatchedFileNames = FileNamesNoDir(~DiscardLogical);
        end

        % Loops through the names in the Directory listing, looking for the text
        % of interest.  Creates the array Match which contains the numbers of the
        % file names that match.
        if strcmp(Directories{i},Pathname)
            ExtraPath = [];
        else
            ExtraPath = Directories{i};
            ExtraPath = ExtraPath(length(Pathname)+1:end);
        end
        for j=1:length(NotYetTextMatchedFileNames)
            FileName = char(NotYetTextMatchedFileNames(j));
            ShouldAddFile = false;
            if ~isempty(TextToFind)
                if strncmpi(ExactOrRegExp,'E',1)
                    % This used to be findstr, but that produced bad results
                    % if the user entered a long string as the text to find,
                    % and there is a filename with part, but not all, of the
                    % name in common with that long string.
                    if ~isempty(strfind(lower(FileName), lower(TextToFind)))||isempty(TextToFind)
                        ShouldAddFile = true;
                    end
                elseif strncmpi(ExactOrRegExp,'R',1)
                    if ~isempty(regexp(lower(FileName), lower(TextToFind)))||isempty(TextToFind)
                        ShouldAddFile = true;
                    end
                end
            else
                ShouldAddFile = true;
            end
            if ShouldAddFile
                if isempty(ExtraPath)
                    FileNames{Count} = FileName;
                else
                    FileNames{Count} = fullfile(ExtraPath, FileName);
                end
                Count = Count + 1;
            end
        end
    end
    if strncmpi(recurse,'S',1) && isfield(handles.Pipeline,'PathNameSubFolders')
        CPwaitbar(i./length(Directories),hWait);
    end
end
if strncmpi(recurse,'S',1) && isfield(handles.Pipeline,'PathNameSubFolders')
    close(hWait)
end
