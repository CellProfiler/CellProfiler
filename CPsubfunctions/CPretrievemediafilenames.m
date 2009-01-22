function FileNames = CPretrievemediafilenames(Pathname, TextToFind, recurse, ExactOrRegExp, ImageOrMovie)

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

%%% If recurse is true, we list all directories
if strncmpi(recurse,'Y',1) || strncmpi(recurse,'S',1)
%     Directories = CPgetdirectorytree(Pathname);
    if strncmpi(recurse,'S',1)
        
        idx = 1;
        More = 'Yes';
        while strcmp(More,'Yes')
            SubDirectory = uigetdir(Pathname);
            Directories{idx} = SubDirectory;
            idx = idx + 1;
            More = CPquestdlg('Do you want to choose another directory?');
        end
        if strcmp(More,'Cancel')
            error('Processing was stopped because the user chose Cancel');
        end
        
        %% CPselectdirectories is still too slow for a lot of subfolders on
        %% bcb_image
%         Directories=CPselectdirectories(Directories);
    end
else
    Directories = {Pathname};
end
FileNames = cell(0);
Count = 1;
for i=1:length(Directories)
    %%% Lists all the contents of that path into a structure which includes the
    %%% name of each object as well as whether the object is a file or
    %%% directory.
    FilesAndDirsStructure = CPdir(Directories{i});
    %%% Puts the names of each object into a list.
    FileAndDirNames = sortrows({FilesAndDirsStructure.name}');
    %%% Puts the logical value of whether each object is a directory into a list.
    LogicalIsDirectory = [FilesAndDirsStructure.isdir];
    %%% Eliminates directories from the list of file names.
    FileNamesNoDir = FileAndDirNames(~LogicalIsDirectory);
    if isempty(FileNamesNoDir) == 1
        continue;
    else
        %%% Makes a logical array that marks with a "1" all file names that start
        %%% with a period (hidden files):
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

        %%% Combines all of the DiscardLogical arrays into one.
        DiscardLogical = DiscardLogical1 | DiscardsByExtension;
        %%% Eliminates filenames to be discarded.
        if isempty(DiscardLogical)
            NotYetTextMatchedFileNames = FileNamesNoDir;
        else NotYetTextMatchedFileNames = FileNamesNoDir(~DiscardLogical);
        end

        %%% Loops through the names in the Directory listing, looking for the text
        %%% of interest.  Creates the array Match which contains the numbers of the
        %%% file names that match.
        if strcmp(Directories{i},Pathname)
            ExtraPath=[];
        else
            ExtraPath=Directories{i};
            ExtraPath=ExtraPath(length(Pathname)+2:end);
        end
        for j=1:length(NotYetTextMatchedFileNames)
            FileName = char(NotYetTextMatchedFileNames(j));
            ShouldAddFile = false;
            if ~isempty(TextToFind)
                if strncmpi(ExactOrRegExp,'E',1)
                    %%% This used to be findstr, but that produced bad results
                    %%% if the user entered a long string as the text to find,
                    %%% and there is a filename with part, but not all, of the
                    %%% name in common with that long string.
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
end
