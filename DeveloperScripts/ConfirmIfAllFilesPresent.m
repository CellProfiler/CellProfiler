function handles = ConfirmIfAllFilesPresent(handles,varargin)

% Help for the ConfirmIfAllFilesPresent module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Given a directory (or subdirectory structure) of input images, find which
% images (or directories) for a representative channel are not matched up 
% with the other channels.
% *************************************************************************
%
% This module is useful for performing quality control on a directory of
% images in order to confirm that all files containing channel information 
% are properly matched.
%
% Settings:
%
% Type the text that identifies the channel/wavelength for one type of
% image
% The text is used to load images (or movies) that have a particular piece
% of text in the name. The entered text is assumed to be an exact match to
% a string in the filename (i.e, no regular expressions).
%
% Analyze all subfolders within the selected folder?
% You may have subfolders within the folder that is being searched, but if
% you are in TEXT mode, the names of the folders themselves must not
% contain the text you are searching for or an error will result.
%
% While this module can recurse subdirectories, it treats each directory as
% a separate experiment and will match on that basis
%
% Enter the path name to the folder where the images to be loaded are
% located
% Relative pathnames can be used. For example, on the Mac platform you
% could leave the folder where images are to be loaded as '.' to choose the
% default image folder. Or, you could type .../AnotherSubfolder (note the 
% three periods: the first is interpreted as a stand-in for the default 
% image folder) as the folder from which images are to be loaded. The above
% also applies for '&' with regards to the default output folder.
%
% Do you want the output information saved to a text file?
% Specify 'Yes' to ouput the module results to a text file, and then
% specify the path and filename in the following box. If you specify only a
% path, the output filename defaults to ConfirmIfAllFilesPresent_output.txt.
% The same notes on pathnames as above apply here also.
%
% See also LoadImages.

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

%textVAR01 = What do you want to call this listing of files within CellProfiler? (Type "Do not use" to ignore)
%defaultVAR01 = FileList
%infotypeVAR01 = filegroup indep
FileListName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%pathnametextVAR02 = Enter the path name to the folder where the images to be loaded are located. Type period (.) for default image folder.
%defaultVAR02 = .
InputPathname = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Type the text that identifies the channel/wavelength for one type of image. Type "Do not use" to ignore:
%defaultVAR03 = w1
TextToFind{1} = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What will these images be called within CellProfiler? (You must have a LoadImages module present for this box to be active)
%infotypeVAR04 = imagegroup
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu

%textVAR05 = Type the text that identifies the channel/wavelength for the second type of image. Type "Do not use" to ignore:
%defaultVAR05 = w2
TextToFind{2} = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = What will these images be called within CellProfiler? (You must have a LoadImages module present for this box to be active)
%infotypeVAR06 = imagegroup
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%inputtypeVAR06 = popupmenu

%textVAR07 = Type the text that identifies the channel/wavelength for the third type of image. Type "Do not use" to ignore:
%defaultVAR07 = Do not use
TextToFind{3} = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = What will these images be called within CellProfiler? (You must have a LoadImages module present for this box to be active)
%infotypeVAR08 = imagegroup
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%textVAR09 = Analyze all subfolders within the selected folder?
%choiceVAR09 = No
%choiceVAR09 = Yes
AnalyzeSubDir = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%textVAR10 = If there are thumbnails, type the text that distinguishes them from the image files. Type "Do not use" to ignore:
%defaultVAR10 = thumb
ThumbText = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = Do you want the output information saved to a text file?
%choiceVAR11 = No
%choiceVAR11 = Yes
SaveOutputFile = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu

%pathnametextVAR12 = If you answered 'Yes' above, enter the path name to the folder where the output file will be saved (use "&" for default output folder). If no filename is specified, the output file defaults to ConfirmIfAllFilesPresent_output.txt.
%defaultVAR12 = &
OutputPathname = char(handles.Settings.VariableValues{CurrentModuleNum,12});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

drawnow;

if handles.Current.SetBeingAnalyzed == 1
    % Get the pathname and check that it exists
    if strncmp(InputPathname,'.',1)
        if length(InputPathname) == 1
            InputPathname = handles.Current.DefaultImageDirectory;
        else
        % If the pathname starts with '.', interpret it relative to
        % the default image dir.
            InputPathname = fullfile(handles.Current.DefaultImageDirectory,strrep(strrep(InputPathname(2:end),'/',filesep),'\',filesep));
        end
    end
    if ~exist(InputPathname,'dir')
        error(['Image processing was canceled in the ', ModuleName, ' module because the directory "',InputPathname,'" does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.'])
    end
    DirectoryTextToExclude = '';

    % Exclude options if user specifies
    ImageName = ImageName(~strcmp('Do not use',TextToFind));
    TextToFind = TextToFind(~strcmp('Do not use',TextToFind));

    % Extract the file names
    
    [AllPathnames,IndivPathnames,IndivFileNames,IndivFileExtensions,idxIndivPaths] = deal(cell(1,length(ImageName)));
    for i = 1:length(ImageName)
        FileList = CPretrievemediafilenames(InputPathname,char(TextToFind(i)),AnalyzeSubDir(1), 'Regular','Image');
        
        % Remove thumbnails, if any
        if ~isempty(FileList) && ~strcmp(ThumbText,'Do not use'),
            thumb_idx = regexp(FileList,ThumbText);
            if ~isempty(thumb_idx), FileList = FileList(cellfun(@isempty,thumb_idx)); end
        end

        % Remove directories, if desired
        if ~isempty(DirectoryTextToExclude)
            for j = 1:length(DirectoryTextToExclude),
                excluded_idx = regexpi(FileList,DirectoryTextToExclude{j});
                if ~isempty(excluded_idx)
                    FileList = FileList(cellfun(@isempty,excluded_idx));
                end
            end
        end
        
        % Checks whether any files are left.
        if isempty(FileList)
            error(['Image processing was canceled in the ', ModuleName, ' module because there are no image files with the text "', TextToFind{i}, '" in the chosen directory (or subdirectories, if you requested them to be analyzed as well).'])
        end

         % ASSUMPTION: Channels are located in the same directory (i.e.
        % IndivPathnames is the same for all channels)
        [IndivPathnames{i},IndivFileNames{i},IndivFileExtensions{i}] = cellfun(@fileparts,FileList,'UniformOutput',false);
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
        fieldname = ['SelectedFileList_',FileListName,'_',ImageName{i}];
        handles.Pipeline.(fieldname).FileList = [];
        for n = 1:length(uniquePaths),
            handles.Pipeline.(fieldname).FileList = cat(2,handles.Pipeline.(fieldname).FileList, NewFileList{n}(m,:));
            handles.Pipeline.(fieldname).Pathname = InputPathname;
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%
    %%% DISPLAY RESULTS %%%
    %%%%%%%%%%%%%%%%%%%%%%%

    % List unmatched directories
    TextString{1} = ['Image directory: ',handles.Current.DefaultImageDirectory];
    TextString{end+1} = '';
    
    TextString{end+1} = 'Unmatched directories found:';
    if all(cellfun(@isempty,UnmatchedDirectories))
        TextString{end+1} = '  None';
    else
        for n = 1:length(UnmatchedDirectories)
            for m = 1:size(UnmatchedDirectories{n},1),
                TextString{end+1} = ['  ',UnmatchedDirectories{n}{m,:}];
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
                    TextString{end+1} = [' Subdirectory: ',uniquePaths{n}];
                end
                for m = 1:size(DuplicateFilenames{n},1),
                    TextString{end+1} = ['  ',DuplicateFilenames{n}{m,1},': ',num2str(DuplicateFilenames{n}{m,2})];
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
                    TextString{end+1} = [' Subdirectory: ',uniquePaths{n}];
                end
                for m = 1:size(UnmatchedFilenames{n},1),
                    TextString{end+1} = ['  ',UnmatchedFilenames{n}{m,1},': ',num2str(UnmatchedFilenames{n}{m,2})];
                end
            end
        end
    end

    drawnow;

    ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);

    if any(findobj == ThisModuleFigureNumber);
        if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
            CPresizefigure('','NarrowText',ThisModuleFigureNumber)
        end    
        % Display list
        currentfig = CPfigure(handles,'Text',ThisModuleFigureNumber);
        for n = 1:length(TextString),
            uicontrol(currentfig,'style','text','units','normalized','fontsize',handles.Preferences.FontSize,'HorizontalAlignment','left','string',TextString{n},'position',[.05 .9-(n-1)*.04 .95 .04],'BackgroundColor',[.7 .7 .9]);
        end
    end

    % Output file if desired
    if strncmpi(SaveOutputFile,'y',1),
        if strncmp(OutputPathname,'&',1)
            % If the pathname is '&', set it to the default output dir.
            if length(OutputPathname) == 1
                OutputPathname = handles.Current.DefaultOutputDirectory;
                OutputFilename = [ModuleName,'_output'];
                OutputExtension = '.txt';
            else
            % If the pathname starts with '&', interpret it relative to
            % the default output dir.
                [OutputExtendedPathname,OutputFilename,OutputExtension] = fileparts(OutputPathname(2:end));
                OutputPathname = fullfile(handles.Current.DefaultOutputDirectory,OutputExtendedPathname,'');
            end
        else
            [OutputPathname,OutputFilename,OutputExtension] = fileparts(OutputPathname);
        end
        if ~exist(OutputPathname,'dir')
            error(['Image processing was canceled in the ', ModuleName, ' module because the directory "',OutputPathname,'" does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.'])
        end

        fid = fopen(fullfile(OutputPathname,[OutputFilename OutputExtension]),'at+');
        if fid > 0,
            fprintf(fid,'%s\n',['Output of ',ModuleName]);
            fprintf(fid,'%s\n','%%%%%%%%%%%%%%%%%%%%%%%%');
            for i = 1:length(TextString)
                fprintf(fid,'%s\n',TextString{i});
            end
            fclose(fid);
        else
            error([ModuleName,': Failed to open the output file for writing']);
        end
    end
end