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

%textVAR02 = (Optional) What directory grouping do you want to use? Select "Do not use" to ignore. (This answer overrides any path-related selections below; see SelectDirectoriesToProcess for details)
%choiceVAR02 = Do not use
%infotypeVAR02 = directorygroup
DirectoryGroupName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%pathnametextVAR03 = If not using a directrory group, enter the path name to the folder where the images to be loaded are located. Type period (.) for default image folder.
%defaultVAR03 = .
InputPathname = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Type the text that identifies the channel/wavelength for one type of image. Type "Do not use" to ignore:
%defaultVAR04 = w1
TextToFind{1} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = What will these images be called within CellProfiler? (You must have a LoadImages module present for this box to be active)
%infotypeVAR05 = imagegroup
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = Type the text that identifies the channel/wavelength for the second type of image. Type "Do not use" to ignore:
%defaultVAR06 = w2
TextToFind{2} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = What will these images be called within CellProfiler? (You must have a LoadImages module present for this box to be active)
%infotypeVAR07 = imagegroup
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = Type the text that identifies the channel/wavelength for the third type of image. Type "Do not use" to ignore:
%defaultVAR08 = Do not use
TextToFind{3} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = What will these images be called within CellProfiler? (You must have a LoadImages module present for this box to be active)
%infotypeVAR09 = imagegroup
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%textVAR10 = Analyze all subfolders within the selected folder?
%choiceVAR10 = No
%choiceVAR10 = Yes
AnalyzeSubDir = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu

%textVAR11 = If there are thumbnails, type the text that distinguishes them from the image files. Type "Do not use" to ignore:
%defaultVAR11 = thumb
ThumbText = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%textVAR12 = Do you want the output information saved to a text file?
%choiceVAR12 = No
%choiceVAR12 = Yes
SaveOutputFile = char(handles.Settings.VariableValues{CurrentModuleNum,12});
%inputtypeVAR12 = popupmenu

%pathnametextVAR13 = If you answered 'Yes' above, enter the path name to the folder where the output file will be saved (use "&" for default output folder). If no filename is specified, the output file defaults to ConfirmIfAllFilesPresent_output.txt.
%defaultVAR13 = &
OutputPathname = char(handles.Settings.VariableValues{CurrentModuleNum,13});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

drawnow;

if handles.Current.SetBeingAnalyzed == 1
    UseUserdefinedDirStructure = ~strcmp(DirectoryGroupName,'Do not use');
    if ~UseUserdefinedDirStructure
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
    else
        % Fill in InputPathname with the directory list
        fieldname = ['SelectedDirectoryList_', DirectoryGroupName];
        InputPathname = handles.Pipeline.(fieldname).ImageDirectory;
        AnalyzeSubDir = handles.Pipeline.(fieldname).AnalyzeSubdirectories;
        DirectoryTextToExclude = handles.Pipeline.(fieldname).TextToExclude;
    end

    % Exclude options if user specifies
    ImageName = ImageName(~strcmp('Do not use',TextToFind));
    TextToFind = TextToFind(~strcmp('Do not use',TextToFind));

    % Extract the file names
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

        [IndivPathnames,IndivFileNames{i}] = cellfun(@fileparts,FileList,'UniformOutput',false);
        [AllPathnames{i},ignore,idx{i}] = unique(IndivPathnames);
    end

    % First, check if directories in which channel images were located are identical. If
    % not, keep track of the different ones for each channel (e.g., if a
    % channel image is found in only some directories and not in others)
    uniquePaths = unique(cat(2,AllPathnames{:}));
    MismatchedDirectories = cellfun(@setdiff,repmat({uniquePaths},[1 length(AllPathnames)]),AllPathnames,'UniformOutput',false);

    % Second, check if the images in each directory/subdirectory match up by
    % channel
    MismatchedFilenames = cell(1,length(ImageName));
    FileNamesForEachChannel = cell(length(idx),length(uniquePaths));
    for m = 1:length(uniquePaths)
        FileNamesForChannelN = cell(1,length(idx));
        for n = 1:length(idx)
            % FileNamesForEachChannel{channel}{subdirectory}: Cell array of strings
            FileNamesForEachChannel{n}{m} = IndivFileNames{n}(idx{n} == m);

            % Find the position of the channel text in the filenames for
            % each subdirectory
            TextToFindIdx = unique(cell2mat(regexp(FileNamesForEachChannel{n}{m},TextToFind{n})));
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
                error(['The specified text for ',FileListName{n},' is not located at a consistent position within the filenames in directory ',uniquePaths{m}]);
            end
        end
        % Compare the filename strings pair-wise through the channel
        % combinations and pull out differences
        combChannels = nchoosek(1:length(idx),2);
        for n = 1:size(combChannels,1)
            chan1 = combChannels(n,1); chan2 = combChannels(n,2);
            [ignore,ia,ib] = setxor(FileNamesForChannelN{[chan1 chan2]},'rows');
            if ~isempty(ia)
                i = find(idx{chan1} == m);
                if all(cellfun(@isempty,IndivPathnames(i(ia)))),
                    separator = repmat(' ',[length(ia) 1]);
                else
                    separator = repmat(filesep,[length(ia) 1]);
                end
                MismatchedFilenames{chan1} = [MismatchedFilenames{chan1}; cellstr(strcat(char(IndivPathnames(i(ia))'),separator,char(IndivFileNames{chan1}(i(ia))')))];
            end
            if ~isempty(ib)
                i = find(idx{chan2} == m);
                if all(cellfun(@isempty,IndivPathnames(i(ib)))),
                    separator = repmat(' ',[length(ib) 1]);
                else
                    separator = repmat(filesep,[length(ib) 1]);
                end
                MismatchedFilenames{chan2} = [MismatchedFilenames{chan2}; cellstr(strcat(char(IndivPathnames(i(ib))'),separator,char(IndivFileNames{chan2}(i(ib))')))];
            end
        end
    end

    % Saves the filelist without the mismatched files to the handles structure      
    for i = 1:length(ImageName),
        fieldname = ['SelectedFileList_',FileListName,'_',ImageName{i}];
        for j = 1:length(MismatchedFilenames{i}),
            FileList(strmatch(MismatchedFilenames{i}{j},FileList)) = [];
        end
        handles.Pipeline.(fieldname).FileList = FileList;
        handles.Pipeline.(fieldname).Pathname = InputPathname;
    end
          
    %%%%%%%%%%%%%%%%%%%%%%%
    %%% DISPLAY RESULTS %%%
    %%%%%%%%%%%%%%%%%%%%%%%

    % List mismatched directories
    TextString{1} = 'Mismatched directories found:';
    if all(cellfun(@isempty,MismatchedDirectories))
        TextString{2} = '  None';
    else
        for n = 1:length(MismatchedDirectories)
            for m = 1:size(MismatchedDirectories{n},1),
                TextString{end+1} = ['  ',MismatchedDirectories{n}{m,:}];
            end
        end
    end

    TextString{end+1} = '';

    % List mismatched filenames
    TextString{end+1} = 'Mismatched filenames found:';
    if cellfun(@isempty,MismatchedFilenames)
        TextString{end+1} = '  None';
    else
        for n = 1:length(MismatchedFilenames)
            if ~isempty(MismatchedFilenames{n}),
                [pathstr,fn] = cellfun(@fileparts,MismatchedFilenames{n},'UniformOutput',false);
                uniquepathstr = unique(pathstr);
                if all(cellfun(@isempty,uniquepathstr)),  % Empty unique path: All files in root directory
                    for m = 1:length(fn),
                        TextString{end+1} = ['   ',fn{m}];
                    end
                else                        % Non-empty unique path: Some files in sub-directories
                    for m = 1:length(uniquepathstr),
                        if ~isempty(uniquepathstr{m}), % At least one file in sub-directory
                            TextString{end+1} = [' Subdirectory:',uniquepathstr{m}];
                        end
                        i = find(~cellfun(@isempty,regexp(pathstr,uniquepathstr{m})));
                        for p = 1:length(i),
                            TextString{end+1} = ['   ',fn{i(p)}];
                        end
                    end
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