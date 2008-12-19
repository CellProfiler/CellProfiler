function handles = SelectDirectoriesAndFiles(handles)

% Help for the SelectDirectoriesAndFiles module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Given a directory (or subdirectory structure) of input images, exclude
% files/directories and find which images (or directories) for a 
% representative channel are not matched up with the other channels.
% *************************************************************************
%
% This module is to be used for excluding image files and directories from
% later consideration.
% Also intended to be used for performing quality control on a directory of
% images in order to confirm that all files containing channel information 
% are properly matched.
% The output is a FileGroupName named structure that can be passed to
% other modules for use.
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

%textVAR01 = What do you want to call this directory/file listing within CellProfiler?
%defaultVAR01 = FileList
%infotypeVAR01 = filegroup indep
FileGroupName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%pathnametextVAR02 = Enter the path name to the folder where the images to be loaded are located. Type period (.) for default image folder or ampersand (&) for default output folder.
%defaultVAR02 = .
InputPathname = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Analyze all subfolders within the selected folder?
%choiceVAR03 = No
%choiceVAR03 = Yes
AnalyzeSubDir = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = If you want to exclude folders, type the text that the excluded folders have in common. Type "Do not use" to ignore.
%defaultVAR04 = Do not use
TextToExcludeDirectories = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = How do you want to exclude these directories based on the text?
%choiceVAR05 = Exact match
%choiceVAR05 = Regular expressions
DirectoryTextExclusionType = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = Type the text that identifies the channel/wavelength for one type of image. Type "Do not use" to ignore:
%defaultVAR06 = w1
TextToFindFiles{1} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = What will these images be called within CellProfiler? (You must have a LoadImages module present for this box to be active)
%infotypeVAR07 = imagegroup
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = Type the text that identifies the channel/wavelength for the second type of image. Type "Do not use" to ignore:
%defaultVAR08 = w2
TextToFindFiles{2} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = What will these images be called within CellProfiler? (You must have a LoadImages module present for this box to be active)
%infotypeVAR09 = imagegroup
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,9});
%inputtypeVAR09 = popupmenu

%textVAR10 = Type the text that identifies the channel/wavelength for the third type of image. Type "Do not use" to ignore.
%defaultVAR10 = Do not use
TextToFindFiles{3} = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%textVAR11 = What will these images be called within CellProfiler? (You must have a LoadImages module present for this box to be active)
%infotypeVAR11 = imagegroup
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,11});
%inputtypeVAR11 = popupmenu

%textVAR12 = If there are files to exclude, type the text that distinguishes them from the image files. Type "Do not use" to ignore. Separate mulitple text strings with commas.
%defaultVAR12 = Do not use
TextToExcludeFiles = char(handles.Settings.VariableValues{CurrentModuleNum,12});

%textVAR13 = Do you want the output information saved to a text file?
%choiceVAR13 = No
%choiceVAR13 = Yes
SaveOutputFile = char(handles.Settings.VariableValues{CurrentModuleNum,13});
%inputtypeVAR13 = popupmenu

%pathnametextVAR14 = If you answered 'Yes' above, enter the path name to the folder where the output file will be saved (use "&" for default output folder). If no filename is specified, the output file defaults to ConfirmIfAllFilesPresent_output.txt.
%defaultVAR14 = &
OutputPathname = char(handles.Settings.VariableValues{CurrentModuleNum,14});

%%%VariableRevisionNumber = 1


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow;

% Determines which cycle is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

if SetBeingAnalyzed == 1
    
    TextToExcludeFiles = textscan(TextToExcludeFiles,'%s','delimiter',',');            
    % (1) Resolve the input pathname
    % Get the InputPathname and check that it exists
    if strncmp(InputPathname,'.',1)
        if length(InputPathname) == 1
            InputPathname = handles.Current.DefaultImageDirectory;
        else
        % If the pathname start with '.', interpret it relative to
        % the default image dir.
            InputPathname = fullfile(handles.Current.DefaultImageDirectory,strrep(strrep(InputPathname(2:end),'/',filesep),'\',filesep),'');
        end
    elseif strncmp(InputPathname, '&', 1)
        if length(InputPathname) == 1
            InputPathname = handles.Current.DefaultOutputDirectory;
        else
        % If the pathname start with '&', interpret it relative to
        % the default output dir.
            InputPathname = fullfile(handles.Current.DefaultOutputDirectory,strrep(strrep(InputPathname(2:end),'/',filesep),'\',filesep),'');
        end
    end
    if ~exist(InputPathname,'dir')
        error(['Image processing was canceled in the ', ModuleName, ' module because the directory "',InputPathname,'" does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.'])
    end

    % Exclude options if user specifies
    ImageName = ImageName(~strcmp('Do not use',TextToFindFiles));
    TextToFindFiles = TextToFindFiles(~strcmp('Do not use',TextToFindFiles));
    NumberOfExcludedDirectories = cell(1,length(ImageName));
    [NumberOfExcludedDirectories{:}] = deal(0);
    
    % Extract the file names
    FileList = cell(1,length(ImageName));
    for i = 1:length(ImageName)
        FileList{i} = CPretrievemediafilenames(InputPathname,TextToFindFiles{i},AnalyzeSubDir(1), 'Regular','Image');
            
        % (3) Remove files with excluded text, if any
        if ~any(strcmp(TextToExcludeFiles,'Do not use'))
            % Separate out comma-delimited list
            for j = 1:length(TextToExcludeFiles),
                excluded_idx = regexpi(FileList{i},TextToExcludeFiles{j});
                if ~isempty(excluded_idx),
                    FileList{i} = FileList{i}(cellfun(@isempty,excluded_idx));
                end
            end
        end

        % Checks whether any files are left after file exclusion
        if isempty(FileList{i})
            error(['Image processing was canceled in the ', ModuleName, ' module because there are no image files with the text "', TextToFindFiles{i}, '" in the chosen directory (or subdirectories, if you requested them to be analyzed as well).'])
        end

        [IndivPathnames,IndivFileNames{i},IndivExtensions{i}] = cellfun(@fileparts,FileList{i},'UniformOutput',false);
        if ~strcmp(TextToExcludeDirectories,'Do not use')
            switch lower(DirectoryTextExclusionType(1))
                case 'e', 
                    excluded_idx = regexp(IndivPathnames,TextToExcludeDirectories);
                    excluded_idx = find(~cellfun(@isempty,excluded_idx));
                case 'r', 
                    excluded_idx = regexp(IndivPathnames,TextToExcludeDirectories);
                    excluded_idx = find(~cellfun(@isempty,excluded_idx));
            end
            if ~isempty(excluded_idx),
                NumberOfExcludedDirectories{i} = length(unique(IndivPathnames(excluded_idx)));
                FileList{i}(excluded_idx) = [];
                IndivFileNames{i}(excluded_idx) = [];
                IndivExtensions{i}(excluded_idx) = [];
                IndivPathnames(excluded_idx) = [];
            end
        end
        
        % Checks whether any files are left after directory exclusion
        if isempty(FileList{i})
            error(['Image processing was canceled in the ', ModuleName, ' module because there are no folders left after excluding the text "', TextToFindFiles{i}, '" in the chosen directory (or subdirectories, if you requested them to be analyzed as well).'])
        end
        [AllPathnames{i},ignore,idx{i}] = unique(IndivPathnames);
    end

    % (4) Check mismatches in file nomenclature
    % (a) Check if directories in which channel images were located are identical. If
    % not, keep track of the different ones for each channel (e.g., if a
    % channel image is found in only some directories and not in others)
    uniquePaths = unique(cat(2,AllPathnames{:}));
    MismatchedDirectories = cellfun(@setdiff,repmat({uniquePaths},[1 length(AllPathnames)]),AllPathnames,'UniformOutput',false);

    % (b) Check if the images in each directory/subdirectory match up by
    % channel
    MismatchedFilenames = cell(1,length(ImageName));
    FileNamesForEachChannel = cell(length(idx),length(uniquePaths));
    for i = 1:length(uniquePaths)
        FileNamesForChannelN = cell(1,length(idx));
        for j = 1:length(idx)
            % FileNamesForEachChannel{channel}{subdirectory}: Cell array of strings
            FileNamesForEachChannel{j}{i} = IndivFileNames{j}(idx{j} == i);

            % Find the position of the channel text in the filenames for
            % each subdirectory
            TextToFindIdx = unique(cell2mat(regexp(FileNamesForEachChannel{j}{i},TextToFindFiles{j})));
            % If the position is the same for all...
            if isscalar(TextToFindIdx)
                %... drop the filename text after the channel text and use the 
                % remainder for comparision
                % ASSUMPTION: Files from same system share common prefix during
                % the same run
                FileNamesForChannelN{j} = strvcat(FileNamesForEachChannel{j}{i});
                FileNamesForChannelN{j} = FileNamesForChannelN{j}(:,1:TextToFindIdx-1);
            else
                %... otherwise, error
                error(['The specified text for ',FileGroupName,' is not located at a consistent position within the filenames in directory ',uniquePaths{m}]);
            end
        end
        % Compare the filename strings pair-wise through the channel
        % combinations and pull out differences
        combChannels = nchoosek(1:length(idx),2);
        for j = 1:size(combChannels,1)
            chan1 = combChannels(j,1); chan2 = combChannels(j,2);
            [ignore,ia,ib] = setxor(FileNamesForChannelN{[chan1 chan2]},'rows');
            if ~isempty(ia)
                k = find(idx{chan1} == i);
                if all(cellfun(@isempty,IndivPathnames(k(ia)))),
                    separator = repmat(' ',[length(ia) 1]);
                else
                    separator = repmat(filesep,[length(ia) 1]);
                end
                MismatchedFilenames{chan1} = [MismatchedFilenames{chan1}; ...
                    cellstr(strcat(char(IndivPathnames(k(ia))'),separator,char(IndivFileNames{chan1}(k(ia))'),char(IndivExtensions{chan1}(k(ia))')))];
            end
            if ~isempty(ib)
                k = find(idx{chan2} == i);
                if all(cellfun(@isempty,IndivPathnames(k(ib)))),
                    separator = repmat(' ',[length(ib) 1]);
                else
                    separator = repmat(filesep,[length(ib) 1]);
                end
                MismatchedFilenames{chan2} = [MismatchedFilenames{chan2}; ...
                    cellstr(strcat(char(IndivPathnames(k(ib))'),separator,char(IndivFileNames{chan2}(k(ib))'),char(IndivExtensions{chan2}(k(ib))')))];
            end
        end
    end

    % (4) Saves the Directory list without the mismatched files to the handles structure      
    for i = 1:length(ImageName),
        fieldname = ['DefinedFileList_',FileGroupName,'_',ImageName{i}];
        for j = 1:length(MismatchedFilenames{i}),
            FileList{i}(strmatch(MismatchedFilenames{i}{j},FileList{i})) = [];
        end
        handles.Pipeline.(fieldname).FileList = FileList{i};
        handles.Pipeline.(fieldname).Pathname = InputPathname;
        handles.Pipeline.(fieldname).AnalyzeSubdirectories = AnalyzeSubDir;
        handles.Pipeline.(fieldname).TextToExcludeDirectories = TextToExcludeDirectories;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%
    %%% DISPLAY RESULTS %%%
    %%%%%%%%%%%%%%%%%%%%%%%
    
    ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
    % If the figure window isn't open, skip everything.
    if any(findobj == ThisModuleFigureNumber)
        % Create a text string that contains the output info
        % (1) Directories
        TextString{1} = ['Information on ',FileGroupName,' file listing:'];
        TextString{end+1} = [' Number of directories found: ',num2str(length(uniquePaths) + max(cell2mat(NumberOfExcludedDirectories)))];
        TextString{end+1} = [' Number of directories excluded: ',num2str(max(cell2mat(NumberOfExcludedDirectories)))];
        TextString{end+1} = [' Directories remaining: ',num2str(length(uniquePaths))];
        TextString{end+1} = '';
        TextString{end+1} = '';
        
        % (2) Mismatches
        % List mismatched directories
        TextString{end} = 'Mismatched directories found:';
        if all(cellfun(@isempty,MismatchedDirectories))
            TextString{end+1} = '  None';
        else
            for i = 1:length(MismatchedDirectories)
                for j = 1:size(MismatchedDirectories{i},1),
                    TextString{end+1} = ['  ',MismatchedDirectories{i}{j,:}];
                end
            end
        end

        TextString{end+1} = '';

        % List mismatched filenames
        TextString{end+1} = 'Mismatched filenames found:';
        if cellfun(@isempty,MismatchedFilenames)
            TextString{end+1} = '  None';
        else
            for i = 1:length(MismatchedFilenames)
                if ~isempty(MismatchedFilenames{i}),
                    [pathstr,fn] = cellfun(@fileparts,MismatchedFilenames{i},'UniformOutput',false);
                    uniquepathstr = unique(pathstr);
                    if all(cellfun(@isempty,uniquepathstr)),  % Empty unique path: All files in root directory
                        for j = 1:length(fn),
                            TextString{end+1} = ['   ',fn{j}];
                        end
                    else                        % Non-empty unique path: Some files in sub-directories
                        for j = 1:length(uniquepathstr),
                            if ~isempty(uniquepathstr{j}), % At least one file in sub-directory
                                TextString{end+1} = [' Subdirectory:',uniquepathstr{j}];
                            end
                            idx = find(~cellfun(@isempty,regexp(pathstr,uniquepathstr{j})));
                            for k = 1:length(idx),
                                TextString{end+1} = ['   ',fn{idx(k)}];
                            end
                        end
                    end
                end
            end
        end
        
        % Create figure and display list
        if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
            CPresizefigure('','NarrowText',ThisModuleFigureNumber)
        end    
        currentfig = CPfigure(handles,'Text',ThisModuleFigureNumber);
        for i = 1:length(TextString),
            uicontrol(currentfig,'style','text','units','normalized','fontsize',handles.Preferences.FontSize,'HorizontalAlignment','left','string',TextString{i},'position',[.05 .9-(i-1)*.04 .95 .04],'BackgroundColor',[.7 .7 .9]);
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



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTION: SUBDIR %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function file = subdir(cdir)
%SUBDIR List directory and all subdirectories.
% SUBDIR recursively calls itself and uses DIR to find all directories 
% within a specified directory and all its subdirectories.
%
% D = SUBDIR('directory_name') returns the results in an M-by-1
% structure with the fields:
% name -- filename
% dir -- directory containing file
% isdir -- 1 if name is a directory and 0 if not

if nargin == 0 || isempty(cdir)
    cdir = cd; % Current directory is default
end
if cdir(end)==filesep
    cdir(end) = ''; % Remove any trailing \ from directory
end
file = CPdir(cdir); % Read current directory
for n = 1:length(file)
    file(n).dir = cdir; % Assign dir field
    if file(n).isdir && file(n).name(1) ~= '.'
        % Element is a directory -> recursively search this one
        tfile = subdir([cdir filesep file(n).name]); % Recursive call
        file = [file; tfile]; % Append to result to current directory structure
    end
end