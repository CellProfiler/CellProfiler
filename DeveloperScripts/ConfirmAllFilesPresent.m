function handles = ConfirmAllFilesPresent(handles,varargin)

% Help for the ConfirmAllFilesPresent module:
% Category: QualityControl
%
% SHORT DESCRIPTION:
% Given a directory (or subdirectory structure) of input images, find which
% images (or directories) for a representative channel are not matched up 
% with the other channels.
% *************************************************************************
%
% While this module can recurse subdirectories, it treats each directory as
% a separate plate and will match on that basis

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What do you want to call these images within CellProfiler?
%defaultVAR01 = OrigBlue
%infotypeVAR01 = imagegroup indep
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Type the text that identifies the channel/wavelength for one type of image. Type "Do not use" to ignore:
%defaultVAR02 = w1
TextToFind{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What do you want to call these images within CellProfiler?
%defaultVAR03 = OrigGreen
%infotypeVAR03 = imagegroup indep
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Type the text that identifies the channel/wavelength for one type of image. Type "Do not use" to ignore:
%defaultVAR04 = w2
TextToFind{2} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = What do you want to call these images within CellProfiler?
%defaultVAR05 = OrigRed
%infotypeVAR05 = imagegroup indep
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Type the text that identifies the channel/wavelength for one type of image. Type "Do not use" to ignore:
%defaultVAR06 = Do not use
TextToFind{3} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = If there are thumbnails, type the text that distinguishes them from the image files. Type "Do not use" to ignore:
%defaultVAR07 = thumb
ThumbText = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = Analyze all subfolders within the selected folder?
%choiceVAR08 = No
%choiceVAR08 = Yes
AnalyzeSubDir = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%pathnametextVAR09 = Enter the path name to the folder where the images to be loaded are located. Type period (.) for default image folder.
%defaultVAR09 = .
InputPathname = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%pathnametextVAR10 = Enter the path name to the folder where the output file will be saved. Type period (&) for default output folder.
%defaultVAR10 = &
OutputPathname = char(handles.Settings.VariableValues{CurrentModuleNum,10});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

drawnow;

% Get the pathname and check that it exists
if strncmp(InputPathname,'.',1)
    if length(InputPathname) == 1
        InputPathname = handles.Current.DefaultImageDirectory;
    else
    % If the pathname starts with '.', interpret it relative to
    % the default image dir.
        InputPathname = fullfile(handles.Current.DefaultImageDirectory,InputPathname(2:end));
    end
end
if ~exist(InputPathname,'dir')
    error(['Image processing was canceled in the ', ModuleName, ' module because the directory "',InputPathname,'" does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.'])
end

% Exclude options if user specifies
ImageName = ImageName(~strcmp('Do not use',TextToFind));
TextToFind = TextToFind(~strcmp('Do not use',TextToFind));

% Extract the file names
for n = 1:length(ImageName)
    FileList = CPretrievemediafilenames(InputPathname,char(TextToFind(n)),AnalyzeSubDir(1), 'Regular','Image');

    % Remove thumbnails
    if ~isempty(FileList) && ~strcmp(ThumbText,'Do not use'),
        thumb_idx = regexp(FileList,ThumbText);
        if ~isempty(thumb_idx), FileList = FileList(cellfun(@isempty,thumb_idx)); end
    end
    
    % Checks whether any files are left.
    if isempty(FileList)
        error(['Image processing was canceled in the ', ModuleName, ' module because there are no image files with the text "', TextToFind{n}, '" in the chosen directory (or subdirectories, if you requested them to be analyzed as well).'])
    end
    
    [IndivPathnames,IndivFileNames{n}] = cellfun(@fileparts,FileList,'UniformOutput',false);
    [AllPathnames{n},ignore,idx{n}] = unique(IndivPathnames);
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
            error(['The specified text for ',ImageName{n},' is not located at a consistent position within the filenames in directory ',uniquePaths{m}]);
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

% Output file
if strncmp(OutputPathname,'&',1)
    if length(OutputPathname) == 1
        OutputPathname = handles.Current.DefaultOutputDirectory;
    else
    % If the pathname starts with '&', interpret it relative to
    % the default output dir.
        OutputPathname = fullfile(handles.Current.DefaultOutputDirectory,OutputPathname(2:end));
    end
end
if ~exist(OutputPathname,'dir')
    error(['Image processing was canceled in the ', ModuleName, ' module because the directory "',OutputPathname,'" does not exist. Be sure that no spaces or unusual characters exist in your typed entry and that the pathname of the directory begins with /.'])
end

fid = fopen(fullfile(OutputPathname,['qualitycontrolpipeline_output.txt']),'at+');
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