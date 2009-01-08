function handles = ConfirmAllFilesPresent(handles,varargin)

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
% This module should be placed BEFORE the first LoadImages module.
%
% TODO: Currently, it only reports results; it doesn't make changes to the 
% handles structure for later use.
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

%pathnametextVAR01 = Enter the path name to the folder where the images to be loaded are located. Type period (.) for default image folder.
%defaultVAR01 = .
InputPathname = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = Type the text that identifies the channel/wavelength for one type of image. Type "Do not use" to ignore:
%defaultVAR02 = w1
TextToFind{1} = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = What will these images be called within CellProfiler? (You must have a LoadImages module present for this box to be active)
%infotypeVAR03 = imagegroup
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = Type the text that identifies the channel/wavelength for the second type of image. Type "Do not use" to ignore:
%defaultVAR04 = w2
TextToFind{2} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = What will these images be called within CellProfiler? (You must have a LoadImages module present for this box to be active)
%infotypeVAR05 = imagegroup
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = Type the text that identifies the channel/wavelength for the third type of image. Type "Do not use" to ignore:
%defaultVAR06 = Do not use
TextToFind{3} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = What will these images be called within CellProfiler? (You must have a LoadImages module present for this box to be active)
%infotypeVAR07 = imagegroup
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,7});
%inputtypeVAR07 = popupmenu

%textVAR08 = Analyze all subfolders within the selected folder?
%choiceVAR08 = No
%choiceVAR08 = Yes
AnalyzeSubDir = char(handles.Settings.VariableValues{CurrentModuleNum,8});
%inputtypeVAR08 = popupmenu

%textVAR09 = If there are thumbnails, type the text that distinguishes them from the image files. Type "Do not use" to ignore:
%defaultVAR09 = thumb
ThumbText = char(handles.Settings.VariableValues{CurrentModuleNum,9});

%textVAR10 = Do you want the results saved to a text file?
%choiceVAR10 = No
%choiceVAR10 = Yes
SaveOutputFile = char(handles.Settings.VariableValues{CurrentModuleNum,10});
%inputtypeVAR10 = popupmenu

%pathnametextVAR11 = If you answered 'Yes' above, enter the path name to the folder where the output file will be saved (use "&" for default output folder). If no filename is specified, the output file defaults to ConfirmIfAllFilesPresent_output.txt.
%defaultVAR11 = &
OutputPathname = char(handles.Settings.VariableValues{CurrentModuleNum,11});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

drawnow;

% Checking the files occurs only the first time through this module.
if handles.Current.SetBeingAnalyzed ~= 1, return; end

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

    % Checks whether any files are left.
    if isempty(FileList)
        error(['Image processing was canceled in the ', ModuleName, ' module because there are no image files with the text "', TextToFind{i}, '" in the chosen directory (or subdirectories, if you requested them to be analyzed as well).'])
    end

    % Saves the File Lists and Path Names to a temporary handles structure.
    % TODO: Since LoadImages currently errors if there is a file mismatch, I'm
    % using a temporary structure for checking until this is resolved
    fieldname = ['FileList', ImageName{i}];
    tempHandles.Pipeline.(fieldname) = FileList;
    fieldname = ['Pathname', ImageName{i}];
    tempHandles.Pipeline.(fieldname) = InputPathname;
end
        
% TODO: This module requires LoadImages to work, but the information
% requested is redundant with it. If I remove the requested information and
% place it after LoadImages so the FileList can be passed to it, there are
% two problems:
% (1) If there is a file mismatch, LoadImages will fail before reaching
% this module
% (2) The text to search for is not retained in the handles structure
% Once this module is integrated into LoadImages, asking for this info
% on TextToFind and ImageName should be unneccesary. 

% ASSUMPTION: Channels are located in the same directory (i.e.
% paths are the same for all channels)

AllPathnames = cell(1,length(ImageName));
fn = fieldnames(tempHandles.Pipeline);
prefix = 'filelist';
fn = fn(strncmpi(fn,prefix,length(prefix)));
if ~iscell(fn), fn = {fn}; end
for i = 1:length(ImageName)
    % ASSUMPTION: Channels are located in the same directory (i.e.
    % paths are the same for all channels)
    AllPathnames{i} = unique(cellfun(@fileparts,tempHandles.Pipeline.(fn{i}),'UniformOutput',false)); %To be used for output later
end

% Call the main subfunction
[tempHandles,UnmatchedDirectories,DuplicateFilenames,UnmatchedFilenames] = CPconfirmallfilespresent(tempHandles,TextToFind,ImageName);

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

uniquePaths = unique(cat(2,AllPathnames{:}));

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

TextString{end+1} = '';
TextString{end+1} = 'If there are unmatched files, LoadImages will fail. Please check the above files, and remove the unmatched ones.';

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