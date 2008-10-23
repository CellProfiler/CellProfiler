function handles = GroupMovieFrames(handles)

% Help for the GroupMovieFrames module:
% Category: File Processing
%
% SHORT DESCRIPTION:
%
% GroupMovieFrames handle a movie to group movie frames to be processed
% within a cycle. The position of a frame within a group can be specified
% with its ImageName to be used downstream. 
%
% Each loaded movie frame will be treated as an individual image with its
% own ImageName.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003--2008.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
% VARIABLES %%%
%%%%%%%%%%%%%%%%%
[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the movie you want to extract from?
%infotypeVAR01 = imagegroup
MovieName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = How many frames should be extracted each cycle?
%defaultVAR02 = 1
NumGroupFrames = str2num(char(handles.Settings.VariableValues{CurrentModuleNum,2}));

%textVAR03 = Are the frames grouped by cycle interleaved (ABCABC...) or separated (AA..BB..CC..)?
%choiceVAR03 = Interleaved
%choiceVAR03 = Separated
Interleaved = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = What do you want to call frame 1 in each cycle (or "Do not use" to ignore)?
%defaultVAR04 = OrigDAPI
%infotypeVAR04 = imagegroup indep
ImageName{1} = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = What do you want to call frame 2 in each cycle (or "Do not use" to ignore)?
%defaultVAR05 = Do not use
%infotypeVAR05 = imagegroup indep
ImageName{2} = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = What do you want to call frame 3 in each cycle (or "Do not use"  to ignore)?
%defaultVAR06 = Do not use
%infotypeVAR06 = imagegroup indep
ImageName{3} = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = What do you want to call frame 4 in each cycle (or "Do not use"  to ignore)?
%defaultVAR07 = Do not use
%infotypeVAR07 = imagegroup indep
ImageName{4} = char(handles.Settings.VariableValues{CurrentModuleNum,7});

%textVAR08 = What do you want to call frame 5 in each cycle (or "Do not use" to ignore)?
%defaultVAR08 = Do not use
%infotypeVAR08 = imagegroup indep
ImageName{5} = char(handles.Settings.VariableValues{CurrentModuleNum,8});

%textVAR09 = What do you want to call frame 6 in each cycle (or "Do not use" to ignore)?
%defaultVAR09 = Do not use
%infotypeVAR09 = imagegroup indep
ImageName{6} = char(handles.Settings.VariableValues{CurrentModuleNum,9});


%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% GROUP LOADING FROM MOVIE %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Determines which cycle is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

% LoadImages goes through each movie file, finds how many frames
% there are, and stores an entry in handles.Pipeline.(['FileList'
% MovieName]).  It also sets NumberOfImageSets to the total number
% of frames.
%
% Reset the max number of cycles here
% Do this only once -- on the first cycle, and only for the first GroupMovieFrames instance
% 
if SetBeingAnalyzed == 1
    if (str2num(handles.Current.CurrentModuleNumber) == min(strmatch('GroupMovieFrames',handles.Settings.ModuleNames)))
        handles.Current.NumberOfImageSets = floor(handles.Current.NumberOfImageSets/NumGroupFrames);
    else
        % Check that this GroupMovieFrames will create the same NumberOfImageSets
        FileList = handles.Pipeline.(['FileList', MovieName]);
        if size(FileList,2)/NumGroupFrames ~= handles.Current.NumberOfImageSets
            CPerrordlg('There is a problem with the number of image sets.  If there are multiple GroupMovieFrames modules, please be sure that they all produce the same number of image sets after grouping.')
        end
    end
end

% Remove slashes entries with no filename from the input,
% i.e., store only valid entries
ImageName = ImageName(~ strcmp(ImageName, 'Do not use'));

FileList = handles.Pipeline.(['FileList', MovieName]);

% Find the base position of the movie for the current frames
CurrentOffset = (SetBeingAnalyzed - 1) * NumGroupFrames + 1;
CurrentMovieStart = CurrentOffset - FileList{2, CurrentOffset} + 1;
GroupNumberInMovie = (CurrentOffset - CurrentMovieStart) / NumGroupFrames;
MovieFrames = FileList{3, CurrentOffset};


for n = 1:length(ImageName) 
    if strcmp(ImageName{n}, 'Do not use'),
        continue
    end

    % This try/catch will catch any problems in the load images module.
    try
        % The following runs every time through this module (i.e. for every cycle).

        if strcmp(Interleaved, 'Interleaved'),
            Position = CurrentMovieStart + GroupNumberInMovie * NumGroupFrames + n - 1;
        else
            Position = CurrentMovieStart + (n - 1) * (MovieFrames / NumGroupFrames) + GroupNumberInMovie;
        end

        % Determines which movie to analyze.
        fieldname = ['FileList', MovieName];
        FileList = handles.Pipeline.(fieldname);
        % Determines the file name of the movie you want to analyze.
        CurrentFileName = FileList(:,Position);
        % Determines the directory to switch to.
        fieldname = ['Pathname', MovieName];
        Pathname = handles.Pipeline.(fieldname);
        fieldname = ['FileFormat', MovieName];
        FileFormat = handles.Pipeline.(fieldname);         

        if strcmpi(FileFormat,'avi movies') == 1
            % If you do not subtract 1 from the index, as specified 
            % in aviread.m, the  movie will fail to load.  However,
            % the first frame will fail if the index=0.  
            IndexLocation=(cell2mat(CurrentFileName(2)));
            NumberOfImageSets = handles.Current.NumberOfImageSets;
            if (cell2mat(CurrentFileName(2)) ~= NumberOfImageSets)
                LoadedRawImage = aviread(fullfile(Pathname, char(CurrentFileName(1))), (IndexLocation));
                LoadedImage = im2double(LoadedRawImage.cdata);
            else
                LoadedRawImage = aviread(fullfile(Pathname, char(CurrentFileName(1))), (IndexLocation-1));
                LoadedImage = im2double(LoadedRawImage.cdata);
            end
        elseif strcmpi(FileFormat,'stk movies') == 1
            LoadedRawImage = CPtiffread(fullfile(Pathname, char(CurrentFileName(1))), cell2mat(CurrentFileName(2)));
            LoadedImage = im2double(LoadedRawImage.data);
        elseif (strcmpi(FileFormat,'tif,tiff,flex movies') == 1)
            LoadedRawImage = CPimread(fullfile(Pathname, char(CurrentFileName(1))), cell2mat(CurrentFileName(2)));
            LoadedImage = im2double(LoadedRawImage);                
        end
        % Saves the original movie file name to the handles
        % structure.  The field is named appropriately based on
        % the user's input, in the Pipeline substructure so that
        % this field will be deleted at the end of the analysis
        % batch.
        fieldname = ['Filename', ImageName{n}];
        [SubdirectoryPathName,BareFileName,ext] = fileparts(char(CurrentFileName(1))); %#ok Ignore MLint
        CurrentFileNameWithFrame = [BareFileName, '_', num2str(cell2mat(CurrentFileName(2))),ext];
        CurrentFileNameWithFrame = fullfile(SubdirectoryPathName, CurrentFileNameWithFrame);
        
        % Saves the loaded image to the handles structure.  The field is named
        % appropriately based on the user's input, and put into the Pipeline
        % substructure so it will be deleted at the end of the analysis batch.
        handles.Pipeline.(fieldname)(SetBeingAnalyzed) = {CurrentFileNameWithFrame};
        handles.Pipeline.(ImageName{n}) = LoadedImage;
    catch 
        ErrorInfo = lasterror;
        ErrorInfo.message = ['Error occurred when trying to extract frame #', num2str(n),' (', ErrorInfo.message,')'];
        rethrow(ErrorInfo);
    end % Goes with: catch
    FileNames(n) = {CurrentFileNameWithFrame};
end
       
%%%%%%%%%%%%%%%%%%%
% DISPLAY RESULTS %
%%%%%%%%%%%%%%%%%%%

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure('','NarrowText',ThisModuleFigureNumber)
    end
    for n = 1:length(ImageName)
        if strcmp(ImageName{n}, 'Do not use'),
            continue
        end
        % Activates the appropriate figure window.
        currentfig = CPfigure(handles,'Text',ThisModuleFigureNumber);
        if iscell(ImageName)
            TextString = [ImageName{n},': ',FileNames{n}];
        else
            TextString = [ImageName,': ',FileNames];
        end
        uicontrol(currentfig,'style','text','units','normalized','fontsize',handles.Preferences.FontSize,'HorizontalAlignment','left','string',TextString,'position',[.05 .85-(n-1)*.15 .95 .1],'BackgroundColor',[.7 .7 .9])
    end
    drawnow
end

%%%%%%%%%%%%%%%%%%%%%%%%
% SAVE DATA TO HANDLES %
%%%%%%%%%%%%%%%%%%%%%%%%

handles = CPsaveFileNamesToHandles(handles, ImageName, Pathname, FileNames);
