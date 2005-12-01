function handles = SplitOrSpliceMovie(handles)

% Help for the Split Or Splice Movie module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Creates one large movie from several small movies, or creates several
% small movies from one large movie.
% *************************************************************************
%
% This module is only compatible with AVI format movies.
%
% Settings:
%
% For SPLICE, what is the common text in your movie file names?
% The files to be spliced should all be located within a single folder. You
% can choose a subset of movies in the folder to splice by specifying
% common text in their names. To splice all movies in the folder, you can
% just enter the file extension, avi.
%
% For SPLIT, you can split only one movie at a time, and the full file name
% should be entered here.
%
% Note: This module is run by itself in a pipeline; there is no need to use
% a Load Images or Save Images module.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision: 1725 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = Do you want to split (create multiple smaller movies from one large movie) or splice (create one large movie from multiple smaller movies)?
%choiceVAR01 = Split
%choiceVAR01 = Splice
%inputtypeVAR01 = popupmenu
SplitOrSplice = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%pathnametextVAR02 = Where are the existing avi-formatted movie files?
ExistingPath = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%pathnametextVAR03 = Where do you want to put the resulting file(s)?
FinalPath = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = For SPLICE, what is the common text in your movie files? For SPLIT, what is the entire name, including extension, of the movie file to be split?
%defaultVAR04 = GFPstain.avi
TargetMovieFileName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = For SPLIT, how many frames per movie do you want?
%defaultVAR05 = 100
FramesPerSplitMovie = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,5}));

%textVAR06 = For SPLICE, what do you want to call the final movie?
%defaultVAR06 = GFPstainSPLICED.avi
FinalSpliceName = char(handles.Settings.VariableValues{CurrentModuleNum,6});

%textVAR07 = Note: This module is run by itself in a pipeline; there is no need to use a Load Images or Save Images module.

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%
%%% FILE PROCESSING %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    close(ThisModuleFigureNumber)
end

if handles.Current.SetBeingAnalyzed == 1
    if strcmp(SplitOrSplice,'Split')
        AviMovieInfo = aviinfo(fullfile(ExistingPath,TargetMovieFileName));

        NumSplitMovies = ceil(AviMovieInfo.NumFrames/FramesPerSplitMovie);

        LastFrameRead = 0;
        for i = 1:NumSplitMovies
            [Pathname,FilenameWithoutExtension,Extension] = fileparts(fullfile(ExistingPath,TargetMovieFileName)); %#ok Ignore MLint
            NewFileAndPathName = fullfile(FinalPath, [FilenameWithoutExtension, '_', num2str(i),Extension]);
            LastFrameToReadForThisFile = min(i*FramesPerSplitMovie,AviMovieInfo.NumFrames);
            LoadedRawImages = aviread(fullfile(ExistingPath,TargetMovieFileName),LastFrameRead+1:LastFrameToReadForThisFile);
            try movie2avi(LoadedRawImages,NewFileAndPathName)
            catch error(['Image processing was canceled in the ', ModuleName, ' module because a problem was encountered during save of ',NewFileAndPathName,'.'])
                return
            end
            LastFrameRead = i*FramesPerSplitMovie;
        end
    else
        Filenames = CPretrieveMediaFileNames(ExistingPath,TargetMovieFileName,'N','E','Movie');
        %%% Checks whether any files are left.
        if isempty(Filenames)
            error(['Image processing was canceled in the ', ModuleName, ' module because there are no image files with the text "', TargetMovieFileName, '" in the chosen directory (or subdirectories, if you requested them to be analyzed as well).'])
        end

        NewFileAndPathName = fullfile(FinalPath,FinalSpliceName);
        NewAviMovie = avifile(NewFileAndPathName);
        NumMovies = length(Filenames);

        for i = 1:NumMovies
            LoadedRawImages = aviread(fullfile(ExistingPath,char(Filenames(i))));
            try NewAviMovie = addframe(NewAviMovie,LoadedRawImages);
            catch error(['Image processing was canceled in the ', ModuleName, ' module because a problem was encountered during save of ',NewFileAndPathName,'.'])
                return
            end
        end
        close(NewAviMovie)
    end
end