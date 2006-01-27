function handles = RenameOrRenumberFiles(handles)

% Help for the Rename Or Renumber Files module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Renames or renumbers files on the hard drive.
% *************************************************************************
%
% File renaming utility that deletes or adds text anywhere within image
% file names. It is especially useful as a file renumbering utility that
% converts numbers within image file names to solve improper ordering of
% files on Unix/Mac OSX systems.  
%
% Be very careful since you will be renaming (= overwriting) your files!!
% You will have the opportunity to confirm the name change for the first
% cycle only. The folder containing the files must not contain subfolders
% or the subfolders and their contents will also be renamed. It is worth
% doing a practice run with copies of images first.
%
% Examples:
% 
% Renumber:
% DrosDAPI_1.tif    -> DrosDAPI_001.tif
% DrosDAPI_10.tif   -> DrosDAPI_010.tif
% DrosDAPI_100.tif  -> DrosDAPI_100.tif
% (to accomplish this, retain 4 characters at the end, retain 9 characters
% at the beginning, and use 3 numerical digits between).
%
% Rename:
% 1DrosophilaDAPI_1.tif    -> 1DrosDP_1.tif
% 2DrosophilaDAPI_10.tif   -> 2DrosDP_10.tif
% 3DrosophilaDAPI_100.tif  -> 3DrosDP_100.tif
% (to accomplish this, retain 4 characters at the end, retain 5 characters
% at the beginning, enter "DP" as text to place between, and leave
% numerical digits as is).

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
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow


[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the images you want to rename/renumber? Be very careful since you will be renaming (= overwriting) your files!! See the help for this module for other warnings.
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = How many characters at the beginning of the file name do you want to retain?
%defaultVAR02 = 6
NumberCharactersPrefix = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,2}));

%textVAR03 = How many characters at the end do you want to retain, including file extension?
%defaultVAR03 = 8
NumberCharactersSuffix = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

%textVAR04 = Enter any text you want to place between those two portions of filename. Leave "/" to leave as is.
%defaultVAR04 = /
TextToAdd = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%textVAR05 = How many numerical digits would you like to use between those two portions of filename, for renumbering purposes? Leave "/" to leave as is.
%defaultVAR05 = /
NumberDigits = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

Pathname = handles.Current.DefaultImageDirectory;
%%% Retrieves all the image file names and the number of
%%% images per set so they can be used by the module.
fieldname=['FileList',ImageName];
FileNames = handles.Pipeline.(fieldname);
if ~ischar(TextToAdd)
    try TextToAdd = num2str(TextToAdd);
    catch error(['Image processing was canceled in the ', ModuleName, ' module because the text you tried to add could not be converted into text for some reason.'])
    end
end

AmtDigits = str2double(NumberDigits);

for n = 1:length(FileNames)
    OldFilename = char(FileNames(n));
    Prefix = OldFilename(1:NumberCharactersPrefix);
    Suffix = OldFilename((end-NumberCharactersSuffix+1):end);

    %Renumbering Stage
    if ~strcmp(NumberDigits,'/')
        OldNumber = OldFilename(NumberCharactersPrefix+1:end-NumberCharactersSuffix);
        NumberOfZerosToAdd = AmtDigits - length(OldNumber);
        if NumberOfZerosToAdd < 0
            OldNumber = OldNumber(end-AmtDigits+1:end);
            NumberOfZerosToAdd=0;
        end
        ZerosToAdd = num2str(zeros(NumberOfZerosToAdd,1))';
        NewText = [ZerosToAdd,OldNumber];
    else
        NewText = '';
    end

    %Renaming Stage
    if strcmp(TextToAdd,'/')
        NewFilename = [Prefix,NewText,Suffix];
    else
        NewFilename = [Prefix,TextToAdd,NewText,Suffix];
    end

    if n == 1
        DialogText = ['Confirm the file name change. For example, the first file''s name will change from ', OldFilename, ' to ', NewFilename, '.  The remaining files will be converted without asking for confirmation.'];
        Answer = CPquestdlg(DialogText, 'Confirm file name change','OK','Cancel','Cancel');
        if strcmp(Answer, 'Cancel')
            error(['Image processing was canceled in the ', ModuleName, ' module at your request.'])
        end
    end
    if ~strcmp(OldFilename,NewFilename)
        movefile(fullfile(Pathname,OldFilename),fullfile(Pathname,NewFilename))
    end
    drawnow
end

%%% This line will "cancel" processing after the first time through this
%%% module.  All the files are renumbered the first time through. Without
%%% the following cancel line, the module will run X times, where X is the
%%% number of files in the current directory.
set(handles.timertexthandle,'string','Cancel')

%%%%%%%%%%%%%%%%%%%%%
%%% FIGURE WINDOW %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The figure window display is unnecessary for this module, so it is
%%% closed during the starting image cycle.
if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
    ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
    if any(findobj == ThisModuleFigureNumber)
        close(ThisModuleFigureNumber)
    end
end