function handles = RenameOrRenumberFile(handles)

% Help for the RenameOrRenumberFile module:
% Category: File Processing
%
% File renaming utility that deletes or adds text anywhere
% within image file names.
% It can also serve as a File renumbering utility that converts numbers
% within image file names to solve improper ordering of files on
% Unix/Mac OSX systems.  Examples:
%
% Renumber:
% DrosDAPI_1.tif    -> DrosDAPI_001.tif
% DrosDAPI_10.tif   -> DrosDAPI_010.tif
% DrosDAPI_100.tif  -> DrosDAPI_100.tif
%
% Rename:
% DrosDAPI_1.tif    -> D_1.tif
% DrosDAPI_10.tif   -> D_10.tif
% DrosDAPI_100.tif  -> D_100.tif
%
% Be very careful since you will be renaming (=
% overwriting) your files!! You will have the opportunity to
% confirm the name change for the first image set only.  The folder
% containing the files must not contain subfolders or the subfolders
% and their contents will also be renamed.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter
%   Thouis Jones
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

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

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

%textVAR05 = How many characters would you like to allow between those two portions of filename, for renumbering purposes? Leave / to leave as is.
%defaultVAR05 = /
NumberDigits = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

Pathname = handles.Current.DefaultImageDirectory;
%%% Retrieves all the image file names and the number of
%%% images per set so they can be used by the module.
fieldname=['FileList',ImageName];
FileNames = handles.Pipeline.(fieldname);
if strcmp(class(TextToAdd), 'char') ~= 1
    try TextToAdd = num2str(TextToAdd);
    catch error('The text you tried to add could not be converted into text for some reason.')
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
        NewText = OldFilename(NumberCharactersPrefix+1:end-NumberCharactersSuffix);
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
        if strcmp(Answer, 'Cancel') == 1
            error('File renaming was canceled at your request.')
        end
    end
    if strcmp(OldFilename,NewFilename) ~= 1
        movefile(fullfile(Pathname,OldFilename),fullfile(Pathname,NewFilename))
    end
    drawnow
end

%%% This line will "cancel" processing after the first time through this
%%% module.  All the files are renumbered the first time through. Without
%%% the following cancel line, the module will run X times, where X is the
%%% number of files in the current directory.
set(handles.timertexthandle,'string','Cancel')

%%%%%%%%%%%%%%%%%%%%
%%% FIGURE WINDOW %%%
%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The figure window display is unnecessary for this module, so the figure
%%% window is closed if it was previously open.
%%% Determines the figure number.
fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
%%% If the window is open, it is closed.
if any(findobj == ThisModuleFigureNumber) == 1;
    delete(ThisModuleFigureNumber)
end