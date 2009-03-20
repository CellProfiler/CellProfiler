function handles = RenameOrRenumberFiles(handles)

% Help for the Rename Or Renumber Files module:
% Category: File Processing
%
% SHORT DESCRIPTION:
% Renames or renumbers files on the hard drive.
% *************************************************************************
%
% This file renaming utility adjusts text within image file names. 
% Be very careful with this module because its purpose is to rename (=
% overwrite) files!! You will have the opportunity to confirm the name
% change for the first cycle only. The folder containing the files must not
% contain subfolders or the subfolders and their contents will also be
% renamed. It is worth doing a practice run with copies of images first.
%
%
% Settings:
%
% * How many characters to retain at the beginning and end of each
% filename? These are the characters that will remain unaltered and note
% that all other characters in between will be removed.
% * The user may choose to add text or numbers between
% the characters that are to be retained.
%
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
% Renumbering is especially useful when numbers within image filenames do
% not have a minimum number of digits and thus appear out of order when
% listed in some Unix/Mac OSX systems. For example, on some systems, files
% would appear like this and be measured out of expected sequence by
% CellProfiler:
% DrosDAPI_1.tif
% DrosDAPI_10.tif
% DrosDAPI_2.tif
% DrosDAPI_3.tif
% DrosDAPI_4.tif
% ...
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

% 
%
% Klmadden 2009_03_20 I think this warning should be moved to a header
% position in the variable settings:
% This module allows you to rename (overwrite) your files. Please see the help for this module for warnings.
% So then the first variable should just say:
% What did you call the images you want to rename/renumber?
% 
% See additional PyCP settings below...


%textVAR01 = What did you call the images you want to rename/renumber? Be very careful since you will be renaming (= overwriting) your files!! See the help for this module for other warnings.
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = How many characters at the beginning of the file name do you want to retain unaltered?
%defaultVAR02 = 6
NumberCharactersPrefix = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,2}));

%textVAR03 = How many characters at the end do you want to retain unaltered, including the .file extension if present?
%defaultVAR03 = 8
NumberCharactersSuffix = str2double(char(handles.Settings.VariableValues{CurrentModuleNum,3}));

% NEW PyCP setting: What do you want to do with the remaining characters (those not retained above)?
% Options: Delete them all, or Extract the numerical characters
%
% If 'extract numerical characters', setting #5 below should appear.

% NEW PyCP: Should we have the following be "Do you want to add text
% between the beginning and ending portions of filename?" and if yes then
% ask what text?  And, this variable (4) should be placed after (5) - we
% want to address the numerical characters first since we just asked about
% them in the New setting above. I think this means the code should be
% rearranged a bit because I bet right now the text is automatically placed first and the
% numbers second (which is the most useful scenario, I think). So, as I've
% rewritten the settings, rearranging 4 and 5, it appears that numbers will
% come first and text second - should we
% also offer the user the choice of putting text first and numbers second?
%
%textVAR04 = Enter any text you want to place between the beginning and ending portions of the filename. Type  "Do not use" to leave as is.
%defaultVAR04 = Do not use
TextToAdd = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%New PyCP: The text "Type Do not use to leave as is." should no longer be
%necessary (because #5 only appears if 'extract numerical chars" was
%chosen).
%
%textVAR05 = How many numerical digits would you like to use between the beginning and ending portions of the filename, for renumbering purposes? Type "Do not use" to leave as is.
%defaultVAR05 = Do not use
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
    if ~strcmp(NumberDigits,'Do not use')
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
    if strcmp(TextToAdd,'Do not use')
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
CPclosefigure(handles,CurrentModule)