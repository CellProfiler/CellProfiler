function handles = AlgFileRenumber(handles)

% Help for the File Renumber module:
% Sorry, this module has not yet been documented.

% The contents of this file are subject to the Mozilla Public License Version 
% 1.1 (the "License"); you may not use this file except in compliance with 
% the License. You may obtain a copy of the License at 
% http://www.mozilla.org/MPL/
% 
% Software distributed under the License is distributed on an "AS IS" basis,
% WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
% for the specific language governing rights and limitations under the
% License.
% 
% 
% The Original Code is the the File Renumber module.
% 
% The Initial Developer of the Original Code is
% Whitehead Institute for Biomedical Research
% Portions created by the Initial Developer are Copyright (C) 2003,2004
% the Initial Developer. All Rights Reserved.
% 
% Contributor(s):
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;
CurrentAlgorithmNum = str2double(handles.currentalgorithm);

%textVAR01 = How many characters precede the image number?
%defaultVAR01 = 6
NumberCharactersPrefix = str2num(char(handles.Settings.Vvariable{CurrentAlgorithmNum,1}));

%textVAR02 = How many characters follow the image number, including file extension?
%defaultVAR02 = 8
NumberCharactersSuffix = str2num(char(handles.Settings.Vvariable{CurrentAlgorithmNum,2}));

%textVAR03 = How many total digits do you want to use for the image number?
%defaultVAR03 = 3
NumberDigits = str2num(char(handles.Settings.Vvariable{CurrentAlgorithmNum,3}));

%textVAR05 = Be very careful since you will be renaming (= overwriting) your files!!
%textVAR06 = It is recommended to test this on copies of images in a separate directory first.
%textVAR07 = The folder containing the files must not contain any subfolders or the
%textVAR08 = subfolder and its contents will also be renamed.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Retrieves all the image file names and the number of
%%% images per set so they can be used by the algorithm.  
FileNames = handles.Vfilenames;
for n = 1:length(FileNames)
    OldFilename = char(FileNames(n));
    Prefix = OldFilename(1:NumberCharactersPrefix);
    Suffix = OldFilename((end-NumberCharactersSuffix+1):end);
    OldNumber = OldFilename(NumberCharactersPrefix+1:end-NumberCharactersSuffix);
    NumberOfZerosToAdd = NumberDigits - length(OldNumber);
    ZerosToAdd = num2str(zeros(NumberOfZerosToAdd,1))';
    NewNumber = [ZerosToAdd,OldNumber];
    NewFilename = [Prefix,NewNumber,Suffix];
    if n == 1
        DialogText = ['Confirm the file name change. For example, the first file''s name will change from ', OldFilename, ' to ', NewFilename, '.'];
        Answer = questdlg(DialogText, 'Confirm file name change','OK','Cancel','Cancel');
        if strcmp(Answer, 'Cancel') == 1
            error('File renumbering was canceled at your request.')
        end
    end
    if strcmp(OldFilename,NewFilename) ~= 1
        movefile(OldFilename,NewFilename) 
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

%%% The figure window display is unnecessary for this module, so the figure
%%% window is closed if it was previously open.
%%% Determines the figure number.
fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% If the window is open, it is closed.
if any(findobj == ThisAlgFigureNumber) == 1;
    delete(ThisAlgFigureNumber)
end