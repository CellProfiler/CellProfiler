function handles = AlgFileRenamer1(handles)

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

%textVAR01 = How many characters at the beginning of the file name do you want to retain?
%defaultVAR01 = 6
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
NumberCharactersPrefix = str2num(handles.(fieldname));

%textVAR02 = How many characters at the end do you want to retain, including file extension?
%defaultVAR02 = 8
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
NumberCharactersSuffix = str2num(handles.(fieldname));

%textVAR03 = Enter any text you want to place between those two portions of filename
%defaultVAR03 = /
fieldname = ['Vvariable',CurrentAlgorithm,'_03'];
TextToAdd = handles.(fieldname);
%textVAR04 = Leave "/" to not add any text.

%textVAR06 = Be very careful since you will be renaming (=overwriting) your files!!
%textVAR07 = It is recommended to test this on copies of images in a separate directory first.
%textVAR08 = The folder containing the files must not contain any subfolders or the
%textVAR09 = subfolder and its contents will also be renamed.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% The following retrieves all the image file names and the number of
%%% images per set so they can be used by the algorithm.  
FileNames = handles.Vfilenames;
if strcmp(class(TextToAdd), 'char') ~= 1
    try TextToAdd = num2str(TextToAdd);
    catch error('The text you tried to add could not be converted into text for some reason.')
    end
end

for n = 1:length(FileNames)
    OldFilename = char(FileNames(n));
    Prefix = OldFilename(1:NumberCharactersPrefix);
    Suffix = OldFilename((end-NumberCharactersSuffix+1):end);
    if strcmp(TextToAdd,'/') == 1
        NewFilename = [Prefix,Suffix];
    else
        NewFilename = [Prefix,TextToAdd,Suffix];
    end
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
drawnow
%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for File Renamer module:
%%%%% .
