function handles = AlgFileRenumber2(handles)

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%

%textVAR1 = How many characters precede the image number?
%defaultVAR1 = 6
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
NumberCharactersPrefix = str2num(handles.(fieldname));

%textVAR2 = How many characters follow the image number, including file extension?
%defaultVAR2 = 8
fieldname = ['Vvariable',CurrentAlgorithm,'_02'];
NumberCharactersSuffix = str2num(handles.(fieldname));

%textVAR3 = How many total digits do you want to use for the image number?
%defaultVAR3 = 3
fieldname = ['Vvariable',CurrentAlgorithm,'_03'];
NumberDigits = str2num(handles.(fieldname));

%textVAR5 = Be very careful since you will be renaming (=overwriting) your files!!
%textVAR6 = It is recommended to test this on copies of images in a separate directory first.
%textVAR7 = The folder containing the files must not contain any subfolders or the
%textVAR8 = subfolder and its contents will also be renamed.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% The following retrieves all the image file names and the number of
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

%%%%% Help for Renumber Images module:
%%%%% .
