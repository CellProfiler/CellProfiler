function ReplaceTextInModules(TestMode)
%%% Normally this function is run with no arguments; dialog boxes will
%%% guide you through the process. You must prepare two .m files in
%%% advance: one containing the old text, and the other containing the
%%% new text.
%%%
%%% If you would like to run this function in test mode, where the
%%% files will not be altered but you can see which replacements will
%%% be made, run ReplaceTextInModules('test').

CurrentDirectory = pwd;
%%% Changes to the directory where CellProfiler.m resides.
FullPathAndFilename = which('CellProfiler')
[Pathname,Filename] = fileparts(FullPathAndFilename)
cd(Pathname)

[FileName,PathName] = uigetfile('*.m', 'Choose the M-file with the old text');
if FileName == 0
    return
end
TextToRemove = retrievetextfromfile([PathName,FileName])
[FileName,PathName] = uigetfile('*.m', 'Choose the M-file with the new text');
if FileName == 0
    return
end
TextToAddInItsPlace = retrievetextfromfile([PathName,FileName])
PathName = uigetdir(pwd,'Choose the folder in which you want to search and replace')
if PathName == 0
    return
end
cd(PathName)
%%% Retrieves every file in the current directory that ends in ".m"
FilesAndDirsStructure = dir(PathName);
FileAndDirNames = sortrows({FilesAndDirsStructure.name}');
LogicalIsDirectory = [FilesAndDirsStructure.isdir];
FileNamesNoDir = FileAndDirNames(~LogicalIsDirectory);
AlgorithmFileNames = cell(0);
        for i = 1:length(FileNamesNoDir),
            if strncmp(FileNamesNoDir{i}(end-1:end),'.m',2) == 1,
                AlgorithmFileNames(length(AlgorithmFileNames)+1) = {FileNamesNoDir{i}(1:end-2)};
            end
        end

NumberOfAlgorithmFiles = size(AlgorithmFileNames,2)
Answer = questdlg('Do you want to replace all instances of the text or just the first?','','All','First','Cancel','All');
if strcmp(Answer,'All') == 1
    Multiple = 1;
elseif strcmp(Answer,'First') == 1
    Multiple = 0;
else return
end
%%% Loops through each Algorithm.
for i = 1:NumberOfAlgorithmFiles
    PathFileName = fullfile(PathName,[AlgorithmFileNames{i},'.m']);
    %%% Opens each file & reads its contents as a string.
    OriginalAlgorithmContents = retrievetextfromfile(PathFileName);
    PositionsOfLocatedText = strfind(OriginalAlgorithmContents,TextToRemove);
    if isempty(PositionsOfLocatedText)==1
        %%% If a match was not found, run the following line.
        Result(i,:) = {['NO replacement for ', AlgorithmFileNames{i}]};
    else
        if Multiple == 1
            LimitToReplace = length(PositionsOfLocatedText);
        else LimitToReplace = 1;
        end
        NewAlgorithmContents = OriginalAlgorithmContents;
        for j = 1:LimitToReplace
            Number = LimitToReplace+1-j;
            PositionToReplace = PositionsOfLocatedText(Number);
            %%% Piece together the file with the beginning part and the
            %%% ending part and the text to add in the middle.
            PreReplacementText = NewAlgorithmContents(1:PositionToReplace-1);
            PostReplacementText = NewAlgorithmContents(PositionToReplace + length(TextToRemove):end);
            NewAlgorithmContents = [PreReplacementText,TextToAddInItsPlace,PostReplacementText];
        end
        if exist('TestMode') == 1
            Result(NumberOfAlgorithmFiles+1,:) = {'This is test mode only. None of the replacements were actually made'};
        else
            fid=fopen(AlgorithmFileNames{i},'w');
            fwrite(fid,NewAlgorithmContents,'char');
            fclose(fid);
        end
        Result(i,:) = {[num2str(LimitToReplace),' successful replacement(s) for ', AlgorithmFileNames{i}]};
    end
end
%%% Prints the results at the command line.
Result
cd(CurrentDirectory)

%%% SUBFUNCTION
function ExtractedText = retrievetextfromfile(PathAndFileName)
%%% Opens the file and retrieves the TextToRemove.
fid=fopen(PathAndFileName);
ExtractedText = char(fread(fid,inf,'char')');
fclose(fid);