function ReplaceTextInModules

[FileName,PathName] = uigetfile('*.m', 'Choose the M-file with the old text')
TextToRemove = retrievetextfromfile([PathName,FileName])

[FileName,PathName] = uigetfile('*.m', 'Choose the M-file with the new text')
TextToAddInItsPlace = retrievetextfromfile([PathName,FileName])

%%% Retrieves every file in the current directory that begins with
%%% "Alg".
PathName = pwd;
FilesAndDirsStructure = dir(PathName);
FileAndDirNames = sortrows({FilesAndDirsStructure.name}');
LogicalIsDirectory = [FilesAndDirsStructure.isdir];
FileNamesNoDir = FileAndDirNames(~LogicalIsDirectory);
AlgorithmFileNames = FileNamesNoDir(strncmp(FileNamesNoDir,'Alg',3))
NumberOfAlgorithmFiles = size(AlgorithmFileNames,1)

%%% Determines features of the text being searched.
FirstLineOfTextToRemove = TextToRemove(1,:)
NumberOfLinesOfTextToRemove = size(TextToRemove,1)

% %%% Loops through each Algorithm.
% for i = 1:NumberOfAlgorithmFiles
%     %%% Opens each file.
%     fid=fopen(AlgorithmFileNames(i,:));
%     while 1;
%         %%% Reads each line.
%         output = fgetl(fid);
%         %%% Ignores the line if it is not text.
%         if ~ischar(output);
%             break;
%         end;
%         %%% Compares the line to desired text.
%         if strcmp(output,FirstLineOfTextToRemove) == 1;
%             %%% Extract that line plus the next N lines of text
%             %%% (DON'T KNOW HOW TO DO THIS)
%             WholeBlockOfPotentiallyMatchingText = ???;
%             for i = 1:NumberOfLinesOfTextToRemove
%                 if strcmp(TextToRemove(i,:), WholeBlockOfMatchingText(i,:))
%                     MatchingLines(i) = 1;
%                 end
%             end
%             if sum(MatchingLines) == NumberOfLinesOfTextToRemove
%                 %%% Need to remove the text here and replace it.
%                 Result = ['Replacement successful for ', AlgorithmFileNames(i,:)]
%             else
%             end
% 
%             break;
%         end
%     end
%     fclose(fid);
%     %%% IF A MATCH WAS NEVER FOUND, run the following line.
%     Result = ['Replacement FAILED for ', AlgorithmFileNames(i,:)]
% end


%%% SUBFUNCTION
ExtractedText = retrievetextfromfile(PathAndFileName)
%%% Opens the file and retrieves the TextToRemove.
fid=fopen(PathAndFileName);
while 1;
    %%% Reads each line. NEED TO DEFINE LINE NUMBER SOMEHOW.
    ExtractedText(LineNumber) = fgetl(fid);
    %%% Ignores the line if it is not text.
    if ~ischar(output);
        break;
    end;
end