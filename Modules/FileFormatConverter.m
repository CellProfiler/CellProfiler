function handles = AlgFileFormatConverter2(handles)

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%textVAR1 = What did you call the images you want to convert? 
%defaultVAR1 = Images
fieldname = ['Vvariable',CurrentAlgorithm,'_01'];
ImageName = handles.(fieldname);
%textVAR2 = Enter text to append to the image name, or leave "N" to keep
%textVAR3 = the name the same except for the file extension.
%defaultVAR3 = N
fieldname = ['Vvariable',CurrentAlgorithm,'_03'];
Appendage = handles.(fieldname);
%textVAR4 = In what file format do you want to save images? Do not include a period
%defaultVAR4 = tif
fieldname = ['Vvariable',CurrentAlgorithm,'_04'];
FileFormat = handles.(fieldname);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Checks whether the file format the user entered is readable by Matlab.
IsFormat = imformats(FileFormat);
if isempty(IsFormat) == 1
    error('The image file type entered in the File Format Converter module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.','Error')
end
%%% Read (open) the image you want to analyze and assign it to a variable,
%%% "OrigImageToBeAnalyzed".
fieldname = ['dOT', ImageName];
%%% Checks whether image has been loaded.
if isfield(handles, fieldname) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the File Format Converter module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
OrigImageToBeAnalyzed = handles.(fieldname);
        % figure, imshow(OrigImageToBeAnalyzed), title('OrigImageToBeAnalyzed')
%%% Update the handles structure.
%%% Removed for parallel: guidata(gcbo, handles);

%%% Check whether the appendages to be added to the file names of images
%%% will result in overwriting the original file, or in a file name that
%%% contains spaces.
%%% Determine the filename of the image to be analyzed.
fieldname = ['dOTFilename', ImageName];
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Find and remove the file format extension within the original file
%%% name, but only if it is at the end. Strip the original file format extension 
%%% off of the file name, if it is present, otherwise, leave the original
%%% name intact.
CharFileName = char(FileName);
PotentialDot = CharFileName(end-3:end-3);
if strcmp(PotentialDot,'.') == 1
    BareFileName = CharFileName(1:end-4);
else BareFileName = CharFileName;
end
%%% Assemble the new image name.
if strcmp(upper(Appendage), 'N') == 1
    Appendage = [];
end
NewImageName = [BareFileName,Appendage,'.',FileFormat];
%%% Check whether the new image name is going to result in a name with
%%% spaces.
A = isspace(Appendage);
if any(A) == 1
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the image name in the File Format Converter module.')
end
%%% Check whether the new image name is going to result in overwriting the
%%% original file.
B = strcmp(upper(CharFileName), upper(NewImageName));
if B == 1
    error('Image processing was canceled because the specifications in the File Format Converter module will result in image files being overwritten.')
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%

%%% Determines the figure number to display in.
fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% The figure window is closed since there is nothing to display.
if handles.setbeinganalyzed == 1;
    delete(ThisAlgFigureNumber)
end
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE PROCESSED IMAGE TO HARD DRIVE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Save the image to the hard drive.    
imwrite(OrigImageToBeAnalyzed, NewImageName, FileFormat);

%%%%%%%%%%%
%%% HELP %%%
%%%%%%%%%%%

%%%%% Help for the File Format Converter module: 
%%%%% .
%%%%% The only reason you would use the File Format Converter
%%%%% is if you want to save the images to the hard drive in a different
%%%%% format. There is no need to use it to feed images to CellProfiler:
%%%%% the Load Images modules are designed to take in any file format and
%%%%% put it into a format that CellProfiler can read.
%%%%% .
%%%%% This module must be preceded by a Load Images module.
