function handles = AlgSaveImages(handles)

% Help for the Save Images module: 
% 
% This module allows you to save images to the hard drive.  The images
% to be saved can be the original images you loaded (in essence making
% CellProfiler work as a file format converter), or any of the
% processed images created by CellProfiler during the analysis.
% 
% NEED TO INCLUDE DESCRIPTION OF HOW TO SAVE NON-OBVIOUS IMAGES.

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
% The Original Code is the Save Images Module.
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

%textVAR01 = What did you call the images you want to save? 
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = Which image's original filename do you want to use as a base
%textVAR03 = to create the new file name? Type N to use sequential numbers.
%defaultVAR03 = OrigBlue
ImageFileName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});

%textVAR04 = Enter text to append to the image name, or leave "N" to keep
%textVAR05 = the name the same except for the file extension.
%defaultVAR04 = N
Appendage = char(handles.Settings.Vvariable{CurrentAlgorithmNum,4});

%textVAR06 = In what file format do you want to save images? Do not include a period
%defaultVAR06 = tif
FileFormat = char(handles.Settings.Vvariable{CurrentAlgorithmNum,6});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Retrieves the image you want to analyze and assigns it to a variable,
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
    error(['Image processing was canceled because the Save Images module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
OrigImageToBeAnalyzed = handles.(fieldname);

%%% Checks whether the file format the user entered is readable by Matlab.
IsFormat = imformats(FileFormat);
if isempty(IsFormat) == 1
    error('The image file type entered in the Save Images module is not recognized by Matlab. Or, you may have entered a period in the box. For a list of recognizable image file formats, type "imformats" (no quotes) at the command line in Matlab.','Error')
end

%%% Determines the file name.
if strcmp(upper(ImageFileName), 'N') == 1
    %%% Sets the filename to be sequential numbers.
    FileName = num2str(handles.setbeinganalyzed);
    CharFileName = char(FileName);
    BareFileName = CharFileName;
else
    %%% Checks whether the appendages to be added to the file names of images
    %%% will result in overwriting the original file, or in a file name that
    %%% contains spaces.
    %%% Determine the filename of the image to be analyzed.
    fieldname = ['dOTFilename', ImageFileName];
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
end
%%% Assembles the new image name.
if strcmp(upper(Appendage), 'N') == 1
    Appendage = [];
end
NewImageName = [BareFileName,Appendage,'.',FileFormat];
%%% Checks whether the new image name is going to result in a name with
%%% spaces.
A = isspace(Appendage);
if any(A) == 1
    error('Image processing was canceled because you have entered one or more spaces in the box of text to append to the image name in the Save Images module.')
end
%%% Checks whether the new image name is going to result in overwriting the
%%% original file.
B = strcmp(upper(CharFileName), upper(NewImageName));
if B == 1
    error('Image processing was canceled because the specifications in the Save Images module will result in image files being overwritten.')
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%

%%% Determines the figure number.
fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% The figure window is closed since there is nothing to display.
if handles.setbeinganalyzed == 1;
    delete(ThisAlgFigureNumber)
end
drawnow

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE IMAGE TO HARD DRIVE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

try 
%%% Saves the image to the hard drive.    
imwrite(OrigImageToBeAnalyzed, NewImageName, FileFormat);
catch error('The image could not be saved to the hard drive for some reason.')
end