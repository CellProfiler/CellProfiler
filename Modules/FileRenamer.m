function handles = AlgFileRenamer(handles)

% Help for the File Renamer module:
% Category: File Handling
%
% File renaming utility that deletes or adds text anywhere within
% image file names.
%
% See also ALGFILERENUMBER.

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
% The Original Code is the File Renamer module.
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

% PROGRAMMING NOTE
% HELP:
% The first unbroken block of lines will be extracted as help by
% CellProfiler's 'Help for this analysis module' button as well as
% Matlab's built in 'help' and 'doc' functions at the command line. It
% will also be used to automatically generate a manual page for the
% module. An example image demonstrating the function of the module
% can also be saved in tif format, using the same name as the
% algorithm (minus Alg), and it will automatically be included in the
% manual page as well.  Follow the convention of purpose of the
% module, description of the variables and acceptable range for each,
% how it works (technical description), info on which images can be 
% saved, and See also CAPITALLETTEROTHERALGORITHMS. The license/author
% information should be separated from the help lines with a blank
% line so that it does not show up in the help displays.

% PROGRAMMING NOTE
% DRAWNOW:
% The 'drawnow' function allows figure windows to be updated and
% buttons to be pushed (like the pause, cancel, help, and view
% buttons).  The 'drawnow' function is sprinkled throughout the code
% so there are plenty of breaks where the figure windows/buttons can
% be interacted with.  This does theoretically slow the computation
% somewhat, so it might be reasonable to remove most of these lines
% when running jobs on a cluster where speed is important.
drawnow

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

% PROGRAMMING NOTE
% VARIABLE BOXES AND TEXT: 
% The '%textVAR' lines contain the text which is displayed in the GUI
% next to each variable box. The '%defaultVAR' lines contain the
% default values which are displayed in the variable boxes when the
% user loads the algorithm. The line of code after the textVAR and
% defaultVAR extracts the value that the user has entered from the
% handles structure and saves it as a variable in the workspace of
% this algorithm with a descriptive name. The syntax is important for
% the %textVAR and %defaultVAR lines: be sure there is a space before
% and after the equals sign and also that the capitalization is as
% shown.  Don't allow the text to wrap around to another line; the
% second line will not be displayed.  If you need more space to
% describe a variable, you can refer the user to the help file, or you
% can put text in the %textVAR line above or below the one of
% interest, and do not include a %defaultVAR line so that the variable
% edit box for that variable will not be displayed; the text will
% still be displayed. CellProfiler is currently being restructured to
% handle more than 11 variable boxes. Keep in mind that you can have
% several inputs into the same box: for example, a box could be
% designed to receive two numbers separated by a comma, as long as you
% write a little extraction algorithm that separates the input into
% two distinct variables.  Any extraction algorithms like this should
% be within the VARIABLES section of the code, at the end.

%%% Reads the current algorithm number, since this is needed to find 
%%% the variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;
CurrentAlgorithmNum = str2double(handles.currentalgorithm);

%textVAR01 = How many characters at the beginning of the file name do you want to retain?
%defaultVAR01 = 6
NumberCharactersPrefix = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,1}));

%textVAR02 = How many characters at the end do you want to retain, including file extension?
%defaultVAR02 = 8
NumberCharactersSuffix = str2double(char(handles.Settings.Vvariable{CurrentAlgorithmNum,2}));

%textVAR03 = Enter any text you want to place between those two portions of filename
%textVAR04 = Leave "/" to not add any text.
%defaultVAR04 = /
TextToAdd = char(handles.Settings.Vvariable{CurrentAlgorithmNum,4});

%textVAR06 = Be very careful since you will be renaming (= overwriting) your files!!
%textVAR07 = It is recommended to test this on copies of images in a separate directory first.
%textVAR08 = The folder containing the files must not contain any subfolders or the
%textVAR09 = subfolder and its contents will also be renamed.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% Retrieves all the image file names and the number of
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

% PROGRAMMING NOTES THAT ARE UNNECESSARY FOR THIS MODULE:
% PROGRAMMING NOTE
% TO TEMPORARILY SHOW IMAGES DURING DEBUGGING: 
% figure, imshow(BlurredImage, []), title('BlurredImage') 
% TO TEMPORARILY SAVE IMAGES DURING DEBUGGING: 
% imwrite(BlurredImage, FileName, FileFormat);
% Note that you may have to alter the format of the image before
% saving.  If the image is not saved correctly, for example, try
% adding the uint8 command:
% imwrite(uint8(BlurredImage), FileName, FileFormat);

% PROGRAMMING NOTE
% DISPLAYING RESULTS:
% Each module checks whether its figure is open before calculating
% images that are for display only. This is done by examining all the
% figure handles for one whose handle is equal to the assigned figure
% number for this algorithm. If the figure is not open, everything
% between the "if" and "end" is ignored (to speed execution), so do
% not do any important calculations here. Otherwise an error message
% will be produced if the user has closed the window but you have
% attempted to access data that was supposed to be produced by this
% part of the code. If you plan to save images which are normally
% produced for display only, the corresponding lines should be moved
% outside this if statement.

% PROGRAMMING NOTE
% DRAWNOW BEFORE FIGURE COMMAND:
% The "drawnow" function executes any pending figure window-related
% commands.  In general, Matlab does not update figure windows until
% breaks between image analysis modules, or when a few select commands
% are used. "figure" and "drawnow" are two of the commands that allow
% Matlab to pause and carry out any pending figure window- related
% commands (like zooming, or pressing timer pause or cancel buttons or
% pressing a help button.)  If the drawnow command is not used
% immediately prior to the figure(ThisAlgFigureNumber) line, then
% immediately after the figure line executes, the other commands that
% have been waiting are executed in the other windows.  Then, when
% Matlab returns to this module and goes to the subplot line, the
% figure which is active is not necessarily the correct one. This
% results in strange things like the subplots appearing in the timer
% window or in the wrong figure window, or in help dialog boxes.
%
% PROGRAMMING NOTE
% HANDLES STRUCTURE:
% In CellProfiler (and Matlab in general), each independent function
% (module) has its own workspace and is not able to 'see' variables
% produced by other modules. For data or images to be shared from one
% module to the next, they must be saved to what is called the
% 'handles structure'. This is a variable of class structure which is
% called handles. Data which should be saved to the handles structure
% within each module includes: any images, data or measurements which
% are to be eventually saved to the hard drive (either in an output
% file, or using the SaveImages module) or which must be used by a
% later module in the analysis pipeline. It is important to think
% about which of these data should be deleted at the end of an
% analysis run because of the way Matlab saves variables: For example,
% a user might process 12 image sets of nuclei which results in a set
% of 12 measurements ("TotalNucArea") stored in the handles structure.
% In addition, a processed image of nuclei from the last image set is
% left in the handles structure ("SegmNucImg").  Now, if the user uses
% a different algorithm which happens to have the same measurement
% output name "TotalNucArea" to analyze 4 image sets, the 4
% measurements will overwrite the first 4 measurements of the previous
% analysis, but the remaining 8 measurements will still be present.
% So, the user will end up with 12 measurements from the 4 sets.
% Another potential problem is that if, in the second analysis run,
% the user runs only an algorithm which depends on the output
% "SegmNucImg" but does not run an algorithm that produces an image by
% that name, the algorithm will run just fine: it will just repeatedly
% use the processed image of nuclei leftover from the last image set,
% which was left in the handles structure ("SegmNucImg").
%
% INCLUDE FURTHER DESCRIPTION OF MEASUREMENTS PER CELL AND PER IMAGE
% HERE>>>
%
% The data extraction functions of CellProfiler are designed to deal
% with only one "column" of data per named measurement field. So, for
% example, instead of creating a field of XY locations stored in
% pairs, it is better to store a field of X locations and a field of Y
% locations.