function handles = AlgMeasureCorrelation(handles)

% Help for the Measure Correlation module:
% 
% Given two or more images, calculates the correlation between the
% pixel intensities. The correlation can be measured for the entire
% images, or individual correlation measurements can be made for each
% individual object, as defined by another module.
%
% See also ALGMEASUREAREAOCCUPIED,
% ALGMEASUREAREASHAPEINTENSTXTR,
% ALGMEASUREINTENSITYTEXTURE,
% ALGMEASURETOTALINTENSITY.

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
% The Original Code is the Measure Correlation Module.
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

%%% Reads the current algorithm number, since this is needed to find the
%%% variable values that the user entered.
CurrentAlgorithm = handles.currentalgorithm;
CurrentAlgorithmNum = str2double(handles.currentalgorithm);

%textVAR01 = Enter the names of each image type to be compared. If a box is unused, leave "/"
%defaultVAR01 = OrigBlue
Image1Name = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = All pairwise comparisons will be performed.
%defaultVAR02 = OrigGreen
Image2Name = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR03 = 
%defaultVAR03 = OrigRed
Image3Name = char(handles.Settings.Vvariable{CurrentAlgorithmNum,3});

%textVAR04 = 
%defaultVAR04 = /
Image4Name = char(handles.Settings.Vvariable{CurrentAlgorithmNum,4});

%textVAR05 = 
%defaultVAR05 = /
Image5Name = char(handles.Settings.Vvariable{CurrentAlgorithmNum,5});

%textVAR06 = 
%defaultVAR06 = /
Image6Name = char(handles.Settings.Vvariable{CurrentAlgorithmNum,6});

%textVAR07 = 
%defaultVAR07 = /
Image7Name = char(handles.Settings.Vvariable{CurrentAlgorithmNum,7});

%textVAR08 = 
%defaultVAR08 = /
Image8Name = char(handles.Settings.Vvariable{CurrentAlgorithmNum,8});

%textVAR09 = 
%defaultVAR09 = /
Image9Name = char(handles.Settings.Vvariable{CurrentAlgorithmNum,9});

%textVAR10 = What did you call the objects within which to compare the images?
%textVAR11 = Leave "/" to compare the entire images
%defaultVAR10 = /
ObjectName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,10});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if strcmp(Image1Name,'/') ~= 1
    try
        %%% Reads (opens) the image you want to analyze and assigns it to a variable.
        fieldname = ['dOT', Image1Name];
        %%% Checks whether image has been loaded.
        if isfield(handles, fieldname) == 0
            %%% If the image is not there, an error message is produced.  The error
            %%% is not displayed: The error function halts the current function and
            %%% returns control to the calling function (the analyze all images
            %%% button callback.)  That callback recognizes that an error was
            %%% produced because of its try/catch loop and breaks out of the image
            %%% analysis loop without attempting further modules.
            error(['Image processing was canceled because the Measure Correlation module could not find the input image.  It was supposed to be named ', Image1Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
        end
        Image1 = handles.(fieldname);
        %%% Checks that the original image is two-dimensional (i.e. not a color
        %%% image), which would disrupt several of the image functions.
        if ndims(Image1) ~= 2
            error('Image processing was canceled because the Measure Correlation module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
        end
    catch error(['There was a problem loading the image you called ', Image1Name, ' in the Measure Correlation module.'])
    end
end
%%% Repeat for the rest of the images.
if strcmp(Image2Name,'/') ~= 1
    try
        fieldname = ['dOT', Image2Name];
        if isfield(handles, fieldname) == 0
            error(['Image processing was canceled because the Measure Correlation module could not find the input image.  It was supposed to be named ', Image2Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
        end
        Image2 = handles.(fieldname);
        if ndims(Image2) ~= 2
            error('Image processing was canceled because the Measure Correlation module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
        end
    catch error(['There was a problem loading the image you called ', Image2Name, ' in the Measure Correlation module.'])
    end
end
if strcmp(Image3Name,'/') ~= 1
    try
        fieldname = ['dOT', Image3Name];
        if isfield(handles, fieldname) == 0
            error(['Image processing was canceled because the Measure Correlation module could not find the input image.  It was supposed to be named ', Image3Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
        end
        Image3 = handles.(fieldname);
        if ndims(Image3) ~= 2
            error('Image processing was canceled because the Measure Correlation module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
        end
    catch error(['There was a problem loading the image you called ', Image3Name, ' in the Measure Correlation module.'])
    end
end
if strcmp(Image4Name,'/') ~= 1
    try
        fieldname = ['dOT', Image4Name];
        if isfield(handles, fieldname) == 0
            error(['Image processing was canceled because the Measure Correlation module could not find the input image.  It was supposed to be named ', Image4Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
        end
        Image4 = handles.(fieldname);
        if ndims(Image4) ~= 2
            error('Image processing was canceled because the Measure Correlation module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
        end
    catch error(['There was a problem loading the image you called ', Image4Name, ' in the Measure Correlation module.'])
    end
end
if strcmp(Image5Name,'/') ~= 1
    try
        fieldname = ['dOT', Image5Name];
        if isfield(handles, fieldname) == 0
            error(['Image processing was canceled because the Measure Correlation module could not find the input image.  It was supposed to be named ', Image5Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
        end
        Image5 = handles.(fieldname);
        if ndims(Image5) ~= 2
            error('Image processing was canceled because the Measure Correlation module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
        end
    catch error(['There was a problem loading the image you called ', Image5Name, ' in the Measure Correlation module.'])
    end
end
if strcmp(Image6Name,'/') ~= 1
    try
        fieldname = ['dOT', Image6Name];
        if isfield(handles, fieldname) == 0
            error(['Image processing was canceled because the Measure Correlation module could not find the input image.  It was supposed to be named ', Image6Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
        end
        Image6 = handles.(fieldname);
        if ndims(Image6) ~= 2
            error('Image processing was canceled because the Measure Correlation module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
        end
    catch error(['There was a problem loading the image you called ', Image6Name, ' in the Measure Correlation module.'])
    end
end
if strcmp(Image7Name,'/') ~= 1
    try
        fieldname = ['dOT', Image7Name];
        if isfield(handles, fieldname) == 0
            error(['Image processing was canceled because the Measure Correlation module could not find the input image.  It was supposed to be named ', Image7Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
        end
        Image7 = handles.(fieldname);
        if ndims(Image7) ~= 2
            error('Image processing was canceled because the Measure Correlation module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
        end
    catch error(['There was a problem loading the image you called ', Image7Name, ' in the Measure Correlation module.'])
    end
end
if strcmp(Image8Name,'/') ~= 1
    try
        fieldname = ['dOT', Image8Name];
        if isfield(handles, fieldname) == 0
            error(['Image processing was canceled because the Measure Correlation module could not find the input image.  It was supposed to be named ', Image8Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
        end
        Image8 = handles.(fieldname);
        if ndims(Image8) ~= 2
            error('Image processing was canceled because the Measure Correlation module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
        end
    catch error(['There was a problem loading the image you called ', Image8Name, ' in the Measure Correlation module.'])
    end
end
if strcmp(Image9Name,'/') ~= 1
    try
        fieldname = ['dOT', Image9Name];
        if isfield(handles, fieldname) == 0
            error(['Image processing was canceled because the Measure Correlation module could not find the input image.  It was supposed to be named ', Image9Name, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
        end
        Image9 = handles.(fieldname);
        if ndims(Image9) ~= 2
            error('Image processing was canceled because the Measure Correlation module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
        end
    catch error(['There was a problem loading the image you called ', Image9Name, ' in the Measure Correlation module.'])
    end
end
if strcmp(ObjectName,'/') ~= 1
    %%% Retrieves the label matrix image that contains the 
    %%% segmented objects which will be used as a mask. Checks first to see
    %%% whether the appropriate image exists.
    fieldname = ['dOTSegmented',ObjectName];
    %%% Checks whether the image exists in the handles structure.
    if isfield(handles, fieldname) == 0
        error(['Image processing has been canceled. Prior to running the Measure Correlation module, you must have previously run an algorithm that generates an image with the primary objects identified.  You specified in the Measure Correlation module that the objects were named ', ObjectName, ' as a result of a previous algorithm, which should have produced an image called ', fieldname, ' in the handles structure.  The Measure Correlation module cannot locate this image.']);
    end
    MaskLabelMatrixImage = handles.(fieldname);
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Starts out with empty variables.
ImageMatrix = [];
ImageNames = [];
%%% For each image, reshapes the image into a column of numbers, then
%%% places it as a column into the variable ImageMatrix.  Adds its name
%%% to the list of ImageNames, too.
if strcmp(Image1Name,'/') ~= 1
Image1Column = reshape(Image1,[],1);
     % figure, imshow(Image1Column), title('Image1Column'), colormap(gray)
ImageMatrix = horzcat(ImageMatrix,Image1Column);
ImageNames = strvcat(ImageNames,Image1Name); %#ok We want to ignore MLint error checking for this line.
end
if strcmp(Image2Name,'/') ~= 1
Image2Column = reshape(Image2,[],1);
ImageMatrix = horzcat(ImageMatrix,Image2Column);
ImageNames = strvcat(ImageNames,Image2Name); %#ok We want to ignore MLint error checking for this line.
end
if strcmp(Image3Name,'/') ~= 1
Image3Column = reshape(Image3,[],1);
ImageMatrix = horzcat(ImageMatrix,Image3Column);
ImageNames = strvcat(ImageNames,Image3Name); %#ok We want to ignore MLint error checking for this line.
end
if strcmp(Image4Name,'/') ~= 1
Image4Column = reshape(Image4,[],1);
ImageMatrix = horzcat(ImageMatrix,Image4Column);
ImageNames = strvcat(ImageNames,Image4Name); %#ok We want to ignore MLint error checking for this line.
end
if strcmp(Image5Name,'/') ~= 1
Image5Column = reshape(Image5,[],1);
ImageMatrix = horzcat(ImageMatrix,Image5Column);
ImageNames = strvcat(ImageNames,Image5Name); %#ok We want to ignore MLint error checking for this line.
end
if strcmp(Image6Name,'/') ~= 1
Image6Column = reshape(Image6,[],1);
ImageMatrix = horzcat(ImageMatrix,Image6Column);
ImageNames = strvcat(ImageNames,Image6Name); %#ok We want to ignore MLint error checking for this line.
end
if strcmp(Image7Name,'/') ~= 1
Image7Column = reshape(Image7,[],1);
ImageMatrix = horzcat(ImageMatrix,Image7Column);
ImageNames = strvcat(ImageNames,Image7Name); %#ok We want to ignore MLint error checking for this line.
end
if strcmp(Image8Name,'/') ~= 1
Image8Column = reshape(Image8,[],1);
ImageMatrix = horzcat(ImageMatrix,Image8Column);
ImageNames = strvcat(ImageNames,Image8Name); %#ok We want to ignore MLint error checking for this line.
end
if strcmp(Image9Name,'/') ~= 1
Image9Column = reshape(Image9,[],1);
ImageMatrix = horzcat(ImageMatrix,Image9Column);
ImageNames = strvcat(ImageNames,Image9Name); %#ok We want to ignore MLint error checking for this line.
end
%%% Applies the mask, if requested
if strcmp(ObjectName,'/') ~= 1
    %%% Turns the image with labeled objects into a binary image in the shape of
    %%% a column.
    MaskLabelMatrixImageColumn = reshape(MaskLabelMatrixImage,[],1);
    MaskBinaryImageColumn = MaskLabelMatrixImageColumn>0;
    %%% Yields the locations of nonzero pixels.
    ObjectLocations = find(MaskBinaryImageColumn);
    %%% Removes the non-object pixels from the image matrix.
    ObjectImageMatrix = ImageMatrix(ObjectLocations,:);
    %%% Calculates the correlation coefficient.
    Results = corrcoef(ObjectImageMatrix);
else
    %%% Calculates the correlation coefficient.
    Results = corrcoef(ImageMatrix);
end

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the figure number to display in.
fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% Checks whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this algorithm.
%%% Note: Everything between the "if" and "end" is not carried out if the
%%% user has closed the figure window, so do not do any important
%%% calculations here. Otherwise an error message will be produced if the
%%% user has closed the window but you have attempted to access data that
%%% was supposed to be produced by this part of the code.
if any(findobj == ThisAlgFigureNumber) == 1;
    %%% The "drawnow" function executes any pending figure window-related
    %%% commands.  In general, Matlab does not update figure windows until
    %%% breaks between image analysis modules, or when a few select
    %%% commands are used. "figure" and "drawnow" are two of the commands
    %%% that allow Matlab to pause and carry out any pending figure window-
    %%% related commands (like zooming, or pressing timer pause or cancel
    %%% buttons or pressing a help button.)  If the drawnow command is not
    %%% used immediately prior to the figure(ThisAlgFigureNumber) line,
    %%% then immediately after the figure line executes, the other commands
    %%% that have been waiting are executed in the other windows.  Then,
    %%% when Matlab returns to this module and goes to the subplot line,
    %%% the figure which is active is not necessarily the correct one. This
    %%% results in strange things like the subplots appearing in the timer
    %%% window or in the wrong figure window, or in help dialog boxes.
    drawnow
    if handles.setbeinganalyzed == 1;
        %%% Sets the width of the figure window to be appropriate (half width).
        originalsize = get(ThisAlgFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = 350;
        set(ThisAlgFigureNumber, 'position', newsize);
    end
    %%% Activates the appropriate figure window.
    figure(ThisAlgFigureNumber);
    %%% Displays the results.
    Displaytexthandle = uicontrol(ThisAlgFigureNumber,'style','text', 'position', [0 0 335 400],'fontname','fixedwidth','backgroundcolor',[0.7,0.7,0.7]);
    TextToDisplay = ['Image Set # ',num2str(handles.setbeinganalyzed)];
    for i = 1:size(ImageNames,1)-1
        for j = i+1:size(ImageNames,1)
            Value = num2str(Results(i,j));
            TextToDisplay = strvcat(TextToDisplay, [ImageNames(i,:),'/', ImageNames(j,:),' Correlation: ',Value]); %#ok We want to ignore MLint error checking for this line.
        end
    end
    set(Displaytexthandle,'string',TextToDisplay)
    set(ThisAlgFigureNumber,'toolbar','figure')
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow
if strcmp(ObjectName,'/') == 1
ObjectName = 'overall';
else ObjectName = ['within',ObjectName];
end

for i = 1:size(ImageNames,1)-1
    for j = i+1:size(ImageNames,1)
        Value = num2str(Results(i,j));
        HeadingName = [char(cellstr(ImageNames(i,:))),'_', char(cellstr(ImageNames(j,:)))];
        fieldname = ['dMTCorrelation', HeadingName, ObjectName];
        handles.(fieldname)(handles.setbeinganalyzed) = {Value};
    end
end