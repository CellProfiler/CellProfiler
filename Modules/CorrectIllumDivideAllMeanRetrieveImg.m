function handles = AlgCorrectIllumDivideAllMeanRetrieveImg(handles)

% Help for the Correct Illumination Divide All Mean Retrieve Image module: 
% Category: Pre-processing
%
% This module is like the Correct Illumination Divide All Mean module,
% except it allows retrieval of an already calculated illumination
% correction image, e.g. from a previously run output file.  This
% module has not been extensively tested and was written in a quick
% emergency situation.
%
% SAVING IMAGES: The illumination corrected images produced by this
% module can be easily saved using the Save Images module, using the
% name you assign. The mean image can be saved using the name
% MeanImageAD plus whatever you called the corrected image (e.g.
% MeanImageADCorrBlue). The Illumination correction image can be saved
% using the name IllumImageAD plus whatever you called the corrected
% image (e.g. IllumImageADCorrBlue).  Note that using the Save Images
% module saves a copy of the image in an image file format, which has
% lost some of the detail that a matlab file format would contain.  In
% other words, if you want to save the illumination image to use it in
% a later analysis, you should use the settings boxes within this
% module to save the illumination image in '.mat' format. If you want
% to save other intermediate images, alter the code for this module to
% save those images to the handles structure (see the section marked
% SAVE DATA TO HANDLES STRUCTURE) and then use the Save Images module.
%
% See also ALGCORRECTILLUMDIVIDEALLMEAN,
% ALGCORRECTILLUMSUBTRACTALLMIN,
% ALGCORRECTILLUMDIVIDEEACHMIN_9, ALGCORRECTILLUMDIVIDEEACHMIN_10,
% ALGCORRECTILLUMSUBTRACTEACHMIN.

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
% The Original Code is the Correct Illumination Divide All Mean Retrieve Image module.
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

%textVAR01 = What did you call the image to be corrected?
%defaultVAR01 = OrigBlue
ImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,1});

%textVAR02 = What do you want to call the corrected image?
%defaultVAR02 = CorrBlue
CorrectedImageName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,2});

%textVAR08 = To save the illum. corr. image to use later, type a file name + .mat. Else, 'N'
%defaultVAR08 = N
IllumCorrectFileName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,8});

%textVAR09 = Type the pathname and filename of the file from which you wish
%textVAR10 = to retrieve the pre-calculated illumination image:
%defaultVAR11 = /
IllumCorrectPathAndFileName = char(handles.Settings.Vvariable{CurrentAlgorithmNum,11});

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Reads (opens) the image you want to analyze and assigns it to a variable,
%%% "OrigImage".
fieldname = ['dOT', ImageName];
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles, fieldname) == 0
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Correct Illumination module could not find the input image.  It was supposed to be named ', ImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
OrigImage = handles.(fieldname);
        
%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error('Image processing was canceled because the Correct Illumination module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end
%%% Makes note of the current directory so the module can return to it
%%% at the end of this module.
CurrentDirectory = cd;

%%% The first time the module is run, retrieves the image to be used for
%%% correction from a file.
if handles.setbeinganalyzed == 1
    if strcmp(IllumCorrectPathAndFileName, '/') ~= 1
        try StructureIlluminationImage = load(IllumCorrectPathAndFileName);
           fieldname = ['dOTIllumImageAD', ImageName];
           IlluminationImage = StructureIlluminationImage.handles.(fieldname);
        catch error(['Image processing was canceled because there was a problem loading the image ', IllumCorrectPathAndFileName, '. Check that the full path and file name has been typed correctly.'])
        end
    end    
    %%% Stores the mean image and the Illumination image to the handles
    %%% structure.
    if exist('MeanImage','var') == 1
        fieldname = ['dOTMeanImageAD', CorrectedImageName];
        handles.(fieldname) = MeanImage;        
    end
    fieldname = ['dOTIllumImageAD', CorrectedImageName];
    handles.(fieldname) = IlluminationImage;
    %%% Saves the illumination correction image to the hard
    %%% drive if requested.
    if strcmp(upper(IllumCorrectFileName), 'N') == 0
        try
            save(IllumCorrectFileName, 'IlluminationImage')
        catch error(['There was a problem saving the illumination correction image to the hard drive. The attempted filename was ', IllumCorrectFileName, '.'])
        end
    end
end

%%% The following is run for every image set. Retrieves the mean image
%%% and illumination image from the handles structure.  The mean image is
%%% retrieved just for display purposes.
fieldname = ['dOTMeanImageAD', CorrectedImageName];
if isfield(handles, fieldname) == 1
    MeanImage = handles.(fieldname);
end
fieldname = ['dOTIllumImageAD', CorrectedImageName];
IlluminationImage = handles.(fieldname);
%%% Corrects the original image based on the IlluminationImage,
%%% by dividing each pixel by the value in the IlluminationImage.
CorrectedImage = OrigImage ./ IlluminationImage;

%%% Returns to the original directory.
cd(CurrentDirectory)

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Determines the figure number to display in.
fieldname = ['figurealgorithm',CurrentAlgorithm];
ThisAlgFigureNumber = handles.(fieldname);
%%% Check whether that figure is open. This checks all the figure handles
%%% for one whose handle is equal to the figure number for this algorithm.
%%% Note: Everything between the "if" and "end" is not carried out if the 
%%% user has closed the figure window, so do not do any important
%%% calculations here. Otherwise an error message will be produced if the
%%% user has closed the window but you have attempted to access data that
%%% was supposed to be produced by this part of the code.
if any(findobj == ThisAlgFigureNumber) == 1;
    %%% The "drawnow" function executes any pending figure window-related
    %%% commands.  In general, Matlab does not update figure windows
    %%% until breaks between image analysis modules, or when a few select
    %%% commands are used. "figure" and "drawnow" are two of the commands
    %%% that allow Matlab to pause and carry out any pending figure window-
    %%% related commands (like zooming, or pressing timer pause or cancel
    %%% buttons or pressing a help button.)  If the drawnow command is not
    %%% used immediately prior to the figure(ThisAlgFigureNumber) line,
    %%% then immediately after the figure line executes, the other commands
    %%% that have been waiting are executed in the other windows.  Then,
    %%% when Matlab returns to this module and goes to the subplot line,
    %%% the figure which is active is not necessarily the correct one.
    %%% This results in strange things like the subplots appearing in the
    %%% timer window or in the wrong figure window, or in help dialog boxes.
    drawnow
    %%% Activates the appropriate figure window.
    figure(ThisAlgFigureNumber);
    %%% A subplot of the figure window is set to display the original
    %%% image, some intermediate images, and the final corrected image.
    subplot(2,2,1); imagesc(OrigImage);
    title(['Input Image, Image Set # ',num2str(handles.setbeinganalyzed)]);
    %%% The mean image does not absolutely have to be present in order to
    %%% carry out the calculations if the illumination image is provided,
    %%% so the following subplot is only shown if MeanImage exists in the
    %%% workspace.
    subplot(2,2,2); imagesc(CorrectedImage); 
    title('Illumination Corrected Image');
    if exist('MeanImage','var') == 1
        subplot(2,2,3); imagesc(MeanImage); 
        title(['Mean of all ', ImageName, ' images']);
    end
    subplot(2,2,4); imagesc(IlluminationImage); 
    title('Illumination Function'); colormap(gray)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the corrected image to the handles structure so it can be used by
%%% subsequent algorithms.
fieldname = ['dOT', CorrectedImageName];
handles.(fieldname) = CorrectedImage;

%%% Determines the filename of the image to be analyzed.
fieldname = ['dOTFilename', ImageName];
FileName = handles.(fieldname)(handles.setbeinganalyzed);
%%% Saves the original file name to the handles structure in a field named
%%% after the corrected image name.
fieldname = ['dOTFilename', CorrectedImageName];
handles.(fieldname)(handles.setbeinganalyzed) = FileName;