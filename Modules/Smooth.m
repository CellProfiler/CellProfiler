function handles = Smooth(handles)

% Help for the Smooth Image module:
% Category: Image Processing
%
% This module smooths (blurs) the incoming image.
%
% Settings:
%
% Smoothing Method:
% The smoothing can be done by fitting a low-order polynomial to the
% image (option = P), or by applying a median filter to the image
% (option = a number). In filtering mode, the user enters an even
% number for the artifact width, and this number is divided by two to
% obtain the radius of a disk shaped structuring element which is used
% for filtering. Values over ~50 take substantial amounts of time to
% process.
%
% SAVING IMAGES: The smoothed images produced by this module can be
% easily saved using the Save Images module, using the name you
% assign. If you want to save the smoothed image to use it for later
% analysis, you should save the smoothed image in '.mat' format to
% prevent degradation of the data.
%
% See also MAKEPROJECTION_AVERAGEIMAGES, CORRECTILLUMINATION_APPLY,
% CORRECTILLUMINATION_CALCULATEUSINGINTENSITIES,
% CORRECTILLUMINATION_CALCULATEUSINGBACKGROUNDINTENSITIES, CPSMOOTH.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%   Ola Friman     <friman@bwh.harvard.edu>
%   Steve Lowe     <stevelowe@alum.mit.edu>
%   Joo Han Chang  <joohan.chang@gmail.com>
%   Colin Clarke   <colinc@mit.edu>
%   Mike Lamprecht <mrl@wi.mit.edu>
%   Susan Ma       <xuefang_ma@wi.mit.edu>
%
% $Revision$

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);

%textVAR01 = What did you call the image to be smoothed?
%infotypeVAR01 = imagegroup
OrigImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the smoothed image?
%defaultVAR02 = CorrBlue
%infotypeVAR02 = imagegroup indep
SmoothedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Are you using this module to smooth an image that results from processing multiple image sets?  (If so, this module will wait until it sees a flag that the other module has completed its calculations before smoothing is performed).
%choiceVAR03 = Yes
%choiceVAR03 = No
WaitForFlag = char(handles.Settings.VariableValues{CurrentModuleNum,3});
WaitForFlag = WaitForFlag(1);
%inputtypeVAR03 = popupmenu

%textVAR04 = Smoothing method: Enter the width of the artifacts (an even number) that are to be smoothed out by median filtering, or use a low order polynomial fit.
%choiceVAR04 = Polynomial fit
SmoothingMethod = char(handles.Settings.VariableValues{CurrentModuleNum,4});
if strcmp(SmoothingMethod,'Polynomial fit')
    SmoothingMethod='P';
end
%inputtypeVAR04 = popupmenu custom

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% The following checks to see whether it is appropriate to calculate the
%%% smooth image at this time or not.  If not, the return function abandons
%%% the remainder of the code, which will otherwise calculate the smooth
%%% image, save it to the handles, and display the results.
if strncmpi(WaitForFlag,'Y',1) == 1
    fieldname = [OrigImageName,'ReadyFlag'];
    ReadyFlag = handles.Pipeline.(fieldname);
    if strcmp(ReadyFlag, 'NotReady') == 1
        %%% If the projection image is not ready, the module aborts until
        %%% the next cycle.
        return
    elseif strcmp(ReadyFlag, 'Ready') == 1
        %%% If the smoothed image has already been calculated, the module
        %%% aborts until the next cycle.
        if isfield(handles.Pipeline, SmoothedImageName) == 1
            return
        end
        %%% If we make it to this point, it is OK to proceed to calculating the smooth
        %%% image, etc.
    else error(['There is a programming error of some kind. The Smooth Image module was expecting to find the text Ready or NotReady in the field called ', fieldname, ' but that text was not matched for some reason.'])
    end
elseif strncmpi(WaitForFlag,'N',1) == 1
    %%% If we make it to this point, it is OK to proceed to calculating the smooth
    %%% image, etc.
else
    error(['Your response to the question "Are you using this module to smooth a projection image?" was not recognized. Please enter Y or N.'])
end

%%% If we make it to this point, it is OK to proceed to calculating the smooth
%%% image, etc.

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
%%% Checks whether the image to be analyzed exists in the handles structure.
if isfield(handles.Pipeline, OrigImageName)==0,
    %%% If the image is not there, an error message is produced.  The error
    %%% is not displayed: The error function halts the current function and
    %%% returns control to the calling function (the analyze all images
    %%% button callback.)  That callback recognizes that an error was
    %%% produced because of its try/catch loop and breaks out of the image
    %%% analysis loop without attempting further modules.
    error(['Image processing was canceled because the Smooth Image module could not find the input image.  It was supposed to be named ', OrigImageName, ' but an image with that name does not exist.  Perhaps there is a typo in the name.'])
end
%%% Reads the image.
OrigImage = handles.Pipeline.(OrigImageName);

if max(OrigImage(:)) > 1 || min(OrigImage(:)) < 0
    CPwarndlg('The images you have loaded are outside the 0-1 range, and you may be losing data.','Outside 0-1 Range','replace');
end

%%% Checks that the original image is two-dimensional (i.e. not a color
%%% image), which would disrupt several of the image functions.
if ndims(OrigImage) ~= 2
    error('Image processing was canceled because the Smooth Image module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
end

%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Smooths the OrigImage according to the user's specifications.
SmoothedImage = CPsmooth(OrigImage,SmoothingMethod,handles.Current.SetBeingAnalyzed);

%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
if any(findobj == ThisModuleFigureNumber) == 1;

    %%% Sets the width of the figure window to be appropriate (half width),
    %%% the first time through the set.
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet...
            | strncmpi(WaitForFlag,'Y',1) == 1
        originalsize = get(ThisModuleFigureNumber, 'position');
        newsize = originalsize;
        newsize(3) = originalsize(3)/2;
        set(ThisModuleFigureNumber, 'position', newsize);
        drawnow
    end
    drawnow
    %%% Activates the appropriate figure window.
    CPfigure(handles,ThisModuleFigureNumber);
    %%% A subplot of the figure window is set to display the original
    %%% image and the smoothed image.
    subplot(2,1,1); imagesc(OrigImage);
    
    title('Input Image');
    subplot(2,1,2); imagesc(SmoothedImage);
    
    title('Smoothed Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow



%%% Saves the corrected image to the
%%% handles structure so it can be used by subsequent modules.
handles.Pipeline.(SmoothedImageName) = SmoothedImage;
