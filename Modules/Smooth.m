function handles = Smooth(handles)

% Help for the Smooth module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Smooths (blurs) images.
% *************************************************************************
%
% Settings:
%
% Smoothing Method:
% Note that smoothing is a time-consuming process, and fitting a polynomial
% is fastest but does not allow a very tight fit as compared to the slower
% median filtering method. Width of artifacts over ~50 take substantial
% amounts of time to process.
%
% Special note on saving images: If you want to save the smoothed image to
% use it for later analysis, you should save the smoothed image in '.mat'
% format to prevent degradation of the data.
%
% Technical note on the median filtering method: the artifact width is
% divided by two to obtain the radius of a disk shaped structuring element
% which is used for filtering. 
%
% See also Average, CorrectIllumination_Apply,
% CorrectIllumination_Calculate, CPsmooth.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the image to be smoothed?
%infotypeVAR01 = imagegroup
OrigImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the smoothed image?
%defaultVAR02 = CorrBlue
%infotypeVAR02 = imagegroup indep
SmoothedImageName = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = Smoothing method: Enter the width of the artifacts (choose an even number) that are to be smoothed out by median filtering, or choose to smooth by fitting a low order polynomial:
%choiceVAR03 = Fit Polynomial
SmoothingMethod = char(handles.Settings.VariableValues{CurrentModuleNum,3});
if strcmp(SmoothingMethod,'Fit Polynomial')
    SmoothingMethod='P';
end
%inputtypeVAR03 = popupmenu custom

%textVAR04 = Are you using this module to smooth an image that results from processing multiple cycles?  (If so, this module will wait until it sees a flag that the other module has completed its calculations before smoothing is performed).
%choiceVAR04 = Yes
%choiceVAR04 = No
WaitForFlag = char(handles.Settings.VariableValues{CurrentModuleNum,4});
WaitForFlag = WaitForFlag(1);
%inputtypeVAR04 = popupmenu

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS & FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    else error(['Image processing was canceled in the ', ModuleName, ' module because there is a programming error of some kind. The module was expecting to find the text Ready or NotReady in the field called ', fieldname, ' but that text was not matched for some reason.'])
    end
elseif strncmpi(WaitForFlag,'N',1) == 1
    %%% If we make it to this point, it is OK to proceed to calculating the smooth
    %%% image, etc.
else
    error(['Image processing was canceled in the ', ModuleName, ' module because your response to the question "Are you using this module to smooth a projection image?" was not recognized. Please enter Y or N.'])
end

%%% If we make it to this point, it is OK to proceed to calculating the smooth
%%% image, etc.

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
OrigImage = CPretrieveimage(handles,OrigImageName,ModuleName,2,1);

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Smooths the OrigImage according to the user's specifications.
SmoothedImage = CPsmooth(OrigImage,SmoothingMethod,handles.Current.SetBeingAnalyzed);

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Sets the width of the figure window to be appropriate (half width),
    %%% the first time through the set.
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet || strncmpi(WaitForFlag,'Y',1)
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
    subplot(2,1,1); CPimagesc(OrigImage);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    subplot(2,1,2); CPimagesc(SmoothedImage); title('Smoothed Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the processed image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(SmoothedImageName) = SmoothedImage;