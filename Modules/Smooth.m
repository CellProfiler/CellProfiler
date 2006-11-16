function handles = Smooth(handles)

% Help for the Smooth module:
% Category: Image Processing
%
% SHORT DESCRIPTION:
% Smooths (blurs) images.
% *************************************************************************
%
% Sorry, this module's documentation is out of date. It will be documented
% soon.
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
% divided by two to obtain the radius of a disk-shaped structuring element
% which is used for filtering. No longer done this way.
%
% See also CorrectIllumination_Apply, CorrectIllumination_Calculate.

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
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
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

%textVAR03 = Enter the smoothing method you would like to use.
%choiceVAR03 = Fit Polynomial
%choiceVAR03 = Median Filtering
%choiceVAR03 = Sum of squares
%choiceVAR03 = Square of sum
SmoothingMethod = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu

%textVAR04 = If you choose Median Filtering, Sum of squares, or Square of sum as your smoothing method, please specify the approximate width of the objects in your image (in pixels). This will be used to calculate an adequate filter size. If you don't know the width of your objects, you can use the ShowOrHidePixelData image tool to find out or leave the word 'Automatic'.
%defaultVAR04 = Automatic
ObjectWidth = handles.Settings.VariableValues{CurrentModuleNum,4};

%textVAR05 = If you want to use your own filter size (in pixels), please specify it here. Otherwise, leave '/'. If you entered a width for the previous variable, this will override it.
%defaultVAR05 = /
SizeOfSmoothingFilter = char(handles.Settings.VariableValues{CurrentModuleNum,5});

%textVAR06 = Are you using this module to smooth an image that results from processing multiple cycles?  (If so, this module will wait until it sees a flag that the other module has completed its calculations before smoothing is performed).
%choiceVAR06 = No
%choiceVAR06 = Yes
WaitForFlag = char(handles.Settings.VariableValues{CurrentModuleNum,6});
WaitForFlag = WaitForFlag(1);
%inputtypeVAR06 = popupmenu

%%%VariableRevisionNumber = 3

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
        ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
        if any(findobj == ThisModuleFigureNumber)
            CPfigure(handles,'Image',ThisModuleFigureNumber);
            title('Results will be shown after the last image cycle only if this window is left open.')
        end
        return
    elseif strcmp(ReadyFlag, 'Ready') == 1
        %%% If the smoothed image has already been calculated, the module
        %%% aborts until the next cycle. Otherwise we continue in this
        %%% module and calculate the smoothed image.
        if isfield(handles.Pipeline, SmoothedImageName) == 1
            return
        end
        %%% If we make it to this point, it is OK to proceed to calculating the smooth
        %%% image, etc.
    else error(['Image processing was canceled in the ', ModuleName, ' module because there is a programming error of some kind. The module was expecting to find the text Ready or NotReady in the field called ', fieldname, ' but that text was not matched for some reason.'])
    end
end

%%% If we make it to this point, it is OK to proceed to calculating the smooth
%%% image, etc.

%%% Some variable checking:

if strcmp(SmoothingMethod,'Median Filtering')
    SmoothingMethod = 'M';
elseif strcmp(SmoothingMethod,'Fit Polynomial')
    SmoothingMethod='P';
elseif strcmp(SmoothingMethod,'Sum of squares')
    SmoothingMethod = 'S';
elseif strcmp(SmoothingMethod,'Square of sum')
    SmoothingMethod = 'Q';
end

if ~strcmp(SizeOfSmoothingFilter,'/')
    SizeOfSmoothingFilter = str2double(SizeOfSmoothingFilter);
    if isnan(SizeOfSmoothingFilter) || (SizeOfSmoothingFilter < 1)
        if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Smoothing size invalid']))
            CPwarndlg(['The size of the smoothing filter you specified  in the ', ModuleName, ' module is invalid, it is being reset to Automatic.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Smoothing size invalid'],'replace');
        end
        SizeOfSmoothingFilter = 'A';
        WidthFlg = 0;
    else
        SizeOfSmoothingFilter = floor(SizeOfSmoothingFilter);
        WidthFlg = 0;
    end
else
    if ~strcmpi(ObjectWidth,'Automatic')
        ObjectWidth = str2double(ObjectWidth);
        if isnan(ObjectWidth) || ObjectWidth < 0
            if isempty(findobj('Tag',['Msgbox_' ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Object width invalid']))
                CPwarndlg(['The object width you specified  in the ', ModuleName, ' module is invalid, it is being reset to Automatic.'],[ModuleName ', ModuleNumber ' num2str(CurrentModuleNum) ': Object width invalid'],'replace');
            end
            SizeOfSmoothingFilter = 'A';
            WidthFlg = 0;
        else
            SizeOfSmoothingFilter = 2*floor(ObjectWidth/2);
            WidthFlg = 1;
        end
    else
        SizeOfSmoothingFilter = 'A';
        WidthFlg = 0;
    end
end

%%% Reads (opens) the image you want to analyze and assigns it to a
%%% variable.
try
    OrigImage = CPretrieveimage(handles,OrigImageName,ModuleName,'MustBeGray','CheckScale');
catch
    ErrorMessage = lasterr;
    error(['Image processing was canceled in the ' ModuleName ' module because: ' ErrorMessage(33:end)]);
end

%%%%%%%%%%%%%%%%%%%%%%
%%% IMAGE ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Smooths the OrigImage according to the user's specifications.
try
    SmoothedImage = CPsmooth(OrigImage,SmoothingMethod,SizeOfSmoothingFilter,WidthFlg);
catch
    ErrorMessage = lasterr;
    error(['Image processing was canceled in the ' ModuleName ' module becuase: ' ErrorMessage(26:end)]);
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
if any(findobj == ThisModuleFigureNumber)
    %%% Activates the appropriate figure window.
    CPfigure(handles,'Image',ThisModuleFigureNumber);
    if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
        CPresizefigure(OrigImage,'TwoByOne',ThisModuleFigureNumber)
    end
    %%% A subplot of the figure window is set to display the original
    %%% image and the smoothed image.
    subplot(2,1,1);
    CPimagesc(OrigImage,handles);
    title(['Input Image, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
    subplot(2,1,2);
    CPimagesc(SmoothedImage,handles);
    title('Smoothed Image');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% SAVE DATA TO HANDLES STRUCTURE %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Saves the processed image to the handles structure so it can be used by
%%% subsequent modules.
handles.Pipeline.(SmoothedImageName) = SmoothedImage;
