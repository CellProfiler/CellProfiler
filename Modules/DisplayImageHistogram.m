function handles = DisplayImageHistogram(handles)

% Help for the Display Image Histogram module:
% Category: Other
%
% SHORT DESCRIPTION:
% Produces a histogram of the intensity of pixels within an image.
% *************************************************************************
%
% This module creates a histogram that shows the pixel intensity of the
% input image. The histogram can then be saved using the SaveImages module.
%
% Settings:
% 
% How many histograms bins would you like to use?
% Choose how many bins to use (i.e. in how many sets do you want the data
% distributed).
%
% Frequency counts:
% Frequency counts refers to the threshold for the leftmost and rightmost
% bins. The minimum value is the threshold at which any measurements less
% than this value will be combined into the leftmost bin. The maximum value
% is the threshold at which any measurements greater than or equal to this
% value will be combined into the rightmosot bin. 
%
% See also DisplayHistogram, MeasureObjectAreaShape,
% MeasureObjectIntensity, MeasureTexture, MeasureCorrelation,
% MeasureObjectNeighbors, and CalculateRatios modules.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the images whose pixel intensity histograms you want to display?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the generated histograms?
%defaultVAR02 = OrigHist
%infotypeVAR02 = imagegroup indep
HistImage = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = How many histogram bins would you like to use?
%choiceVAR03 = Automatic
%choiceVAR03 = 16
%choiceVAR03 = 2
%choiceVAR03 = 10
%choiceVAR03 = 50
%choiceVAR03 = 100
%choiceVAR03 = 256
NumBins = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu custom

%textVAR04 = Enter the range for frequency counts on the Y axis ('Min Max'):
%choiceVAR04 = Automatic
FreqRange = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu custom

%textVAR05 = Log transform the X axis of the histogram?
%choiceVAR05 = No
%choiceVAR05 = Yes
LogOption = char(handles.Settings.VariableValues{CurrentModuleNum,5});
%inputtypeVAR05 = popupmenu

%textVAR06 = Enter any optional commands or leave a period.
OptionalCmds = char(handles.Settings.VariableValues{CurrentModuleNum,6});
%defaultVAR06 = .

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Nothing really needed here

%%%%%%%%%%%%%%%%%%%%%
%%% DATA ANALYSIS %%%
%%%%%%%%%%%%%%%%%%%%%
drawnow

OrigImage=handles.Pipeline.(ImageName);

if strcmp(LogOption,'Yes')
    OrigImage(OrigImage == 0) = min(OrigImage(OrigImage > 0));
    OrigImage = log(OrigImage);
    hi = max(OrigImage(:));
    lo = min(OrigImage(:));
    OrigImage = (OrigImage - lo)/(hi - lo);
end

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
drawnow
%%% Activates the appropriate figure window.
CPfigure(handles,'Image',ThisModuleFigureNumber);

h = subplot(1,1,1);

if strcmpi(NumBins,'Automatic')
    imhist(OrigImage);
else
    imhist(OrigImage, str2double(NumBins));
end

%%% Restore window settings that are lost in call to imhist
HistHandle = CPfigure(handles,'Image',ThisModuleFigureNumber);
set(get(h,'Title'),'String',[ImageName ' pixel intensity histogram'])
set(get(h,'XLabel'),'String','Pixel Intensities')
set(get(h,'YLabel'),'String','Number of Pixels')

set(ThisModuleFigureNumber,'Name',[ModuleName ' Display, cycle # '])
drawnow

if ~strcmpi(FreqRange,'Automatic')
    try
        YRange = strread(FreqRange);
    catch
        error(['Image processing was canceled in the ', ModuleName, ' module because your entry for the range for frequency counts on the Y axis was invalid. Please follow the specified format.']);
    end
    ylim(YRange);
end

if ~strcmp(OptionalCmds, '.')
    eval(OptionalCmds);
end

%%% Store into handles structure
OneFrame = getframe(HistHandle);
handles.Pipeline.(HistImage)=OneFrame.cdata;