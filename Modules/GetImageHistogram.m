function handles = GetImageHistogram(handles)
% Help for the Get Image Histogram module:
% Category: Other

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter
%   Thouis Jones
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
% $Revision: 2535 $

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the images you want to include?
%infotypeVAR01 = imagegroup
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = What do you want to call the generated histograms?
%defaultVAR02 = OrigHist
%infotypeVAR02 = imagegroup indep
HistImage = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = How many bins do you want?
%choiceVAR03 = Automatic
%choiceVAR03 = 2
%choiceVAR03 = 16
%choiceVAR03 = 256
NumBins = char(handles.Settings.VariableValues{CurrentModuleNum,3});
%inputtypeVAR03 = popupmenu custom

%textVAR04 = Set the range for frequency counts
%choiceVAR04 = Automatic
FreqRange = char(handles.Settings.VariableValues{CurrentModuleNum,4});
%inputtypeVAR04 = popupmenu custom

%textVAR05 = Log transform the histogram?
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

%%% Determines which cycle is being analyzed.
SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;
NumberOfImageSets = handles.Current.NumberOfImageSets;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% FIRST CYCLE FILE HANDLING %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

HistHandle = CPfigure(handles,ThisModuleFigureNumber);
if strcmp(NumBins,'Automatic')
    imhist(OrigImage);
else
    imhist(OrigImage, str2double(NumBins));
end

if ~strcmp(FreqRange,'Automatic')
    YRange = strread(FreqRange);
    ylim(YRange);
end

if ~strcmp(OptionalCmds, '.')
    eval(OptionalCmds);
end

OneFrame = getframe(HistHandle);
handles.Pipeline.(HistImage)=OneFrame.cdata;
close(ThisModuleFigureNumber);