function handles = CalculateStatistics(handles)

% Help for the Calculate Statistics module:
% Category: Other
%
% See also <nothing relevant>.

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
% $Revision: 1725 $

%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%
drawnow

%%% Reads the current module number, because this is needed to find
%%% the variable values that the user entered.
CurrentModule = handles.Current.CurrentModuleNumber;
CurrentModuleNum = str2double(CurrentModule);
ModuleName = char(handles.Settings.ModuleNames(CurrentModuleNum));

%textVAR01 = What did you call the grouping values (must be loaded by LoadText)?
%infotypeVAR01 = datagroup
%inputtypeVAR01 = popupmenu
DataName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What measure do you want to use?
%choiceVAR02 = Intensity
%inputtypeVAR02 = popupmenu
MeasureType = char(handles.Settings.VariableValues{CurrentModuleNum,2});

%textVAR03 = If using Intensity or Texture, what image did you use for calculations?
%infotypeVAR03 = imagegroup
%inputtypeVAR03 = popupmenu
ImageName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = What objects do you want to use (or Image for whole image calculations)?
%infotypeVAR04 = objectgroup
%choiceVAR04 = Image
%inputtypeVAR04 = popupmenu
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%%VariableRevisionNumber = 1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets
    GroupingStrings = handles.Measurements.(DataName);
    %%% Need column vector
    GroupingValues = str2num(char(GroupingStrings'));

    if strcmp(MeasureType,'Intensity') || strcmp(MeasureType,'Texture')
        MeasureType = [MeasureType,'_',ImageName];
    end

    fieldname = [MeasureType,'Features'];
    MeasureFeatures = handles.Measurements.(ObjectName).(fieldname);


    Ymatrix = zeros(length(handles.Current.NumberOfImageSets),length(MeasureFeatures));
    for i = 1:handles.Current.NumberOfImageSets
        for j = 1:length(MeasureFeatures)
            Ymatrix(i,j) = mean(handles.Measurements.Cells.Intensity_CropBlue{i}(:,j));
        end
    end
    [v, z] = VZfactors(GroupingValues,Ymatrix)
end
%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%

function [v, z] = VZfactors(xcol, ymatr)
% xcol is (Nobservations,1) column vector of grouping values
% (in terms of dose curve it may be Dose).
% ymatr is (Nobservations, Nmeasures) matrix, where rows correspond to observations
% and columns corresponds to different measures.
% v, z are (1, Nmeasures) row vectors containing V- and Z-factors 
% for the corresponding measures.
[xs, avers, stds] = LocShrinkMeanStd(xcol, ymatr);
range = max(avers) - min(avers);
cnstns = find(range == 0);
if (length(cnstns) > 0) range(cnstns) = 0.000001; end
vstd = mean(stds);
v = 1 - 6 .* (vstd ./ range);  
zstd = stds(1, :) + stds(length(xs), :);
z = 1 - 3 .* (zstd ./ range);  
return
% ================================== Local Functions:
% ================================== LocShrinkMeanStd
function [xs, avers, stds] = LocShrinkMeanStd(xcol, ymatr)
% 
[nrows, ncols] = size(ymatr);
[labels, labnum, xs] = LocVectorLabels(xcol);
avers = zeros(labnum, ncols);
stds = avers;
for ilab = 1 : labnum
    labinds = find(labels == ilab);
    labmatr = ymatr(labinds, :);
    avers(ilab, :) = mean(labmatr);
    stds(ilab, :) = std(labmatr, 1);
end
% ================================== LocVectorLabels
function [labels, labnum, uniqsortvals] = LocVectorLabels(x)
%
n = length(x);
labels = zeros(1, n);
[srt, inds] = sort(x); 
prev = srt(1) - 1; % absent value
labnum = 0;
uniqsortvals = labels;
for i = 1 : n
    nextval = srt(i);
    if (nextval ~= prev) % 1-st time for sure
        prev = srt(i);
        labnum = labnum + 1;
        uniqsortvals(labnum) = nextval;
    end
    labels(inds(i)) = labnum;
end
uniqsortvals = uniqsortvals(1 : labnum);
% ==================================