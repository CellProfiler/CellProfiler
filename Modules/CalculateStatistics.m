function handles = CalculateStatistics(handles)

% Help for the Calculate Statistics module:
% Category: Other
%
% SHORT DESCRIPTION:
% Calculates the V and Z' factors for measurements made from images.
% *************************************************************************
%
% See also <nothing relevant>.

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
% $Revision: 1725 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
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

%textVAR02 = In order to run this module, you must load grouping values which correspond to 1 value per image set. All measured values (Intensity, AreaShape) will be calculated for both Z and V factors. When analysis is finished, you can export the Experiment group to see all V and Z factors in excel.

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

if handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets

    %%% Get all fieldnames in Measurements
    ObjectFields = fieldnames(handles.Measurements);

    GroupingStrings = handles.Measurements.(DataName);
    %%% Need column vector
    GroupingValues = str2num(char(GroupingStrings'));

    for i = 1:length(ObjectFields)

        ObjectName = char(ObjectFields(i));

        %%% Filter out Experiment and Image fields
        if ~strcmp(ObjectName,'Experiment') && ~strcmp(ObjectName,'Image')

            try
                %%% Get all fieldnames in Measurements.(ObjectName)
                MeasureFields = fieldnames(handles.Measurements.(ObjectName));
            catch %%% Must have been text field and ObjectName is class 'cell'
                continue
            end

            for j = 1:length(MeasureFields)

                MeasureFeatureName = char(MeasureFields(j));

                if length(MeasureFeatureName) > 7
                    if strcmp(MeasureFeatureName(end-7:end),'Features')

                        %%% Not placed with above if statement since
                        %%% MeasureFeatureName may not be 8 characters long
                        if ~strcmp(MeasureFeatureName(1:8),'Location')

                            %%% Get Features
                            MeasureFeatures = handles.Measurements.(ObjectName).(MeasureFeatureName);

                            %%% Get Measure name
                            MeasureName = MeasureFeatureName(1:end-8);
                            %%% Check for measurements
                            if ~isfield(handles.Measurements.(ObjectName),MeasureName)
                                error(['The ',ModuleName,' module could not find the measurements you specified.']);
                            end

                            Ymatrix = zeros(length(handles.Current.NumberOfImageSets),length(MeasureFeatures));
                            for k = 1:handles.Current.NumberOfImageSets
                                for l = 1:length(MeasureFeatures)
                                    Ymatrix(k,l) = mean(handles.Measurements.(ObjectName).(MeasureName){k}(:,l));
                                end
                            end

                            [v, z] = VZfactors(GroupingValues,Ymatrix);

                            measurefield = [ObjectName,'Statistics'];
                            featuresfield = [ObjectName,'StatisticsFeatures'];
                            if isfield(handles.Measurements,'Experiment')
                                if isfield(handles.Measurements.Experiment,measurefield)
                                    OldEnd = length(handles.Measurements.Experiment.(measurefield));
                                    for a = 1:length(z)
                                        handles.Measurements.Experiment.(measurefield){1}(1,OldEnd+a) = z(a);
                                        handles.Measurements.Experiment.(measurefield){1}(1,OldEnd+length(z)+a) = v(a);
                                        handles.Measurements.Experiment.(featuresfield){OldEnd+a} = ['Zfactor_',MeasureFeatures{a}];
                                        handles.Measurements.Experiment.(featuresfield){OldEnd+length(z)+a} = ['Vfactor_',MeasureFeatures{a}];
                                    end
                                else
                                    for a = 1:length(z)
                                        handles.Measurements.Experiment.(measurefield){1}(1,a) = z(a);
                                        handles.Measurements.Experiment.(measurefield){1}(1,length(z)+a) = v(a);
                                        handles.Measurements.Experiment.(featuresfield){a} = ['Zfactor_',MeasureFeatures{a}];
                                        handles.Measurements.Experiment.(featuresfield){length(z)+a} = ['Vfactor_',MeasureFeatures{a}];
                                    end
                                end
                            else
                                for a = 1:length(z)
                                    handles.Measurements.Experiment.(measurefield){1}(1,a) = z(a);
                                    handles.Measurements.Experiment.(measurefield){1}(1,length(z)+a) = v(a);
                                    handles.Measurements.Experiment.(featuresfield){a} = ['Zfactor_',MeasureFeatures{a}];
                                    handles.Measurements.Experiment.(featuresfield){length(z)+a} = ['Vfactor_',MeasureFeatures{a}];
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%
%%% DISPLAY RESULTS %%%
%%%%%%%%%%%%%%%%%%%%%%%
drawnow

fieldname = ['FigureNumberForModule',CurrentModule];
ThisModuleFigureNumber = handles.Current.(fieldname);
%%% Closes the window if it is open.
if any(findobj == ThisModuleFigureNumber) == 1;
    close(ThisModuleFigureNumber)
end

%%%%%%%%%%%%%%%%%%%%
%%% SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%
drawnow

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
if (length(cnstns) > 0)
    range(cnstns) = 0.000001;
end
vstd = mean(stds);
v = 1 - 6 .* (vstd ./ range);
zstd = stds(1, :) + stds(length(xs), :);
z = 1 - 3 .* (zstd ./ range);
return

function [xs, avers, stds] = LocShrinkMeanStd(xcol, ymatr)

ncols = size(ymatr,2);
[labels, labnum, xs] = LocVectorLabels(xcol);
avers = zeros(labnum, ncols);
stds = avers;
for ilab = 1 : labnum
    labinds = find(labels == ilab);
    labmatr = ymatr(labinds,:);
    avers(ilab, :) = mean(labmatr);
    stds(ilab, :) = std(labmatr, 1);
end

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