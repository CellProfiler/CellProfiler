function handles = CalculateStatistics(handles)

% Help for the Calculate Statistics module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Calculates measures of assay quality (V and Z' factors) for all
% measured features made from images.
% *************************************************************************
%
% The V and Z' factors are statistical measures of assay quality and are
% calculated for each per-cell and per-image measurement that you have made
% in the pipeline. For example, the Z' factor indicates how well-separated
% the positive and negative controls are. Calculating these values by
% placing this module at the end of a pipeline allows you to choose which
% measured features are most powerful for distinguishing positive and
% negative control samples, or for accurately quantifying the assay's
% response to dose. Both Z' and V factors will be calculated for all
% measured values (Intensity, AreaShape, Texture, etc.). These measurements
% can be exported as the "Experiment" set of data.
%
% For both Z' and V factors, the highest possible value (best assay
% quality) = 1 and they can range into negative values (for assays where
% distinguishing between positive and negative controls is difficult or
% impossible). A Z' factor > 0 is potentially screenable; A Z' factor > 0.5
% is considered an excellent assay.
%
% The Z' factor is based only on positive and negative controls. The V
% factor is based on an entire dose-response curve rather than on the
% minimum and maximum responses. When there are only two doses in the assay
% (positive and negative controls only), the V factor will equal the Z'
% factor.
%
% Note that if the standard deviation of a measured feature is zero for a
% particular set of samples (e.g. all the positive controls), the Z' and V
% factors will equal 1 despite the fact that this is not a useful feature
% for the assay. This occurs when you have only one sample at each dose.
% This also occurs for some non-informative measured features, like the
% number of Cytoplasm compartments per Cell which is always equal to 1.
%
% You must load a simple text file with one entry per cycle (using the Load
% Text module) that tells this module either which samples are positive and
% negative controls, or the concentrations of the sample-perturbing reagent
% (e.g., drug dosage). This text file would look something like this:
%
% [For the case where you have only positive or negative controls; in this
% example the first three images are negative controls and the last three
% are positive controls. They need not be labeled 0 and 1; the calculation
% is based on whichever samples have minimum and maximum dose, so it would
% work just as well to use -1 and 1, or indeed any pair of values:]
% DESCRIPTION Doses
% 0
% 0
% 0
% 1
% 1
% 1
%
% [For the case where you have samples of varying doses; using decimal
% values:]
% DESCRIPTION Doses
% .0000001
% .00000003
% .00000001
% .000000003
% .000000001
% (Note that in this examples, the Z' and V factors will be meaningless because
% there is only one sample at the each dose, so the standard deviation of
% measured features at each dose will be zero).
%
% [Another example where you have samples of varying doses; this time using
% exponential notation:]
% DESCRIPTION Doses
% 10^-7
% 10^-7.523
% 10^-8
% 10^-8.523
% 10^-9
%
%
% The reference for Z' factor is: JH Zhang, TD Chung, et al. (1999) "A
% simple statistical parameter for use in evaluation and validation of high
% throughput screening assays." J Biomolecular Screening 4(2): 67-73.
%
% The reference for V factor is: I Ravkin (2004): Poster #P12024 - Quality
% Measures for Imaging-based Cellular Assays. Society for Biomolecular
% Screening Annual Meeting Abstracts. This is likely to be published 
%
% Code for the calculation of Z' and V factors was kindly donated by Ilya
% Ravkin: http://www.ravkin.net
%
% This module currently contains code copyrighted by Carlos Evangelista.

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
% $Revision: 1725 $

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the grouping values you loaded for each image cycle? See help for details.
%infotypeVAR01 = datagroup
DataName = char(handles.Settings.VariableValues{CurrentModuleNum,1});
%inputtypeVAR01 = popupmenu

%textVAR02 = Would you like to log-transform the grouping values before attempting to fit a sigmoid curve?
%choiceVAR02 = Yes
%choiceVAR02 = No
LogOrLinear = char(handles.Settings.VariableValues{CurrentModuleNum,2});
%inputtypeVAR02 = popupmenu

%textVAR03 = To save the plotted dose response data as an interactive figure in the default output folder, enter the filename here (.fig extension will be automatically added):
%defaultVAR03 = Do not save
%infotypeVAR03 = imagegroup indep
FigureName = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%%%VariableRevisionNumber = 3

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% PRELIMINARY CALCULATIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

%%% Checks whether the user has the Image Processing Toolbox.
LicenseStats = license('test','statistics_toolbox');
if LicenseStats ~= 1
    CPwarndlg('It appears that you do not have a license for the Statistics Toolbox of Matlab.  You will be able to calculate V and Z'' factors, but not EC50 values. Typing ''ver'' or ''license'' at the Matlab command line may provide more information about your current license situation.');
end

if handles.Current.SetBeingAnalyzed == handles.Current.NumberOfImageSets

    %%% Get all fieldnames in Measurements
    ObjectFields = fieldnames(handles.Measurements);

    GroupingStrings = handles.Measurements.Image.(DataName);
    %%% Need column vector
    GroupingValues = str2num(char(GroupingStrings'));

    for i = 1:length(ObjectFields)

        ObjectName = char(ObjectFields(i));

        if strcmp(ObjectName,'Results')
            test=eps;
        end
        %%% Filter out Experiment and Image fields
        if ~strcmp(ObjectName,'Experiment')

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

                            if strcmp(MeasureFeatureName,'ModuleErrorFeatures')
                                continue;
                            end


                            %%% Get Features
                            MeasureFeatures = handles.Measurements.(ObjectName).(MeasureFeatureName);

                            %%% Get Measure name
                            MeasureName = MeasureFeatureName(1:end-8);
                            %%% Check for measurements
                            if ~isfield(handles.Measurements.(ObjectName),MeasureName)
                                error(['Image processing was canceled in the ', ModuleName, ' module because it could not find the measurements you specified.']);
                            end

                            Ymatrix = zeros(length(handles.Current.NumberOfImageSets),length(MeasureFeatures));
                            for k = 1:handles.Current.NumberOfImageSets
                                for l = 1:length(MeasureFeatures)
                                    if isempty(handles.Measurements.(ObjectName).(MeasureName){k})
                                        Ymatrix(k,l) = 0;
                                    else
                                        Ymatrix(k,l) = mean(handles.Measurements.(ObjectName).(MeasureName){k}(:,l));
                                    end
                                end
                            end

                            [GroupingValueRows,n] = size(GroupingValues);
                            [YmatrixRows, n] = size(Ymatrix);
                            if GroupingValueRows ~= YmatrixRows
                                CPwarndlg('There was an error in the Calculate Statistics module involving the number of text elements loaded for it.  CellProfiler will proceed but this module will be skipped.');
                                return;
                            else
                                [v, z] = VZfactors(GroupingValues,Ymatrix);
                                if LicenseStats == 1
                                    if ~strcmpi(FigureName,'Do not save')
                                        PartialFigureName = fullfile(handles.Current.DefaultOutputDirectory,FigureName);
                                    else PartialFigureName = FigureName;
                                    end
                                    ec50stats = CPec50(GroupingValues,Ymatrix,LogOrLinear,PartialFigureName,ModuleName);
                                    ec = ec50stats(:,3);
                                end
                            end

                            measurefield = [ObjectName,'Statistics'];
                            featuresfield = [ObjectName,'StatisticsFeatures'];
                            if isfield(handles.Measurements,'Experiment')
                                if isfield(handles.Measurements.Experiment,measurefield)
                                    OldEnd = length(handles.Measurements.Experiment.(featuresfield));
                                else OldEnd = 0;
                                end
                            else OldEnd = 0;
                            end
                            for a = 1:length(z)
                                handles.Measurements.Experiment.(measurefield){1}(1,OldEnd+a) = z(a);
                                handles.Measurements.Experiment.(measurefield){1}(1,OldEnd+length(z)+a) = v(a);
                                handles.Measurements.Experiment.(featuresfield){OldEnd+a} = ['Zfactor_',MeasureName,'_',MeasureFeatures{a}];
                                handles.Measurements.Experiment.(featuresfield){OldEnd+length(z)+a} = ['Vfactor_',MeasureName,'_',MeasureFeatures{a}];
                                if LicenseStats == 1
                                    handles.Measurements.Experiment.(measurefield){1}(1,OldEnd+2*length(z)+a) = ec(a);
                                    handles.Measurements.Experiment.(featuresfield){OldEnd+2*length(z)+a} = ['EC50_',MeasureName,'_',MeasureFeatures{a}];
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

%%% The figure window display is unnecessary for this module, so it is
%%% closed during the starting image cycle.
if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
    ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
    if any(findobj == ThisModuleFigureNumber)
        close(ThisModuleFigureNumber)
    end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% V AND Z SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
drawnow

% Code for the calculation of Z' and V factors was kindly donated by Ilya
% Ravkin: http://www.ravkin.net

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


function [xs, avers, stds] = LocShrinkMeanStd(xcol, ymatr)
ncols = size(ymatr,2);
[labels, labnum, xs] = LocVectorLabels(xcol);
avers = zeros(labnum, ncols);
stds = avers;
for ilab = 1 : labnum
    labinds = find(labels == ilab);
    labmatr = ymatr(labinds,:);
    if size(labmatr,1) == 1
        for j = 1:size(labmatr,2)
            avers(ilab,j) = labmatr(j);
        end
    else
        avers(ilab, :) = mean(labmatr);
        stds(ilab, :) = std(labmatr, 1);
    end
end


function [labels, labnum, uniqsortvals] = LocVectorLabels(x)
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


%%%%%%%%%%%%%%%%%%%%%%%%%
%%% EC50 SUBFUNCTIONS %%%
%%%%%%%%%%%%%%%%%%%%%%%%%

function results=CPec50(conc,responses,LogOrLinear,PartialFigureName,ModuleName)
% EC50 Function to fit a dose-response data to a 4 parameter dose-response
%   curve.
% 
% Requirements: nlinfit function in the Statistics Toolbox
%           and accompanying m.files: calc_init_params.m and sigmoid.m
% Inputs: 1. a 1 dimensional array of drug concentrations
%         2. the corresponding m x n array of responses 
% Algorithm: generate a set of initial coefficients including the Hill
%               coefficient
%            fit the data to the 4 parameter dose-response curve using
%               nonlinear least squares
% Output: a matrix of the 4 parameters
%         results[m,1]=min
%         results[m,2]=max
%         results[m,3]=ec50
%         results[m,4]=Hill coefficient
%
% Copyright 2004 Carlos Evangelista 
% send comments to CCEvangelista@aol.com
% Version 1.0    01/07/2004

%%% If we are using a log-domain set of doses, we have a better chance of
%%% fitting a sigmoid to the curve if the concentrations are
%%% log-transformed.
if strcmpi(LogOrLinear,'Yes')
    conc = log(conc);
end

[m,n]=size(responses);
results=zeros(n,4);
for i=1:n
    response=responses(:,i);
    initial_params=calc_init_params(conc,response);
    % OPTIONS = statset('display','iter');
    %%% Turns off MATLAB-level warnings but saves the previous warning state.
    PreviousWarningState = warning('off', 'all');
    [coeffs,r,J]=nlinfit(conc,response,'CPsigmoid',initial_params);
    nlintool(conc,response,'CPsigmoid',initial_params);
    if ~strcmpi(PartialFigureName,'Do not save')
        try
            FigureHandle = gcf;
            saveas(FigureHandle,[PartialFigureName,num2str(i),'.fig'],'fig');
        catch
            errordlg(['Image processing was NOT canceled in the ', ModuleName, ' module, but the figure could not be saved to the hard drive for some reason. Check your settings.  The error is: ', lasterr])
        end
    end

    %%% Turns MATLAB-level warnings back on.
    warning(PreviousWarningState);
    for j=1:4
        results(i,j)=coeffs(j);
    end
end



function init_params = calc_init_params(x,y)
% This generates the min, max, x value at the mid-y value, and Hill
% coefficient. These values are used for calculating dose responses using
% nlinfit.

%%% Parameters 1&2
init_params(1) = min(y);
init_params(2) = max(y);
%%% Parameter 3
% OLD:  parms(3)=(min(x)+max(x))/2;
YvalueAt50thPercentile = (min(y)+max(y))/2;
PairedValues = [y,x];
DistanceToCentralYValue = abs(PairedValues(:,1) - YvalueAt50thPercentile);
LocationOfNearest = find(DistanceToCentralYValue == min(DistanceToCentralYValue));
XvalueAt50thPercentile = PairedValues(LocationOfNearest(1),2);
init_params(3) = XvalueAt50thPercentile;
%%% Parameter 4
sizey=size(y);
sizex=size(x);
if (y(1)-y(sizey))./(x(2)-x(sizex))>0
    init_params(4)=(y(1)-y(sizey))./(x(2)-x(sizex));
else
    init_params(4)=1;
end