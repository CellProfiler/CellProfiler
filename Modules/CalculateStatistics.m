function handles = CalculateStatistics(handles)

% Help for the Calculate Statistics module:
% Category: Measurement
%
% SHORT DESCRIPTION:
% Calculates measures of assay quality (V and Z' factors) and dose response
% data (EC50) for all measured features made from images.
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
% Features measured:   Feature Number:
% Zfactor            |      1
% Vfactor            |      2
% EC50               |      3
%
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
% (Note that in this example, the Z' and V factors will be meaningless because
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
% $Revision$

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

%textVAR03 = If you want to save the plotted dose response data for each feature as an interactive figure in the default output folder, enter the filename here (.fig extension will be automatically added). Note: the figures do not stay open during processing because it tends to cause memory issues when so many windows are open. Note: This option is not compatible with running the pipeline on a cluster of computers.
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

    handles = CPcalculateStatistics(handles,DataName,LogOrLinear,FigureName,ModuleName,LicenseStats);
    
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

drawnow
