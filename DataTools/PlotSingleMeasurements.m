function handles = PlotSingleMeasurements(handles)

% Help for the Plot Single Measurements tool:
% Category: Data Tools
%
% This tool allows you to plot measurements where there is one value
% per image (like MeanAreaNuclei, or TotalIntensityBlue), using data
% in a CellProfiler output file.  As prompted, select the output file
% containing the measurements, then choose the measurement parameter
% to be displayed.
%
% POTENTIAL IMPROVEMENTS (see code for PlotOrExportHistograms to get
% started): 
% - You will then have the option of loading names
% for each image so that each histogram you make will be labeled with
% those names (if the measurement file does not already have names
% embedded).
% - Change plots/change bars buttons: These buttons allow you to change
% properties of the plots or the bars within the plots for either
% every plot in the window ('This window'), the current plot only
% ('Current'), or every plot inevery open window ('All windows').
% This include colors, axis limits and other properties.
%
% See also PLOTOREXPORTHISTOGRAMS.

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
%
% $Revision$

    %%% Restored this code, because the uigetfile function does not seem
    %%% to work properly.  It goes to the parent of the directory that was
    %%% specified.  I have asked Mathworks about this issue 3/23/05. -Anne
CurrentDir = pwd;
    cd(handles.Current.DefaultOutputDirectory)


%%% Ask the user to choose the file from which to extract
%%% measurements. The window opens in the default output directory.
% [RawFileName, RawPathname] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'*.mat'),'Select the raw measurements file');
[RawFileName, RawPathname] = uigetfile('*.mat','Select the raw measurements file');
    %%% Restored this code, because the uigetfile function does not seem
    %%% to work properly.  It goes to the parent of the directory that was
    %%% specified.  I have asked Mathworks about this issue 3/23/05. -Anne

%%% Allows canceling.
if RawFileName == 0
    return
end
load(fullfile(RawPathname, RawFileName));

Fieldnames = fieldnames(handles.Measurements);
MeasFieldnames = Fieldnames(strncmp(Fieldnames,'Image',5)==1);
%%% Error detection.
if isempty(MeasFieldnames)
    errordlg('No measurements were found in the file you selected.  They would be found within the output file''s handles.Measurements structure preceded by ''Image''.')
    return
end
%%% Removes the 'Image' prefix from each name for display purposes.
for Number = 1:length(MeasFieldnames)
    EditedMeasFieldnames{Number} = MeasFieldnames{Number}(6:end);
end
%%% Allows the user to select a measurement from the list.
[Selection, ok] = listdlg('ListString',EditedMeasFieldnames, 'ListSize', [300 600],...
    'Name','Select measurement',...
    'PromptString','Choose a measurement to display as histograms','CancelString','Cancel',...
    'SelectionMode','single');
if ok ~= 0,
    EditedMeasurementToExtract = char(EditedMeasFieldnames(Selection));
    MeasurementToExtract = ['Image', EditedMeasurementToExtract];
    figure;
    h = bar(cell2mat(handles.Measurements.(MeasurementToExtract)));
    axis tight;
    set(get(h, 'Children'), 'EdgeAlpha', 0);
    xlabel(gca,'Image set number')
    ylabel(gca,EditedMeasurementToExtract)
end

cd(CurrentDir)
    %%% Restored this code, because the uigetfile function does not seem
    %%% to work properly.  It goes to the parent of the directory that was
    %%% specified.  I have asked Mathworks about this issue 3/23/05. -Anne
