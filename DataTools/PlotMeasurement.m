function handles = PlotMeasurement(handles)

% Help for the Plot Measurements tool:
% Category: Data Tools
%
% This tool makes a bar plot of the mean and standard deviation of a measurement. 
% As prompted, select a CellProfiler output file containing the measurements, 
% then choose the measurement parameter to be displayed.
%
% See also HISTOGRAM.

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

%%% Ask the user to choose the file from which to extract
%%% measurements. The window opens in the default output directory.
[RawFileName, RawPathname] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'*.mat'),'Select the raw measurements file');
%%% Allows canceling.
if RawFileName == 0
    return
end
load(fullfile(RawPathname, RawFileName));

%%% Call the function GetFeature(), which opens a series of list dialogs and
%%% lets the user choose a feature. The feature can be identified via 'ObjectTypename',
%%% 'FeatureType' and 'FeatureNo'.
[ObjectTypename,FeatureType,FeatureNo] = GetFeature(handles);
if isempty(ObjectTypename),return,end

%%% Extract the measurement and calculate mean and standard deviation
tmp = handles.Measurements.(ObjectTypename).(FeatureType);
MeasurementMean = zeros(length(tmp),1);
MeasurementStd = zeros(length(tmp),1);
for k = 1:length(tmp)
    if ~isempty(tmp{k})
        MeasurementsMean(k) = mean(tmp{k}(:,FeatureNo));
        MeasurementsStd(k)  = std(tmp{k}(:,FeatureNo));
    end
end

%%% Do the plotting
fig = figure;
titlestr = [handles.Measurements.(ObjectTypename).([FeatureType,'Features']){FeatureNo},' feature for ', ObjectTypename];
set(fig,'Numbertitle','off','name',['Plot Measurement: ',titlestr])
hold on
bar(MeasurementsMean);
colormap([.7 .7 .7])
shading flat
for k = 1:length(MeasurementsMean)
    plot([k k],[MeasurementsMean(k)-MeasurementsStd(k),MeasurementsMean(k)+MeasurementsStd(k)],'k','linewidth',1)
    plot([k-0.075,k+0.075],[MeasurementsMean(k)-MeasurementsStd(k),MeasurementsMean(k)-MeasurementsStd(k)],'k','linewidth',1)
    plot([k-0.075,k+0.075],[MeasurementsMean(k)+MeasurementsStd(k),MeasurementsMean(k)+MeasurementsStd(k)],'k','linewidth',1)
end
hold off
set(gca,'xtick',[1:length(MeasurementsMean)])
FontSize = get(0,'UserData');
set(gca,'fontname','times','fontsize',FontSize)
xlabel(gca,'Image set number','Fontname','times','fontsize',FontSize+2)
ylabel(gca,'Mean and standard deviation','fontname','times','fontsize',FontSize+2)
title(titlestr,'Fontname','times','fontsize',FontSize+2)
axis([0 length(MeasurementsMean)+1 ylim])




function [ObjectTypename,FeatureType,FeatureNo] = GetFeature(handles)
%
%   This function takes the user through three list dialogs where a
%   specific feature is chosen. It is possible to go back and forth
%   between the list dialogs. The chosen feature can be identified
%   via the output variables
%


%%% Extract the fieldnames of measurements from the handles structure.
MeasFieldnames = fieldnames(handles.Measurements);

% Remove the 'GeneralInfo' field
index = setdiff(1:length(MeasFieldnames),strmatch('GeneralInfo',MeasFieldnames));
MeasFieldnames = MeasFieldnames(index);

%%% Error detection.
if isempty(MeasFieldnames)
    errordlg('No measurements were found.')
    ObjectTypename = [];FeatureType = [];FeatureNo = [];
    return
end

dlgno = 1;                            % This variable keeps track of which list dialog is shown
while dlgno < 4
    switch dlgno
        case 1
            [Selection, ok] = listdlg('ListString',MeasFieldnames, 'ListSize', [300 400],...
                'Name','Select measurement',...
                'PromptString','Choose an object type',...
                'CancelString','Cancel',...
                'SelectionMode','single');
            if ok == 0
                ObjectTypename = [];FeatureType = [];FeatureNo = [];
                return
            end
            ObjectTypename = MeasFieldnames{Selection};

            % Get the feature types, remove all fields that contain
            % 'Features' in the name
            FeatureTypes = fieldnames(handles.Measurements.(ObjectTypename));
            tmp = {};
            for k = 1:length(FeatureTypes)
                if isempty(strfind(FeatureTypes{k},'Features'))
                    tmp = cat(1,tmp,FeatureTypes(k));
                end
            end
            FeatureTypes = tmp;
            dlgno = 2;                      % Indicates that the next dialog box is to be shown next
        case 2
            [Selection, ok] = listdlg('ListString',FeatureTypes, 'ListSize', [300 400],...
                'Name','Select measurement',...
                'PromptString',['Choose a feature type for ', ObjectTypename],...
                'CancelString','Back',...
                'SelectionMode','single');
            if ok == 0
                dlgno = 1;                  % Back button pressed, go back one step in the menu system
            else
                FeatureType = FeatureTypes{Selection};
                Features = handles.Measurements.(ObjectTypename).([FeatureType 'Features']);
                dlgno = 3;                  % Indicates that the next dialog box is to be shown next
            end
        case 3
            [Selection, ok] = listdlg('ListString',Features, 'ListSize', [300 400],...
                'Name','Select measurement',...
                'PromptString',['Choose a ',FeatureType,' feature for ', ObjectTypename],...
                'CancelString','Back',...
                'SelectionMode','single');
            if ok == 0
                dlgno = 2;                  % Back button pressed, go back one step in the menu system
            else
                FeatureNo = Selection;
                dlgno = 4;                  % dlgno = 4 will exit the while-loop
            end
    end
end
