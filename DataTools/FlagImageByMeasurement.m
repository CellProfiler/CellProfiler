function FlagImageByMeasurement(handles)

% Help for the FilterImageByQCMeasure data tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
%
% *************************************************************************
%
% This data tool can take any measurements in a CellProfiler output file
% and 
% 
%
% The new measurements will be stored under the first object's data, under
% the name Ratio.
%
% See also

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
% $Revision: 5776 $

[FileName, Pathname] = CPuigetfile('*.mat', 'Select the raw measurements file',handles.Current.DefaultOutputDirectory);

if FileName == 0
    return
end

% Load the specified CellProfiler output file
try
    temp = load(fullfile(Pathname, FileName));
    handles = CP_convert_old_measurements(temp.handles);
catch
    CPerrordlg(['Unable to load file ''', fullfile(Pathname, FileName), ''' (possibly not a CellProfiler output file).'])
    return
end

[ImageOrExperiment,Feature] = CPgetfeature(handles);

promptLoop = 0;
while promptLoop == 0
    Answers = inputdlg({'What is the lower limit on the feature you chose? "Do not use" to ignore', 'What is the upper limit on this feature? "Do not use" to ignore'}, 'Limits', 1, {'Do not use', 'Do not use'}); 
    LowLim = Answers{1};
    UpperLim = Answers{2};

    if isnan(str2double(LowLim)) && ~strcmpi(LowLim,'Do not use')
        uiwait(CPerrordlg('Error: there was a problem with your choice for the lower limit. Please enter a numeric value.'));
        continue
    end
    if isnan(str2double(UpperLim)) && ~strcmpi(UpperLim,'Do not use')
        uiwait(CPerrordlg('Error: there was a problem with your choice for the upper limit. Please enter a numeric value.'));
        continue
    end
    promptLoop = 1;
end


[FeatureName,CategoryAndImageName] = strtok(Feature,'_');
[CategoryName,ImageName]=strtok(CategoryAndImageName,'_');



MeasureInfo = handles.Measurements.(ImageOrExperiment).(Feature);

if strcmpi(LowLim, 'Do not use')
    LowLim = -Inf;
else
    LowLim = str2double(LowLim);
end

if strcmpi(UpperLim, 'Do not use')
    UpperLim = Inf;
else
    UpperLim = str2double(UpperLim);
end

if strcmpi(LowLim, 'Do not use') && strcmpi(UpperLim, 'Do not use')
CPwarndlg(['No images are being filtered using your current settings'])
end
   
    % Do filtering
MeasureInfo = cell2mat(MeasureInfo);

Filter = find((MeasureInfo < LowLim) | (MeasureInfo > UpperLim));
QCFlag = zeros(size(MeasureInfo));
for i = 1:numel(Filter)
    QCFlag(Filter(i)) = 1;
end


% Record the new measure in the handles structure.
    
   
    
    try
        handles = CPaddmeasurements(handles,'Experiment','QCFlag',QCFlag);    
    catch
        uiwait(CPerrordlg(['Could not add new measurements:\n' lasterr]));
        return;
    end
% Save the updated CellProfiler output file
try
    save(fullfile(Pathname, FileName),'handles');
    CPmsgbox(['Updated ',FileName,' successfully saved.']);
catch
    CPwarndlg(['Could not save updated ',FileName,' file.']);
end
