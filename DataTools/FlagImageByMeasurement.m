function FlagImageByMeasurement(handles)

% Help for the FlagImageByMeasurement data tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
%
% *************************************************************************
%
% This data tool can take any per-image measurements in a CellProfiler output file
% and flag the measurements based on user-inputted values.
% 
%
% The new measurements will be stored under Experiment, with the name "QC"
% flag.

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

Values = QCFlag;

%Prompt what to save file as, and where to save it.
filename = '*.txt';
SavePathname = handles.Current.DefaultOutputDirectory;
[filename,SavePathname] = CPuiputfile(filename, 'Save QCFlag As...',SavePathname);
if filename == 0
    CPmsgbox('You have canceled the option to save the pipeline as a text file, but your pipeline will still be saved in .mat format.');
    return
end
fid = fopen(fullfile(SavePathname,filename),'w');
if fid == -1
    error('Cannot create the output file %s. There might be another program using a file with the same name.',filename);
end

for row = 1:size(Values, 1),
        for col = 1:size(Values, 2),
            if col ~= 1,
                fprintf(fid, '\t');
            end
            val = Values(row, col);
            if isempty(val) || (isnumeric(val) && isnan(val)),
                fprintf(fid, '');
            elseif ischar(val),
                fprintf(fid, '%s', val);
            else
                fprintf(fid, '%d', val);        
            end
        end
        fprintf(fid, '\n');
end




