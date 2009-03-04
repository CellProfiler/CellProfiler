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

dlgno = 1;                            % This variable keeps track of which list dialog is shown

while dlgno < 5
    switch dlgno
        case 1
            Input1 = {'Creating a new QCFlag','Appending an existing QCFlag'};
            [selection,ok] = CPlistdlg('ListString',Input1,'PromptString','What are you doing?');
            if ok == 0
                dlgno = 1;
            elseif strcmp(Input1{selection},'Creating a new QCFlag')
                dlgno = 2;
            elseif strcmp(Input1{selection}, 'Appending an existing QCFlag')
                dlgno = 3;
            end
        case 2
            Answers1 = inputdlg({'What do you what want to call this new QCFlag?'});
            FlagNameNew = Answers1{1};
            dlgno = 4;
        case 3
            Answers2 = inputdlg({'What did you call the existing QCFlag you would like to append?'});
            FlagNameOld = Answers2{1};
            dlgno = 4;
        case 4
            Answers3 = inputdlg({'What is the lower limit on the feature you chose? "Do not use" to ignore', 'What is the upper limit on this feature? "Do not use" to ignore'}, 'Limits', 1, {'Do not use', 'Do not use'});
            LowLim = Answers3{1};
            UpperLim = Answers3{2};

            if isnan(str2double(LowLim)) && ~strcmpi(LowLim,'Do not use')
                uiwait(CPerrordlg('Error: there was a problem with your choice for the lower limit. Please enter a numeric value.'));
                continue
            end
            if isnan(str2double(UpperLim)) && ~strcmpi(UpperLim,'Do not use')
                uiwait(CPerrordlg('Error: there was a problem with your choice for the upper limit. Please enter a numeric value.'));
                continue
            end
            dlgno = 5;
    end

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
%QCFlag = cell(size(MeasureInfo));
if strcmpi(Input1{selection}, 'Appending an existing QCFlag')
    FlagNameOld = CPjoinstrings('QCFlag',FlagNameOld);
else
    FlagNameNew = CPjoinstrings('QCFlag',FlagNameNew);
end
for i = 1:handles.Current.NumberOfImageSets
    if (MeasureInfo{i} < LowLim) || (MeasureInfo{i} > UpperLim)
        QCFlag{i} = 1;
    else QCFlag{i} = 0;
    end
    % Record the new measure in the handles structure.
    if strcmpi(Input1{selection}, 'Appending an existing QCFlag')
        try
           FlagToAppend = handles.Measurements.Image.(FlagNameOld);
        catch
           error([lasterrr 'CellProfiler could not find the QCFlag you asked to append.']);
        end
        try
            QCFlag{i}  = QCFlag{i} + FlagToAppend{i};
        catch
            error([lasterr 'CellProfiler could not append the QCFlag you specified. Perhaps it is not the same size as the one you are trying to append it with?']);
        end
        if QCFlag{i} == 2
            QCFlag{i} = 1;
        end
    end

    if strcmpi(Input1{selection}, 'Appending an existing QCFlag')
        handles.Measurements.Image.(FlagNameOld){i} = QCFlag{i};
    elseif strcmpi(Input1{selection},'Creating a new QCFlag')
        handles = CPaddmeasurements(handles,'Image',FlagNameNew,QCFlag{i},i);
    end

end


 
% Save the updated CellProfiler output file
try
    save(fullfile(Pathname, FileName),'handles');
    CPmsgbox(['Updated ',FileName,' successfully saved.']);
catch
    CPwarndlg(['Could not save updated ',FileName,' file.']);
end

Values = cell2mat(QCFlag);

%Prompt what to save file as, and where to save it.
filename = '*.txt';
SavePathname = handles.Current.DefaultOutputDirectory;
[filename,SavePathname] = CPuiputfile(filename, 'Save QCFlag As...',SavePathname);
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




