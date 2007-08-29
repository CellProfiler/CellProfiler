function CalculateRatiosDataTool(handles)

% Help for the Calculate Ratios data tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Calculates the product, ratio, sum, or difference between any
% measurements already measured (e.g. Intensity of green staining in
% cytoplasm/Area of cells)
% *************************************************************************
%
% This data tool can take any measurements in a CellProfiler output file
% and multiply, divide, add, or subtract them. Resulting measurements can
% also be saved and used to calculate other measurements.
%
% The data tool currently works on an object-by-object basis (it calculates
% the ratio for each object). If you need to calculate image-by-image
% ratios or ratios for object measurements by whole image measurements (to
% allow normalization), use the CalculateRatios module until this data tool
% is updated to handle such calculations. Be careful with your denominator 
% data. Any 0's found in it may corrupt your output, especially when 
% dividing measurements.
%
% The new measurements will be stored under the first object's data, under
% the name Ratio.
%
% See also CalculateRatios and all Measure modules.

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

[FileName, Pathname] = CPuigetfile('*.mat', 'Select the raw measurements file',handles.Current.DefaultOutputDirectory);

if FileName == 0
    return
end

%%% Load the specified CellProfiler output file
try
    load(fullfile(Pathname, FileName));
catch
    CPerrordlg('Selected file is not a CellProfiler or MATLAB file (it does not have the extension .mat).')
    return
end

promptLoop = 0;
while promptLoop == 0
    Answers = inputdlg({'Operation? Multiply or Divide, Add Or Subtract', 'Take Log10? Yes or no', 'Raise to what power? "/" to ignore'}, 'Operation', 1, {'Multiply', '/', '/'}); 
    Operation = Answers{1};
    Log = Answers{2};
    Power = Answers{3};
    if ~strcmpi(Operation,'Multiply') && ~strcmpi(Operation,'Divide') && ~strcmpi(Operation, 'Add') && ~strcmpi(Operation, 'Subtract') 
        uiwait(CPerrordlg('Error: there was a problem with your choice for operation'));
        continue
    end

    if ~strcmpi(Log, 'yes') &&  ~strcmpi(Log, 'y') && ~strcmpi(Log, 'no') && ~strcmpi(Log, 'n') && ~strcmpi(Log, '/')
        uiwait(CPerrordlg('Error: there was a problem with your choice for log10'));
        continue
    elseif strcmpi(Log, 'no') || strcmpi(Log, 'n')
        Log = '/';
    end

    if isnan(str2double(Power)) && ~strcmpi(Power,'/')
        uiwait(CPerrordlg('Error: there was a problem with your choice for power'));
        continue
    end

    promptLoop = 1;
end

menuLoop=0;
while menuLoop == 0
    try
        [Measure1Object,Measure1fieldname,Measure1featurenumber] = CPgetfeature(handles,1);
        [Measure2Object,Measure2fieldname,Measure2featurenumber] = CPgetfeature(handles,1);
    catch
        ErrorMessage = lasterr;
        CPerrordlg(['An error occurred in CalculateRatiosDataTool. ' ErrorMessage(30:end)]);
        return
    end
    if isempty(Measure1Object),return,end
    if isempty(Measure2Object),return,end

    menuLoop = 1;
end

Measure1 = handles.Measurements.(Measure1Object).(Measure1fieldname);
Measure2 = handles.Measurements.(Measure2Object).(Measure2fieldname);
if length(Measure1) ~= length(Measure2)
    CPerrordlg(['Processing cannot continue because the specified object names ',Measure1Object,' and ',Measure2Object,' do not have the same amount of measurements.']);
    return
end
%SetBeingAnalyzed = handles.Current.SetBeingAnalyzed;

for i = 1:length(Measure1)
       
    %% Extract the measures of interest
    NumeratorMeasurements = handles.Measurements.(Measure1Object).(Measure1fieldname){i};
    NumeratorMeasurements = NumeratorMeasurements(:,Measure1featurenumber);
    DenominatorMeasurements = handles.Measurements.(Measure2Object).(Measure2fieldname){i};
    DenominatorMeasurements = DenominatorMeasurements(:,Measure2featurenumber);
    
    %%% Calculate the new measure. 
    if strcmpi(Operation,'Multiply')
        FinalMeasurements = NumeratorMeasurements.*DenominatorMeasurements;
    elseif strcmpi(Operation,'Divide')
        FinalMeasurements = NumeratorMeasurements./DenominatorMeasurements;
    elseif strcmpi(Operation,'Add')
        FinalMeasurements = NumeratorMeasurements+DenominatorMeasurements;
    elseif strcmpi(Operation, 'Subtract')
        FinalMeasurements = NumeratorMeasurements-DenominatorMeasurements;
    end
    
    if ~strcmp(Log, '/')
        FinalMeasurements = log10(FinalMeasurements);
    end
    
    if ~strcmp(Power,'/')
    	FinalMeasurements = FinalMeasurements .^ str2double(Power);
    end
    
    %%% Record the new measure in the handles structure.
    NewFieldName = [Measure1Object,'_',Measure1fieldname(1),'_',num2str(Measure1featurenumber),'_',char(Operation),'_',Measure2Object,'_',Measure2fieldname(1),'_',num2str(Measure2featurenumber)];
    handles.Current.SetBeingAnalyzed = i;
    handles = CPaddmeasurements(handles,Measure1Object,'Ratio',NewFieldName,FinalMeasurements);    
end

%%% Save the updated CellProfiler output file
try
    save(fullfile(Pathname, FileName),'handles');
    CPmsgbox(['Updated ',FileName,' successfully saved.'])
catch
    CPwarndlg(['Could not save updated ',FileName,' file.']);
end