function CalculateRatiosDataTool2(handles)

% Help for the Calculate Ratios data tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Calculates the ratio between any measurements already measured (e.g.
% Intensity of green staining in cytoplasm/Area of cells)
% *************************************************************************
%
% This module can take any measurements produced by previous modules and
% calculate a ratio. Resulting ratios can also be used to calculate other
% ratios and be used in Classify Objects.
%
% This module currently works on an object-by-object basis (it calculates
% the ratio for each object) but can also calculate ratios for measurements
% made for entire images (but only for measurements produced by the
% Correlation module).
%
% Feature Number:
% The feature number specifies which features from the Measure module(s)
% will be used for the ratio. See each Measure module's help for the
% numbered list of the features measured by that module.
%
% See also all Measure modules.

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
% $Revision: 3534 $

if exist(handles.Current.DefaultOutputDirectory, 'dir')
    [FileName, Pathname] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'.','*.mat'),'Select the raw measurements file');
else
    [FileName, Pathname] = uigetfile('*.mat','Select the raw measurements file');
end

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
    
    
    if isempty(str2num(Power)) && ~strcmpi(Power,'/')
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
    	FinalMeasurements = FinalMeasurements .^ str2num(Power);
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
    CPwarndlg = (['Could not save updated ',FileName,' file.']);
end