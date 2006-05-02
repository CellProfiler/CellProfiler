function CalculateRatiosDataTool2(handles)

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

[Measure1Object,Measure1fieldname,Measure1featurenumber] = CPgetfeature(handles,1);
[Measure2Object,Measure2fieldname,Measure2featurenumber] = CPgetfeature(handles,1);

x = 0;
while x == 0
    Operation = inputdlg('Operation? Multiply or Divide, Add Or Subtract', 'Operation', 1, {'Multiply'}); 
    if ~strcmpi(Operation,'Multiply') && ~strcmpi(Operation,'Divide') && ~strcmpi(Operation, 'Add') && ~strcmpi(Operation, 'Subtract') 
        continue
    end
    x=1;
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
    
    %%% Record the new measure in the handles structure.
    NewFieldName = [Measure1Object,'_',Measure1fieldname(1),'_',num2str(Measure1featurenumber),'_',char(Operation),'_',Measure2Object,'_',Measure2fieldname(1),'_',num2str(Measure2featurenumber)];
    handles.Current.SetBeingAnalyzed = i;
    handles = CPaddmeasurements(handles,Measure1Object,'Ratio',NewFieldName,FinalMeasurements);
end