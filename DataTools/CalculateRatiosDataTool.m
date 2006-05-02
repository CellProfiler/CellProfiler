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
    
    x = 1;
end

menuLoop=0;
while v == 0
    [Measure1Object,Measure1fieldname,Measure1featurenumber] = CPgetfeature(handles,1);
    if isempty(Measure2Object),return,end
    
    [Measure2Object,Measure2fieldname,Measure2featurenumber] = CPgetfeature(handles,1);
    if isempty(Measure1Object),return,end

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
catch
    errors{FileName} = ['Could not save updated ',FileName,' file.'];
end