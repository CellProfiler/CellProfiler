function handles = ExportData(handles)

% Help for the Export Data tool:
% Category: Data Tools
%
% Once image analysis is complete, use this tool to select the
% output file to extract the measurements and other information about
% the analysis.  The data will be converted to a tab-delimited text file
% which can be read by for example Excel or in a text editor.

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

%%% Ask the user to choose the file from which to extract measurements.
if exist(handles.Current.DefaultOutputDirectory, 'dir')
    [RawFileName, RawPathname] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'.','*.mat'),'Select the raw measurements file');
    PathToSave = handles.Current.DefaultOutputDirectory;
else
    [RawFileName, RawPathname] = uigetfile('*.mat','Select the raw measurements file');
    PathToSave = RawPathname;
end

if RawFileName == 0
    return
end

%%% Load the specified CellProfiler output file
load(fullfile(RawPathname, RawFileName));

%%% Quick check if it seems to be a CellProfiler file or not
if ~exist('handles','var')
    errordlg('The selected file does not seem to be a CellProfiler output file.')
    return
end

%%% Opens a window that lets the user chose what to export
ExportInfo = ObjectsToExport(handles,RawFileName);

%%% Indicates that the Cancel button was pressed
if isempty(ExportInfo.ObjectNames)
    return
end

%%% Create a waitbarhandle that can be accessed from the functions below
global waitbarhandle
waitbarhandle = waitbar(0,'');

%%% Export process info
if strcmp(ExportInfo.ExportProcessInfo,'Yes')
    WriteProcessInfo(handles,ExportInfo,RawFileName,RawPathname);
end

%%% Export measurements
if ~isempty(ExportInfo.MeasurementFilename)
    WriteMeasurements(handles,ExportInfo,RawPathname);
end

%%% Done!
close(waitbarhandle)
CPmsgbox('Exporting is completed.')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function WriteProcessInfo(handles,ExportInfo,RawFileName,RawPathname)
%%% This function extracts info about the process that generated a
%%% CellProfiler output file, and writes this info to a textfile.

%%% Get the handle to the waitbar and update the text in the waitbar
global waitbarhandle
waitbar(0,waitbarhandle,'Exporting process info')

%%% Open file for writing
%%% Add dot in extension if it's not there
if ExportInfo.ProcessInfoExtension(1) ~= '.';
    ExportInfo.ProcessInfoExtension = ['.',ExportInfo.ProcessInfoExtension];
end
filename = [ExportInfo.ProcessInfoFilename ExportInfo.ProcessInfoExtension];
fid = fopen(fullfile(RawPathname,filename),'w');
if fid == -1
    errordlg(sprintf('Cannot create the output file %s. There might be another program using a file with the same name.',filename));
    return
end

fprintf(fid,'Processing info for file: %s\n',fullfile(RawPathname, RawFileName));
fprintf(fid,'Processed: %s\n\n',handles.Current.TimeStarted);

fprintf(fid,'Pipeline:\n');
for module = 1:length(handles.Settings.ModuleNames)
    fprintf(fid,'\t%s\n',handles.Settings.ModuleNames{module});
end

fprintf(fid,'\n\nVariable Values:\n');
for module = 1:length(handles.Settings.ModuleNames)
    fprintf(fid,'\t%s  - revision %s\n',handles.Settings.ModuleNames{module},num2str(handles.Settings.VariableRevisionNumbers(module)));
    variables = [];
    variables = handles.Settings.VariableValues(module,:);
    for varnum = 1:length(variables)
        if ~isempty(variables{varnum})
            fprintf(fid,'\t\tVariable %s Value: %s\n',num2str(varnum),variables{varnum});
        end
    end
end
    
    

fprintf(fid,'\n\nPixel size: %s micrometer(s)\n',handles.Settings.PixelSize);

% Get variable names used

try
    VariableNames = fieldnames(handles.Measurements.Image);
catch
    errordlg('The output file does not contain a field called Measurements.');
    return;
end
Variableindex = find(cellfun('isempty',strfind(VariableNames,'Filename'))==0);
VariableNames = VariableNames(Variableindex);

%%% Get number of processed sets
if ~isempty(VariableNames)
    NbrOfProcessedSets = length(handles.Measurements.Image.(VariableNames{1}));
else
    NbrOfProcessedSets = 0;
end
fprintf(fid,'Number of processed image sets: %d\n\n',NbrOfProcessedSets);

%%% Get segmented objects, don't count the 'Image' field
ObjectNames = fieldnames(handles.Measurements);
ObjectNames = ObjectNames(find(cellfun('isempty',strfind(ObjectNames,'Image'))));
 

%%% Report info for each image set
for imageset = 1:NbrOfProcessedSets

    % Update waitbar handle
    waitbar(imageset/NbrOfProcessedSets,waitbarhandle)

    % Write info about image set
    fprintf(fid,'Image set #%d ---------------------------------------\n',imageset);
    fprintf(fid,'\tVariables:\n');
    for k = 1:length(VariableNames)
        ImageName = handles.Measurements.Image.(VariableNames{k}){imageset};
        fprintf(fid,'\t\t%s: %s\n',VariableNames{k}(9:end),ImageName);
    end
    fprintf(fid,'\n');
    fprintf(fid,'\tObjects:\n');
    for k = 1:length(ObjectNames)
        fprintf(fid,'\t\t%s\t',ObjectNames{k});
        fields = fieldnames(handles.Measurements.Image);
        fields = fields(~cellfun('isempty',strfind(fields,'Features')));
        for j = 1:length(fields)
            column = find(~cellfun('isempty',strfind(handles.Measurements.Image.(fields{j}),ObjectNames{k})));
            if ~isempty(column)
                
                fprintf(fid,'\t %s: %g',fields{j}(1:end-8),handles.Measurements.Image.(fields{j}(1:end-8)){imageset}(column));
            end
        end
    fprintf(fid,'\n');
    end
    fprintf(fid,'\n');
end
fclose(fid);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function WriteMeasurements(handles,ExportInfo,RawPathname)
%%% This function exports full and summary measurement reports

%%% Get the handle to the waitbar and update the text in the waitbar
global waitbarhandle
waitbar(0,waitbarhandle,'')


%%% Step 1: Create a cell array containing matrices with all measurements for each object type
%%% concatenated.
SuperMeasurements = cell(length(ExportInfo.ObjectNames),1);
SuperFeatureNames = cell(length(ExportInfo.ObjectNames),1);
for Object = 1:length(ExportInfo.ObjectNames)
    ObjectName = ExportInfo.ObjectNames{Object};

    %%% Get fields in handles.Measurements
    fields = fieldnames(handles.Measurements.(ObjectName));

    %%% Organize features in format suitable for exportation. This
    %%% piece of code creates a super measurement matrix, where
    %%% all features for all objects are stored. There will be one such
    %%% matrix for each image set, and each column in such a matrix
    %%% corresponds to one feature, e.g. Area or IntegratedIntensity.
    FeatureNames = {};
    Measurements = {};
    for k = 1:length(fields)
        if ~isempty(strfind(fields{k},'Features'))                              % Found a field with feature names

            % Get the associated cell array of measurements
            try
                CellArray = handles.Measurements.(ObjectName).(fields{k}(1:end-8));
            catch
                errordlg('Error in handles.Measurements structure. The field ',fields{k},' does not have an associated measurement field.');
            end

            % Concatenate
            if length(Measurements) == 0
                Measurements = CellArray;                             % The first measurement structure encounterered
            else
                for j = 1:length(CellArray)
                    Measurements(j) = {cat(2,Measurements{j},CellArray{j})};
                end
            end

            % Construct informative feature names
            tmp = handles.Measurements.(ObjectName).(fields{k});     % Get the feature names
            for j = 1:length(tmp)
                tmp{j} = [tmp{j} ' (', ObjectName,', ',fields{k}(1:end-8),')'];
            end
            FeatureNames = cat(2,FeatureNames,tmp);
        end
    end
    SuperMeasurements{Object} = Measurements;
    SuperFeatureNames{Object} = FeatureNames;
end

%%% Step 2: Write the measurements to file
for Object = 1:length(ExportInfo.ObjectNames)

    ObjectName = ExportInfo.ObjectNames{Object};

    % Open a file for exporting the measurements
    % Add dot in extension if it's not there
    if ExportInfo.MeasurementExtension(1) ~= '.';
        ExportInfo.MeasurementExtension = ['.',ExportInfo.MeasurementExtension];
    end
    filename = [ExportInfo.MeasurementFilename,'_',ObjectName,ExportInfo.MeasurementExtension];
    fid = fopen(fullfile(RawPathname,filename),'w');
    if fid == -1
        errordlg(sprintf('Cannot create the output file %s. There might be another program using a file with the same name.',filename));
        return
    end

    % Get the measurements and feature names to export
    Measurements = SuperMeasurements{Object};
    FeatureNames = SuperFeatureNames{Object};

    % The 'Image' object type is gets some special treatement.
    % First, add the image file names as features,
    % then, add the average values for all other measurements too
    if strcmp(ObjectName,'Image')
        for k = 1:length(ExportInfo.ObjectNames)
            if ~strcmp('Image',ExportInfo.ObjectNames{k})
                FeatureNames = cat(2,FeatureNames,SuperFeatureNames{k});
                tmpMeasurements = SuperMeasurements{k};
                for imageset = 1:length(Measurements)
                    Measurements{imageset} = cat(2,Measurements{imageset},mean(tmpMeasurements{imageset},1));
                end
            end
        end
    else     % If not the 'Image' field, add a Object Nbr feature instead
        FeatureNames = cat(2,{'Object Nbr'},FeatureNames);
    end

    %%% Write tab-separated file that can be imported into Excel
    % Header part
    fprintf(fid,'%s\n\n', ObjectName);

    if strcmp(ExportInfo.SwapRowsColumnInfo,'No')
        % Write feature names in one row
        % Interleave feature names with commas and write to file
        str = cell(2*length(FeatureNames),1);
        str(1:2:end) = {'\t'};
        str(2:2:end) = FeatureNames;
        fprintf(fid,sprintf('%s\n',cat(2,str{:})));

        % Get the filenames
        fields = fieldnames(handles.Measurements.Image);
        Filenames  = fields(strmatch('Filename',fields));

        % Loop over the images sets
        for imageset = 1:length(Measurements)

            % Update waitbar
            waitbar(imageset/length(ExportInfo.ObjectNames),waitbarhandle,sprintf('Exporting %s',ObjectName));

            % Write info about the image set (a little unnecessary code here)
            ImageName = handles.Measurements.Image.(Filenames{1}){imageset};
            fprintf(fid,'Set #%d, %s',imageset,ImageName);

            % Write measurements row by row
            if ~isempty(Measurements{imageset})
                for row = 1:size(Measurements{imageset},1)                        % Loop over the rows

                    % If not the 'Image' field, write an object number
                    if ~strcmp(ObjectName,'Image')
                        fprintf(fid,'\t%d',row);
                    end

                    % Write measurements
                    tmp = cellstr(num2str(Measurements{imageset}(row,:)','%g'));  % Create cell array with measurements
                    str = cell(2*length(tmp),1);                           % Interleave with tabs
                    str(1:2:end) = {'\t'};
                    str(2:2:end) = tmp;
                    fprintf(fid,sprintf('%s\n',cat(2,str{:})));            % Write to file
                end
            end
        end
    else
        fprintf(fid,'\t%d',[]);
        for imageset= 1:length(Measurements)
            fields = fieldnames(handles.Measurements.Image);
            Filenames  = fields(strmatch('Filename',fields));
            fprintf(fid,'Set #%d, %s',imageset,handles.Measurements.Image.(Filenames{1}){imageset});
            str = cell(size(Measurements{imageset}-1,1),1);
            str(1:end)={'\t'};
            fprintf(fid,sprintf('%s',cat(2,str{:})));
        end
        fprintf(fid,'\n%d',[]);
        if ~strcmp(ObjectName,'Image')
            for imageset= 1:length(Measurements)
                Measurements{imageset} = cat(2,[1:size(Measurements{imageset},1)]',Measurements{imageset});
            end
        end
        for i = 1:length(Measurements{1})
            
            % Update waitbar
            waitbar(i/length(Measurements{1}),waitbarhandle,sprintf('Exporting %s',ObjectName));
            
            try % In case things in Measurements are not the same length
                fprintf(fid,'%s',FeatureNames{i});
            end
            tmp = {};
            for imageset = 1:length(Measurements)
                try % In case things in Measurements are not the same length
                    tmp = cat(1,tmp,cellstr(num2str(Measurements{imageset}(:,i),'%g')));
                end
            end
               str = cell(2*length(tmp),1);
               str(1:2:end) = {'\t'};
               str(2:2:end) = tmp;
               fprintf(fid,sprintf('%s\n',cat(2,str{:})));
        end  
    end
    fclose(fid);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ExportInfo = ObjectsToExport(handles,RawFileName)
% This function displays a window so that lets the user choose which
% measurements to export. If the return variable 'ObjectNames' is empty
% it means that either no measurements were found or the user pressed
% the Cancel button (or the window was closed). 'Summary' takes on the values'yes'
% or 'no', depending if the user only wants a summary report (mean and std)
% or a full report.

% Initialize output variables
ExportInfo.ObjectNames = [];
ExportInfo.MeasurementFilename = [];
ExportInfo.ProcessInfoFilename = [];

% The fontsize is stored in the 'UserData' property of the main Matlab window
FontSize = get(0,'UserData');

% Get measurement object fields
fields = fieldnames(handles.Measurements);
if length(fields) > 20
    errordlg('There are more than 20 different objects in the chosen file. There is probably something wrong in the handles.Measurement structure.')
    return
end

% Create Export window
ETh = figure;
set(ETh,'units','inches','resize','on','menubar','none','toolbar','none','numbertitle','off','Name','Export window','Color',[.7 .7 .9],'CloseRequestFcn','set(gcf,''UserData'',0);uiresume()');

% Some variables controling the sizes of uicontrols
uiheight = 0.3;

% Set window size in inches, depends on the number of objects
pos = get(ETh,'position');
Height = 2.5+length(fields)*uiheight;
Width  = 4.2;
set(ETh,'position',[pos(1)+1 pos(2) Width Height]);

if ~isempty(fields)
    % Top text
    uicontrol(ETh,'style','text','String','The following measurements were found:','FontName','Times','FontSize',FontSize,...
        'HorizontalAlignment','left','units','inches','position',[0.2 Height-0.3 4 0.15],'BackgroundColor',get(ETh,'color'))

    % Radio buttons for extracted measurements
    h = [];
    for k = 1:length(fields)
        uicontrol(ETh,'style','text','String',fields{k},'FontName','Times','FontSize',FontSize,'HorizontalAlignment','left',...
            'units','inches','position',[0.6 Height-0.35-uiheight*k 3 0.15],'BackgroundColor',get(ETh,'color'))
        h(k) = uicontrol(ETh,'Style','checkbox','units','inches','position',[0.2 Height-0.35-uiheight*k uiheight uiheight],...
            'BackgroundColor',get(ETh,'color'),'Value',1);
    end

    % Filename, remove 'OUT' and '.mat' extension from filename
    basey = 1.5;
    ProposedFilename = RawFileName;
    indexOUT = strfind(ProposedFilename,'OUT');
    if ~isempty(indexOUT),ProposedFilename = [ProposedFilename(1:indexOUT(1)-1) ProposedFilename(indexOUT(1)+3:end)];end
    indexMAT = strfind(ProposedFilename,'mat');
    if ~isempty(indexMAT),ProposedFilename = [ProposedFilename(1:indexMAT(1)-2) ProposedFilename(indexMAT(1)+3:end)];end
    ProposedFilename = [ProposedFilename,'_Export'];
    uicontrol(ETh,'style','text','String','Choose base of output filename:','FontName','Times','FontSize',FontSize,...
        'HorizontalAlignment','left','units','inches','position',[0.2 basey+0.2 1.8 uiheight],'BackgroundColor',get(ETh,'color'))
    EditMeasurementFilename = uicontrol(ETh,'Style','edit','units','inches','position',[0.2 basey 2.5 uiheight],...
        'backgroundcolor',[1 1 1],'String',ProposedFilename);
    uicontrol(ETh,'style','text','String','Choose extension:','FontName','Times','FontSize',FontSize,...
        'HorizontalAlignment','left','units','inches','position',[2.9 basey+0.2 1.2 uiheight],'BackgroundColor',get(ETh,'color'))
    EditMeasurementExtension = uicontrol(ETh,'Style','edit','units','inches','position',[2.9 basey 0.7 uiheight],...
        'backgroundcolor',[1 1 1],'String','.xls');

else  % No measurements found
    uicontrol(ETh,'style','text','String','No measurements found!','FontName','Times','FontSize',FontSize,...
        'units','inches','position',[0 Height-0.5 6 0.15],'BackgroundColor',get(ETh,'color'),'fontweight','bold')
end

%%% Process info
basey = 0.65;
% Propose a filename. Remove 'OUT' and '.mat' extension from filename
ProposedFilename = RawFileName;
indexOUT = strfind(ProposedFilename,'OUT');
if ~isempty(indexOUT),ProposedFilename = [ProposedFilename(1:indexOUT(1)-1) ProposedFilename(indexOUT(1)+3:end)];end
indexMAT = strfind(ProposedFilename,'mat');
if ~isempty(indexMAT),ProposedFilename = [ProposedFilename(1:indexMAT(1)-2) ProposedFilename(indexMAT(1)+3:end)];end
ProposedFilename = [ProposedFilename,'_ProcessInfo'];
uicontrol(ETh,'style','text','String','Export process info?','FontName','Times','FontSize',FontSize,...
        'HorizontalAlignment','left','units','inches','position',[0.2 basey+0.3 1.6 uiheight],'BackgroundColor',get(ETh,'color'));
ExportProcessInfo = uicontrol(ETh,'style','popupmenu','String',{'No','Yes'},'FontName','Times','FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[1.7 basey+0.35 0.6 uiheight],'BackgroundColor',get(ETh,'color'));
uicontrol(ETh,'style','text','String','Swap Rows/Columns?','FontName','Times','FontSize',FontSize,...
        'HorizontalAlignment','left','units','inches','position',[2.5 basey+1.6 1.8 uiheight],'BackgroundColor',get(ETh,'color'));
SwapRowsColumnInfo = uicontrol(ETh,'style','popupmenu','String',{'No','Yes'},'FontName','Times','FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[3 basey+1.4 0.6 uiheight],'BackgroundColor',get(ETh,'color'));
EditProcessInfoFilename = uicontrol(ETh,'Style','edit','units','inches','position',[0.2 basey 2.5 uiheight],...
    'backgroundcolor',[1 1 1],'String',ProposedFilename);
uicontrol(ETh,'style','text','String','Choose extension:','FontName','Times','FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[2.9 basey+0.2 1.2 uiheight],'BackgroundColor',get(ETh,'color'))
EditProcessInfoExtension = uicontrol(ETh,'Style','edit','units','inches','position',[2.9 basey 0.7 uiheight],...
    'backgroundcolor',[1 1 1],'String','.txt');


% Export and Cancel pushbuttons
posx = (Width - 1.7)/2;               % Centers buttons horizontally
exportbutton = uicontrol(ETh,'style','pushbutton','String','Export','FontName','Times','FontSize',FontSize,'units','inches',...
    'position',[posx 0.1 0.75 0.3],'Callback','[foo,fig] = gcbo;set(fig,''UserData'',1);uiresume(fig);clear fig foo','BackgroundColor',[.7 .7 .9]);
cancelbutton = uicontrol(ETh,'style','pushbutton','String','Cancel','FontName','Times','FontSize',FontSize,'units','inches',...
    'position',[posx+0.95 0.1 0.75 0.3],'Callback','close(gcf)','BackgroundColor',[.7 .7 .9]);

uiwait(ETh)                         % Wait until window is destroyed or uiresume() is called

if get(ETh,'Userdata') == 1     % The user pressed the Export button

    % File names
    if ~isempty(fields)
        ExportInfo.MeasurementFilename = get(EditMeasurementFilename,'String');
        ExportInfo.MeasurementExtension = get(EditMeasurementExtension,'String');
    end
    ExportInfo.ProcessInfoFilename = get(EditProcessInfoFilename,'String');
    ExportInfo.ProcessInfoExtension = get(EditProcessInfoExtension,'String');
    if get(ExportProcessInfo,'Value') == 1                                       % Indicates a 'No' (equals 2 if 'Yes')                       
        ExportInfo.ExportProcessInfo = 'No';
    else
        ExportInfo.ExportProcessInfo = 'Yes';
    end
    if get(SwapRowsColumnInfo,'Value') == 1
        ExportInfo.SwapRowsColumnInfo = 'No';
    else
        ExportInfo.SwapRowsColumnInfo = 'Yes';
    end
    
    % Get measurements to export
    if ~isempty(fields)
        buttonchoice = get(h,'Value');
        if iscell(buttonchoice)                              % buttonchoice will be a cell array if there are several objects
            buttonchoice = cat(1,buttonchoice{:});
        end
        ExportInfo.ObjectNames = fields(find(buttonchoice));  % Get the fields for which the radiobuttons are enabled
    end
    delete(ETh)
else
    delete(ETh);
    ExportInfo.ObjectNames = [];
end


