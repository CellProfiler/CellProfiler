function handles = ExportData(handles)

% Help for the Export Mean Data tool:
% Category: Data Tools
%
% Once image analysis is complete, use this tool to select the
% output file to extract the measurements and other information about
% the analysis.  The data will be converted to a comma-delimited text file
% which can be read by most programs.

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

cd(handles.Current.DefaultOutputDirectory)
%%% Ask the user to choose the file from which to extract measurements.
[RawFileName, RawPathname] = uigetfile('*.mat','Select the raw measurements file');

LoadedHandles = load(fullfile(RawPathname, RawFileName));

%%% Opens a window that lets the user chose what to export and if
%%% only a summary (mean and std) should be exported

[ObjectNames,Summary] = ObjectsToExport(handles);

for Object = 1:length(ObjectNames)
    ObjectName = ObjectNames{Object};

    % Open a file for exporting the measurements
    if strcmp(Summary,'no')
        ExportFileName = [RawFileName(1:end-4),'_',ObjectName,'.xls'];
    else
        ExportFileName = [RawFileName(1:end-4),'_',ObjectName,'_Summary.xls'];
    end
    fid = fopen(fullfile(RawPathname,ExportFileName),'w');


    fields = fieldnames(handles.Measurements.(ObjectName));

    %%% Get total number of objects
    % THIS CODE SHOULD BE USED WHEN WE WANT TO ADD MORE INFO ON THE
    % TOP ROW
    %NumObjects = [];
    %for k = 1:length(fields)
    %    if isempty(strfind(fields{k},'Features'))
    %        for j = 1:length(handles.Measurements.(ObjectName).(fields{k}))
    %            NumObjects = [NumObjects size(handles.Measurements.(ObjectName).(fields{k}){j},1)];
    %        end
    %    end
    %end

    %%% Write comma-separated file that can be imported into Excel
    %%%% Header part, general information goes here. Different for
    % summary vs full report.
    if strcmp(Summary,'no')
        fprintf(fid,'%s: Full report\n', ObjectName);
        fprintf(fid,'\n');                                 % ~Image names should be here
    else
        fprintf(fid,'%s: Summary\n', ObjectName);
        fprintf(fid,'\n');
    end

    %%% Feature part
    % Organize features in format suitable for exportation. Create
    % a cell array Measurements where all features are concatenated
    % with each column corresponding to a separate feature
    FeatureNames = {};
    Measurements = {};
    
    for k = 1:length(fields)
        if ~isempty(strfind(fields{k},'Features'))
            % Concatenate measurement and feature name matrices
            tmp = handles.Measurements.(ObjectName).(fields{k}(1:end-8));   % Get the associate cell array of measurements
            if isempty(Measurements)
                Measurements = tmp;
            else
                for j = 1:length(tmp)
                    Measurements(j) = {cat(2,Measurements{j},tmp{j})};
                end
            end
            
            % Construct informative feature names
            tmp = handles.Measurements.(ObjectName).(fields{k});
            for j = 1:length(tmp)
                tmp{j} = [tmp{j} ' (' , fields{k}(1:end-8),')'];
            end
            FeatureNames = cat(2,FeatureNames,tmp);
        end
    end

    % Count objects
    NumObjects = zeros(length(Measurements),1);
    for k = 1:length(Measurements)
        NumObjects(k) = size(Measurements{k},1);
    end

    % If only summary report desired, replace the entries in Measurements with
    % mean, median and standard deviation of the measurements
    if strcmp(Summary,'yes')
        for k = 1:length(Measurements)
            Measurements{k} = [mean(Measurements{k});median(Measurements{k});std(Measurements{k})];
        end
    end

    % Interleave feature names with commas and write to file
    str = cell(2*length(FeatureNames),1);
    str(1:2:end) = {','};
    str(2:2:end) = FeatureNames;
    fprintf(fid,'%s\n',cat(2,str{:}));
    
    % Write measurements one row at the time
    SummaryInfo = {'Mean','Median','Std'};
    for k = 1:length(Measurements)                                 % Loop over image sets
        fprintf(fid,'Set #%d %d objects\n',k,NumObjects(k));
        for row = 1:size(Measurements{k},1)                        % Loop over the rows
            if strcmp(Summary,'yes')
                fprintf(fid,'%s',SummaryInfo{row});
            end
            tmp = cellstr(num2str(Measurements{k}(row,:)','%g'));  % Create cell array with measurements
            str = cell(2*length(tmp),1);                           % Interleave with commas
            str(1:2:end) = {','};
            str(2:2:end) = tmp;
            fprintf(fid,'%s\n',cat(2,str{:}));                     % Write to file
        end
        fprintf(fid,'\n');                                         % Separate image sets with a blank row
    end

    fclose(fid);
    cd(handles.Current.StartupDirectory);

end



function [ObjectNames,Summary] = ObjectsToExport(handles)
% This function displays a window so that lets the user choose which
% measurements to export. If the return variable 'ObjectNames' is empty
% it means that either no measurements were found or the user pressed
% the Cancel button (or the window was closed). 'Summary' takes on the values'yes'
% or 'no', depending if the user only wants a summary report (mean and std)
% or a full report.

fields = fieldnames(handles.Measurements);

% Remove fields that should be ignored
Ignorefields = {'Pathname' 'Filename' 'ImageThreshold' 'TimeElapsed'};
tmp = {};
for k = 1:length(fields)
    test = 0;
    for j = 1:length(Ignorefields)
        if ~isempty(strfind(fields{k},Ignorefields{j}))
            test = test + 1;
        end
    end
    if test == 0,tmp = cat(1,tmp,fields(k));end           % Field is not among the Ignorefields, store it
end
fields = tmp;

% Create Export window
ETh = figure;
set(ETh,'units','inches','resize','off','menubar','none','toolbar','none','numbertitle','off','Name','Export window');
pos = get(ETh,'position');
Height = 1.5+length(fields)*0.25;                       % Window height in inches
set(ETh,'position',[pos(1) pos(2) 3 Height]);

% Top text
uicontrol(ETh,'style','text','String','The following measurements were found:','FontName','Times','FontSize',8,...
    'units','inches','position',[0 Height-0.2 3 0.15],'BackgroundColor',get(ETh,'color'),'fontweight','bold')

% Radio buttons for extracted measurements
h = [];
for k = 1:length(fields)
    uicontrol(ETh,'style','text','String',fields{k},'FontName','Times','FontSize',8,'HorizontalAlignment','left',...
        'units','inches','position',[0.6 Height-0.35-0.2*k 3 0.15],'BackgroundColor',get(ETh,'color'))
    h(k) = uicontrol(ETh,'Style','Radiobutton','units','inches','position',[0.2 Height-0.35-0.2*k 0.2 0.2],...
        'BackgroundColor',get(ETh,'color'),'Value',1);
end

% Summary? button
uicontrol(ETh,'style','text','String','Only export summary data','FontName','Times','FontSize',8,'HorizontalAlignment','left',...
    'units','inches','position',[0.6 0.7 2 0.15],'BackgroundColor',get(ETh,'color'))
summarybutton = uicontrol(ETh,'Style','Radiobutton','units','inches','position',[0.2 0.7 0.2 0.2],...
    'BackgroundColor',get(ETh,'color'),'Value',0);

% Export and Cancel pushbuttons
uicontrol(ETh,'style','pushbutton','String','Export','FontName','Times','FontSize',8,'units','inches',...
    'position',[0.65 0.1 0.75 0.3],'Callback','[foo,fig] = gcbo;set(fig,''UserData'',1);uiresume(fig)')
uicontrol(ETh,'style','pushbutton','String','Cancel','FontName','Times','FontSize',8,'units','inches',...
    'position',[1.6 0.1 0.75 0.3],'Callback','[foo,fig] = gcbo;set(fig,''UserData'',0);uiresume(fig)')

uiwait(ETh)                         % Wait until window is destroyed or uiresume() is called

if get(summarybutton,'value') == 0
    Summary = 'no';
else
    Summary = 'yes';
end

if get(ETh,'Userdata') == 0
    ObjectNames = [];                                   % The user pressed the Cancel button
    close(ETh)
elseif get(ETh,'Userdata') == 1
    buttonchoice = get(h,'Value');
    buttonchoice = cat(1,buttonchoice{:});
    ObjectNames = fields(find(buttonchoice));            % Get the fields for which the radiobuttons are enabled
    close(ETh)
else
    ObjectNames = [];
end

