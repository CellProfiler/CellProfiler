function handles = ProcessingInfo(handles)

% Help for the Processing Info tool:
% Category: Data Tools
%
% Extracts processing info from a CellProfiler output file
% and writes it to a text file.


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
load(fullfile(RawPathname, RawFileName));

%%% Let the user choose a filename
Filename = GetFilename(handles,RawFileName);

%%% Open file for writing
fid = fopen(fullfile(RawPathname,Filename),'w');
if fid == -1
    errordlg('Cannot create the output file. There might be another program using a file with the same name.')
    return
end

fprintf(fid,'Processing info for file: %s\n',fullfile(RawPathname, RawFileName));
fprintf(fid,'Processed: %s\n\n',handles.Current.TimeStarted);

fprintf(fid,'Pipeline:\n');
for module = 1:length(handles.Settings.ModuleNames)
    fprintf(fid,'\t%s\n',handles.Settings.ModuleNames{module});
    varibles = handles.Settings.VariableValues(module,:);
    % Can add info about what the module did
    %switch handles.Settings.ModuleNames{module}
    %    case 'LoadImagesText'
    %    case 'RGBSplit'
    %    case 'IdentifyEasy'
    %end
end

fprintf(fid,'\nPixel size: %s\n',handles.Settings.PixelSize);
fprintf(fid,'Number of image sets: %d\n\n',handles.Current.NumberOfImageSets);

% Get variable names used
VariableNames = fieldnames(handles.Pipeline);
Variableindex = find(cellfun('isempty',strfind(VariableNames,'FileList'))==0);
VariableNames = VariableNames(Variableindex);

% Get objects segmented
if isfield(handles,'Measurements') && isfield(handles.Measurements,'GeneralInfo')
    ObjectNames   = fieldnames(handles.Measurements.GeneralInfo);
    Thresholdindex = find(cellfun('isempty',strfind(ObjectNames,'ImageThreshold'))==0);
    ObjectNames = ObjectNames(Thresholdindex);
else
    ObjectNames = [];
end

% Report info for each image set
for imageset = 1:handles.Current.NumberOfImageSets
    fprintf(fid,'Image set #%d ---------------------------------------\n',imageset);
    fprintf(fid,'\tVariables:\n');
    for k = 1:length(VariableNames)
        fprintf(fid,'\t\t%s: %s\n',VariableNames{k}(9:end),handles.Pipeline.(VariableNames{k}){imageset});
    end
    fprintf(fid,'\n');
    fprintf(fid,'\tObjects:\n');
    for k = 1:length(ObjectNames)
        fprintf(fid,'\t\t%s',ObjectNames{k}(15:end));
        fprintf(fid,'\t Threshold: %g\n',handles.Measurements.GeneralInfo.(ObjectNames{k}){imageset});
    end
    fprintf(fid,'\n'); 
end
fclose(fid);



function Filename = GetFilename(handles,RawFileName)
% This function displays a window so that lets the user choose a filename


% The fontsize is stored in the 'UserData' property of the main Matlab window
FontSize = get(0,'UserData');

% Create Export window
window = figure;
set(window,'units','inches','resize','on','menubar','none','toolbar','none','numbertitle','off','Name','Processing Info window');
pos = get(window,'position');
set(window,'position',[pos(1) pos(2) 3 1.3]);

% Filename
% Remove 'OUT' and '.mat' extension from filename
ProposedFilename = RawFileName;
indexOUT = strfind(ProposedFilename,'OUT');
if ~isempty(indexOUT),ProposedFilename = [ProposedFilename(1:indexOUT(1)-1) ProposedFilename(indexOUT(1)+3:end)];end
indexMAT = strfind(ProposedFilename,'mat');
if ~isempty(indexMAT),ProposedFilename = [ProposedFilename(1:indexMAT(1)-2) ProposedFilename(indexMAT(1)+3:end)];end
ProposedFilename = [ProposedFilename,'_ProcessInfo.txt'];
uicontrol(window,'style','text','String','Chose output filename:','FontName','Times','FontSize',FontSize,...
    'HorizontalAlignment','left','units','inches','position',[0.4 0.9 2.2 0.2],'BackgroundColor',get(window,'color'))
editfilename = uicontrol(window,'Style','edit','units','inches','position',[0.4 0.7 2.2 0.2],...
    'backgroundcolor',[1 1 1],'String',ProposedFilename);

% Export and Cancel pushbuttons
exportbutton = uicontrol(window,'style','pushbutton','String','OK','FontName','Times','FontSize',FontSize,'units','inches',...
    'position',[0.65 0.1 0.75 0.3],'Callback','[foo,fig] = gcbo;set(fig,''UserData'',1);uiresume(fig)');
cancelbutton = uicontrol(window,'style','pushbutton','String','Cancel','FontName','Times','FontSize',FontSize,'units','inches',...
    'position',[1.6 0.1 0.75 0.3],'Callback','[foo,fig] = gcbo;set(fig,''UserData'',0);uiresume(fig)');

uiwait(window)                         % Wait until window is destroyed or uiresume() is called

if get(window,'Userdata') == 0         % The user pressed the Cancel button
    close(window)
    Filename = [];
elseif get(window,'Userdata') == 1

    % File name
    Filename = get(editfilename,'String');
    close(window)
else
    errordlg('Bug in Processing Info tool')
end
