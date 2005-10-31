function MeasurementCalculator(handles)

% Help for the Measurement Calculator tool:
% Category: Data Tools
%
% This tool allows the user to perform multiplications and
% divisions of already extracted measurements.
%
%
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
%   Ola Friman     <friman@bwh.harvard.edu>
%   Steve Lowe     <stevelowe@alum.mit.edu>
%   Joo Han Chang  <joohan.chang@gmail.com>
%   Colin Clarke   <colinc@mit.edu>
%   Mike Lamprecht <mrl@wi.mit.edu>
%   Susan Ma       <xuefang_ma@wi.mit.edu>
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
%%% This function returns a UserInput structure with the
%%% information required to carry of the calculations.
UserInput = UserInputWindow(handles);

% If Cancel button pressed, return
if isempty(UserInput),return,end

% Get measurements
Measurements1 = handles.Measurements.(UserInput.ObjectTypename1).(UserInput.FeatureType1);
Measurements2 = handles.Measurements.(UserInput.ObjectTypename2).(UserInput.FeatureType2);

% Do the calculation for all image sets
NewMeasurement = cell(1,length(Measurements1));
for ImageSetNbr = 1:length(Measurements1)
    
    % Get the measurement for this image set in two temporary variables
    tmp1 = Measurements1{ImageSetNbr}(:,UserInput.FeatureNo1);
    tmp2 = Measurements2{ImageSetNbr}(:,UserInput.FeatureNo2);
    
    % If Operation2 indicates mean, replace all entries in tmp2 with
    % the image average, and proceed by doing "objectwise" multiplication/division
    if strcmp(UserInput.Operation2,'mean')
        tmp2 = mean(tmp2)*ones(size(tmp1));    % Important to give the new vector the size of tmp1
    elseif strcmp(UserInput.Operation2,'median')
        tmp2 = median(tmp2)*ones(size(tmp1));    % Important to give the new vector the size of tmp1
    end
    
    % Check so tmp1 and tmp2 have the same size
    if length(tmp1) ~= length(tmp2)
        errordlg('The selected measurements do not have the same number of objects.')
        return
    end
    
    % Do the calculation
    if strcmp(UserInput.Operation1,'multiplication')
        NewMeasurement{ImageSetNbr} = tmp1.*tmp2;
    elseif strcmp(UserInput.Operation1,'division')
        NewMeasurement{ImageSetNbr} = tmp1./tmp2;
    end
end


% Add the new measurement to the handles structure. If the 'UserDefined'
% field doesn't exist this is easy. If it already exists we have append
% for each image set.
if ~isfield(handles.Measurements.(UserInput.SaveLocation),'UserDefined')
    handles.Measurements.(UserInput.SaveLocation).UserDefined = NewMeasurement;
    handles.Measurements.(UserInput.SaveLocation).UserDefinedFeatures = {UserInput.FeatureDescription};
else
    for ImageSetNbr = 1:length(NewMeasurement)
        handles.Measurements.(UserInput.SaveLocation).UserDefined{ImageSetNbr}(:,end+1) = NewMeasurement{ImageSetNbr};
    end
    handles.Measurements.(UserInput.SaveLocation).UserDefinedFeatures(end+1) = {UserInput.FeatureDescription};
end

[ignore,Attributes] = fileattrib(fullfile(RawPathname, RawFileName));
if Attributes.UserWrite == 0
    error(['You do not have permission to write ',fullfile(RawPathname, RawFileName),'!']);
else
    save(fullfile(RawPathname, RawFileName),'handles');
end
CPmsgbox('Calculation complete!')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function UserInput = UserInputWindow(handles)
% This function displays a window so that lets the user choose which
% measurements to export. If the return variable 'ObjectNames' is empty
% it means that either no measurements were found or the user pressed
% the Cancel button (or the window was closed). 'Summary' takes on the values'yes'
% or 'no', depending if the user only wants a summary report (mean and std)
% or a full report.


% Create window
Window = figure;
Width = 4.5;  % inches
Height = 3.5; % inches
uiheight = 0.2;
set(Window,'units','inches','resize','off','menubar','none','toolbar','none','numbertitle','off',...
    'Name','Measurement Calculator','Color',[.7 .7 .9],'Position',[4 4 Width Height]);

BaseY = 3.2;
uicontrol(Window,'style','text','String','Calculate','FontName','Times','Fontweight','bold',...
    'FontSize',handles.Current.FontSize,'units','inches','position',[0 BaseY Width uiheight],'backgroundcolor',[.7 .7 .9]);

%%% FEATURE 1
BaseY = 2.7;
uicontrol(Window,'style','pushbutton','String','Get feature','FontName','Times','FontSize',handles.Current.FontSize,'units','inches',...
    'position',[0.1 BaseY 0.7 uiheight],'Callback',...
    ['[cobj,cfig] = gcbo;',...
    'UserData = get(cfig,''UserData'');',...
    '[ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(UserData.handles,0);',...
    'if ~isempty(ObjectTypename),',...
    '   UserData.ObjectTypename1 = ObjectTypename;',...
    '   UserData.FeatureType1 = FeatureType;',...
    '   UserData.FeatureNo1 = FeatureNo;',...
    '   FeatureName = UserData.handles.Measurements.(ObjectTypename).([FeatureType,''Features'']){FeatureNo};',...
    '   set(UserData.Feature1,''string'',[ObjectTypename,'', '',FeatureType,'', '',FeatureName]);',...
    '   set(cfig,''UserData'',UserData);',...
    '   str = get(UserData.SaveLocation,''string'');',...
    '   str{1} = ObjectTypename;',...
    '   set(UserData.SaveLocation,''string'',str);',...
    'end,',...
    'clear variables;']);

uicontrol(Window,'style','text','String','Feature:','FontName','Times','Fontweight','bold','horizontalalignment','left',...
    'FontSize',handles.Current.FontSize,'units','inches','position',[0.1 BaseY+uiheight 1 uiheight],'backgroundcolor',[.7 .7 .9]);
Feature1 = uicontrol(Window,'style','text','String','','FontName','Times',...
    'Fontsize',handles.Current.FontSize,'units','inches','position',[0.9 BaseY 3.5 uiheight],'backgroundcolor',[.8 .8 1]);

%%% OPERATION
BaseY = 2.25;
Operation1 = uicontrol(Window,'Style','popupmenu','String',{'multiplied','divided'},'FontName','Times','FontSize',8,...
    'backgroundcolor',[1 1 1],'units','inches','position',[0.1 BaseY 0.8 uiheight]);
Operation2 = uicontrol(Window,'Style','popupmenu','String',{'objectwise by','by the image mean of','by the image median of'},...
    'backgroundcolor',[1 1 1],'FontName','Times','FontSize',8,'units','inches','position',[1 BaseY 1.4 uiheight]);

%%% FEATURE 2
BaseY = 1.7;
uicontrol(Window,'style','pushbutton','String','Get feature','FontName','Times','FontSize',handles.Current.FontSize,'units','inches',...
    'position',[0.1 BaseY 0.7 uiheight],'Callback',...
    ['[cobj,cfig] = gcbo;',...
    'UserData = get(cfig,''UserData'');',...
    '[ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(UserData.handles,0);',...
    'if ~isempty(ObjectTypename),',...
    '   UserData.ObjectTypename2 = ObjectTypename;',...
    '   UserData.FeatureType2 = FeatureType;',...
    '   UserData.FeatureNo2 = FeatureNo;',...
    '   FeatureName = UserData.handles.Measurements.(ObjectTypename).([FeatureType,''Features'']){FeatureNo};',...
    '   set(UserData.Feature2,''string'',[ObjectTypename,'', '',FeatureType,'', '',FeatureName]);',...
    '   set(cfig,''UserData'',UserData);',...
    '   str = get(UserData.SaveLocation,''string'');',...
    '   str{2} = ObjectTypename;',...
    '   set(UserData.SaveLocation,''string'',str);',...
    'end,',...
    'clear Variables;']);

uicontrol(Window,'style','text','String','Feature:','FontName','Times','Fontweight','bold','horizontalalignment','left',...
    'FontSize',handles.Current.FontSize,'units','inches','position',[0.1 BaseY+uiheight 1 uiheight],'backgroundcolor',[.7 .7 .9]);
Feature2 = uicontrol(Window,'style','text','String','','FontName','Times',...
    'FontSize',handles.Current.FontSize,'units','inches','position',[0.9 BaseY 3.5 uiheight],'backgroundcolor',[.8 .8 1]);

%%% MEASUREMENT DESCRIPTION AND WHERE TO SAVE 
BaseY = 0.9;
uicontrol(Window,'style','text','String','Enter description and where to save:','FontName','Times','Fontweight','bold','horizontalalignment','left',...
    'FontSize',handles.Current.FontSize,'units','inches','position',[0.1 BaseY+uiheight 3.5 uiheight],'backgroundcolor',[.7 .7 .9]);
FeatureDescription = uicontrol(Window,'style','edit','string','Default measurement description','FontName','Times','FontSize',handles.Current.FontSize,...
    'backgroundcolor',[1 1 1],'units','inches','position',[0.1 BaseY+0.03 Width-0.2 uiheight]);
uicontrol(Window,'style','text','String','handles.Measurements.','FontName','Times','horizontalalignment','left',...
    'FontSize',handles.Current.FontSize,'units','inches','position',[0.1 BaseY-uiheight-0.05 1.5 uiheight],'backgroundcolor',[.7 .7 .9]);
SaveLocation = uicontrol(Window,'Style','popupmenu','string',{'',''},'FontName','Times','FontSize',handles.Current.FontSize,...
    'backgroundcolor',[1 1 1],'units','inches','position',[1.23 BaseY-uiheight 0.9 uiheight]);
uicontrol(Window,'style','text','String','.UserDefined','FontName','Times','horizontalalignment','left',...
    'FontSize',handles.Current.FontSize,'units','inches','position',[2.15 BaseY-uiheight-0.05 1 uiheight],'backgroundcolor',[.7 .7 .9]);

%%% EXPORT AND CANCEL BUTTONS
posx = (Width - 1.7)/2;               % Centers buttons horizontally
calculatebutton = uicontrol(Window,'style','pushbutton','String','Calculate','FontName','Times','FontSize',handles.Current.FontSize,'units','inches',...
    'position',[posx 0.1 0.75 0.3],'Callback','[cobj,cfig] = gcbo;set(cobj,''UserData'',1);uiresume(cfig);clear variables;','BackgroundColor',[.7 .7 .9]);
uicontrol(Window,'style','pushbutton','String','Cancel','FontName','Times','FontSize',handles.Current.FontSize,'units','inches',...
    'position',[posx+0.95 0.1 0.75 0.3],'Callback','[cobj,cfig] = gcbo;set(cobj,''UserData'',1);uiresume(cfig);clear variables;','BackgroundColor',[.7 .7 .9]);

% Store some variables in the current figure's UserData propery
UserData.handles = handles;
UserData.Feature1 = Feature1;
UserData.Feature2 = Feature2;
UserData.SaveLocation = SaveLocation;
set(gcf,'UserData',UserData);
clear UserData;

% Repeat until valid input has been entered or the window is destroyed
while 1
    
    % Wait until window is destroyed or uiresume() is called
    uiwait(Window)

    % Action depending on the user input
    if get(calculatebutton,'UserData') == 1                  % The Calculate button pressed
        UserInput = get(Window,'UserData');
        if ~isfield(UserInput,'FeatureNo1') | ~isfield(UserInput,'FeatureNo2')
            errordlg('Please choose two features!')          % Check that both feature fields are filled out...
            set(calculatebutton,'UserData',0);               % Reset button press 
        else
             % If both features are selected we can continue
            UserInput = rmfield(UserInput,{'handles','Feature1','Feature2'});    % Remove some unnecessary fields and       
            str = get(SaveLocation,'String');
            UserInput.SaveLocation = str{get(SaveLocation,'Value')};             % Where should the new measurement be stored?
            UserInput.FeatureDescription = get(FeatureDescription,'String');     % The description of the new measurement
            
            % Operation1 indicates multiplication or division
            if  get(Operation1,'Value') == 1                                     
                UserInput.Operation1 = 'multiplication';
            else
                UserInput.Operation1 = 'division';
            end
            
            % Operation2 indicates if operation should be perform objectwise, or based on the image mean
            if  get(Operation2,'Value') == 1                                     
                UserInput.Operation2 = 'objectwise';
            elseif get(Operation2,'Value') == 2
                UserInput.Operation2 = 'mean';
            elseif get(Operation2,'Value') == 3
                UserInput.Operation2 = 'median';
            end
                
            close(Window);
            return
        end
    else                                                      % The user pressed the cancel button or closed the window.
        UserInput = [];
        if ishandle(Window),close(Window);end
        return
    end
end