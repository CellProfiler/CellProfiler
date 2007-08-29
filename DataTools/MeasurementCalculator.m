function MeasurementCalculator(handles)

% Help for the Measurement Calculator tool:
% Category: Data Tools
%
% SHORT DESCRIPTION:
% Multiplies or divides measurements in output files.
% *************************************************************************
% Note: this tool is beta-version and has not been thoroughly checked.
%
% This tool allows you to multiply or divide data taken from CellProfiler
% output files. You can choose the two measurements you wish to use, and
% choose whether to multiply or divide them either objectwise, by the image
% mean, or by the image median. You can give a name to your new measurement
% and save it for later use.
%
% See also CalculateRatiosDataTool, CalculateRatios.

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

%%% Ask the user to choose the file from which to extract measurements.
[RawFileName, RawPathname] = CPuigetfile('*.mat', 'Select the raw measurements file');
if RawFileName == 0
    return
end

%%% Load the specified CellProfiler output file
load(fullfile(RawPathname, RawFileName));

%%% Quick check if it seems to be a CellProfiler file or not
if ~exist('handles','var')
    CPerrordlg('The selected file does not seem to be a CellProfiler output file.')
    return
end

%%% Opens a window that lets the user choose what to export
%%% This function returns a UserInput structure with the
%%% information required to carry of the calculations.
try UserInput = UserInputWindow(handles);
catch CPerrordlg(lasterr)
    return
end

% If Cancel button pressed, return
if ~isfield(UserInput, 'SaveLocation')
    return
end

% Get measurements
Measurements1 = handles.Measurements.(UserInput.ObjectTypename1).(UserInput.FeatureType1);
Measurements2 = handles.Measurements.(UserInput.ObjectTypename2).(UserInput.FeatureType2);

% Do the calculation for all image sets
NewMeasurement = cell(1,length(Measurements1));
for ImageSetNbr = 1:length(Measurements1)

    try
        % Get the measurement for this image set in two temporary variables
        tmp1 = Measurements1{ImageSetNbr}(:,UserInput.FeatureNo1);
        tmp2 = Measurements2{ImageSetNbr}(:,UserInput.FeatureNo2);
    catch
        CPerrordlg('use the data tool MergeOutputFiles or ConvertBatchFiles to convert the data first');
        return
    end
   
    % If Operation2 indicates mean, replace all entries in tmp2 with
    % the image average, and proceed by doing "objectwise" multiplication/division
    if strcmp(UserInput.Operation2,'mean')
        tmp2 = mean(tmp2)*ones(size(tmp1));    % Important to give the new vector the size of tmp1
    elseif strcmp(UserInput.Operation2,'median')
        tmp2 = median(tmp2)*ones(size(tmp1));    % Important to give the new vector the size of tmp1
    end

    % Check so tmp1 and tmp2 have the same size
    if length(tmp1) ~= length(tmp2)
        CPerrordlg('The selected measurements do not have the same number of objects.')
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
% field doesn't exist this is easy. If it already exists we have to append
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
save(fullfile(RawPathname, RawFileName),'handles');
CPmsgbox(['The calculation is complete. Your new measurement has been saved in the output file ', RawFileName, ' in the default directory ', RawPathname]);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function UserInput = UserInputWindow(handles)
% This function displays a window so that lets the user choose which
% measurements to export. If the return variable 'UserInput' is empty
% it means that either no measurements were found or the user pressed
% the Cancel button (or the window was closed). 

% Create window
Window = CPfigure;
Width = 4.7;  % inches
Height = 4; % inches
uiheight = 0.3;
FontSize = handles.Preferences.FontSize;

set(Window,'units','inches','resize','off','menubar','none','toolbar','none','numbertitle','off',...
    'Name','Measurement Calculator','Color',[.7 .7 .9],'Position',[4 3 Width Height]);

BaseY = 3.5;
uicontrol(Window,'style','text','String','Calculate','Fontweight','bold',...
    'FontSize',FontSize,'units','inches','position',[0 BaseY Width uiheight],'backgroundcolor',[.7 .7 .9]);

%%% FEATURE 1
BaseY = BaseY - uiheight;
uicontrol(Window,'style','text','String','A feature:','Fontweight','bold','horizontalalignment','left',...
    'FontSize',FontSize,'units','inches','position',[0.1 BaseY 1 uiheight],'backgroundcolor',[.7 .7 .9]);
BaseY = BaseY - uiheight;
uicontrol(Window,'style','pushbutton','String','Get feature','BackGroundColor',[.7 .7 .9],'FontSize',FontSize,'units','inches',...
    'position',[0.1 BaseY 0.9 uiheight],'Callback',...
    ['[cobj,cfig] = gcbo;',...
    'UserData = get(cfig,''UserData'');',...
    'try,',...
    '[ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(UserData.handles,0);',...
    'catch,',...
    'ErrorMessage = lasterr;',...
    'CPerrordlg([''An error occurred in the MeasurementCalculator Data Tool. '' ErrorMessage(30:end)]);',...
    'return;',...
    'end;',...
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

Feature1 = uicontrol(Window,'style','text','String','',...
    'Fontsize',FontSize,'units','inches','position',[1.1 BaseY 3.5 uiheight],'backgroundcolor',[.8 .8 1]);

%%% OPERATION
BaseY = BaseY - uiheight*1.5;
uicontrol(Window,'style','text','String','will be:','Fontweight','bold','horizontalalignment','left',...
    'FontSize',FontSize,'units','inches','position',[0.1 BaseY 1 uiheight],'backgroundcolor',[.7 .7 .9]);
Operation1 = uicontrol(Window,'Style','popupmenu','String',{'multiplied','divided'},'FontSize',FontSize,...
    'backgroundcolor',[.7 .7 .9],'units','inches','position',[1.1 BaseY 1.3 uiheight]);
Operation2 = uicontrol(Window,'Style','popupmenu','String',{'objectwise by','by the image mean of','by the image median of'},...
    'backgroundcolor',[.7 .7 .9],'FontSize',FontSize,'units','inches','position',[2.5 BaseY 2.1 uiheight]);

%%% FEATURE 2
BaseY = BaseY - uiheight*1.5;
uicontrol(Window,'style','text','String','another feature:','Fontweight','bold','horizontalalignment','left',...
    'FontSize',FontSize,'units','inches','position',[0.1 BaseY 2 uiheight],'backgroundcolor',[.7 .7 .9]);
BaseY = BaseY - uiheight;
uicontrol(Window,'style','pushbutton','String','Get feature','BackGroundColor',[.7 .7 .9],'FontSize',FontSize,'units','inches',...
    'position',[0.1 BaseY 0.9 uiheight],'Callback',...
    ['[cobj,cfig] = gcbo;',...
    'UserData = get(cfig,''UserData'');',...
    'try,',...
    '[ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(UserData.handles,0);',...
    'catch,',...
    'ErrorMessage = lasterr;',...
    'CPerrordlg([''An error occurred in the MeasurementCalculator Data Tool. '' ErrorMessage(30:end)]);',...
    'return;',...
    'end;',...
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

Feature2 = uicontrol(Window,'style','text','String','',...
    'FontSize',FontSize,'units','inches','position',[1.1 BaseY 3.5 uiheight],'backgroundcolor',[.8 .8 1]);

%%% MEASUREMENT DESCRIPTION AND WHERE TO SAVE
BaseY = BaseY - uiheight*2.5;
uicontrol(Window,'style','text','String','Name your new measurement:','Fontweight','bold','horizontalalignment','left',...
    'FontSize',FontSize,'units','inches','position',[0.1 BaseY+uiheight 3.5 uiheight],'backgroundcolor',[.7 .7 .9]);
FeatureDescription = uicontrol(Window,'style','edit','string','Default measurement description','FontSize',FontSize,...
    'backgroundcolor',[1 1 1],'units','inches','position',[0.1 BaseY+0.03 Width-0.2 uiheight]);
BaseY = BaseY - uiheight*1.5;
uicontrol(Window,'style','text','String','Associate this measurement with:','Fontweight','bold','horizontalalignment','left',...
    'FontSize',FontSize,'units','inches','position',[0.1 BaseY 2.5 uiheight],'backgroundcolor',[.7 .7 .9]);
SaveLocation = uicontrol(Window,'Style','popupmenu','string',{'',''},'FontSize',FontSize,...
    'backgroundcolor',[.7 .7 .9],'units','inches','position',[2.6 BaseY 1.8 uiheight]);

%HELP BUTTON
Help_Callback = 'CPhelpdlg(''Depending on [your choice] in this drop down menu, the measurements will be saved in handles.Measurements.[your choice].UserDefined.'')';
uicontrol(Window,'style','pushbutton','String','?','FontSize',FontSize,...
    'HorizontalAlignment','center','units','inches','position',[4.4 BaseY 0.2 uiheight],'BackgroundColor',[.7 .7 .9],'FontWeight', 'bold',...
    'Callback', Help_Callback);

%%% CALCULATE AND CANCEL BUTTONS
posx = (Width - 1.7)/2;               % Centers buttons horizontally
calculatebutton = uicontrol(Window,'style','pushbutton','String','Calculate','Fontweight','bold','FontSize',FontSize,'units','inches',...
    'position',[posx 0.1 0.75 0.3],'Callback','[cobj,cfig] = gcbo;set(cobj,''UserData'',1);uiresume(cfig);clear cobj cfig;','BackgroundColor',[.7 .7 .9]);
cancelbutton = uicontrol(Window,'style','pushbutton','String','Cancel','Fontweight','bold','FontSize',FontSize,'units','inches',...
    'position',[posx+0.95 0.1 0.75 0.3],'Callback','close(gcf)','BackgroundColor',[.7 .7 .9]);

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
    if ishandle(calculatebutton)                  % The Calculate button pressed
        UserInput = get(Window,'UserData');
        if ~isfield(UserInput,'FeatureNo1') || ~isfield(UserInput,'FeatureNo2')
            warnfig=CPwarndlg('Please choose two features!');          % Check that both feature fields are filled out...
            uiwait(warnfig);
            set(calculatebutton,'UserData',[]);               % Reset button press
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

            delete(Window);
            return
        end
    else                                                      % The user pressed the cancel button or closed the window.
        UserInput = [];
        if ishandle(Window),delete(Window);end
        return
    end
end