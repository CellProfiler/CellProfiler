function handles = ShowDataOnImage(handles)

% Help for the Show Data on Image tool:
% Category: Image Tools
%
% This allows you to extract measurements from an output file and
% overlay any measurements that you have made on any image. For
% example, you could look at the DNA content (e.g.
% IntegratedIntensityOrigBlue) of each cell on an image of nuclei.
% Or, you could look at cell area on an image of nuclei.  
% 
% First, you are asked to select the measurement you want to be
% displayed on the image.  Next, you are asked to select the X and
% then the Y locations where these measurements should be displayed.
% Typically, your options are the XY locations of the nuclei, or the
% XY locations of the cells, and these are usually named something
% like 'CenterXNuclei'.  If your output file has measurements from
% many images, you then select which sample number to view.
% 
% Then, CellProfilerTM tries to guide you to find the image that
% corresponds to this sample number.  First, it asks which file name
% would be most helpful for you to find the image. CellProfilerTM
% uses whatever you enter here to look up the exact file name you are
% looking for, so that you can browse to find the image. Once the
% image is selected, extraction ensues and eventually the image will
% be shown with the measurements on top.
% 
% You can use the tools at the top to zoom in on this image. If the
% text is overlapping and not easily visible, you can change the
% number of decimal places shown with the 'Fewer significant digits'
% button, or you can change the font size with the 'Text Properties'.
% You can also change the font style, color, and other properties with
% this button.  
% 
% The resulting figure can be saved in Matlab format (.fig) or
% exported in a traditional image file format.
%
% See also SHOWIMAGE, SHOWPIXELDATA.

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

%%% Asks the user to choose the file from which to extract measurements.
[RawFileName, RawPathname] = uigetfile(fullfile(handles.Current.DefaultOutputDirectory,'.','*.mat'),'Select the raw measurements file');

if RawFileName == 0,return,end

load(fullfile(RawPathname,RawFileName));

%%% Call the function CPgetfeature(), which opens a series of list dialogs and
%%% lets the user choose a feature. The feature can be identified via 'ObjectTypename',
%%% 'FeatureType' and 'FeatureNo'.
[ObjectTypename,FeatureType,FeatureNo] = CPgetfeature(handles);
if isempty(ObjectTypename),return,end

%%% Prompts the user to choose a sample number to be displayed.
Answer = inputdlg({'Which sample number do you want to display?'},'Choose sample number',1,{'1'});
if isempty(Answer)
    return
end
SampleNumber = str2double(Answer{1});

%TotalNumberImageSets = length(handles.Measurements.(MeasurementToExtract));
%if SampleNumber > TotalNumberImageSets
%    error(['The number you entered exceeds the number of samples in the file.  You entered ', num2str(SampleNumber), ' but there are only ', num2str(TotalNumberImageSets), ' in the file.'])
%end

%%% Looks up the corresponding image file name.
Fieldnames = fieldnames(handles.Measurements.Image);
PotentialImageNames = Fieldnames(strncmp(Fieldnames,'Filename',8)==1);
%%% Error detection.
if isempty(PotentialImageNames)
    h = errordlg('CellProfiler was not able to look up the image file names used to create these measurements to help you choose the correct image on which to display the results. You may continue, but you are on your own to choose the correct image file.')
    ImageFileName = [];
else
    %%% Allows the user to select a filename from the list.
    [Selection, ok] = listdlg('ListString',PotentialImageNames, 'ListSize', [300 300],...
        'Name','Choose the image whose filename you want to display',...
        'PromptString','Choose the image whose filename you want to display','CancelString','Cancel',...
        'SelectionMode','single');
    if ok == 0
        return
    end
    SelectedImageName = char(PotentialImageNames(Selection));
    ImageFileName = handles.Measurements.Image.(SelectedImageName){SampleNumber};
    %%% Prompts the user with the image file name.
    h = CPmsgbox(['Browse to find the image called ', ImageFileName,'.']);
end

%%% Opens a user interface window which retrieves a file name and path
%%% name for the image to be displayed.
[FileName,Pathname] = uigetfile(fullfile(handles.Current.DefaultImageDirectory,'.','*.*'),'Select the image to view');
try delete(h), end

%%% If the user presses "Cancel", the FileName will = 0 and nothing will happen.
if FileName == 0,return,end

%%% Opens and displays the image, with pixval shown.
[ImageToDisplay, handles] = CPimread(fullfile(Pathname,FileName), handles);

%%% Extracts the measurement values.
tmp = handles.Measurements.(ObjectTypename).(FeatureType){SampleNumber};
ListOfMeasurements = tmp(:,FeatureNo);
StringListOfMeasurements = cellstr(num2str(ListOfMeasurements));

%%% Extracts the XY locations. This is temporarily hard-coded
Xlocations = handles.Measurements.(ObjectTypename).Location{SampleNumber}(:,1);
Ylocations = handles.Measurements.(ObjectTypename).Location{SampleNumber}(:,2);
 

%%% Create window
ImageFileName = strrep(ImageFileName,'_','\_');
FigureHandle = CPfigure; imagesc(ImageToDisplay), colormap(gray)
title([ObjectTypename,', ',handles.Measurements.(ObjectTypename).([FeatureType,'Features']){FeatureNo} ' on ', ImageFileName])
set(FigureHandle,'toolbar', 'figure')

%%% Overlays the values in the proper location in the image.
TextHandles = text(Xlocations , Ylocations , StringListOfMeasurements,...
    'HorizontalAlignment','center', 'color', [0 0 1],'fontsize',8);

%%% Create structure and save it to the UserData property of the window
Info = get(FigureHandle,'UserData');
Info.ListOfMeasurements = ListOfMeasurements;
Info.TextHandles = TextHandles;
set(FigureHandle,'UserData',Info);

%%% A button is created in the display window which
%%% allows altering the properties of the text.
StdUnit = 'point';
StdColor = get(0,'DefaultUIcontrolBackgroundColor');
PointsPerPixel = 72/get(0,'ScreenPixelsPerInch');
if strcmp(computer,'MAC') == 1
    DisplayButtonCallback1 = 'CPmsgbox(''A bug in Matlab is preventing this function from working on the Mac platform. Service Request #1-RR6M1''), drawnow';
else
    DisplayButtonCallback1 = 'CurrentTextHandles = getfield(get(gcbf,''Userdata''),''TextHandles''); try, propedit(CurrentTextHandles,''v6''); catch, CPmsgbox(''A bug in Matlab is preventing this function from working. Service Request #1-RR6M1''), end; drawnow';
end
uicontrol('Parent',FigureHandle, ...
    'Unit',StdUnit, ...
    'BackgroundColor',StdColor, ...
    'CallBack',DisplayButtonCallback1, ...
    'Position',PointsPerPixel*[2 2 90 22], ...
    'Units','Normalized',...
    'String','Text Properties', ...
    'Style','pushbutton', ...
    'FontSize',handles.Current.FontSize);

DisplayButtonCallback2 = 'NumberOfDecimals = inputdlg(''Enter the number of decimal places to display'',''Enter the number of decimal places'',1,{''0''}); CurrentTextHandles = getfield(get(gcbf,''Userdata''),''TextHandles''); NumberValues = getfield(get(gcbf,''Userdata''),''ListOfMeasurements''); Command = [''%.'',num2str(NumberOfDecimals{1}),''f'']; NewNumberValues = num2str(NumberValues,Command); CellNumberValues = cellstr(NewNumberValues); PropName(1) = {''string''}; set(CurrentTextHandles,PropName, CellNumberValues); drawnow';
uicontrol('Parent',FigureHandle, ...
    'Unit',StdUnit, ...
    'BackgroundColor',StdColor, ...
    'CallBack',DisplayButtonCallback2, ...
    'Position',PointsPerPixel*[100 2 135 22], ...
    'Units','Normalized',...
    'String','Significant digits', ...
    'Style','pushbutton', ...
    'FontSize',handles.Current.FontSize);

DisplayButtonCallback3 = 'CurrentTextHandles = getfield(get(gcbf,''Userdata''),''TextHandles''); ListOfMeasurements = getfield(get(gcbf,''Userdata''),''ListOfMeasurements''); StringListOfMeasurements = cellstr(num2str(ListOfMeasurements)); PropName(1) = {''string''}; set(CurrentTextHandles,PropName, StringListOfMeasurements);drawnow';
uicontrol('Parent',FigureHandle, ...
    'Unit',StdUnit, ...
    'BackgroundColor',StdColor, ...
    'CallBack',DisplayButtonCallback3, ...
    'Position',PointsPerPixel*[240 2 135 22], ...
    'Units','Normalized',...
    'String','Restore labels', ...
    'Style','pushbutton', ...
    'FontSize',handles.Current.FontSize);

DisplayButtonCallback4 = 'CurrentTextHandles = getfield(get(gcbf,''Userdata''),''TextHandles''); set(CurrentTextHandles, ''visible'', ''off''); drawnow';
uicontrol('Parent',FigureHandle, ...
    'Unit',StdUnit, ...
    'BackgroundColor',StdColor, ...
    'CallBack',DisplayButtonCallback4, ...
    'Position',PointsPerPixel*[380 2 85 22], ...
    'Units','Normalized',...
    'String','Hide labels', ...
    'Style','pushbutton', ...
    'FontSize',handles.Current.FontSize);

DisplayButtonCallback5 = 'CurrentTextHandles = getfield(get(gcbf,''Userdata''),''TextHandles''); set(CurrentTextHandles, ''visible'', ''on''); drawnow';
uicontrol('Parent',FigureHandle, ...
    'Unit',StdUnit, ...
    'BackgroundColor',StdColor, ...
    'CallBack',DisplayButtonCallback5, ...
    'Position',PointsPerPixel*[470 2 85 22], ...
    'Units','Normalized',...
    'String','Show labels', ...
    'Style','pushbutton', ...
    'FontSize',handles.Current.FontSize);