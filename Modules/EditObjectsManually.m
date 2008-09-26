function handles = EditObjectsManually(handles)

% Help for the Edit Objects Manually module:
% Category: Object Processing
%
% SHORT DESCRIPTION:
% User interface for removing objects manually from an image
% *************************************************************************
%
% This module allows you to remove objects through a user interface. The
% module displays three images: the objects as originally segmented,
% the objects that have not been removed and the objects that have been
% removed.
% If you click on an object in the "not removed" image, it moves to the
% "removed" image and will be removed. If you click on an object in the
% "removed" image, it moves to the "not removed" image and will not be
% removed. If you click on an object in the original image, it will
% toggle its "removed" state.
%
% The pipeline pauses once per processed image when it reaches this module.
% You have to press the continue button to accept the selected objects
% and continue the pipeline.
%
% See also FilterByObjectMeasurement, Exclude, OverlayOutlines, ConvertToImage.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

%%%%%%%%%%%%%%%%%
%%% VARIABLES %%%
%%%%%%%%%%%%%%%%%
drawnow

[CurrentModule, CurrentModuleNum, ModuleName] = CPwhichmodule(handles);

%textVAR01 = What did you call the objects you want to filter?
%infotypeVAR01 = objectgroup
%inputtypeVAR01 = popupmenu
ObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,1});

%textVAR02 = What do you want to call the remaining objects?
%defaultVAR02 = FilteredObjects
%infotypeVAR02 = objectgroup indep
RemainingObjectName = char(handles.Settings.VariableValues{CurrentModuleNum,2});
CPvalidfieldname(RemainingObjectName);

%textVAR03 = What do you want to call the outlines of the remaining objects (optional)?
%defaultVAR03 = Do not use
%infotypeVAR03 = outlinegroup indep
SaveOutlines = char(handles.Settings.VariableValues{CurrentModuleNum,3});

%textVAR04 = Do you want to renumber the objects created by this module or retain the original numbering?
%defaultVAR04 = Renumber
%choiceVAR04 = Renumber
%choiceVAR04 = Retain
%inputtypeVAR04 = popupmenu
RenumberOrRetain = char(handles.Settings.VariableValues{CurrentModuleNum,4});

%%%VariableRevisionNumber = 2

%%%%%%%%%%%%%%%%%%
%%% SETUP      %%%
%%%%%%%%%%%%%%%%%%

%%% Operate on this image
SegmentedObjectImage = CPretrieveimage(handles,['Segmented', ObjectName],ModuleName);
%%% The following is only relevant for objects identified using
%%% Identify Primary modules, not Identify Secondary modules.
fieldname = ['UneditedSegmented',ObjectName];
if isfield(handles.Pipeline, fieldname)
    UneditedSegmentedObjectImage = CPretrieveimage(handles,fieldname,ModuleName);
else
    UneditedSegmentedObjectImage = [];
end

fieldname = ['SmallRemovedSegmented',ObjectName];
if isfield(handles.Pipeline, fieldname)
    SmallRemovedSegmentedObjectImage = CPretrieveimage(handles,fieldname,ModuleName);
else
    SmallRemovedSegmentedObjectImage = [];
end

%%% Set up the three display images
img = CPlabel2rgb(handles,SegmentedObjectImage);
my_data = struct(...
	'SegmentedObjectImage', SegmentedObjectImage,...
    'Filter',AllLabels(SegmentedObjectImage),...
    'IncludedSegmentedObjectImage',SegmentedObjectImage,...
    'ExcludedSegmentedObjectImage',zeros(size(SegmentedObjectImage,1),size(SegmentedObjectImage,2)),...
    'handles',handles);

%%%%%%%%%%%%%%%%%%
%%% DISPLAY UI %%%
%%%%%%%%%%%%%%%%%%
drawnow;

ThisModuleFigureNumber = handles.Current.(['FigureNumberForModule',CurrentModule]);
hFigure = CPfigure(handles,'Image',ThisModuleFigureNumber);
my_data.hFigure = ThisModuleFigureNumber;
set(hFigure,'Name',['EditObjectsManually, cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);

if handles.Current.SetBeingAnalyzed == handles.Current.StartingImageSet
    CPresizefigure(img,'TwoByTwo',ThisModuleFigureNumber);
end
%%% A subplot of the figure window is set to display the original image.
hAx=subplot(2,2,1,'Parent',ThisModuleFigureNumber);
my_data.hOriginal=CPimagesc(img,handles,hAx);
set(my_data.hOriginal,'ButtonDownFcn',{@ImageClickCallback,'toggle'});
title(hAx,['Previously identified ', ObjectName,', cycle # ',num2str(handles.Current.SetBeingAnalyzed)]);
my_data.hAxIncluded=subplot(2,2,2,'Parent',ThisModuleFigureNumber);
title(my_data.hAxIncluded,'Objects to keep');
my_data.hAxExcluded=subplot(2,2,4,'Parent',ThisModuleFigureNumber);
title(my_data.hAxExcluded,'Objects to remove');
%%%
%%% The continue button exits the modal state
%%%
hContinue=uicontrol('Position',[30 30 100 20],'String','Continue',...
                    'Callback',{@Continue_Callback,ThisModuleFigureNumber},...
                    'UserData',struct('operation','none'),...
                    'DeleteFcn','uiresume(gcbf)',...
                    'Parent',ThisModuleFigureNumber...
                );
my_data.hContinue = hContinue;
uibuttons = [...
uicontrol('Position', [30,120,100,20],...
          'String',   'Keep all',...
          'Callback', {@CommandCallback, hContinue,'keepall'},...
          'Parent',   ThisModuleFigureNumber),...
uicontrol('Position', [30,90,100,20],...
          'String',   'Remove all',...
          'Callback', {@CommandCallback, hContinue,'removeall'},...
          'Parent',   ThisModuleFigureNumber),...
uicontrol('Position', [30,60,100,20],...
          'String',   'Reverse selection',...
          'Callback', {@CommandCallback, hContinue,'toggleall'},...
          'Parent',   ThisModuleFigureNumber)];
set(my_data.hOriginal,'UserData',my_data);
my_data = Update('keepall',[1,1],my_data,handles);
set(ThisModuleFigureNumber,'CloseRequestFcn','uiresume(gcbf);closereq');
while (ishandle(hContinue) && isstruct(get(hContinue,'UserData')))
    x = get(hContinue,'UserData');
    if ~ strcmp(x.operation,'none')
        my_data = Update(x.operation,x.pos,my_data,handles);
    end
    drawnow;
    uiwait(ThisModuleFigureNumber);
end;
set(hContinue,'Visible','off');
set(uibuttons,'Visible','off');
set(my_data.hOriginal,'ButtonDownFcn',@CPimagetool);
set(my_data.hIncluded,'ButtonDownFcn',@CPImagetool);
set(my_data.hExcluded,'ButtonDownFcn',@CPImagetool);

%%%%%%%%%%%%%%%%%%%%%%%
%%% Add object      %%%
%%%%%%%%%%%%%%%%%%%%%%%

OutputImages = struct('image', my_data.IncludedSegmentedObjectImage,...
                      'fieldname',['Segmented',RemainingObjectName]);

%%% Keep points in optional image whose labels are in the filter array
if ~ isempty(UneditedSegmentedObjectImage)
    OutputImages(length(OutputImages)+1) = ...
        struct('image',UneditedSegmentedObjectImage,...
               'fieldname',['UneditedSegmented',RemainingObjectName]);
end

if ~ isempty(SmallRemovedSegmentedObjectImage)
    OutputImages(length(OutputImages)+1) = ...
        struct('image',SmallRemovedSegmentedObjectImage,...
               'fieldname',['SmallRemovedSegmented',RemainingObjectName]);
end

%%% Saves images to the handles structure so they can be saved to the hard
%%% drive, if the user requested.
if ~strcmp(SaveOutlines,'Do not use')
    OutputImages(length(OutputImages)+1) = struct(...
        'image',CPlabelperim(my_data.IncludedSegmentedObjectImage,8),...
        'fieldname',SaveOutlines);
end

for imgstruct =OutputImages
    if strcmp(RenumberOrRetain,'Renumber')
        handles.Pipeline.(imgstruct.fieldname) = bwlabel(imgstruct.image);
    else
        handles.Pipeline.(imgstruct.fieldname) = imgstruct.image;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Continue_Callback - handle the continue button
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function Continue_Callback(hObject, eventdata, ThisModuleFigureNumber) %#ok<INUSL>
    set(hObject,'UserData',0);
    uiresume(ThisModuleFigureNumber);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% ImageClickCallback - handle a click on one of the images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function ImageClickCallback(hObject,eventdata,operation) %#ok<INUSL>
    pos=get(get(gcbo,'Parent'),'CurrentPoint');
    pos=floor(pos(1,1:2));
    hData = get(hObject,'UserData');
    if ishandle(hData.hContinue)
        x=struct('operation',operation,'pos',pos);
        set(hData.hContinue,'UserData',x);
    end
    uiresume(gcf);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% CommandCallback - handle a command button
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function CommandCallback(hObject, eventdata, hContinue, operation) %#ok<INUSL>
    set(hContinue,'UserData',struct('operation',operation,'pos',[]));
    uiresume(gcf);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Update - update the GUI and data for a image click
%%%    operation - 'toggle','keep' or 'remove'
%%%    pos - position of click in image coords
%%%    hData - my_data structure above
%%%    handles - main handles for CP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function hDataOut=Update(operation, pos, hData, handles)

    if ~isempty(pos)
        label = hData.SegmentedObjectImage(pos(2),pos(1));
        if label == 0
            operation='none';
        end
    end
    if strcmp(operation,'toggle')
        if ~ isempty(find(hData.Filter==label,1))
            operation='remove';
        else
            operation='keep';
        end
    end
    if strcmp(operation,'keep')
        if isempty(find(hData.Filter==label, 1))
            hData.Filter(length(hData.Filter)+1) = label;
        end
    elseif strcmp(operation,'remove')
        hData.Filter=hData.Filter(hData.Filter~=label);
    elseif strcmp(operation,'keepall')
        hData.Filter=AllLabels(hData.SegmentedObjectImage);
    elseif strcmp(operation,'removeall')
        hData.Filter = [];
    elseif strcmp(operation,'toggleall')
        NewLabels=AllLabels(hData.SegmentedObjectImage);
        hData.Filter=NewLabels(~ ismember(NewLabels,hData.Filter));
    elseif ~strcmp(operation,'none')
        error(['Unsupported operation: ',operation]);
    end
    hData.IncludedSegmentedObjectImage = hData.SegmentedObjectImage;
    hData.ExcludedSegmentedObjectImage = hData.SegmentedObjectImage;
    hData.IncludedSegmentedObjectImage(~ismember(hData.IncludedSegmentedObjectImage,hData.Filter))=0;
    hData.ExcludedSegmentedObjectImage(ismember(hData.ExcludedSegmentedObjectImage,hData.Filter))=0;
    if max(hData.IncludedSegmentedObjectImage(:)) == 0
        img = zeros(size(hData.IncludedSegmentedObjectImage,1),...
                    size(hData.IncludedSegmentedObjectImage,2),3);
    else
        img = CPlabel2rgb(handles, hData.IncludedSegmentedObjectImage);
    end
    hData.hIncluded=CPimagesc(img, handles,hData.hAxIncluded);
    set(hData.hIncluded,'ButtonDownFcn',{@ImageClickCallback,'remove'});
    if max(hData.ExcludedSegmentedObjectImage(:)) == 0
        img = zeros(size(hData.ExcludedSegmentedObjectImage,1),...
                    size(hData.ExcludedSegmentedObjectImage,2),3);
    else
        img = CPlabel2rgb(handles, hData.ExcludedSegmentedObjectImage);
    end
    hData.hExcluded=CPimagesc(img, handles,hData.hAxExcluded);
    set(hData.hExcluded,'ButtonDownFcn',{@ImageClickCallback,'keep'});
    set(hData.hOriginal,'UserData',hData);
    set(hData.hIncluded,'UserData',hData);
    set(hData.hExcluded,'UserData',hData);
    hDataOut = hData;

function Filter=AllLabels(SegmentedObjectImage)
    Filter=unique(SegmentedObjectImage(SegmentedObjectImage ~= 0));
