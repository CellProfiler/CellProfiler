function ImageTool(handle)
%
% This function opens or updates the Image Tool window.
% It should be invoked when the user clicks on an image produced
% by a module. 
%
% Example of usage:
% ImageHandle = imagesc(SegmentedObjects);
% set(ImageHandle,'ButtonDownFcn','ImageTool(gco)')
%
% This function calls Callback functions located in the file ImageTool_Callbacks()
%


% Check that the input argument is an image handle
if ~strcmp(get(handle,'Type'),'image')
    error('Tried to open the Image Tool for a non-image object. This message should never appear.')
end

% Check if the Image Tool window already is open
ITh = findobj('Tag','Image Tool');
if ~isempty(ITh)
    figure(ITh)
    set(ITh,'UserData',handle);                                  % Store the new handle in the UserData property
    th = findobj(get(ITh,'children'),'style','text');            % Get handle to text object
    
    Title = get(get(get(handle,'Parent'),'Title'),'String');      % Get title of image
    set(th,'string',Title);                                       % Put new text

    % Enable histogram function if 2D image
    if ndims(get(handle,'Cdata')) == 2
        set(findobj(get(ITh,'children'),'tag','Histogram'),'Enable','on')
    else
        set(findobj(get(ITh,'children'),'tag','Histogram'),'Enable','off')
    end

else  
    drawnow
    % Create Image Tool window
    ITh = figure;
    set(ITh,'units','inches','resize','off','menubar','none','toolbar','none','numbertitle','off','Tag','Image Tool','Name','Image Tool');
    set(ITh,'UserData',handle);
    pos = get(ITh,'position');
    set(ITh,'position',[pos(1) pos(2) 1.5 2.5]);
    
    % Get title of image
    Title = get(get(get(handle,'Parent'),'Title'),'String');
    
    % Create buttons
    Text = uicontrol(ITh,'style','text','units','normalized','position',[.1 .8 .8 .1],'string',Title);
    set(Text,'Backgroundcolor',get(ITh,'Color'))
    NewWindow = uicontrol(ITh,'style','pushbutton','units','normalized','position',[.1 .6 .8 .1],'string','Open in new window');
    Histogram = uicontrol(ITh,'style','pushbutton','units','normalized','position',[.1 .45 .8 .1],'string','Show histogram','Tag','Histogram');
    MatlabWS  = uicontrol(ITh,'style','pushbutton','units','normalized','position',[.1 .3 .8 .1],'string','Save to work space');
    Cancel    = uicontrol(ITh,'style','pushbutton','units','normalized','position',[.1 .05 .8 .1],'string','Cancel');

    % Assign callback functions
    set(NewWindow,'Callback','ImageTool_Callbacks(''NewWindow'')')
    set(Histogram,'Callback','ImageTool_Callbacks(''Histogram'')')
    set(MatlabWS,'Callback','ImageTool_Callbacks(''MatlabWS'')')
    set(Cancel,'Callback','[foo,ITh] = gcbo;close(ITh); clear foo ITh')

    % Currently, no histogram function for RGB images
    if ndims(get(handle,'Cdata')) ~= 2
        set(Histogram,'Enable','off')
    end
      
    
end






