function CPImageTool(varargin)

% This function opens or updates the Image Tool window.
% It should be invoked when the user clicks on an image produced
% by a module. 
%
% Example of usage:
% ImageHandle = CPimagesc(SegmentedObjects);
% set(ImageHandle,'ButtonDownFcn','CPImageTool')

% Check that the input argument is an action in the form of a string

if ~isempty(varargin)
    action = varargin{1};
    [foo, ITh] = gcbo;
    if ishandle(get(ITh,'UserData'))  % The user might have closed the figure with the current image handle, check that it exists!
        switch action
            case {'NewWindow'}        % Show image in a new window
                drawnow
                CPfigure;
                data = get(get(ITh,'UserData'),'Cdata');
                if ndims(data) == 2
                    CPimagesc(data),axis image,colormap gray    % Scalar image
                else
                    image(data),axis image                    % RGB image
                end
                title(get(get(ITh,'UserData'),'Tag'))
            case {'Histogram'}                                % Produce histogram (only for scalar images)
                drawnow
                CPfigure
                data = get(get(ITh,'UserData'),'Cdata');
                hist(data(:),min(200,round(length(data(:))/150)));
                title(['Histogram for ' get(get(ITh,'UserData'),'Tag')])
                grid on
            case {'MatlabWS'}                                 % Store image in Matlab base work space
                assignin('base','ImageToolIm',get(get(ITh,'UserData'),'Cdata'));
            otherwise
                disp('Unknown action')                        % Should never get here, but just in case.
        end
    end
else
    handle = gcbo;
    % Check if the Image Tool window already is open
    ITh = findobj('Tag','Image Tool');
    if ~isempty(ITh)
        CPfigure(ITh);
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
        ITh = CPfigure;
        set(ITh,'units','inches','resize','off','menubar','none','toolbar','none','numbertitle','off','Tag','Image Tool','Name','Image Tool');
        set(ITh,'UserData',handle);
        pos = get(ITh,'position');
        set(ITh,'position',[pos(1) pos(2) 1.5 3]);

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
        set(NewWindow,'Callback','CPImageTool(''NewWindow'')')
        set(Histogram,'Callback','CPImageTool(''Histogram'')')
        set(MatlabWS,'Callback','CPImageTool(''MatlabWS'')')
        set(Cancel,'Callback','[foo,ITh] = gcbo;close(ITh); clear foo ITh')

        % Currently, no histogram function for RGB images
        if ndims(get(handle,'Cdata')) ~= 2
            set(Histogram,'Enable','off')
        end
    end
end