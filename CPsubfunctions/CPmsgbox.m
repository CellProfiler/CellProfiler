function h = CPmsgbox(varargin)

h = msgbox(varargin{:});
%%% I think we need to render for the following code to work.
drawnow;

%% This allows message boxes to be closed with 'Windows -> Close All'
userData.Application = 'CellProfiler';
set(h,'UserData',userData);

try
    % Find the ui elements (not the most portable way to do this, hence the "try"
    children = get(h, 'Children');
    ax = children(1);
    okbutton = children(2);
    oldtext = get(ax, 'Children');
    
    % Compute the rescaling
    oldfontsize = get(ax, 'FontSize');
    newfontsize = get(0, 'defaultuicontrolfontsize');
    ratio = (newfontsize+1) / oldfontsize;

    % change fonts, and switch everything to normalized layout
    set(okbutton, 'FontSize', newfontsize, 'units', 'normalized');
    set(ax, 'units', 'normalized', 'FontSize', newfontsize);

    % resize the msgbox
    oldPos = get(h, 'Position');
    set(h, 'Position', [oldPos(1:2), oldPos(3:4)*ratio], 'Resize', 'on', 'Units', 'characters');


    % why are both of these necessary? (I hate matlab gui)
    drawnow;
    refresh(h);
end