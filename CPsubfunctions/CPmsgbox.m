function h = CPmsgbox(varargin)

h = msgbox(varargin{:});
%%% I think we need to render for the following code to work.
drawnow;


try
    % Find the ui elements (not the most portable way to do this, hence the "try"
    children = get(h, 'Children');
    ax = children(1);
    okbutton = children(2);
    textui = get(ax, 'Children');
    
    % Compute the rescaling
    oldfontsize = get(textui, 'FontSize');
    newfontsize = get(0, 'defaultuicontrolfontsize');
    ratio = newfontsize / oldfontsize;
    
    % change fonts, and switch everything to normalized layout
    set(ax, 'units', 'normalized');
    set(okbutton, 'FontSize', newfontsize, 'units', 'normalized');
    set(textui, 'FontSize', newfontsize, 'units', 'normalized');
    
    % resize the msgbox
    oldPos = get(h, 'Position');
    set(h, 'Position', [oldPos(1:2), oldPos(3:4)*ratio]);
    % why are both of these necessary? (I hate matlab gui)
    drawnow;
    refresh(h);
end