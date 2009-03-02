function h = CPmsgbox(varargin)

% $Revision$

h = msgbox(varargin{:});
%%% I think we need to render for the following code to work.
drawnow;

%% This allows message boxes to be closed with 'Windows -> Close All'
userData.Application = 'CellProfiler';
tempstring = varargin{1};
if iscell(tempstring),
    for i=1:length(tempstring),
        % add a newline to each line of the input
        tempstring{i} = [tempstring{i} 10];
    end
    userData.WarningText = [tempstring{:}];
else
    userData.WarningText = tempstring;
end
set(h,'UserData',userData);

if nargin > 2 && ~strcmp(varargin(3),'help')
    try
        % Find the ui elements (not the most portable way to do this, hence the "try"
        ax = findobj(h,'Type','axes','Tag',''); %% IconAxes will have a Tagg that says so
        assert(length(ax) == 1)
        okbutton = findobj(h,'Tag','OKButton');
        assert(length(okbutton) == 1)
        txt = get(ax, 'Children');

        % Compute the rescaling
        oldfontsize = get(ax, 'FontSize');
        newfontsize = get(0, 'defaultuicontrolfontsize');
        ratio = (newfontsize+1) / oldfontsize;


        % move the ok button to the left to make room for the copy-to-clipboard button
        set(ax, 'Units', 'points');
        okPos = get(okbutton, 'Position');
        axPos = get(ax, 'Position');
        % 3.5 is 1 + 2.5, 2.5 is the width of the Copy button below
        okPos(1) = (axPos(3) - 3.5 * okPos(3)) / 2;
        set(okbutton, 'Position', okPos);

        % change fonts, and switch everything to normalized layout
        set(okbutton, 'FontSize', newfontsize, 'units', 'normalized');
        set(ax, 'units', 'normalized', 'FontSize', newfontsize);
        set(txt, 'units', 'normalized', 'Fontsize', newfontsize);



        % resize the msgbox
        oldPos = get(h, 'Position');
        set(h, 'Position', [oldPos(1:2), oldPos(3:4)*ratio], 'Resize', 'on', 'Units', 'characters');

        Font.FontUnits=get(okbutton, 'FontUnits');
        Font.FontSize=get(okbutton,'FontSize');
        Font.FontName=get(okbutton,'FontName');
        Font.FontWeight=get(okbutton,'FontWeight');

        okPos = get(okbutton, 'Position');
        copypos = okPos;
        % set the position next to the OK button
        copypos(1) = copypos(1) + copypos(3);
        % make it 2.5 times wider
        copypos(3) = copypos(3) * 2.5;

        CopyHandle = uicontrol(h                                   , ...
            Font                                                   , ...
            'Style'              ,'pushbutton'                     , ...
            'Units'              ,'normalized'                     , ...
            'Position'           , copypos                         , ...
            'CallBack'           ,'ud = get(gcbf, ''UserData''); clipboard(''copy'', ud.WarningText);'                    , ...
            'KeyPressFcn'        ,@doKeyPress                       , ...
            'String'             ,'Copy To Clipboard'               , ...
            'HorizontalAlignment','center'                          , ...
            'Tag'                ,'CopyButton'                        ...
            );

        % why are both of these necessary? (I hate matlab gui)
        drawnow;
        refresh(h);
    end
end