function Dirname = CPuigetdir(start_path,txt)
% This CP function is present only because Matlab's uigetdir 
% does not display the expected text in thte title bar.
% This adds a dialog box which displays the text which belongs 
% in the title bar.
% See documentation for uigetdir for more information.

% $Revision$

if ismac
    [ScreenWidth,ScreenHeight] = CPscreensize;
    h = CPmsgbox(txt,'','help');
    old_pos = get(h,'position');
    set(h,'Position',[0 ScreenHeight,old_pos(3:4)])
end

Dirname = uigetdir(start_path,txt);

if ismac && ishandle(h),
    close(h);
end
