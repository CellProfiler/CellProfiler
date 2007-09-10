function fout = CPwaitbar(varargin)

% This CP function is present only so we can easily replace the
% waitbar if necessary, and also to change colors and fonts to match
% CP's preferences.  See documentation for waitbar for usage.


fout = waitbar(varargin{:});
userData.Application = 'CellProfiler';
userData.ImageFlag = 0;
set(fout, 'Color', [0.7 0.7 0.9], 'UserData',userData);

ax = get(fout, 'Children');
ttl = get(ax, 'Title');
try 
    handles = guidata(gcbo);
    axFontSize = handles.Preferences.FontSize;
    set(ax, 'FontSize', axFontSize);
    set(ttl, 'FontSize', axFontSize);
catch     
    set(ax, 'FontSize', 12);
    set(ttl, 'FontSize', 12);
end

p = findobj(fout,'Type','patch');
l = findobj(fout,'Type','line');
set(p, 'FaceColor', [0.3 0.3 0.5]);
set(l, 'Color', [0.3 0.3 0.5]);

return;
