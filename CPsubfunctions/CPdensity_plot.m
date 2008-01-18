function figurenumber = CPdensity_plot(xvals, yvals, figurenumber)

if nargin == 3,
    CPfigure(figurenumber);
else
    figurenumber = CPfigure;
end
   
minx = min(xvals);
maxx = max(xvals);
xedges = linspace(minx, maxx, 100);
xvals = (xvals - minx) / (maxx - minx);

miny = min(yvals);
maxy = max(yvals);
yedges = linspace(miny, maxy, 100);
yvals = (yvals - miny) / (maxy - miny);

counts = full(sparse(round(99 * yvals + 1), round(99 * xvals + 1), 1, 100, 100));
counts = log(counts + 1);
ax = newplot;
surf(xedges, yedges, -counts, counts, 'EdgeColor','none', 'FaceColor','interp');
view(ax,2);
colormap(ax,flipud(gray(256)))
grid(ax,'off');
drawnow
