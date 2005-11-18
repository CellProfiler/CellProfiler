function h = CPimagesc(Image)

h = imagesc(Image);
set(h,'ButtonDownFcn','CPImageTool(gco)');