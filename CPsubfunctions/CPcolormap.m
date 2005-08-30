function CPcolormap(handles)

ax = gca;

try
    im=get(get(ax,'Children'),'CData');
catch
    return;
end

if size(im,3)~=1
    return
else
    if all(im<=1) %then intensity image
        colormap(handles.Preferences.IntensityColorMap);
    else %then label matrix
        cmap = eval([handles.Preferences.LabelColorMap '(max(64,max(im(:))))']);
        im = label2rgb(im, cmap, 'k', 'shuffle');
        set(get(ax,'Children'),'CData',im);
    end
end