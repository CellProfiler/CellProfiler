function im=CPlabel2rgb(handles,im)

if sum(sum(im)) >= 1
    cmap = eval([handles.Preferences.LabelColorMap '(max(64,max(im(:))))']);
    im = label2rgb(im, cmap, 'k', 'shuffle');
else
    im=im;
end