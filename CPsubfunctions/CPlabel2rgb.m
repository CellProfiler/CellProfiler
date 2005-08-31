function im=CPlabel2rgb(handles,im)

cmap = eval([handles.Preferences.LabelColorMap '(max(64,max(im(:))))']);
im = label2rgb(im, cmap, 'k', 'shuffle');