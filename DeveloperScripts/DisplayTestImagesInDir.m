Filelist = dir('./*_OUT.mat');

figure

for i = 1:length(Filelist)
    Filename = Filelist(i).name;
    load(Filename);
    subplot(7,5,5*(i-1)+1), imagesc(handles.Pipeline.ProjectedOrigBlue), colormap(gray), axis image, colorbar
    if i == 1, title({' ProjectedOrigBlue'}), end    
    subplot(7,5,5*(i-1)+2), imagesc(handles.Pipeline.SmoothProjectedOrigBlue), colormap(gray),  axis image, colorbar
    if i == 1,     title({' SmoothProjectedOrigBlue'}), end
    subplot(7,5,5*(i-1)+3), imagesc(handles.Pipeline.RawProjectedThreshBlue), colormap(gray),  axis image, colorbar
    if i == 1,     title({' RawProjectedThreshBlue'}), end
    subplot(7,5,5*(i-1)+4), imagesc(handles.Pipeline.ProjectedThreshBlue), colormap(gray),  axis image, colorbar
    if i == 1,     title({' ProjectedThreshBlue'}),end
    subplot(7,5,5*(i-1)+5), imagesc(handles.Pipeline.SmoothProjectedThreshBlue), colormap(gray),  axis image, colorbar
    if i == 1,     title({' SmoothProjectedThreshBlue'}), end
end

%         if exist('Image','var')
%         subplot(2,7,i), imagesc(Image), colormap(gray), title(Filename), axis image
%     else
