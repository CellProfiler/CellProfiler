Filelist = dir('./*.mat');

figure

for i = 1:length(Filelist)
    Filename = Filelist(i).name;
    Image = load(Filename); %%% NEED TO FIX VARIABLES!!!!!!!!!!!!
    subplot(2,7,i), imagesc(Image.Image), colormap(gray), title(Filename), axis image
end
