function DisplayTestImages

% Gets all images with a particular text in the name.
Filelist = dir('./Illum*e*.mat');


% Filelist(1).name = 'AS_09125_050115070001_A01f00d0LINE.jpg';
% Filelist(2).name = 'AS_09125_050115070001_A01f00d1LINE.jpg';
% Filelist(3).name = 'AS_09125_050115070001_A01f01d0LINE.jpg';
% Filelist(4).name = 'AS_09125_050115070001_A01f01d1LINE.jpg';
% Filelist(5).name = 'AS_09125_050116000001_A01f02d0LINE.jpg';
% Filelist(6).name = 'AS_09125_050116000001_A01f02d1LINE.jpg';
% Filelist(7).name = 'AS_09125_050116000001_A01f03d0LINE.jpg';
% Filelist(8).name = 'AS_09125_050116000001_A01f03d1LINE.jpg';
% Filelist(9).name = 'AS_09125_050116180001_H10f02d0LINE.jpg';
% Filelist(10).name = 'AS_09125_050116180001_H10f02d1LINE.jpg';
% Filelist(11).name = 'AS_09125_050116180001_H10f03d0LINE.jpg';
% Filelist(12).name = 'AS_09125_050116180001_H10f03d1LINE.jpg';
% Filelist(13).name = 'AS_09125_050117110001_H10f02d0LINE.jpg';
% Filelist(14).name = 'AS_09125_050117110001_H10f02d1LINE.jpg';
% Filelist(15).name = 'AS_09125_050117110001_H10f03d0LINE.jpg';
% Filelist(16).name = 'AS_09125_050117110001_H10f03d1LINE.jpg';

figure
for i = 1:length(Filelist)
    Img = CPimread(Filelist(i).name,'mat');
    subplot(1,3,i),
    imagesc(Img),
    colormap(gray)
    title(Filelist(i).name)
    axis('image')
end

% TO DISPLAY EVERY OTHER ONE:
% figure
% for i = 1:2:length(Filelist)
%     Img = CPimread(Filelist(i).name);
%     subplot(2,4,(i+1)/2),
%     imagesc(Img),
%     colormap(gray)
%     title(Filelist(i).name)
% end
% figure
% for i = 2:2:length(Filelist)
%     Img = CPimread(Filelist(i).name);
%     subplot(2,4,i/2),
%     imagesc(Img),
%     colormap(gray)
%     title(Filelist(i).name)
% end