function DisplayTestImages
FileList{1} = 'AS_09125_050115070001_A01f00d0LINE.jpg';
FileList{2} = 'AS_09125_050115070001_A01f00d1LINE.jpg';
FileList{3} = 'AS_09125_050115070001_A01f01d0LINE.jpg';
FileList{4} = 'AS_09125_050115070001_A01f01d1LINE.jpg';
FileList{5} = 'AS_09125_050116000001_A01f02d0LINE.jpg';
FileList{6} = 'AS_09125_050116000001_A01f02d1LINE.jpg';
FileList{7} = 'AS_09125_050116000001_A01f03d0LINE.jpg';
FileList{8} = 'AS_09125_050116000001_A01f03d1LINE.jpg';
FileList{9} = 'AS_09125_050116180001_H10f02d0LINE.jpg';
FileList{10} = 'AS_09125_050116180001_H10f02d1LINE.jpg';
FileList{11} = 'AS_09125_050116180001_H10f03d0LINE.jpg';
FileList{12} = 'AS_09125_050116180001_H10f03d1LINE.jpg';
FileList{13} = 'AS_09125_050117110001_H10f02d0LINE.jpg';
FileList{14} = 'AS_09125_050117110001_H10f02d1LINE.jpg';
FileList{15} = 'AS_09125_050117110001_H10f03d0LINE.jpg';
FileList{16} = 'AS_09125_050117110001_H10f03d1LINE.jpg';

figure
for i = 1:2:length(FileList)
    Img = imread(FileList{i});
    subplot(2,4,(i+1)/2),
    imagesc(Img),
    colormap(gray)
    title(FileList{i})
end
figure
for i = 2:2:length(FileList)
    Img = imread(FileList{i});
    subplot(2,4,i/2),
    imagesc(Img),
    colormap(gray)
    title(FileList{i})
end