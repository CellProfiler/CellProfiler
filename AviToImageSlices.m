function AviToImageSlices(MovieFilename,FileFormat)

Movie = aviread(MovieFilename);
BaseFilename = MovieFilename(1:end-4);

for i = 1:size(Movie,2)
Filename = [BaseFilename,num2str(i),'.',FileFormat];
StructureImage = Movie(i);
Image = StructureImage.cdata;
imwrite(Image, Filename, FileFormat);
end