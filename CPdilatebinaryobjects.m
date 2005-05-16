function OutputImage = CPdilatebinaryobjects(InputImage, NumericalObjectDilationRadius)

%%% This filter acts as if we had dilated each object by a certain
%%% number of pixels prior to making the projection. It is faster
%%% to do this convolution when the entire projection is completed
%%% rather than dilating each object as each image is processed.
if  NumericalObjectDilationRadius ~= 0
    StructuringElement = fspecial('gaussian',3*NumericalObjectDilationRadius,NumericalObjectDilationRadius);
    OutputImage = filter2(StructuringElement,InputImage,'same');
end