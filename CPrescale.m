function OutputImage = CPrescale(InputImage,RescaleOption,OrigImage)

% See the help for RESCALEINTENSITY for details. 

if strncmpi(RescaleOption,'N',1) == 1
    OutputImage = InputImage;
elseif strncmpi(RescaleOption,'S',1) == 1
    %%% The minimum of the image is brought to zero, whether it
    %%% was originally positive or negative.
    IntermediateImage = InputImage - min(min(InputImage));
    %%% The maximum of the image is brought to 1.
    OutputImage = IntermediateImage ./ max(max(IntermediateImage));
elseif strncmpi(RescaleOption,'M',1) == 1
    %%% Rescales the image so the max equals the max of
    %%% the original image.
    IntermediateImage = InputImage ./ max(max(InputImage));
    OutputImage = IntermediateImage .* max(max(OrigImage));
elseif strncmpi(RescaleOption,'G',1) == 1
    %%% Rescales the image so that all pixels are equal to or greater
    %%% than one. This is done by dividing each pixel of the image by
    %%% a scalar: the minimum pixel value anywhere in the smoothed
    %%% image. (If the minimum value is zero, .0001 is substituted
    %%% instead.) This rescales the image from 1 to some number. This
    %%% is useful in cases where other images will be divided by this
    %%% image, because it ensures that the final, divided image will
    %%% be in a reasonable range, from zero to 1.
    drawnow
    OutputImage = InputImage ./ max([min(min(InputImage)); .0001]);
else error(['For the rescaling option, you must enter N, S, M, or E for the method by which to rescale the image. Your entry was ', RescaleOption])
end