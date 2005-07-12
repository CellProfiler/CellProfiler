if ndims(OrigImage) ~= 2
    s = size(OrigImage);
    if (length(s) == 3 && s(3) == 3)
        OrigImage = OrigImage(:,:,1)+OrigImage(:,:,2)+OrigImage(:,:,3);
    else
        error('Image processing was canceled because the Identify Primary Intensity module requires an input image that is two-dimensional (i.e. X vs Y), but the image loaded does not fit this requirement.  This may be because the image is a color image.')
    end
end