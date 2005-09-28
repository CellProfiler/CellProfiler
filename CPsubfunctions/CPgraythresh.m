function [level] = CPgraythresh(varargin)

if nargin == 1
    im = varargin{1};
else
    im = varargin{1};
    handles = varargin{2};
    ImageName = varargin{3};
    %%% If the image was produced using a cropping mask, we do not
    %%% want to include the Masked part in the calculation of the
    %%% proper threshold, because there will be many zeros in the
    %%% image.  So, we check to see whether there is a field in the
    %%% handles structure that goes along with the image of interest.
    fieldname = ['CropMask', ImageName];
    if isfield(handles.Pipeline,fieldname) == 1
        %%% Retrieves previously selected cropping ellipse from handles
        %%% structure.
        BinaryCropImage = handles.Pipeline.(fieldname);
        %%% Masks the image and I think turns it into a linear
        %%% matrix.
        im = im(logical(BinaryCropImage));
    end
end

%%% The threshold is calculated using the matlab function graythresh
%%% but with our modifications that work in log space, and take into
%%% account the max and min values in the image.
im = double(im(:));

if max(im) == min(im),
    level = im(0);
else,
    %%% We want to limit the dynamic range of the image to 256.
    %%% Otherwise, an image with almost all values near zero can give a
    %%% bad result.
    minval = max(im)/256;
    im(im < minval) = minval;
    im = log(im);
    minval = min (im);
    maxval = max (im);
    im = (im - minval) / (maxval - minval);
    level = exp(minval + (maxval - minval) * graythresh(im));
end
