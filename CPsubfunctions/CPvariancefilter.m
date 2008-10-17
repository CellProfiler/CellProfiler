function out = CPvariancefilter(im, WINDOWSIZE, C)
% $Revision$
    %tic
    WIND = floor((WINDOWSIZE-1)/2);
    im = CPjustify(double(im));
    
    [x, y] = meshgrid(-WIND:WIND,-WIND:WIND);
    c = exp(-(x.^2+y.^2)/(2*C^2)) / (sqrt(2*pi)*C);
    
    out = CPjustify(sqrt(im.*(sum(c(:))*im - 2*imfilter(im,c,'symmetric')) + imfilter(im.^2,c,'symmetric')));
%     out = sqrt(im.*(sum(c(:))*im - 2*imfilter(im,c,'symmetric')) + imfilter(im.^2,c,'symmetric'));
    %toc
