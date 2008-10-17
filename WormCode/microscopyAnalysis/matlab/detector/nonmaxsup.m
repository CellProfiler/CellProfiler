% NONMAXSUP - Non-maxima suppression
%
% Usage:
%          [im,location] = nonmaxsup(inimage, orient, radius);
%
% Function for performing non-maxima suppression on an image using an
% orientation image.  It is assumed that the orientation image gives 
% feature normal orientation angles in degrees (0-180).
%
% Input:
%   inimage - Image to be non-maxima suppressed.
% 
%   orient  - Image containing feature normal orientation angles in degrees
%             (0-180), angles positive anti-clockwise.
% 
%   radius  - Distance in pixel units to be looked at on each side of each
%             pixel when determining whether it is a local maxima or not.
%             This value cannot be less than 1.
%             (Suggested value about 1.2 - 1.5)
%
% Returns:
%   im        - Non maximally suppressed image.
%   location  - Complex valued image holding subpixel locations of edge
%               points. For any pixel the real part holds the subpixel row
%               coordinate of that edge point and the imaginary part holds
%               the column coordinate.  (If a pixel value is 0+0i then it
%               is not an edgepoint.)
%               (Note that if this function is called without 'location'
%               being specified as an output argument is not computed)
%
% Notes:
%
% The suggested radius value is 1.2 - 1.5 for the following reason. If the
% radius parameter is set to 1 there is a chance that a maxima will not be
% identified on a broad peak where adjacent pixels have the same value.  To
% overcome this one typically uses a radius value of 1.2 to 1.5.  However
% under these conditions there will be cases where two adjacent pixels will
% both be marked as maxima.  Accordingly there is a final morphological
% thinning step to correct this.
%
% This function is slow.  It uses bilinear interpolation to estimate
% intensity values at ideal, real-valued pixel locations on each side of
% pixels to determine if they are local maxima.

% Copyright (c) 1996-2005 Peter Kovesi
% School of Computer Science & Software Engineering
% The University of Western Australia
% http://www.csse.uwa.edu.au/
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
%
% The Software is provided "as is", without warranty of any kind.

% December  1996 - Original version
% September 2004 - Subpixel localization added
% August    2005 - Made Octave compatible


function [im, location] = nonmaxsup(inimage, orient, radius)

if any(size(inimage) ~= size(orient))
  error('image and orientation image are of different sizes');
end

if radius < 1
  error('radius must be >= 1');
end

v = version; Octave = v(1)<'5';  % Crude Octave test    

[rows,cols] = size(inimage);
im = zeros(rows,cols);        % Preallocate memory for output image

if nargout == 2
    location = zeros(rows,cols);
end

iradius = ceil(radius);

% Precalculate x and y offsets relative to centre pixel for each orientation angle 

angle = [0:180].*pi/180;    % Array of angles in 1 degree increments (but in radians).
xoff = radius*cos(angle);   % x and y offset of points at specified radius and angle
yoff = radius*sin(angle);   % from each reference position.

hfrac = xoff - floor(xoff); % Fractional offset of xoff relative to integer location
vfrac = yoff - floor(yoff); % Fractional offset of yoff relative to integer location

orient = fix(orient)+1;     % Orientations start at 0 degrees but arrays start
                            % with index 1.

% Now run through the image interpolating grey values on each side
% of the centre pixel to be used for the non-maximal suppression.

for row = (iradius+1):(rows - iradius)
  for col = (iradius+1):(cols - iradius) 

    or = orient(row,col);   % Index into precomputed arrays
    x = col + xoff(or);     % x, y location on one side of the point in question
    y = row - yoff(or);

    fx = floor(x);          % Get integer pixel locations that surround location x,y
    cx = ceil(x);
    fy = floor(y);
    cy = ceil(y);
    tl = inimage(fy,fx);    % Value at top left integer pixel location.
    tr = inimage(fy,cx);    % top right
    bl = inimage(cy,fx);    % bottom left
    br = inimage(cy,cx);    % bottom right

    upperavg = tl + hfrac(or) * (tr - tl);  % Now use bilinear interpolation to
    loweravg = bl + hfrac(or) * (br - bl);  % estimate value at x,y
    v1 = upperavg + vfrac(or) * (loweravg - upperavg);

  if inimage(row, col) > v1 % We need to check the value on the other side...

    x = col - xoff(or);     % x, y location on the `other side' of the point in question
    y = row + yoff(or);

    fx = floor(x);
    cx = ceil(x);
    fy = floor(y);
    cy = ceil(y);
    tl = inimage(fy,fx);    % Value at top left integer pixel location.
    tr = inimage(fy,cx);    % top right
    bl = inimage(cy,fx);    % bottom left
    br = inimage(cy,cx);    % bottom right
    upperavg = tl + hfrac(or) * (tr - tl);
    loweravg = bl + hfrac(or) * (br - bl);
    v2 = upperavg + vfrac(or) * (loweravg - upperavg);

    if inimage(row,col) > v2            % This is a local maximum.
      im(row, col) = inimage(row, col); % Record value in the output
                                        % image.
					

      % Code for sub-pixel localization if it was requested					
      if nargout == 2
         % Solve for coefficients of parabola that passes through 
	 % [-1, v1]  [0, inimage] and [1, v2]. 
	 % v = a*r^2 + b*r + c
	 c = inimage(row,col);
	 a = (v1 + v2)/2 - c;
	 b = a + c - v1;
	 
	 % location where maxima of fitted parabola occurs
	 r = -b/(2*a);
	 location(row,col) = complex(row + r*yoff(or), col - r*xoff(or));      
      end
      
    end

   end
  end
end


% Finally thin the 'nonmaximally suppressed' image by pointwise
% multiplying itself with a morphological skeletonization of itself.
%
% I know it is oxymoronic to thin a nonmaximally supressed image but 
% fixes the multiple adjacent peaks that can arise from using a radius
% value > 1.

if Octave
    skel = bwmorph(im>0,'thin',Inf);   % Octave bwmorph only works on binary
                                       % images and 'thin' seems to produce
                                       % better results.
else
    skel = bwmorph(im,'skel',Inf);
end
im = im.*skel;
if nargout == 2
    location = location.*skel;
end
