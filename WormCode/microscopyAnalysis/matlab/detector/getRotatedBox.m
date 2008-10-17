function box = getRotatedBox(I,x,y,sz,angle)
% extract from the B/W image a rotated box of size "size" at
% counter-clockwise angle "angle" centered at x,y
%
% I is a cell array with the rotated versions of the image. The rotations
% are in the range [0,90). Other rotations are generated using reflections.
% x and y should be given as an offset relative to the center of the image.
    
    global as;
    
    if(mod(sz,2) ~= 1)
        error(['sz= ' num2str(sz) ', it should be an odd number']);
    end
    half_sz = (sz-1)/2;
    
    angle_index = round(angle/as);
    ai = mod(angle_index,90/as);        % the angle index, modulo 90 degrees
    angle_quadrant = mod(floor(angle_index/(90/as)),4); % the quadrant of the angle
    rounded_angle = ai*as;
    
   
    [h,w] = size(I{ai+1});
    mid_x = w/2;
    mid_y = h/2;
    
    d =  [x, y] * [cosd(rounded_angle)  -sind(rounded_angle); sind(rounded_angle)  cosd(rounded_angle)];
    rx = round(mid_x + d(1));
    ry = round(mid_y + d(2));
    
    [my,mx] = size(I{ai+1});
    if ((ry-half_sz)<1 || (ry+half_sz)>my || (rx-half_sz)<1 || (rx+half_sz)>mx )
        fprintf('problem\n');
    end
    box = I{ai+1}(ry-half_sz:ry+half_sz,rx-half_sz:rx+half_sz);
      

    % perform large rotations through invertion of indexes and transposition.
    switch angle_quadrant
      case 1
        box = box(:,end:-1:1)';
      case 2
        box = box(end:-1:1, end:-1:1);
      case 3
        box = box(end:-1:1, :)';
    end
    
    
