function box = getRotatedDerivBox(D,x,y,sz,angle)
% extract from a deriv image pair rotated box of size "size" at
% counter-clockwise angle "angle" centered at x,y
%
% D is a cell array with the rotated versions of the derivative. The rotations
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
    
    [h,w] = size(D{ai+1}.mag);
    mid_x = w/2;
    mid_y = h/2;
    
    d =  [x, y] * [cosd(rounded_angle)  -sind(rounded_angle); sind(rounded_angle)  cosd(rounded_angle)];
    rx = round(mid_x + d(1));
    ry = round(mid_y + d(2));
    
    box.mag   = D{ai+1}.mag(ry-half_sz:ry+half_sz,rx-half_sz:rx+half_sz);
    box.angle = D{ai+1}.angle(ry-half_sz:ry+half_sz,rx-half_sz:rx+half_sz);
    
    % perform large rotations through invertion of indexes and transposition.
    switch angle_quadrant
      case 1
        box.mag = box.mag(:,end:-1:1)';
        box.angle = mod(box.angle(:,end:-1:1)'-0.25,1);
      case 2
        box.mag = box.mag(end:-1:1, end:-1:1);
        box.angle = mod(box.angle(end:-1:1, end:-1:1)-0.5,1);
      case 3
        box.mag = box.mag(end:-1:1, :)';
        box.angle = mod(box.angle(end:-1:1, :)'-0.75,1);
    end
    
    
