function out = CPminimizeenergy2(I, L1, L2, ITERATIONS, EPS, inMethod, DIRECTION, inK)
% $Revision$
	global b K method sqrFilter smoothFilter integrableFilter frame
	im = CPjustify(double(I));
	method = inMethod;
    
    if strcmp(method,'abs2') || strcmp(method,'sin')
        K = inK;
    elseif ~strcmp(method,'abs') && ~strcmp(method,'sqr')
        error(sprintf('Unknown method %s', method));
        return
    end
    
	[H, W] = size(im);
    
    if strcmpi(DIRECTION,'diagonal')
    	sx = 1 / sqrt(2);
    	sy = 1 / sqrt(2);
    elseif strcmpi(DIRECTION,'vertical')
        sx = 1;
        sy = 0;
    end
    
    b = zeros(H,W);
    i2 = im(3:H  ,2:W-1);
    i4 = im(2:H-1,3:W  );
    i6 = im(2:H-1,1:W-2);
    i8 = im(1:H-2,2:W-1);
    b(2:H-1,2:W-1) = sx*i2 + sy*i4 - sy*i6 - sx*i8;
    sqrFilter = [-sx*sy,-2*sx^2,sx*sy;-2*sy^2,4*(sx^2+sy^2),-2*sy^2;sx*sy,-2*sx^2,-sx*sy];
    if ~strcmpi(method,'sin')
        if strcmp(method,'abs2')
            b = -K * b;
            sqrFilter = -K * sqrFilter;
        end
    else
        b = (pi/K).*b;
        sqrFilter = (pi/K).*sqrFilter;
        [y,x] = meshgrid(1:W,1:H);
        frame = find(x < 3 | x > H-2 | y < 3 | y > W-2);
    end
    smoothFilter = L1*[sx*sy,-2*sy^2,-sx*sy;-2*sx^2,4*(sx^2+sy^2),-2*sx^2;-sx*sy,-2*sy^2,sx*sy];
    integrableFilter = L2*[0,sx,0;sy,0,-sy;0,-sx,0];
    
	out = zeros(H,W);
	for itr = 1:ITERATIONS
        out = out - EPS*grad(out);
	end
	
	out = CPjustify(out);
end

function out = grad(f)
    global b K method sqrFilter smoothFilter integrableFilter frame
    
    if strcmp(method,'sqr')
        out = b + imfilter(f,sqrFilter+smoothFilter+integrableFilter);
    elseif strcmp(method,'abs')
        out = sign(b+imfilter(f,sqrFilter)) + imfilter(f,smoothFilter+integrableFilter);
    elseif strcmp(method,'abs2')
        out = 2 ./ (1 + exp(b+imfilter(f,sqrFilter))) - 1 + imfilter(f,smoothFilter+integrableFilter);
    elseif strcmp(method,'sin')
        baseError = b + imfilter(f,sqrFilter);
        error = (pi/K).*sin(baseError);
        error(abs(baseError) > pi) = 0;
        out = error + imfilter(f,smoothFilter+integrableFilter);
        out(frame) = 0;
    end
end