function out = CPminimizeenergy(im, LAMBDA1, LAMBDA2, ITERATIONS, DIRECTION)
% $Revision$
	global H W A B C T1 T2 b
	im = CPjustify(double(im));
	[H, W] = size(im);
    if strcmpi(DIRECTION,'diagonal')
    	sx = 1 / sqrt(2);
    	sy = 1 / sqrt(2);
    elseif strcmpi(DIRECTION,'vertical')
        sx = 1;
        sy = 0;
    end
	A = sx^2*1 + sy^2*LAMBDA1;
	B = sy^2*1 + sx^2*LAMBDA1;
	C = .5*sx*sy*(1-LAMBDA1);
	D = 2*(A + B);
    A = A/D; B = B/D; C = C/D;
    T1 = .5*sx*LAMBDA2/D;
    T2 = .5*sy*LAMBDA2/D;
	
    b = zeros(H, W);
	i2 = im(3:H  ,2:W-1);
	i4 = im(2:H-1,3:W  );
	i6 = im(2:H-1,1:W-2);
	i8 = im(1:H-2,2:W-1);
	b(2:H-1,2:W-1) = .5*sx/D*(i8-i2) + .5*sy/D*(i6-i4);
	
    out = zeros(H, W);
    for itr = 1:ITERATIONS
        out = b + imfilter(out, [C, A-T1, -C;B-T2, 0, B+T2;-C, A+T1, C]);
    end
	
	out = CPjustify(out);