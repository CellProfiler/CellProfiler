function out = CPhilberttransform(im, ITERATIONS, ALPHA, EXP, DIRECTION)
% $Revision$
	global H W A E DIAG
    DIAG = strcmpi(DIRECTION,'diagonal');
	A = ALPHA;
    E = EXP;
	[H, W] = size(im);
	im = CPjustify(double(im));
	
	tim = ht(im,0);
	out = tim;
	for i = 2:ITERATIONS
        nim = im - dif(tim);
		tim = ht(nim,i-1);
		out = out + tim;
		im = nim;
	end
	
	out = CPjustify(out);

function out = ht(im,n)
	global H W A E DIAG
    [u, v] = meshgrid([1:ceil(W/2),-floor(W/2):-1]*2*pi/W, [1:ceil(H/2),-floor(H/2):-1]*2*pi/H);
    if DIAG
        out = real(ifft2(-i * A * sign(u+v) ./ (A + n*(abs(u-v) + abs(u+v)).^E) .* fft2(im)));
    else
        out = real(ifft2(-i * A * sign(v) ./ (A + n*(abs(v)).^E) .* fft2(im)));
    end

function out = dif(im)
    global DIAG
	[t1,t2] = gradient(im);
    if DIAG
    	out = (t1+t2)/-2;
    else
        out = -t2;
    end
