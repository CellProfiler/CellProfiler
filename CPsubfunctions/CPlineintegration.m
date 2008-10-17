function out = CPlineintegration(im, EXP, EXPD, DIRECTION)
% $Revision$
	im = CPjustify(double(im));
	[H, W] = size(im);
    L = max(H,W);
    
    pexp = cumprod(ones(L, 1)*EXP);
    pexpd = cumprod(ones(L, 1)*EXPD);
    sumd = ones(L, 1);
    sumd(2:L) = cumsum(pexpd(1:L-1)) + 1;
    
    if strcmpi(DIRECTION, 'diagonal')
        memd1 = im;
        for i = 2:H
            for j = 1:W-1
                memd1(i,j) = im(i,j) + EXPD*memd1(i-1,j+1);
            end
        end
        memd2 = im;
        for i = H-1:-1:1
            for j = 2:W
                memd2(i,j) = im(i,j) + EXPD*memd2(i+1,j-1);
            end
        end

        [dj, di] = meshgrid(1:W,1:H);
        r = di + H*dj - 1;

        d1 = min(di-1, W-dj);
        d2 = min(H-di, dj-1);
        d = min(d1, d2);

        t = memd1 + memd2 - im;
        t1 = d+1;
        t2 = (H-1)*d;
        t3 = pexpd(d+1);

        ind = find(d < d1);
        for i = 1:length(ind)
            t(ind(i)) = t(ind(i)) - t3(ind(i)) * memd1(r(ind(i))+t2(ind(i)));
        end
%         t(ind) = t(ind) - t3(ind) .* memd1(r(ind)+t2(ind));
        ind = find(d < d2);
        for i = 1:length(ind)
            t(ind(i)) = t(ind(i)) - t3(ind(i)) * memd2(r(ind(i))-t2(ind(i))+(2-2*H));
        end
%         t(ind) = t(ind) - t3(ind) .* memd2(r(ind)-t2(ind)+(2-2*H));
        t = t ./ (2*sumd(t1)-1);

        mem1 = im;
        for i = 2:H
            for j = 2:W
                mem1(i,j) = mem1(i-1,j-1)*EXP + t(i,j);
            end
        end

        mem2 = im;
        for i = H-1:-1:1
            for j = 1:W-1
                mem2(i,j) = mem2(i+1,j+1)*EXP + t(i,j);
            end
        end

        d1 = min(di,dj)-1;
        d2 = min(H-di,W-dj);
        d = min(d1,d2);

        t1 = pexp(d+1);
        t2 = (1+H)*d;

        out = mem1 - mem2 + mean(mean(im));
        ind = find(d < d1 & t1 .* max(max(mem1)) >= .001);
        for i = 1:length(ind)
            out(ind(i)) = out(ind(i)) - t1(ind(i)) * mem1(r(ind(i))-t2(ind(i))-2*H);
        end
%         out(ind) = out(ind) - t1(ind) .* mem1(r(ind)-t2(ind)-2*H);
        ind = find(d < d2 & t1 .* max(max(mem2)) >= .001);
        for i = 1:length(ind)
            out(ind(i)) = out(ind(i)) + t1(ind(i)) * mem2(r(ind(i))+t2(ind(i))+2);
        end
%         out(ind) = out(ind) + t1(ind) .* mem2(r(ind)+t2(ind)+2);

        out = CPjustify(out);
    elseif strcmpi(DIRECTION,'vertical')
        memd1 = im;
        for i = 1:H
            for j = 2:W
                memd1(i,j) = im(i,j) + EXPD*memd1(i,j-1);
            end
        end
        memd2 = im;
        for i = 1:H
            for j = W-1:-1:1
                memd2(i,j) = im(i,j) + EXPD*memd2(i,j+1);
            end
        end
        
        [dj, di] = meshgrid(1:W,1:H);
        d1 = dj-1;
        d2 = W-dj;
        d = min(d1, d2);
        
        t = memd1 + memd2 - im;
        tp = pexpd(d+1);

        ind = find(d < d1);
        for i = 1:length(ind)
            t(ind(i)) = t(ind(i)) - tp(ind(i)) * memd1(ind(i)-H);
        end
%         t(ind) = t(ind) - t3(ind) .* memd1(ind-1);
        ind = find(d < d2);
        for i = 1:length(ind)
            t(ind(i)) = t(ind(i)) - tp(ind(i)) * memd2(ind(i)+H);
        end
%         t(ind) = t(ind) - t3(ind) .* memd2(ind+1);
        t = t ./ (2*sumd(d+1)-1);

        mem1 = im;
        for i = 2:H
            for j = 1:W
                mem1(i,j) = mem1(i-1,j)*EXP + t(i,j);
            end
        end

        mem2 = im;
        for i = H-1:-1:1
            for j = 1:W
                mem2(i,j) = mem2(i+1,j)*EXP + t(i,j);
            end
        end

        d1 = di-1;
        d2 = H-di;
        d = min(d1,d2);

        tp = pexp(d+1);

        out = mem1 - mem2 + mean(mean(im));
        ind = find(d < d1 & tp .* max(max(mem1)) >= .001);
        for i = 1:length(ind)
            out(ind(i)) = out(ind(i)) - tp(ind(i)) * mem1(ind(i)-1);
        end
%         out(ind) = out(ind) - tp(ind) .* mem1(ind-H);
        ind = find(d < d2 & tp .* max(max(mem2)) >= .001);
        for i = 1:length(ind)
            out(ind(i)) = out(ind(i)) + tp(ind(i)) * mem2(ind(i)+1);
        end
%         out(ind) = out(ind) + tp(ind) .* mem2(ind+H);

        out = CPjustify(out);
    end
