function b = CPblkproc(varargin)
% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Please see the AUTHORS file for credits.
%
% Website: http://www.cellprofiler.org
%
% $Revision$

% This function is very similar to Matlab's function blkproc.  The only
% difference is that CPblkproc can take a third dimension for the block
% size.  This function is necessary for CPthreshold.
   
[a, block, border, fun, params, padval] = parse_inputs(varargin{:});

% Expand A: Add border, pad if size(a) is not divisible by block.
[ma,na,pa] = size(a);
mpad = rem(ma,block(1)); if mpad>0, mpad = block(1)-mpad; end
npad = rem(na,block(2)); if npad>0, npad = block(2)-npad; end
ppad = rem(pa,block(3)); if ppad>0, ppad = block(3)-ppad; end
if (isa(a, 'uint8'))
    if (padval == 1)
        aa = repmat(uint8(1), ma+mpad+2*border(1),na+npad+2*border(2),pa+ppad+2*border(3));
    else
        aa = repmat(uint8(0), ma+mpad+2*border(1),na+npad+2*border(2),pa+ppad+2*border(3));
    end
elseif isa(a, 'uint16')
    if (padval == 1)
        aa = repmat(uint16(1), ma+mpad+2*border(1),na+npad+2*border(2),pa+ppad+2*border(3));
    else
        aa = repmat(uint16(0), ma+mpad+2*border(1),na+npad+2*border(2),pa+ppad+2*border(3));
    end
else
    if (padval == 1)
        aa = ones(ma+mpad+2*border(1),na+npad+2*border(2),pa+ppad+2*border(3));
    else
        aa = zeros(ma+mpad+2*border(1),na+npad+2*border(2),pa+ppad+2*border(3));
    end
end
aa(border(1)+(1:ma),border(2)+(1:na),border(3)+(1:pa)) = a;

%
% Process first block.
%
m = block(1) + 2*border(1);
n = block(2) + 2*border(2);
p = block(3) + 2*border(3);
mblocks = (ma+mpad)/block(1);
nblocks = (na+npad)/block(2);
pblocks = (pa+ppad)/block(3);
arows = 1:m; acols = 1:n; avecs = 1:p;
for i=1:p
    x(:,:,i) = aa(arows, acols, i);
end
firstBlock = feval(fun,x,params{:});

[a,b,c] = size(firstBlock);

firstBlock = feval(fun,x,params{:});
if (isempty(firstBlock))
  style = 'e'; % empty
  b = [];
elseif (all([a b c] == size(x)))
  style = 's'; % same
  % Preallocate output.
  if (isa(firstBlock, 'uint8'))
     b = repmat(uint8(0), ma+mpad, na+npad, pa+ppad);
  elseif (isa(firstBlock, 'uint16'))
     b = repmat(uint16(0), ma+mpad, na+npad, pa+ppad);
  else
     b = zeros(ma+mpad, na+npad, pa+ppad);
  end
  brows = 1:block(1);
  bcols = 1:block(2);
  bvecs = 1:block(3);
  mb = block(1);
  nb = block(2);
  pb = block(3);
  xrows = brows + border(1);
  xcols = bcols + border(2);
  xvecs = bvecs + border(3);
  b(brows, bcols, bcols) = firstBlock(xrows, xcols, xvecs);
elseif (all([a b c] == (size(x)-2*border)))
  style = 'b'; % border
  % Preallocate output.
  if (isa(firstBlock, 'uint8'))
      b = repmat(uint8(0), ma+mpad, na+npad, pa+ppad);
  elseif (isa(firstBlock, 'uint16'))
      b = repmat(uint16(0), ma+mpad, na+npad, pa+ppad);
  else
      b = zeros(ma+mpad, na+npad, pa+ppad);
  end
  brows = 1:block(1);
  bcols = 1:block(2);
  bvecs = 1:block(3);
  b(brows, bcols, bvecs) = firstBlock;
  mb = block(1);
  nb = block(2);
  pb = block(3);
else
  style = 'd'; % different
  [P,Q,R] = size(firstBlock);
  brows = 1:P;
  bcols = 1:Q;
  bvecs = 1:R;
  mb = P;
  nb = Q;
  pb = R;
  if (isa(firstBlock, 'uint8'))
      b = repmat(uint8(0), mblocks*P, nblocks*Q, pblocks*R);
  elseif (isa(firstBlock, 'uint16'))
      b = repmat(uint16(0), mblocks*P, nblocks*Q, pblocks*R);
  else
      b = zeros(mblocks*P, nblocks*Q, pblocks*R);
  end
  b(brows, bcols, bvecs) = firstBlock;
end

[rr,cc, ll] = meshgrid(0:(mblocks-1), 0:(nblocks-1), 0:(pblocks-1));
rr = rr(:);
cc = cc(:);
ll = ll(:);
mma = block(1);
nna = block(2);
ppa = block(3);
for k = 2:length(rr)
  x = aa(rr(k)*mma+arows,cc(k)*nna+acols, ll(k)*ppa+avecs);
  c = feval(fun,x,params{:});
  if (style == 's')
    b(rr(k)*mb+brows,cc(k)*nb+bcols,ll(k)*pb+bvecs) = c(xrows,xcols,xvecs);
  elseif (style == 'b')
    b(rr(k)*mb+brows,cc(k)*nb+bcols,ll(k)*pb+bvecs) = c;
  elseif (style == 'd')
    b(rr(k)*mb+brows,cc(k)*nb+bcols,ll(k)*pb+bvecs) = c;
  end
end

if ((style == 's') || (style == 'b'))
  b = b(1:ma,1:na,1:pa);
end

%%%
%%% Function parse_inputs
%%%
function [a, block, border, fun, params, padval] = parse_inputs(varargin)

iptchecknargin(2,Inf,nargin,mfilename);

switch nargin
case 3
    % BLKPROC(A, [m n p], 'fun')
    a = varargin{1};
    block = varargin{2};
    border = [0 0 0];
    fun = fcnchk(varargin{3});
    params = cell(0,0,0);
    padval = 0;
    
case 4
    if (strcmp(varargin{2}, 'indexed'))
        % BLKPROC(X, 'indexed', [m n], 'fun')
        a = varargin{1};
        block = varargin{3};
        border = [0 0 0];
        fun = fcnchk(varargin{4});
        params = cell(0,0);
        padval = 1;
        
    else
        params = varargin(4);
        [fun,msg] = fcnchk(varargin{3}, length(params));
        if isempty(msg)
            % BLKPROC(A, [m n p], 'fun', P1)
            a = varargin{1};
            block = varargin{2};
            border = [0 0 0];
            padval = 0;
            
        else
            % BLKPROC(A, [m n], [mb nb], 'fun')
            a = varargin{1};
            block = varargin{2};
            border = varargin{3};
            fun = fcnchk(varargin{4});
            params = cell(0,0);
            padval = 0;
        end
    end
    
otherwise
    if (strcmp(varargin{2}, 'indexed'))
        params = varargin(5:end);
        [fun,msg] = fcnchk(varargin{4},length(params));
        if isempty(msg)
            % BLKPROC(A, 'indexed', [m n], 'fun', P1, ...)
            a = varargin{1};
            block = varargin{3};
            border = [0  0 0];
            padval = 1;
            
        else
            % BLKPROC(A, 'indexed', [m n], [mb nb], 'fun', P1, ...)
            a = varargin{1};
            block = varargin{3};
            border = varargin{4};
            params = varargin(6:end);
            fun = fcnchk(varargin{5},length(params));
            padval = 1;
            
        end
        
    else
        params = varargin(4:end);
        [fun,msg] = fcnchk(varargin{3},length(params));
        if isempty(msg)
            % BLKPROC(A, [m n], 'fun', P1, ...)
            a = varargin{1};
            block = varargin{2};
            border = [0 0 0];
            padval = 0;
            
        else
            % BLKPROC(A, [m n], [mb nb], 'fun', P1, ...)
            a = varargin{1};
            block = varargin{2};
            border = varargin{3};
            params = varargin(5:end);
            fun = fcnchk(varargin{4}, length(params));
            padval = 0;
            
        end
        
    end
end
    
if (islogical(a) || isa(a,'uint8') || isa(a, 'uint16'))
    padval = 0;
end

