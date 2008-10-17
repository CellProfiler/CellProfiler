function zz = zero_crossings(b)

% function zz = zero_crossings(b)

[m, n] = size(b);
rr = 2:m-1; cc=2:n-1;

zz = false(size(b));
[rx,cx] = find( b(rr,cc) < 0 & b(rr,cc+1) > 0);   % [- +]
zz((rx+1) + cx*m) = true;
[rx,cx] = find( b(rr,cc-1) > 0 & b(rr,cc) < 0 );   % [+ -]
zz((rx+1) + cx*m) = true;
[rx,cx] = find( b(rr,cc) < 0 & b(rr+1,cc) > 0 );   % [- +]'
zz((rx+1) + cx*m) = true;
[rx,cx] = find( b(rr-1,cc) > 0 & b(rr,cc) < 0 );   % [+ -]'
zz((rx+1) + cx*m) = true;

[rz,cz] = find( b(rr,cc)==0 );
if ~isempty(rz)
  % Look for the zero crossings: +0-, -0+ and their transposes
  % The edge lies on the Zero point
  zero = (rz+1) + cz*m;   % Linear index for zero points
  ee = find(b(zero-1) < 0 & b(zero+1) > 0 );
  zz(zero(ee)) = 1;

  ee = find(b(zero-1) > 0 & b(zero+1) < 0);      % [+ 0 -]'
  zz(zero(ee)) = 1;
  
  ee = find(b(zero-m) < 0 & b(zero+m) > 0 );     % [- 0 +]
  zz(zero(ee)) = 1;

  ee = find(b(zero-m) > 0 & b(zero+m) < 0 );     % [+ 0 -]
  zz(zero(ee)) = 1;

end
