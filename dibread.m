function im = dibread(filename, width, height, channels, depth)

% The contents of this file are subject to the Mozilla Public License Version 
% 1.1 (the "License"); you may not use this file except in compliance with 
% the License. You may obtain a copy of the License at 
% http://www.mozilla.org/MPL/
% 
% Software distributed under the License is distributed on an "AS IS" basis,
% WITHOUT WARRANTY OF ANY KIND, either express or implied. See the License
% for the specific language governing rights and limitations under the
% License.
% 
% 
% The Original Code is the DIB Reader.
% 
% The Initial Developer of the Original Code is
% Whitehead Institute for Biomedical Research
% Portions created by the Initial Developer are Copyright (C) 2003,2004
% the Initial Developer. All Rights Reserved.
% 
% Contributor(s):
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$


fid = fopen(filename, 'r');
if (fid == -1),
  error(['DIBREAD could not open ' filename]);
end

ignore = fread(fid, 52, 'uchar');

im = zeros(height,width,channels);

for c=1:channels,
  [data, count] = fread(fid, width * height, 'uint16', 0, 'l');
  if count < (width * height),
    fclose(fid);
    error(['end-of-file encountered in DIBREAD while reading ' filename]);
  end
  im(:,:,c) = reshape(data, [width height])' / (2^depth - 1);
end

fclose(fid);
