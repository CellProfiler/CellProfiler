function imout = clearborder(im)

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne E. Carpenter
%   Thouis Ray Jones
%   In Han Kang
%   Ola Friman
%   Steve Lowe
%   Joo Han Chang
%   Colin Clarke
%   Mike Lamprecht
%   Susan Ma
%   Wyman Li
%
% Website: http://www.cellprofiler.org
%
% $Revision: 2802 $

map = 0:max(im(:));
map(im(1,:)+1)=0;
map(im(end,:)+1)=0;
map(im(:,1)+1)=0;
map(im(:,end)+1)=0;
imout = map(im+1);
