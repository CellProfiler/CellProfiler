function handles = ShowPixelData(handles)

% Help for the Show Pixel Data tool:
% Category: Image Tools
%
% Use this tool if you have an image already displayed in a figure
% window and would like to determine the X, Y position or the
% intensity at a particular pixel.
%
% See also SHOWIMAGE, SHOWDATAONIMAGE.

% CellProfiler is distributed under the GNU General Public License.
% See the accompanying file LICENSE for details.
%
% Developed by the Whitehead Institute for Biomedical Research.
% Copyright 2003,2004,2005.
%
% Authors:
%   Anne Carpenter <carpenter@wi.mit.edu>
%   Thouis Jones   <thouis@csail.mit.edu>
%   In Han Kang    <inthek@mit.edu>
%
% $Revision$

FigureNumber = inputdlg('In which figure number would like to see pixel data?','',1);
if ~isempty(FigureNumber)
    FigureNumber = str2double(FigureNumber{1});
    pixval(FigureNumber,'on')
end