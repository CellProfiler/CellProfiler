function handles = ShowPixelData(handles)

% Help for the SShow Pixel Data tool:
% Category: Image Tools
%
% This module has not yet been documented.
%
% See also SHOWIMAGE, SHOWDATAONIMAGE (a Data tool).

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