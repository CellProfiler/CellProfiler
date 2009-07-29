function ShowOrHidePixelData(handles)

% Help for the Show or Hide Pixel Data tool:
% Category: Image Tools
%
% SHORT DESCRIPTION:
% Shows X,Y pixel location and intensity information in the figure window.
% *************************************************************************
%
% This tool shows the pixel intensity at each X,Y location as you hover 
% over points within an image. The pixels are displayed via a small box at 
% the lower left corner of the figure window. If the image is color (RGB), 
% three intensity values are shown: Red, Green, and Blue.
%
% Currently, it can also measure lengths if you click the mouse at a
% starting point and hold the button down while dragging, although this
% could also be done with the Measure Length tool, accessible by clicking
% on the image of interest and choosing Measure Length from the resulting
% Image Tool window.
%
% To exit the tool, click the 'x' in the pixel intensity information panel.

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

if verLessThan('matlab', '7.8');
    warning off Images:pixval:obsoleteFunction
    pixval
    warning on Images:pixval:obsoleteFunction
else
    %% For now, we can't turn this one off
    impixelinfo
end