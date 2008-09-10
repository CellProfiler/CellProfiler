function LabelBoundaryImage = CPlabelperim(LabelMatrixImage, conn)

% A fast fuction to obtain the label boundary image from a label matrix image. 
% The Matlab function 'bwperim' only works for a binary image, i.e.,
% bwperim ignores the labels. An exmaple of LabelBoundaryImage looks like this: 
%
%   LabelBoundaryImage = 0     0     0     1     1     1
%                        0     0     0     1     0     1
%                        0     0     0     1     1     1
%                        2     2     2     2     0     0
%                        2     0     0     2     0     0
%                        2     0     0     2     0     0
%                        2     2     2     2     0     0      
%
% Second, optional argument is neghborhood connectivity, defaulting to 4.
%
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

if nargin == 1,
    conn = 4;
end

[sr sc] = size(LabelMatrixImage);        
ShiftLeft = zeros(sr,sc);
ShiftRight = zeros(sr,sc);
ShiftUp = zeros(sr,sc);
ShiftDown = zeros(sr,sc);
ShiftLeft(:,1:end-1) = LabelMatrixImage(:,2:end);
ShiftRight(:,2:end) = LabelMatrixImage(:,1:end-1);
ShiftUp(1:end-1,:) = LabelMatrixImage(2:end,:);
ShiftDown(2:end,:) = LabelMatrixImage(1:end-1,:);
if conn == 4,
    InnerOuterBoundaryImage = ((ShiftLeft~=LabelMatrixImage) | (ShiftRight~=LabelMatrixImage) | ...
        (ShiftUp~=LabelMatrixImage) | (ShiftDown~=LabelMatrixImage));
else
    ShiftLeftUp = zeros(sr, sc);
    ShiftLeftDown = zeros(sr, sc);
    ShiftRightUp = zeros(sr, sc);
    ShiftRightDown = zeros(sr, sc);
    ShiftLeftUp(1:end-1,:) = ShiftLeft(2:end,:);
    ShiftRightUp(1:end-1,:) = ShiftRight(2:end,:);
    ShiftLeftDown(2:end,:) = ShiftLeft(1:end-1,:);
    ShiftRightDown(2:end,:) = ShiftRight(1:end-1,:);
    InnerOuterBoundaryImage = ((ShiftLeft~=LabelMatrixImage) | (ShiftRight~=LabelMatrixImage) | ...
        (ShiftUp~=LabelMatrixImage) | (ShiftDown~=LabelMatrixImage) | ...
        (ShiftRightUp~=LabelMatrixImage) | (ShiftRightDown~=LabelMatrixImage) | ...
        (ShiftLeftUp~=LabelMatrixImage) | (ShiftLeftDown~=LabelMatrixImage));
end
    
BoundaryImage = LabelMatrixImage & InnerOuterBoundaryImage;
LabelBoundaryImage = BoundaryImage .* LabelMatrixImage;
