function Grid = CPmakegrid(GridInfo)

% This subfunction will take GridInfo and create the grid lines
% neccessary for any module which requires a grid
%
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
%   Peter Swire
%   Rodrigo Ipince
%   Vicky Lay
%   Jun Liu
%   Chris Gang
%
% Website: http://www.cellprofiler.org
%
% $Revision: 2802 $

XLocationOfLowestXSpot = GridInfo.XLocationOfLowestXSpot;
YLocationOfLowestYSpot= GridInfo.YLocationOfLowestYSpot;
XSpacing = GridInfo.XSpacing;
YSpacing = GridInfo.YSpacing;
Rows = GridInfo.Rows;
Columns = GridInfo.Columns;
TotalHeight = GridInfo.TotalHeight;
TotalWidth = GridInfo.TotalWidth;
LeftOrRight = GridInfo.LeftOrRight;
TopOrBottom = GridInfo.TopOrBottom;
RowsOrColumns = GridInfo.RowsOrColumns;

%%% Calculates and arranges the integer numbers for labeling.
LinearNumbers = 1:Rows*Columns;
if strcmp(RowsOrColumns,'Columns')
    SpotTable = reshape(LinearNumbers,Rows,Columns);
elseif strcmp(RowsOrColumns,'Rows')
    SpotTable = reshape(LinearNumbers,Columns,Rows);
    SpotTable = SpotTable';
end
if strcmp(LeftOrRight,'Right')
    SpotTable = fliplr(SpotTable);
end
if strcmp(TopOrBottom,'Bottom')
    SpotTable = flipud(SpotTable);
end

%%% Calculates the locations for all the sample labelings (whether it is
%%% numbers, spot identifying information, or coordinates).
GridXLocations = SpotTable;
for g = 1:size(GridXLocations,2)
    GridXLocations(:,g) = XLocationOfLowestXSpot + (g-1)*XSpacing - round(XSpacing/2);
end
%%% Converts to a single column.
XLocations = reshape(GridXLocations, 1, []);
%%% Same routine for Y.
GridYLocations = SpotTable;
for h = 1:size(GridYLocations,1)
    GridYLocations(h,:) = YLocationOfLowestYSpot + (h-1)*YSpacing - round(YSpacing/2);
end
YLocations = reshape(GridYLocations, 1, []);

%%% Calculates the lines.
%%% Adds extra spaced line to end of X locations
VertLinesX(1,:) = [GridXLocations(1,:),GridXLocations(1,end)+XSpacing];
VertLinesX(2,:) = [GridXLocations(1,:),GridXLocations(1,end)+XSpacing];
VertLinesY(1,:) = repmat(0,1,size(GridXLocations,2)+1); %%% Same as zeros(1,size(GridXLocations,2)+1)
VertLinesY(2,:) = repmat(TotalHeight,1,size(GridXLocations,2)+1);
%%% Adds extra spaced line to end of X locations
HorizLinesY(1,:) = [GridYLocations(:,1)',GridYLocations(end,1)+YSpacing];
HorizLinesY(2,:) = [GridYLocations(:,1)',GridYLocations(end,1)+YSpacing];
HorizLinesX(1,:) = repmat(0,1,size(GridYLocations,1)+1); %%% Same as zeros(1,size(GridYLocations,2)+1)
HorizLinesX(2,:) = repmat(TotalWidth,1,size(GridYLocations,1)+1);

Grid.XLocationOfLowestXSpot = XLocationOfLowestXSpot;
Grid.YLocationOfLowestYSpot = YLocationOfLowestYSpot;
Grid.XSpacing = XSpacing;
Grid.YSpacing = YSpacing;
Grid.Rows = Rows;
Grid.Columns = Columns;
Grid.VertLinesX = VertLinesX;
Grid.VertLinesY = VertLinesY;
Grid.HorizLinesX = HorizLinesX;
Grid.HorizLinesY = HorizLinesY;
Grid.SpotTable = SpotTable;
Grid.TotalHeight = TotalHeight;
Grid.TotalWidth = TotalWidth;
Grid.GridXLocations = GridXLocations;
Grid.GridYLocations = GridYLocations;
Grid.YLocations = YLocations;
Grid.XLocations = XLocations;
Grid.LeftOrRight = LeftOrRight;
Grid.TopOrBottom = TopOrBottom;